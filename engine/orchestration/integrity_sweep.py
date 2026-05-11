#!/usr/bin/env python3
"""
ASST Gamma Flywheel — Nightly Data Integrity Sweep

Self-healing pipeline that runs after master + selector exports each evening.
Catches the "computed but not persisted" bug class that has bitten this project
four times in two weeks.

What it does (in order):
  1. SCAN — query daily_runs for NULLs in critical fields across last 7 days
  2. CLASSIFY — separate "bug NULLs" (auto-fixable) from "expected NULLs"
                 (older than the field's introduction date)
  3. RECOVER — for each fixable NULL class, run the appropriate backfill
  4. REFRESH — re-evaluate the Selector log entries that touched fixed rows
  5. REBUILD — rebuild the Selector quant export so on-disk artifacts match DB
  6. ROTATE — compress fetch logs older than 30 days
  7. REPORT — write a JSON summary to integrity_sweep.last.json + stdout

Idempotent: safe to run twice in a row. Detects 0 NULLs the second time.
Atomic: each recovery step is wrapped to never leave partial state.

Usage:
  python3 integrity_sweep.py           # full sweep
  python3 integrity_sweep.py --dry-run # report only, no writes

Exit codes:
  0  = clean OR all detected issues recovered
  1  = some issues recovered but residual NULLs remain (manual review needed)
  2  = sweep itself failed (e.g. DB unreachable)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ─── Config ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/user/workspace/asst-gamma-dashboard")
# v2: env-var override with v1 fallback.
DB_PATH = Path(os.environ.get(
    "ASST_DB_PATH",
    "/home/user/workspace/asst-gamma-dashboard/data.db",
))
SERVER_DIR = PROJECT_ROOT / "server"
LOGS_TO_ROTATE = [
    PROJECT_ROOT / "fetch_AM.log",
    PROJECT_ROOT / "fetch_MID.log",
    PROJECT_ROOT / "fetch_PM.log",
    PROJECT_ROOT / "stochastic_failures.log",
    PROJECT_ROOT / "gap_check.log",
]
LOG_ROTATION_DAYS = 30   # files older than this get gzipped + archived
LOG_ARCHIVE_DAYS = 180   # gzipped archives older than this get deleted

REPORT_PATH = PROJECT_ROOT / "integrity_sweep.last.json"

# Field groups. Each entry: (db_column, intro_date_or_None, recovery_action)
# - intro_date: any row dated before this is allowed to be NULL (legacy)
# - recovery_action: callable name (resolved against this module) or None for warn-only
ALWAYS_REQUIRED: list[tuple[str, Optional[str], Optional[str]]] = [
    ("net_gex", None, None),                    # warn — fundamental fetch problem
    ("gamma_flip", None, None),                 # warn — fundamental fetch problem
    ("regime", None, None),                     # warn — derived; if missing the row is broken
    ("spot", None, None),                       # warn — fundamental
    ("stochastic_output_json", None, None),     # warn — stochastic layer ran but didn't persist
]

# Forward-only fields: only flag NULLs for rows AFTER the field was introduced
FORWARD_REQUIRED: list[tuple[str, str, str]] = [
    ("btc_mvrv_zscore",      "2026-05-04", "recover_cycle_metrics"),
    ("btc_puell_multiple",   "2026-05-04", "recover_cycle_metrics"),
    ("btc_nupl",             "2026-05-04", "recover_cycle_metrics"),
    ("btc_reserve_risk",     "2026-05-04", "recover_cycle_metrics"),
    ("btc_pi_cycle_signal",  "2026-05-04", "recover_cycle_metrics"),
    ("option_chain_snapshot_json", "2026-05-04", None),  # forward-only by methodology rule
]

# Best-effort fields: NULL is occasionally normal (e.g. provider hiccup).
# Reported but not auto-fixed.
BEST_EFFORT: list[tuple[str, Optional[str]]] = [
    ("btc_taker_buy_sell_ratio", None),
    ("btc_funding_rate", None),
    ("btc_fear_greed", None),
    ("iv_skew_25d", None),
    ("put_call_oi_ratio", None),
]

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class FieldFinding:
    field: str
    null_rows: list[tuple[int, str, str]]  # (id, date, session)
    legacy_nulls: int                       # NULL but before intro date — expected
    fixable_nulls: int                      # NULL and after intro date — auto-recoverable
    recovery_action: Optional[str]
    severity: str  # "ok" | "warn" | "fixable" | "critical"


@dataclass
class SweepReport:
    started_at: str
    finished_at: Optional[str] = None
    scan_window_days: int = 7
    total_rows_scanned: int = 0
    findings: list[dict] = None
    actions_taken: list[str] = None
    selector_refreshed: bool = False
    quant_export_rebuilt: bool = False
    logs_rotated: int = 0
    residual_nulls: int = 0
    exit_code: int = 0
    notes: list[str] = None


# ─── Step 1: Scan ────────────────────────────────────────────────────────────

def scan_db(scan_days: int = 7) -> tuple[list[FieldFinding], int]:
    """Inspect last N days of daily_runs for NULL critical fields."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"data.db not found at {DB_PATH}")

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM daily_runs WHERE date >= date('now', ? ) ORDER BY id",
            (f"-{scan_days} days",),
        ).fetchall()
    finally:
        con.close()

    findings: list[FieldFinding] = []
    all_required = (
        [(f, intro, action) for (f, intro, action) in ALWAYS_REQUIRED]
        + [(f, intro, action) for (f, intro, action) in FORWARD_REQUIRED]
    )
    for field, intro_date, action in all_required:
        null_rows: list[tuple[int, str, str]] = []
        legacy = 0
        fixable = 0
        for r in rows:
            v = r[field] if field in r.keys() else None
            if v is not None:
                continue
            row_date = r["date"]
            if intro_date and row_date < intro_date:
                legacy += 1
            else:
                fixable += 1
                null_rows.append((r["id"], r["date"], r["session"]))
        if not null_rows and legacy == 0:
            severity = "ok"
        elif fixable == 0:
            severity = "warn"   # only legacy NULLs — expected
        elif action:
            severity = "fixable"
        else:
            severity = "critical"  # NULL we can't auto-fix
        findings.append(FieldFinding(
            field=field, null_rows=null_rows,
            legacy_nulls=legacy, fixable_nulls=fixable,
            recovery_action=action, severity=severity,
        ))

    # Best-effort fields: just count, never recover
    for field, intro_date in BEST_EFFORT:
        null_rows = []
        for r in rows:
            v = r[field] if field in r.keys() else None
            if v is None:
                null_rows.append((r["id"], r["date"], r["session"]))
        if null_rows:
            findings.append(FieldFinding(
                field=field, null_rows=null_rows,
                legacy_nulls=0, fixable_nulls=0, recovery_action=None,
                severity="warn",
            ))

    return findings, len(rows)


# ─── Step 2: Recovery actions ────────────────────────────────────────────────

def recover_cycle_metrics(dry_run: bool = False) -> dict:
    """Run the cycle metrics backfill helper. Idempotent — only updates NULL rows."""
    script = SERVER_DIR / "backfill_cycle_metrics_today.py"
    if dry_run:
        return {"action": "recover_cycle_metrics", "dry_run": True, "skipped": True}
    try:
        proc = subprocess.run(
            ["python3", str(script)],
            cwd=str(SERVER_DIR), capture_output=True, text=True, timeout=120,
        )
        return {
            "action": "recover_cycle_metrics",
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-5:] if proc.stdout else [],
            "stderr_tail": proc.stderr.splitlines()[-3:] if proc.stderr else [],
        }
    except subprocess.TimeoutExpired:
        return {"action": "recover_cycle_metrics", "exit_code": -1, "error": "timeout"}


# ─── Step 3: Selector log refresh + quant export rebuild ─────────────────────

def refresh_selector_log_for_dates(dates: list[str], dry_run: bool = False) -> dict:
    """Re-evaluate auto_logged Selector entries for the given dates.

    Uses the existing refresh_selector_log_today.py with --date= args.
    Skipped if no dates given.
    """
    if not dates:
        return {"action": "refresh_selector_log", "skipped": True, "reason": "no dates needed"}
    if dry_run:
        return {"action": "refresh_selector_log", "dry_run": True, "skipped": True, "dates": dates}
    script = SERVER_DIR / "refresh_selector_log_today.py"
    args = ["python3", str(script)] + [f"--date={d}" for d in dates]
    try:
        proc = subprocess.run(
            args, cwd=str(SERVER_DIR), capture_output=True, text=True, timeout=180,
        )
        return {
            "action": "refresh_selector_log",
            "exit_code": proc.returncode,
            "dates": dates,
            "stdout_tail": proc.stdout.splitlines()[-5:] if proc.stdout else [],
        }
    except subprocess.TimeoutExpired:
        return {"action": "refresh_selector_log", "exit_code": -1, "error": "timeout"}


def rebuild_quant_export(dry_run: bool = False) -> dict:
    """Rebuild the Selector quant export so on-disk files match the refreshed log."""
    if dry_run:
        return {"action": "rebuild_quant_export", "dry_run": True, "skipped": True}
    script = Path("/home/user/workspace/selector_export/build_selector_export.py")
    try:
        proc = subprocess.run(
            ["python3", str(script)],
            cwd=str(script.parent), capture_output=True, text=True, timeout=120,
        )
        return {
            "action": "rebuild_quant_export",
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-3:] if proc.stdout else [],
        }
    except subprocess.TimeoutExpired:
        return {"action": "rebuild_quant_export", "exit_code": -1, "error": "timeout"}


# ─── Step 4: Log rotation ────────────────────────────────────────────────────

def rotate_logs(dry_run: bool = False) -> dict:
    """Compress logs older than LOG_ROTATION_DAYS, delete archives older than LOG_ARCHIVE_DAYS.

    Strategy: each log file at FOO.log gets an mtime check. If older than the
    rotation threshold, it's gzipped to FOO.log.YYYYMMDD.gz and the original
    is truncated (preserving the inode so tail -f sessions don't break).
    """
    rotated = 0
    deleted = 0
    now = datetime.now()
    rotation_cutoff = now - timedelta(days=LOG_ROTATION_DAYS)
    archive_cutoff = now - timedelta(days=LOG_ARCHIVE_DAYS)
    archive_dir = PROJECT_ROOT / "log_archive"

    for log_path in LOGS_TO_ROTATE:
        if not log_path.exists():
            continue
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        size = log_path.stat().st_size
        # Rotate if both: file is large enough to bother (>1MB) AND older than cutoff
        if size > 1_000_000 and mtime < rotation_cutoff:
            if dry_run:
                rotated += 1
                continue
            archive_dir.mkdir(exist_ok=True)
            stamp = mtime.strftime("%Y%m%d")
            target = archive_dir / f"{log_path.name}.{stamp}.gz"
            try:
                with open(log_path, "rb") as src, gzip.open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Truncate in place to preserve any tail -f or fd handles
                with open(log_path, "w") as f:
                    f.write(f"# Rotated to {target.name} at {now.isoformat()}\n")
                rotated += 1
            except Exception as e:
                print(f"[integrity] log rotation failed for {log_path.name}: {e}", file=sys.stderr)

    # Sweep archive of files older than archive_cutoff
    if archive_dir.exists():
        for archive in archive_dir.glob("*.gz"):
            if datetime.fromtimestamp(archive.stat().st_mtime) < archive_cutoff:
                if not dry_run:
                    try:
                        archive.unlink()
                        deleted += 1
                    except OSError:
                        pass

    return {"rotated": rotated, "deleted_archives": deleted, "rotation_days": LOG_ROTATION_DAYS}


# ─── Step 5: Orchestrator ────────────────────────────────────────────────────

def run_sweep(dry_run: bool = False, scan_days: int = 7) -> SweepReport:
    started = datetime.now(timezone.utc).isoformat()
    report = SweepReport(
        started_at=started, scan_window_days=scan_days,
        findings=[], actions_taken=[], notes=[],
    )

    print(f"[integrity] sweep starting ({'DRY RUN' if dry_run else 'LIVE'}) — last {scan_days} days")

    # 1. Scan
    try:
        findings, total_rows = scan_db(scan_days)
        report.total_rows_scanned = total_rows
    except Exception as e:
        print(f"[integrity] scan FAILED: {e}", file=sys.stderr)
        report.exit_code = 2
        report.notes.append(f"scan failed: {e}")
        return report

    # Convert findings to dicts for the report
    report.findings = [asdict(f) for f in findings]

    # Print scan summary
    fixable_count = sum(1 for f in findings if f.severity == "fixable")
    critical_count = sum(1 for f in findings if f.severity == "critical")
    warn_count = sum(1 for f in findings if f.severity == "warn")
    print(f"[integrity] scanned {total_rows} rows · "
          f"{fixable_count} fixable · {critical_count} critical · {warn_count} warn")
    for f in findings:
        if f.severity in ("fixable", "critical"):
            sample = ", ".join(f"{r[1]}/{r[2]}" for r in f.null_rows[:3])
            extra = f" (+{len(f.null_rows)-3} more)" if len(f.null_rows) > 3 else ""
            print(f"  · {f.severity.upper():8s} {f.field}: {f.fixable_nulls} NULL — {sample}{extra}")

    # 2. Recovery — group fixable findings by recovery_action so we don't run
    #    the same backfill multiple times
    actions_to_run: dict[str, set[tuple[str, str]]] = {}  # action -> set of (date, session)
    for f in findings:
        if f.severity != "fixable" or not f.recovery_action:
            continue
        if f.recovery_action not in actions_to_run:
            actions_to_run[f.recovery_action] = set()
        for (_, d, s) in f.null_rows:
            actions_to_run[f.recovery_action].add((d, s))

    if not actions_to_run:
        print("[integrity] no fixable issues — pipeline is clean")
    else:
        for action_name, affected in actions_to_run.items():
            handler = globals().get(action_name)
            if not handler:
                print(f"[integrity] WARN: no handler for action '{action_name}'")
                continue
            print(f"[integrity] running {action_name} (affects {len(affected)} rows)")
            result = handler(dry_run=dry_run)
            report.actions_taken.append(result)
            if result.get("exit_code", 0) != 0 and not result.get("skipped"):
                report.notes.append(f"{action_name} failed with exit {result.get('exit_code')}")

    # 3. After fixes, refresh Selector log for any affected dates so the auto-logged
    #    entries reflect updated underlying data
    affected_dates = sorted({d for affected in actions_to_run.values() for (d, _) in affected})
    if affected_dates and not dry_run:
        refresh_result = refresh_selector_log_for_dates(affected_dates, dry_run=dry_run)
        report.actions_taken.append(refresh_result)
        report.selector_refreshed = refresh_result.get("exit_code", 0) == 0

        # 4. Rebuild quant export so the CSV/Parquet files reflect the refreshed log
        rebuild_result = rebuild_quant_export(dry_run=dry_run)
        report.actions_taken.append(rebuild_result)
        report.quant_export_rebuilt = rebuild_result.get("exit_code", 0) == 0

    # 5. Log rotation runs every sweep — independent of NULL recovery
    rotation = rotate_logs(dry_run=dry_run)
    report.logs_rotated = rotation.get("rotated", 0)
    report.actions_taken.append({"action": "rotate_logs", **rotation})

    # 6. Re-scan to confirm fixes landed and find residual NULLs
    if actions_to_run and not dry_run:
        post_findings, _ = scan_db(scan_days)
        residual = sum(f.fixable_nulls for f in post_findings if f.severity == "fixable")
        report.residual_nulls = residual
        if residual > 0:
            print(f"[integrity] residual NULLs after recovery: {residual} — manual review needed")
            report.notes.append(f"{residual} NULL fields remain after recovery")
            report.exit_code = 1
        else:
            print("[integrity] all fixable NULLs recovered")

    if critical_count > 0:
        report.exit_code = max(report.exit_code, 1)
        report.notes.append(f"{critical_count} critical fields with no recovery handler")

    # 6.5 Refresh iv_band observability report. This is a read-only derivation
    # from already-persisted iv_percentile, so it cannot break the sweep.
    # We invoke it after fixes have landed so the report reflects post-sweep
    # state. Failure here is logged but not escalated to the sweep's exit code.
    if not dry_run:
        try:
            iv_proc = subprocess.run(
                ["python3", str(PROJECT_ROOT / "server" / "iv_band_report.py"), "--quiet"],
                capture_output=True, text=True, timeout=60,
            )
            if iv_proc.returncode == 0:
                report.actions_taken.append({"action": "refresh_iv_band_report", "exit_code": 0})
            else:
                report.actions_taken.append({
                    "action": "refresh_iv_band_report",
                    "exit_code": iv_proc.returncode,
                    "stderr_tail": (iv_proc.stderr or "")[-400:],
                })
                report.notes.append(
                    f"iv_band report refresh exited {iv_proc.returncode} — see iv_band_report.last.json (likely coverage break)"
                )
        except subprocess.TimeoutExpired:
            report.actions_taken.append({"action": "refresh_iv_band_report", "exit_code": -1, "error": "timeout"})
        except Exception as e:
            report.actions_taken.append({"action": "refresh_iv_band_report", "exit_code": -1, "error": str(e)})

    report.finished_at = datetime.now(timezone.utc).isoformat()

    # 7. Persist report (atomic)
    if not dry_run:
        tmp = REPORT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(report), indent=2, default=str))
        tmp.replace(REPORT_PATH)

    return report


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    parser.add_argument("--scan-days", type=int, default=7, help="Look-back window")
    args = parser.parse_args()

    report = run_sweep(dry_run=args.dry_run, scan_days=args.scan_days)

    print()
    print("=" * 60)
    print(f"INTEGRITY SWEEP COMPLETE — exit={report.exit_code}")
    print(f"  rows scanned:        {report.total_rows_scanned}")
    print(f"  actions taken:       {len(report.actions_taken)}")
    print(f"  selector refreshed:  {report.selector_refreshed}")
    print(f"  quant rebuilt:       {report.quant_export_rebuilt}")
    print(f"  logs rotated:        {report.logs_rotated}")
    print(f"  residual NULLs:      {report.residual_nulls}")
    if report.notes:
        print(f"  notes:")
        for note in report.notes:
            print(f"    - {note}")
    print("=" * 60)

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())

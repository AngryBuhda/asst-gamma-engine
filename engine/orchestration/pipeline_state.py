"""Pipeline state machine — single source of truth for recurring-task health.

Why this exists:
  Per the architecture review (2026-05-09), the system has 9 recurring tasks
  with implicit ordering (tiingo → master_export → selector_export →
  integrity_sweep). When one fails silently, downstream stages may proceed
  with stale data and never know. Symptom: integrity sweep clean, but the
  selector quant export was actually built off Friday's data because Monday's
  master export crashed.

  This module gives every cron a place to record:
    - "I started at T"
    - "I finished at T+Δ with status=ok|failed|skipped"
    - "My prerequisites were/weren't satisfied"

  Downstream stages check prerequisites BEFORE running. The /api/health
  endpoint reads from this table instead of file mtimes. Failures become
  a cascade we can see, not a silent compounding of errors.

Schema (sqlite, alongside daily_runs):

    CREATE TABLE IF NOT EXISTS pipeline_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      stage TEXT NOT NULL,        -- "fetch_am" | "tiingo_eod" | "master_export" | ...
      run_date TEXT NOT NULL,     -- ISO YYYY-MM-DD (NY trading date)
      started_at TEXT NOT NULL,   -- UTC ISO 8601
      finished_at TEXT,           -- NULL if still running
      status TEXT NOT NULL,       -- "running" | "ok" | "failed" | "skipped"
      error TEXT,                 -- error message when status=failed
      depends_on TEXT,            -- comma-sep prerequisite stage names
      depends_satisfied INTEGER,  -- 1=all prereqs ok, 0=missing/failed prereq
      metadata_json TEXT          -- stage-specific blob (counts, durations, ...)
    );
    CREATE INDEX IF NOT EXISTS pipeline_runs_stage_date
      ON pipeline_runs(stage, run_date);

Usage from a cron's command:

    python3 -c "
    import sys
    sys.path.insert(0, '/home/user/workspace/asst-gamma-dashboard/server')
    from pipeline_state import begin_stage, end_stage_ok, end_stage_failed
    run_id = begin_stage('fetch_am', depends_on=[])
    try:
        # ... actual work ...
        end_stage_ok(run_id, metadata={'rows_written': 1})
    except Exception as e:
        end_stage_failed(run_id, error=str(e))
        raise
    "

CLI usage (for shell-based crons that don't want Python wrapping):

    python3 server/pipeline_state.py begin --stage fetch_am
    # ... work ...
    python3 server/pipeline_state.py end --run-id $RUN_ID --status ok
"""
from __future__ import annotations
import json
import sqlite3
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import os as _os
# v2: env-var override with v1 fallback (see engine/persist.py).
DB_PATH = _os.environ.get(
    "ASST_DB_PATH",
    "/home/user/workspace/asst-gamma-dashboard/data.db",
)


def _conn():
    """Open a connection to the dashboard DB.

    We connect read-write here even when the rest of the dashboard uses
    read-only connections, because pipeline_state writes its own table.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_table():
    """Create the pipeline_runs table if it doesn't exist. Idempotent."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              stage TEXT NOT NULL,
              run_date TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              status TEXT NOT NULL,
              error TEXT,
              depends_on TEXT,
              depends_satisfied INTEGER,
              metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS pipeline_runs_stage_date
              ON pipeline_runs(stage, run_date);
        """)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    """NY trading date — uses local server clock. Matches daily_runs.date."""
    return date.today().isoformat()


def _check_prerequisites(depends_on: list[str], run_date: str) -> bool:
    """Return True iff every named prerequisite has status='ok' for run_date.

    A missing run record (cron didn't fire yet) counts as unsatisfied.
    A 'skipped' status counts as satisfied — explicit no-op is acceptable.
    """
    if not depends_on:
        return True
    placeholders = ",".join("?" * len(depends_on))
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT stage, status FROM pipeline_runs "
            f"WHERE run_date = ? AND stage IN ({placeholders}) "
            f"AND status IN ('ok', 'skipped') "
            f"ORDER BY started_at DESC",
            (run_date, *depends_on),
        ).fetchall()
    satisfied_stages = {r[0] for r in rows}
    return all(d in satisfied_stages for d in depends_on)


def begin_stage(
    stage: str,
    depends_on: Optional[list[str]] = None,
    run_date: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Record stage start. Returns the run_id used by end_stage_*.

    Checks prerequisites and records depends_satisfied so downstream
    troubleshooting can answer "did this stage start with a clean upstream?"
    even if it later crashed.
    """
    _ensure_table()
    rd = run_date or _today_iso()
    deps = depends_on or []
    deps_ok = _check_prerequisites(deps, rd)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO pipeline_runs "
            "(stage, run_date, started_at, status, depends_on, depends_satisfied, metadata_json) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (stage, rd, _now_iso(), ",".join(deps) if deps else None,
             1 if deps_ok else 0,
             json.dumps(metadata) if metadata else None),
        )
        return cur.lastrowid


def end_stage_ok(run_id: int, metadata: Optional[dict] = None):
    """Mark the stage successful. Optionally append/replace metadata."""
    with _conn() as conn:
        if metadata is not None:
            # Merge with existing metadata if present
            existing_row = conn.execute(
                "SELECT metadata_json FROM pipeline_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            existing = json.loads(existing_row[0]) if existing_row and existing_row[0] else {}
            existing.update(metadata)
            md_json = json.dumps(existing)
        else:
            md_json = None
        if md_json:
            conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = 'ok', metadata_json = ? WHERE id = ?",
                (_now_iso(), md_json, run_id),
            )
        else:
            conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = 'ok' WHERE id = ?",
                (_now_iso(), run_id),
            )


def end_stage_failed(run_id: int, error: str, metadata: Optional[dict] = None):
    """Mark the stage failed with an error message."""
    with _conn() as conn:
        md_json = json.dumps(metadata) if metadata else None
        conn.execute(
            "UPDATE pipeline_runs SET finished_at = ?, status = 'failed', error = ?, metadata_json = COALESCE(?, metadata_json) WHERE id = ?",
            (_now_iso(), error[:2000], md_json, run_id),
        )


def end_stage_skipped(run_id: int, reason: str):
    """Mark the stage as skipped (intentional no-op)."""
    with _conn() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET finished_at = ?, status = 'skipped', error = ? WHERE id = ?",
            (_now_iso(), reason[:500], run_id),
        )


def latest_status(stage: str, run_date: Optional[str] = None) -> Optional[dict]:
    """Return the most recent run record for a stage on a given date."""
    _ensure_table()
    rd = run_date or _today_iso()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE stage = ? AND run_date = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (stage, rd),
        ).fetchone()
    return dict(row) if row else None


def all_stages_today() -> list[dict]:
    """Latest record for every stage today — used by /api/health."""
    _ensure_table()
    rd = _today_iso()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_date = ? ORDER BY started_at DESC",
            (rd,),
        ).fetchall()
    # Dedup by stage, keeping only the latest per stage
    seen = {}
    for r in rows:
        if r["stage"] not in seen:
            seen[r["stage"]] = dict(r)
    return list(seen.values())


# ─── CLI for shell-based crons ───────────────────────────────────────────────

def _cli_main():
    """Tiny CLI so bash crons can call us without writing Python wrappers."""
    args = sys.argv[1:]
    if not args:
        print("usage: pipeline_state.py [begin|end|status|list] ...", file=sys.stderr)
        sys.exit(2)

    cmd = args[0]
    if cmd == "begin":
        # python3 pipeline_state.py begin --stage fetch_am [--depends a,b]
        stage = None
        depends = []
        i = 1
        while i < len(args):
            if args[i] == "--stage":
                stage = args[i + 1]; i += 2
            elif args[i] == "--depends":
                depends = [s for s in args[i + 1].split(",") if s]; i += 2
            else:
                i += 1
        if not stage:
            print("--stage required", file=sys.stderr); sys.exit(2)
        run_id = begin_stage(stage, depends_on=depends)
        print(run_id)  # caller captures this in $RUN_ID

    elif cmd == "end":
        # python3 pipeline_state.py end --run-id N --status ok|failed [--error MSG]
        run_id = None
        status = None
        error = None
        i = 1
        while i < len(args):
            if args[i] == "--run-id":
                run_id = int(args[i + 1]); i += 2
            elif args[i] == "--status":
                status = args[i + 1]; i += 2
            elif args[i] == "--error":
                error = args[i + 1]; i += 2
            else:
                i += 1
        if run_id is None or status is None:
            print("--run-id and --status required", file=sys.stderr); sys.exit(2)
        if status == "ok":
            end_stage_ok(run_id)
        elif status == "failed":
            end_stage_failed(run_id, error=error or "(no message)")
        elif status == "skipped":
            end_stage_skipped(run_id, reason=error or "(no reason)")
        else:
            print(f"unknown status: {status}", file=sys.stderr); sys.exit(2)

    elif cmd == "status":
        stage = args[1] if len(args) > 1 else None
        if not stage:
            print("usage: status STAGE", file=sys.stderr); sys.exit(2)
        s = latest_status(stage)
        print(json.dumps(s, indent=2) if s else "null")

    elif cmd == "list":
        rows = all_stages_today()
        print(json.dumps(rows, indent=2))

    else:
        print(f"unknown command: {cmd}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    _cli_main()

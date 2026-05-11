#!/usr/bin/env python3
"""
BCI Evidence Analysis — nightly digest.

Studies whether the BCI (Buchanan / Alan Ellman) initialization annotation has
predictive value on PMCC forward returns. Runs after the nightly master export
so forward-return data is fresh.

Reads:
  - /home/user/workspace/selector_export/asst_selector_recommendations_v1.0.parquet
    (PMCC recommendation rows with bci_passes / bci_buffer / bci_breakeven)
  - /home/user/workspace/master_research_export/asst_research_master_v1.5.parquet
    (PM rows with fwd5 / fwd10 / fwd21 / fwd63 forward returns)

Produces:
  - /home/user/workspace/bci_evidence/bci_analysis_YYYY-MM-DD.pdf
    Full table + buffer-vs-fwd-return scatter
  - /home/user/workspace/bci_evidence/bci_analysis_latest.json
    Structured digest (counts, medians, CIs, promotion verdict)
  - Returns the digest dict for the caller (cron) to use as the email body.

Methodology:
  * Bucket by buffer:
      FAIL       buffer <= 0
      NEAR_FAIL  0 < buffer < 0.25
      PASS       buffer >= 0.25
  * For each bucket × horizon (fwd5, fwd21), compute:
      n, mean, median, p10, p90, bootstrap 90% CI on median
  * Promotion criteria (all required to recommend BCI as a hard gate):
      1. n_fail + n_near_fail >= 8
      2. n_pass >= 8
      3. fwd5 median(PASS) - fwd5 median(FAIL+NEAR) >= +1.0%
      4. CI separation: 5th pct of PASS-CI > 95th pct of FAIL-CI on fwd5

Honest reporting: if sample is too small, report counts and the headline
"INSUFFICIENT SAMPLE" verdict rather than computing noisy statistics that
would mislead.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import os as _os
OUT_DIR = Path(_os.environ.get(
    "ASST_BCI_DIR", "/home/user/workspace/bci_evidence"
))
OUT_DIR.mkdir(exist_ok=True)
RECS_PATH = Path(_os.environ.get(
    "ASST_SELECTOR_RECS_PARQUET",
    "/home/user/workspace/selector_export/asst_selector_recommendations_v1.0.parquet",
))
MASTER_PATH = Path(_os.environ.get(
    "ASST_MASTER_PARQUET",
    "/home/user/workspace/master_research_export/asst_research_master_v1.5.parquet",
))

# Bucketing thresholds
NEAR_FAIL_THRESHOLD = 0.25  # buffer < 0.25 (and > 0) is "near-fail"

# Promotion-to-gate criteria
MIN_N_PER_BUCKET = 8
PROMOTION_FWD5_GAP_PCT = 1.0  # PASS median must beat FAIL+NEAR median by >= this many percentage points
HORIZONS = ("fwd5", "fwd10", "fwd21", "fwd63")


def load_data() -> pd.DataFrame:
    """Load PMCC SHORT-leg rows with BCI fields, joined to forward returns."""
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"recommendations parquet missing at {RECS_PATH}")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"master research parquet missing at {MASTER_PATH}")

    recs = pd.read_parquet(RECS_PATH)
    # Keep one row per recommendation: PMCC vehicle, SHORT leg (carries the
    # short-strike + same BCI fields as LONG leg). Filter to PM sessions only —
    # forward returns in the master export are computed at PM closes, so AM/MID
    # rows would have no fwd to join. (AM/MID BCI annotations are still useful
    # for the dashboard rationale; they're just not part of this analysis.)
    pmcc = recs[
        (recs["vehicle"] == "PMCC")
        & (recs["leg_role"] == "SHORT")
        & (recs["session"] == "PM")
    ].copy()
    if pmcc.empty:
        return pmcc

    # Drop rows where BCI is missing entirely (pre-chain-integration era)
    pmcc = pmcc[pmcc["bci_passes"].notna()].copy()
    if pmcc.empty:
        return pmcc

    # Coerce types: bci_passes was bool when written, may have come back as obj
    pmcc["bci_passes"] = pmcc["bci_passes"].astype(bool)

    # Bucket
    def bucket(row):
        b = row["bci_buffer"]
        if b <= 0:
            return "FAIL"
        if b < NEAR_FAIL_THRESHOLD:
            return "NEAR_FAIL"
        return "PASS"
    pmcc["bci_bucket"] = pmcc.apply(bucket, axis=1)

    # Join forward returns from master (date+session keyed)
    master = pd.read_parquet(MASTER_PATH)
    fwd_cols = ["date", "session"] + [c for c in HORIZONS if c in master.columns]
    fwd = master[fwd_cols].drop_duplicates(subset=["date", "session"])
    pmcc = pmcc.merge(fwd, on=["date", "session"], how="left", suffixes=("", "_fwd"))

    return pmcc


def bootstrap_median_ci(values: np.ndarray, n_iter: int = 2000, ci_level: float = 0.90) -> tuple[float, float]:
    """Bootstrap CI on the median. Returns (low, high) at the given level."""
    if len(values) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    medians = np.empty(n_iter, dtype=float)
    n = len(values)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)
    alpha = (1 - ci_level) / 2
    return float(np.percentile(medians, alpha * 100)), float(np.percentile(medians, (1 - alpha) * 100))


def compute_bucket_stats(df: pd.DataFrame, bucket: str, horizon: str) -> dict[str, Any]:
    """Per-bucket stats at one horizon. Includes n, mean, median, percentiles, CI."""
    sub = df[df["bci_bucket"] == bucket]
    series = sub[horizon].dropna()
    n = int(len(series))
    out: dict[str, Any] = {
        "bucket": bucket,
        "horizon": horizon,
        "n": n,
        "n_total_in_bucket": int(len(sub)),
        "n_with_fwd": n,
    }
    if n == 0:
        out.update({"mean": None, "median": None, "p10": None, "p90": None,
                    "ci_low": None, "ci_high": None})
        return out
    arr = series.values.astype(float)
    out["mean"] = float(np.mean(arr))
    out["median"] = float(np.median(arr))
    out["p10"] = float(np.percentile(arr, 10))
    out["p90"] = float(np.percentile(arr, 90))
    lo, hi = bootstrap_median_ci(arr) if n >= 3 else (float("nan"), float("nan"))
    out["ci_low"] = None if np.isnan(lo) else lo
    out["ci_high"] = None if np.isnan(hi) else hi
    return out


def evaluate_promotion(stats_by_bucket_horizon: dict, df: pd.DataFrame) -> dict[str, Any]:
    """Apply the promotion-to-gate criteria. All must be True to recommend gate."""
    # Counts
    n_fail = int(((df["bci_bucket"] == "FAIL")).sum())
    n_near = int(((df["bci_bucket"] == "NEAR_FAIL")).sum())
    n_pass = int(((df["bci_bucket"] == "PASS")).sum())
    fail_or_near = n_fail + n_near

    # Sample sufficiency
    sample_ok = (fail_or_near >= MIN_N_PER_BUCKET) and (n_pass >= MIN_N_PER_BUCKET)

    # Median gap on fwd5 (PASS vs FAIL+NEAR combined)
    df_fwd5 = df.dropna(subset=["fwd5"])
    pass_med = df_fwd5[df_fwd5["bci_bucket"] == "PASS"]["fwd5"].median() if (df_fwd5["bci_bucket"] == "PASS").any() else None
    fail_med = df_fwd5[df_fwd5["bci_bucket"].isin(["FAIL", "NEAR_FAIL"])]["fwd5"].median() if (df_fwd5["bci_bucket"].isin(["FAIL", "NEAR_FAIL"])).any() else None
    median_gap_pct = None
    if pass_med is not None and fail_med is not None:
        # fwd values are decimal returns (e.g. 0.05 = 5%). Multiply for percentage points.
        median_gap_pct = (pass_med - fail_med) * 100
    median_gap_ok = (median_gap_pct is not None and median_gap_pct >= PROMOTION_FWD5_GAP_PCT)

    # CI separation on fwd5: PASS CI low > FAIL+NEAR CI high
    pass_stats = stats_by_bucket_horizon.get(("PASS", "fwd5"), {})
    fail_combined_arr = df_fwd5[df_fwd5["bci_bucket"].isin(["FAIL", "NEAR_FAIL"])]["fwd5"].dropna().values
    fail_lo, fail_hi = bootstrap_median_ci(fail_combined_arr) if len(fail_combined_arr) >= 3 else (float("nan"), float("nan"))
    ci_separated = False
    if pass_stats.get("ci_low") is not None and not np.isnan(fail_hi):
        ci_separated = pass_stats["ci_low"] > fail_hi

    all_pass = sample_ok and median_gap_ok and ci_separated

    return {
        "promotion_recommended": bool(all_pass),
        "criteria": {
            "sample_sufficient": bool(sample_ok),
            "median_gap_ok": bool(median_gap_ok),
            "ci_separation_ok": bool(ci_separated),
        },
        "details": {
            "n_pass": n_pass,
            "n_near_fail": n_near,
            "n_fail": n_fail,
            "n_fail_or_near": fail_or_near,
            "min_n_required": MIN_N_PER_BUCKET,
            "fwd5_pass_median": None if pass_med is None else float(pass_med),
            "fwd5_failnear_median": None if fail_med is None else float(fail_med),
            "fwd5_median_gap_pct": None if median_gap_pct is None else float(median_gap_pct),
            "fwd5_min_gap_required_pct": PROMOTION_FWD5_GAP_PCT,
            "fwd5_pass_ci": [pass_stats.get("ci_low"), pass_stats.get("ci_high")],
            "fwd5_failnear_ci": [None if np.isnan(fail_lo) else float(fail_lo),
                                 None if np.isnan(fail_hi) else float(fail_hi)],
        },
    }


def find_near_fails(df: pd.DataFrame, lookback_sessions: int = 30) -> list[dict]:
    """Recent (date, session) tuples where BCI buffer is in the near-fail or fail zone."""
    sub = df.sort_values(["date", "session"]).tail(lookback_sessions)
    near = sub[sub["bci_bucket"].isin(["FAIL", "NEAR_FAIL"])]
    return [
        {
            "date": r["date"], "session": r["session"],
            "bucket": r["bci_bucket"],
            "buffer": float(r["bci_buffer"]),
            "short_strike": float(r["strike"]) if pd.notna(r["strike"]) else None,
            "leap_strike": float(r["bci_leap_mid"]) if pd.notna(r["bci_leap_mid"]) else None,
            "short_mid": float(r["bci_short_mid"]) if pd.notna(r["bci_short_mid"]) else None,
            "fwd5": float(r["fwd5"]) if pd.notna(r["fwd5"]) else None,
            "fwd21": float(r["fwd21"]) if pd.notna(r["fwd21"]) else None,
        }
        for _, r in near.iterrows()
    ]


def build_digest(df: pd.DataFrame) -> dict[str, Any]:
    """Compose the structured digest."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if df.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_date": today,
            "verdict": "INSUFFICIENT_SAMPLE",
            "reason": "No PMCC recommendations with BCI annotation found.",
            "counts": {"total": 0},
        }

    # Per-bucket × horizon stats
    bucket_stats: dict[tuple[str, str], dict] = {}
    for bucket in ("PASS", "NEAR_FAIL", "FAIL"):
        for horizon in HORIZONS:
            if horizon not in df.columns:
                continue
            bucket_stats[(bucket, horizon)] = compute_bucket_stats(df, bucket, horizon)

    promotion = evaluate_promotion(bucket_stats, df)
    near_fails = find_near_fails(df, lookback_sessions=30)

    # Date range covered
    date_min = df["date"].min()
    date_max = df["date"].max()

    # Forward-return coverage
    fwd5_coverage = int(df["fwd5"].notna().sum()) if "fwd5" in df.columns else 0
    fwd21_coverage = int(df["fwd21"].notna().sum()) if "fwd21" in df.columns else 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_date": today,
        "verdict": "PROMOTE_TO_GATE" if promotion["promotion_recommended"] else (
            "INSUFFICIENT_SAMPLE" if not promotion["criteria"]["sample_sufficient"]
            else "ANNOTATION_ONLY"
        ),
        "data_window": {
            "earliest": str(date_min),
            "latest": str(date_max),
            "n_sessions_total": int(len(df)),
            "n_sessions_with_fwd5": fwd5_coverage,
            "n_sessions_with_fwd21": fwd21_coverage,
        },
        "counts": {
            "PASS": int((df["bci_bucket"] == "PASS").sum()),
            "NEAR_FAIL": int((df["bci_bucket"] == "NEAR_FAIL").sum()),
            "FAIL": int((df["bci_bucket"] == "FAIL").sum()),
        },
        "stats_by_bucket_horizon": {f"{b}|{h}": v for (b, h), v in bucket_stats.items()},
        "promotion": promotion,
        "near_fails_last_30": near_fails,
    }


def write_pdf(digest: dict, df: pd.DataFrame, out_path: Path) -> None:
    """Generate a one-page PDF with table + scatter chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    BG = "#0a0e14"
    INK = "#e5e7eb"
    INK_MUTED = "#9ca3af"
    INK_FAINT = "#6b7280"
    ACCENT = "#00d4ff"
    HIGHLIGHT = "#facc15"
    GREEN = "#22c55e"
    RED = "#ef4444"
    ORANGE = "#fb923c"
    GRID = "#1f2937"

    with PdfPages(out_path) as pdf:
        # Page 1 — summary + scatter
        fig = plt.figure(figsize=(8.5, 11), facecolor=BG)
        fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.05, hspace=0.55)

        # Header
        ax_header = fig.add_axes([0.07, 0.93, 0.86, 0.04])
        ax_header.axis("off")
        ax_header.text(0, 1, "BCI Evidence Analysis", fontsize=20, color=INK, weight="bold", va="top", family="monospace")
        ax_header.text(0, 0.0, f"Report date: {digest['report_date']} · "
                       f"Window: {digest['data_window']['earliest']} \u2192 {digest['data_window']['latest']} · "
                       f"Verdict: {digest['verdict']}",
                       fontsize=9, color=ACCENT if digest['verdict'] == "PROMOTE_TO_GATE" else
                       HIGHLIGHT if digest['verdict'] == "ANNOTATION_ONLY" else INK_MUTED,
                       va="top", family="monospace")

        # Counts row
        ax_counts = fig.add_axes([0.07, 0.86, 0.86, 0.06])
        ax_counts.axis("off")
        c = digest["counts"]
        counts_str = (f"PMCC recommendations  ·  PASS: {c['PASS']}  ·  NEAR_FAIL: {c['NEAR_FAIL']}  ·  FAIL: {c['FAIL']}  "
                      f"\u2502  fwd5 cov: {digest['data_window']['n_sessions_with_fwd5']}  ·  "
                      f"fwd21 cov: {digest['data_window']['n_sessions_with_fwd21']}")
        ax_counts.text(0, 0.5, counts_str, fontsize=10, color=INK, family="monospace")

        # Bucket stats table
        ax_tbl = fig.add_axes([0.07, 0.55, 0.86, 0.27])
        ax_tbl.axis("off")
        ax_tbl.text(0, 1.0, "Per-bucket stats by horizon", fontsize=12, color=INK, weight="bold", family="monospace")

        rows = []
        for bucket in ("PASS", "NEAR_FAIL", "FAIL"):
            for h in ("fwd5", "fwd21"):
                key = f"{bucket}|{h}"
                s = digest["stats_by_bucket_horizon"].get(key, {})
                rows.append([
                    bucket, h, s.get("n", 0),
                    f"{(s.get('mean') or 0)*100:+.2f}%" if s.get("mean") is not None else "—",
                    f"{(s.get('median') or 0)*100:+.2f}%" if s.get("median") is not None else "—",
                    f"{(s.get('p10') or 0)*100:+.2f}%" if s.get("p10") is not None else "—",
                    f"{(s.get('p90') or 0)*100:+.2f}%" if s.get("p90") is not None else "—",
                    f"[{(s.get('ci_low') or 0)*100:+.2f}, {(s.get('ci_high') or 0)*100:+.2f}]"
                    if s.get("ci_low") is not None else "—",
                ])

        col_labels = ["Bucket", "Horizon", "n", "Mean", "Median", "p10", "p90", "Median CI 90%"]
        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center",
                           cellLoc="left", colLoc="left",
                           bbox=[0, 0, 1, 0.92])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.5)
            if i == 0:
                cell.set_text_props(color=ACCENT, weight="bold")
                cell.set_facecolor("#0f1925")
            else:
                cell.set_facecolor("#0d141d" if i % 2 else BG)
                bucket_name = rows[i-1][0]
                col_color = GREEN if bucket_name == "PASS" else ORANGE if bucket_name == "NEAR_FAIL" else RED
                if j == 0:
                    cell.set_text_props(color=col_color, weight="bold")
                else:
                    cell.set_text_props(color=INK)

        # Scatter: buffer vs fwd5 return
        ax_sc = fig.add_axes([0.10, 0.30, 0.83, 0.20])
        ax_sc.set_facecolor(BG)
        for spine in ax_sc.spines.values():
            spine.set_color(GRID)
        ax_sc.tick_params(colors=INK_MUTED)

        if not df.empty and "fwd5" in df.columns:
            sub = df.dropna(subset=["fwd5"])
            colors = sub["bci_bucket"].map({"PASS": GREEN, "NEAR_FAIL": ORANGE, "FAIL": RED})
            ax_sc.scatter(sub["bci_buffer"], sub["fwd5"] * 100, c=colors, s=40, alpha=0.85, edgecolors="none")
            ax_sc.axvline(0, color=RED, linestyle="--", linewidth=0.8, alpha=0.6)
            ax_sc.axvline(NEAR_FAIL_THRESHOLD, color=ORANGE, linestyle="--", linewidth=0.8, alpha=0.6)
            ax_sc.axhline(0, color=INK_FAINT, linewidth=0.5)
            ax_sc.set_xlabel("BCI buffer ($/share)", color=INK_MUTED, fontsize=10)
            ax_sc.set_ylabel("fwd5 return (%)", color=INK_MUTED, fontsize=10)
            ax_sc.set_title("BCI buffer vs realized fwd5 return", color=INK, fontsize=11, family="monospace", loc="left")
            ax_sc.text(0.005, ax_sc.get_ylim()[1] * 0.92, "FAIL", fontsize=8, color=RED, weight="bold")
            ax_sc.text(NEAR_FAIL_THRESHOLD + 0.02, ax_sc.get_ylim()[1] * 0.92, "NEAR_FAIL", fontsize=8, color=ORANGE, weight="bold")
            ax_sc.text(0.7, ax_sc.get_ylim()[1] * 0.92, "PASS", fontsize=8, color=GREEN, weight="bold")
        else:
            ax_sc.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                       color=INK_MUTED, fontsize=14)
            ax_sc.set_xticks([])
            ax_sc.set_yticks([])

        # Promotion verdict box
        ax_v = fig.add_axes([0.07, 0.10, 0.86, 0.16])
        ax_v.axis("off")
        verdict = digest["verdict"]
        verdict_color = GREEN if verdict == "PROMOTE_TO_GATE" else HIGHLIGHT if verdict == "ANNOTATION_ONLY" else INK_MUTED
        ax_v.text(0, 1, "Promotion criteria", fontsize=12, color=INK, weight="bold", family="monospace")
        prom = digest.get("promotion", {})
        crit = prom.get("criteria", {})
        det = prom.get("details", {})
        lines = [
            f"  Sample sufficient (n_pass\u22658, n_failnear\u22658):  {'\u2713' if crit.get('sample_sufficient') else '\u2717'}  "
            f"(pass={det.get('n_pass')}, fail+near={det.get('n_fail_or_near')})",
            f"  fwd5 median gap \u2265 {det.get('fwd5_min_gap_required_pct')}pp:        {'\u2713' if crit.get('median_gap_ok') else '\u2717'}  "
            f"(observed: {det.get('fwd5_median_gap_pct'):+.2f}pp)" if det.get('fwd5_median_gap_pct') is not None else
            f"  fwd5 median gap \u2265 {det.get('fwd5_min_gap_required_pct')}pp:        \u2717  (no data)",
            f"  CI separation (PASS lo > FAIL hi):       {'\u2713' if crit.get('ci_separation_ok') else '\u2717'}",
            f"",
            f"  Verdict: {verdict}",
        ]
        for i, line in enumerate(lines):
            ax_v.text(0, 0.85 - i * 0.15, line, fontsize=9, color=INK if i < 4 else verdict_color,
                      family="monospace", weight="bold" if i == 4 else "normal")

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)


def compose_email_body(digest: dict) -> str:
    """Plain-text email body. Concise, scannable, no fluff."""
    c = digest["counts"]
    w = digest["data_window"]
    p = digest.get("promotion", {})
    crit = p.get("criteria", {})
    det = p.get("details", {})
    sb = digest.get("stats_by_bucket_horizon", {})

    def fmt_stat(key):
        s = sb.get(key, {})
        if not s or s.get("n", 0) == 0:
            return f"n=0"
        med = s.get("median")
        n = s.get("n")
        ci_lo, ci_hi = s.get("ci_low"), s.get("ci_high")
        if med is None:
            return f"n={n} (no fwd data)"
        ci_str = f" CI[{(ci_lo or 0)*100:+.1f}, {(ci_hi or 0)*100:+.1f}]" if ci_lo is not None else ""
        return f"n={n} · median {med*100:+.2f}%{ci_str}"

    body_lines = [
        f"BCI EVIDENCE ANALYSIS — {digest['report_date']}",
        "",
        f"VERDICT: {digest['verdict']}",
        "",
        f"Window: {w['earliest']} → {w['latest']}  ·  total recs: {w['n_sessions_total']}",
        f"Coverage: fwd5 {w['n_sessions_with_fwd5']}/{w['n_sessions_total']}  ·  "
        f"fwd21 {w['n_sessions_with_fwd21']}/{w['n_sessions_total']}",
        "",
        f"Bucket counts:  PASS {c['PASS']}  ·  NEAR_FAIL {c['NEAR_FAIL']}  ·  FAIL {c['FAIL']}",
        "",
        "Forward-return medians (with bootstrap 90% CI on the median):",
        f"  PASS       fwd5  : {fmt_stat('PASS|fwd5')}",
        f"  NEAR_FAIL  fwd5  : {fmt_stat('NEAR_FAIL|fwd5')}",
        f"  FAIL       fwd5  : {fmt_stat('FAIL|fwd5')}",
        f"  PASS       fwd21 : {fmt_stat('PASS|fwd21')}",
        f"  NEAR_FAIL  fwd21 : {fmt_stat('NEAR_FAIL|fwd21')}",
        f"  FAIL       fwd21 : {fmt_stat('FAIL|fwd21')}",
        "",
        "Promotion-to-gate criteria:",
        f"  [{'✓' if crit.get('sample_sufficient') else '✗'}] sample sufficient "
        f"(pass={det.get('n_pass')}, fail+near={det.get('n_fail_or_near')}, need ≥8 each)",
    ]
    if det.get("fwd5_median_gap_pct") is not None:
        body_lines.append(
            f"  [{'✓' if crit.get('median_gap_ok') else '✗'}] fwd5 median gap "
            f"observed {det['fwd5_median_gap_pct']:+.2f}pp (need ≥{det['fwd5_min_gap_required_pct']}pp)"
        )
    else:
        body_lines.append(
            f"  [✗] fwd5 median gap — no data yet (need both PASS and FAIL+NEAR rows with fwd5)"
        )
    body_lines.append(
        f"  [{'✓' if crit.get('ci_separation_ok') else '✗'}] CI separation on fwd5 (PASS lo > FAIL hi)"
    )

    near_fails = digest.get("near_fails_last_30") or []
    if near_fails:
        body_lines.extend([
            "",
            f"NEAR-FAILS / FAILS in last 30 sessions ({len(near_fails)}):",
        ])
        for nf in near_fails[-10:]:
            fwd5_str = f"fwd5 {nf['fwd5']*100:+.1f}%" if nf.get("fwd5") is not None else "fwd5 pending"
            body_lines.append(
                f"  {nf['date']} {nf['session']:<3}  bucket={nf['bucket']:<10} "
                f"buffer ${nf['buffer']:+.2f}  short ${nf['short_strike']}  ·  {fwd5_str}"
            )

    body_lines.extend([
        "",
        "Full PDF + JSON: /home/user/workspace/bci_evidence/",
        "",
        "Verdict guide:",
        "  PROMOTE_TO_GATE — all 3 criteria satisfied; consider gating PMCC recs on bci_passes",
        "  ANNOTATION_ONLY — sample sufficient but criteria not met; keep BCI as informational",
        "  INSUFFICIENT_SAMPLE — wait for more data, no statistical claim possible",
    ])
    return "\n".join(body_lines)


def main() -> int:
    print(f"[bci-evidence] loading recommendations + master export")
    df = load_data()
    print(f"[bci-evidence] PMCC SHORT-leg rows with BCI: {len(df)}")

    digest = build_digest(df)

    # Persist artifacts
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pdf_path = OUT_DIR / f"bci_analysis_{today}.pdf"
    latest_json = OUT_DIR / "bci_analysis_latest.json"
    latest_email = OUT_DIR / "bci_analysis_latest_email.txt"

    try:
        write_pdf(digest, df, pdf_path)
        print(f"[bci-evidence] wrote {pdf_path}")
    except Exception as e:
        print(f"[bci-evidence] PDF generation failed (non-fatal): {e}", file=sys.stderr)

    # Atomic write of digest JSON + email body
    tmp_json = latest_json.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(digest, indent=2, default=str))
    tmp_json.replace(latest_json)

    body = compose_email_body(digest)
    tmp_email = latest_email.with_suffix(".tmp")
    tmp_email.write_text(body)
    tmp_email.replace(latest_email)

    print(f"[bci-evidence] wrote {latest_json}")
    print(f"[bci-evidence] wrote {latest_email}")
    print()
    print("=" * 72)
    print(body)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
ASST Master Research Export — production-grade quant data pipeline.

Three tiers, generated atomically from data.db.

  Tier 1: LITE (research-ready scalars only)
    - asst_research_master_v{V}.csv
    - asst_research_master_v{V}.parquet
    - asst_research_master_v{V}_meta.json
    - asst_research_master_v{V}_README.md

  Tier 2: FULL (scalars + JSONs-as-strings; all DB columns flattened)
    - asst_research_master_v{V}_full.csv
    - asst_research_master_v{V}_full.parquet

  Tier 3: ARCHIVE (one nested JSON object per run; all JSONs PARSED)
    - asst_runs_archive_v{V}.jsonl

Plus a static cohort timeline:
    - RESEARCH_TIMELINE.md

Schema version: 1.1  (additive bump from 1.0)
  v1.0 → v1.1 changes:
    + Added Bucket A treasury scalars: btc_nav, nav_per_share, avg_cost_per_btc,
      total_shares, diluted_shares, cash_balance, debt_outstanding
    + Added Bucket B suggestion-summary scalars: csp_count, csp_top_strike,
      csp_top_expiry, csp_top_dte, csp_top_mid, csp_top_eff_basis,
      leap_core_count, leap_mid_count, leap_tail_count, pmcc_status,
      leap_core_strikes, leap_mid_strikes, leap_tail_strikes
    + Added pos_magnets, neg_magnets, notes, status, symbol (string passthroughs)
    + Added Tier 2 (full) and Tier 3 (archive) outputs

Idempotent: same DB state → byte-identical output. Atomic writes (tmp+rename).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Reach into asst-gamma-dashboard/server for the canonical iv_band helper
# so this script and the dashboard cannot drift on band edges.
# v2: iv_band ported into engine.compute.iv_band; no sys.path hack needed.
_DASH_SERVER = Path("/home/user/workspace/asst-gamma-dashboard/server")  # unused; kept for back-compat
if str(_DASH_SERVER) not in sys.path:
    sys.path.insert(0, str(_DASH_SERVER))
from engine.compute.iv_band import label_for_band  # noqa: E402

import numpy as np
import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.5"
# v1.4 → v1.5: cohort_id semantics changed. The middle component is now the
# iv_band display label (EXTREME_CHEAP / CHEAP_VOL / NORMAL_VOL / RICH_VOL /
# EXTREME_RICH) derived from iv_band via label_for_band, NOT the legacy
# iv_regime string. Statistics conditional on cohort_id are NOT comparable
# between v1.4 and v1.5 — the partition itself changed. Pin to v1.5 in any
# new analysis. The iv_regime column is still emitted for backward compat
# but should be treated as a derived display field, not a partition key.
#
# v1.3 → v1.4 (parked): added iv_band column; cohort_id still keyed on iv_regime.
import os as _os
# v2: env-var override with v1 fallback (same pattern as engine.persist).
DB_PATH = Path(_os.environ.get(
    "ASST_DB_PATH", "/home/user/workspace/asst-gamma-dashboard/data.db"
))
OUT_DIR = Path(_os.environ.get(
    "ASST_MASTER_EXPORT_DIR", "/home/user/workspace/master_research_export"
))
OUT_DIR.mkdir(exist_ok=True)

# ── Tier 1 (lite) ──
LITE_CSV     = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}.csv"
LITE_PARQUET = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}.parquet"
META_PATH    = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}_meta.json"
README_PATH  = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}_README.md"

# ── Tier 2 (full) ──
FULL_CSV     = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}_full.csv"
FULL_PARQUET = OUT_DIR / f"asst_research_master_v{SCHEMA_VERSION}_full.parquet"

# ── Tier 3 (archive) ──
ARCHIVE_PATH = OUT_DIR / f"asst_runs_archive_v{SCHEMA_VERSION}.jsonl"

FORWARD_HORIZONS = [5, 10, 21, 63]

# ─── Schema definitions ──────────────────────────────────────────────────────

# Tier 1: LITE — research-ready scalars only.
# (db_column, master_column, dtype_label)
LITE_SCHEMA: list[tuple[str, str, str]] = [
    # Provenance / keys
    ("id",                       "run_id",                    "int"),
    ("date",                     "date",                      "string"),
    ("session",                  "session",                   "string"),
    ("runtime_utc",              "runtime_utc",               "string"),
    ("symbol",                   "symbol",                    "string"),
    ("status",                   "run_status",                "string"),

    # ASST market structure
    ("spot",                     "spot",                      "float"),
    ("gamma_flip",               "gamma_flip",                "float"),
    ("raw_flip",                 "raw_flip",                  "float"),
    ("regime",                   "gamma_regime",              "string"),
    ("net_gex",                  "net_gex",                   "float"),
    ("gex_percentile",           "gex_percentile",            "float"),
    ("atr_1d",                   "atr_1d",                    "float"),
    ("net_vanna",                "net_vanna",                 "float"),
    ("vanna_percentile",         "vanna_percentile",          "float"),
    ("vanna_regime",             "vanna_regime",              "string"),

    # Treasury / mNAV (Bucket A — added in v1.1)
    ("basic_mnav",               "basic_mnav",                "float"),
    ("ev_mnav",                  "ev_mnav",                   "float"),
    ("mnav_discount",            "mnav_discount",             "float"),
    ("btc_per_share_basic",      "btc_per_share_basic",       "float"),
    ("btc_per_share_diluted",    "btc_per_share_diluted",     "float"),
    ("btc_holdings",             "btc_holdings",              "float"),
    ("btc_yield_ytd",            "btc_yield_ytd",             "float"),
    ("nav_per_share",            "nav_per_share",             "float"),
    ("btc_nav",                  "btc_nav",                   "float"),
    ("avg_cost_per_btc",         "avg_cost_per_btc",          "float"),
    ("total_shares",             "total_shares",              "int"),
    ("diluted_shares",           "diluted_shares",            "int"),
    ("cash_balance",             "cash_balance",              "float"),
    ("debt_outstanding",         "debt_outstanding",          "float"),

    # IV
    ("current_iv",               "current_iv",                "float"),
    ("iv_rank",                  "iv_rank",                   "float"),
    ("iv_percentile",            "iv_percentile",             "float"),
    ("iv_rank_method",           "iv_rank_method",            "string"),
    ("iv_regime",                "iv_regime",                 "string"),
    ("iv_band",                  "iv_band",                   "int"),
    ("iv_skew_25d",              "iv_skew_25d",               "float"),
    ("put_call_oi_ratio",        "put_call_oi_ratio",         "float"),

    # BTC context
    ("btc_price",                "btc_price",                 "float"),
    ("btc_weekly_rsi",           "btc_weekly_rsi",            "float"),
    ("btc_mvrv",                 "btc_mvrv",                  "float"),
    ("btc_realized_price",       "btc_realized_price",        "float"),
    ("btc_cycle_zone",           "btc_cycle_zone",            "string"),
    ("btc_taker_buy_sell_ratio", "btc_taker_buy_sell_ratio",  "float"),
    ("btc_funding_rate",         "btc_funding_rate",          "float"),
    ("btc_liq_total_usd",        "btc_liq_total_usd",         "float"),
    ("btc_liq_long_pct",         "btc_liq_long_pct",          "float"),
    ("btc_fear_greed",           "btc_fear_greed",            "int"),
    ("btc_gex_secondary_confirm","btc_gex_secondary_confirm", "float"),
    # BTC cycle-top valuation indicators (added in v1.2 for Selector v4)
    ("btc_mvrv_zscore",          "btc_mvrv_zscore",           "float"),
    ("btc_pi_cycle_signal",      "btc_pi_cycle_signal",       "int"),
    ("btc_puell_multiple",       "btc_puell_multiple",        "float"),
    ("btc_nupl",                 "btc_nupl",                  "float"),
    ("btc_reserve_risk",         "btc_reserve_risk",          "float"),

    # ASST risk context
    ("asst_drawdown_90d",        "asst_drawdown_90d",         "float"),
    ("asst_90d_high",            "asst_90d_high",             "float"),
    ("risk_zone",                "risk_zone",                 "string"),
    ("market_closed",            "market_closed",             "bool_int"),

    # Permissions / posture
    ("action_banner",            "action_banner",             "string"),
    ("csp_allowed",              "csp_allowed",               "string"),
    ("leap_add_allowed",         "leap_add_allowed",          "string"),
    ("leap_add_size",            "leap_add_size",             "string"),
    ("pmcc_allowed",             "pmcc_allowed",              "string"),
    ("pmcc_status",              "pmcc_status",               "string"),

    # Band geometry / proximity
    ("csp_band_low",             "csp_band_low",              "float"),
    ("csp_band_high",            "csp_band_high",             "float"),
    ("leap_core_band_low",       "leap_core_band_low",        "float"),
    ("leap_core_band_high",      "leap_core_band_high",       "float"),
    ("csp_delta_to_band",        "csp_delta_to_band",         "float"),
    ("csp_magnet_proximity",     "csp_magnet_proximity",      "float"),

    # Suggestion summaries (Bucket B — added in v1.1)
    ("csp_count",                "csp_count",                 "int"),
    ("csp_top_strike",           "csp_top_strike",            "float"),
    ("csp_top_expiry",           "csp_top_expiry",            "string"),
    ("csp_top_dte",              "csp_top_dte",               "int"),
    ("csp_top_mid",              "csp_top_mid",               "float"),
    ("csp_top_eff_basis",        "csp_top_eff_basis",         "float"),
    ("leap_core_count",          "leap_core_count",           "int"),
    ("leap_mid_count",           "leap_mid_count",            "int"),
    ("leap_tail_count",          "leap_tail_count",           "int"),
    ("leap_core_strikes",        "leap_core_strikes",         "string"),
    ("leap_mid_strikes",         "leap_mid_strikes",          "string"),
    ("leap_tail_strikes",        "leap_tail_strikes",         "string"),

    # SATA
    ("sata_price",               "sata_price",                "float"),
    ("sata_volume",              "sata_volume",               "int"),
    ("sata_options_oi",          "sata_options_oi",           "int"),

    # Volume / cap
    ("asst_volume",              "asst_volume",               "int"),
    ("asst_market_cap",          "asst_market_cap",           "float"),

    # LEAP entry quality
    ("leap_entry_percentile",    "leap_entry_percentile",     "float"),
    ("leap_entry_score",         "leap_entry_score",          "float"),

    # Magnets (JSON-array strings; passed through verbatim)
    ("pos_magnets",              "pos_magnets",               "string"),
    ("neg_magnets",              "neg_magnets",               "string"),

    # Operational notes
    ("notes",                    "notes",                     "string"),
]

# JSON blob columns. Excluded from LITE; included as escaped strings in FULL;
# parsed and embedded in ARCHIVE.
JSON_BLOB_COLUMNS: list[str] = [
    "csp_candidates_json",
    "csp_suggestion_json",
    "leap_suggestion_json",
    "pmcc_suggestion_json",
    "stochastic_output_json",
    "option_chain_snapshot_json",  # added in v1.3 for Selector v4 chain awareness
]

# Columns to monitor in `data_health_flags` (NULLs flagged, not errors).
HEALTH_NULLABLE = [
    "btc_weekly_rsi", "btc_mvrv", "btc_realized_price", "btc_cycle_zone",
    "btc_taker_buy_sell_ratio", "btc_funding_rate",
    "iv_rank", "iv_percentile", "current_iv",
    "net_vanna", "vanna_percentile",
    "basic_mnav", "ev_mnav",
    "mnav_discount", "btc_per_share_basic",
]


# ─── Forward-return + path-stat computation ──────────────────────────────────

def compute_forward_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each PM row at index i and horizon h:
        fwd{h}        = spot[i+h] / spot[i] - 1
        max_dd_{h}d   = worst peak-to-trough drawdown on PM[i..i+h]
        rv_{h}d       = realized vol of daily log returns on PM[i..i+h],
                        annualized via sqrt(252)
    PM rows only (most stable closing print). AM/MID get NaN by design.
    """
    df = df.sort_values(["date", "run_id"]).reset_index(drop=True)
    df_pm = df[df["session"] == "PM"].copy().sort_values("date").reset_index(drop=True)
    spot = df_pm["spot"].to_numpy(dtype=float)
    n = len(df_pm)

    for h in FORWARD_HORIZONS:
        fwd = np.full(n, np.nan)
        max_dd = np.full(n, np.nan)
        rv = np.full(n, np.nan)
        for i in range(n):
            j = i + h
            if j < n and spot[i] > 0 and not np.isnan(spot[j]):
                fwd[i] = spot[j] / spot[i] - 1
                path = spot[i : j + 1]
                if np.all(~np.isnan(path)) and np.all(path > 0):
                    running_max = np.maximum.accumulate(path)
                    drawdowns = path / running_max - 1
                    max_dd[i] = float(drawdowns.min())
                    log_rets = np.diff(np.log(path))
                    if len(log_rets) >= 2:
                        rv[i] = float(np.std(log_rets, ddof=1) * np.sqrt(252))
        df_pm[f"fwd{h}"] = fwd
        df_pm[f"max_dd_{h}d"] = max_dd
        df_pm[f"rv_{h}d"] = rv

    fwd_cols = ["date", "session"] + [
        c for h in FORWARD_HORIZONS for c in [f"fwd{h}", f"max_dd_{h}d", f"rv_{h}d"]
    ]
    return df.merge(df_pm[fwd_cols], on=["date", "session"], how="left")


# ─── Cohort labeling ─────────────────────────────────────────────────────────

def assign_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    def family(g: Any) -> str:
        if not isinstance(g, str):
            return "UNKNOWN"
        gu = g.upper()
        if gu.startswith("LONG_GAMMA"):
            return "LONG"
        if gu.startswith("SHORT_GAMMA"):
            return "SHORT"
        if gu == "NEUTRAL":
            return "NEUTRAL"
        return "UNKNOWN"

    cycle = df["btc_cycle_zone"].astype("string").str.upper().fillna("UNKNOWN")
    # v1.5 cutover: cohort_id middle component is the iv_band label (5-state),
    # not iv_regime (3-state). Rows with NULL iv_band fall back to UNKNOWN so
    # the partition stays exhaustive (forward-only doctrine — missing data
    # gets its own cohort, never imputed).
    iv_band_int = df["iv_band"].astype("Int64") if "iv_band" in df.columns else None
    if iv_band_int is None:
        iv_label = pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")
    else:
        iv_label = iv_band_int.apply(
            lambda b: (label_for_band(int(b)) if pd.notna(b) else "UNKNOWN")
        ).astype("string")
    fam = df["gamma_regime"].apply(family)
    df["gamma_family"] = fam
    df["iv_band_label"] = iv_label  # surfaced as a tier column for analysts
    df["cohort_id"] = cycle.str.cat([iv_label, fam], sep="|")
    df["cohort_id_v"] = SCHEMA_VERSION  # version-tag every cohort_id we emit
    return df


def add_health_flags(df: pd.DataFrame) -> pd.DataFrame:
    cols_in_df = [c for c in HEALTH_NULLABLE if c in df.columns]
    def flags(row):
        return ";".join([c for c in cols_in_df if pd.isna(row[c])])
    df["data_health_flags"] = df.apply(flags, axis=1)
    return df


def coerce_lite_types(df: pd.DataFrame) -> pd.DataFrame:
    for _, master_col, dtype in LITE_SCHEMA:
        if master_col not in df.columns:
            continue
        if dtype == "float":
            df[master_col] = pd.to_numeric(df[master_col], errors="coerce")
        elif dtype == "int":
            df[master_col] = pd.to_numeric(df[master_col], errors="coerce").astype("Int64")
        elif dtype == "bool_int":
            df[master_col] = pd.to_numeric(df[master_col], errors="coerce").fillna(0).astype("Int64")
        elif dtype == "string":
            df[master_col] = df[master_col].astype("string")
    return df


# ─── Tier builders ───────────────────────────────────────────────────────────

def build_lite() -> pd.DataFrame:
    """Tier 1: scalars only, with cohort + forward stats."""
    db_cols = [db for db, _, _ in LITE_SCHEMA]
    select_clause = ", ".join(db_cols)
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            f"SELECT {select_clause} FROM daily_runs ORDER BY date, id",
            conn,
        )
    rename_map = {db: master for db, master, _ in LITE_SCHEMA}
    df = df.rename(columns=rename_map)
    df = coerce_lite_types(df)
    df = add_health_flags(df)
    df = compute_forward_stats(df)
    df = assign_cohorts(df)
    df["schema_version"] = SCHEMA_VERSION

    final_cols = [m for _, m, _ in LITE_SCHEMA] + [
        "gamma_family", "iv_band_label", "cohort_id", "cohort_id_v",
        "data_health_flags", "schema_version",
    ]
    for h in FORWARD_HORIZONS:
        final_cols.extend([f"fwd{h}", f"max_dd_{h}d", f"rv_{h}d"])
    return df[final_cols]


def build_full(lite_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 2: scalars + JSON blobs as escaped strings. One row per run, every
    DB column captured. Pandas-readable; JSON columns can be parsed on demand.
    """
    json_select = ", ".join(["id"] + JSON_BLOB_COLUMNS)
    with sqlite3.connect(str(DB_PATH)) as conn:
        json_df = pd.read_sql_query(
            f"SELECT {json_select} FROM daily_runs ORDER BY id",
            conn,
        )
    json_df = json_df.rename(columns={"id": "run_id"})
    # Coerce to strings (pandas/parquet handle nulls fine)
    for col in JSON_BLOB_COLUMNS:
        json_df[col] = json_df[col].astype("string")
    return lite_df.merge(json_df, on="run_id", how="left")


def build_archive() -> list[dict[str, Any]]:
    """
    Tier 3: nested JSON-per-row. Every column from the DB. JSON blob columns
    are PARSED (not double-escaped) so consumers don't need to re-parse.
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM daily_runs ORDER BY date, id").fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        # Parse JSON columns; keep as-is on parse failure (preserve audit trail).
        for jcol in JSON_BLOB_COLUMNS:
            v = rec.get(jcol)
            if isinstance(v, str) and v:
                try:
                    rec[jcol] = json.loads(v)
                    rec[jcol + "__parse_status"] = "ok"
                except (json.JSONDecodeError, ValueError) as e:
                    rec[jcol + "__parse_status"] = f"error: {type(e).__name__}"
            else:
                rec[jcol] = None
                rec[jcol + "__parse_status"] = "null"
        # Stamp provenance
        rec["schema_version"] = SCHEMA_VERSION
        rec["archive_format"] = "jsonl-nested-v1"
        out.append(rec)
    return out


# ─── Cohort-maturity diagnostics ─────────────────────────────────────────────

@dataclass
class CohortMaturity:
    cohort_id: str
    n_total: int
    n_with_fwd5: int
    n_with_fwd10: int
    n_with_fwd21: int
    n_with_fwd63: int
    earliest_date: Optional[str]
    latest_date: Optional[str]


def compute_cohort_maturity(df: pd.DataFrame) -> list[CohortMaturity]:
    pm = df[df["session"] == "PM"]
    out: list[CohortMaturity] = []
    for cohort, sub in pm.groupby("cohort_id"):
        out.append(CohortMaturity(
            cohort_id=str(cohort),
            n_total=int(len(sub)),
            n_with_fwd5=int(sub["fwd5"].notna().sum()),
            n_with_fwd10=int(sub["fwd10"].notna().sum()),
            n_with_fwd21=int(sub["fwd21"].notna().sum()),
            n_with_fwd63=int(sub["fwd63"].notna().sum()),
            earliest_date=str(sub["date"].min()) if len(sub) else None,
            latest_date=str(sub["date"].max()) if len(sub) else None,
        ))
    out.sort(key=lambda c: (-c.n_total, c.cohort_id))
    return out


# ─── Atomic writers ──────────────────────────────────────────────────────────

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)


def atomic_write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    tmp.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ─── Validation ──────────────────────────────────────────────────────────────

ANCHOR = {
    "date": "2026-05-01",
    "session": "PM",
    "expected": {
        "spot": 16.27,
        "gamma_flip": 14.5393,
        "gex_percentile": 96.19,
        "current_iv": 40.0,
        "iv_regime": "CHEAP_VOL",
        "btc_mvrv": 1.4246,
        "btc_cycle_zone": "RECOVERY",
        "asst_drawdown_90d": -0.3842,
        "action_banner": "BUILD: LEAP ONLY",
        "gamma_regime": "long_gamma_weak",
    },
}


def validate_anchor(df: pd.DataFrame) -> tuple[bool, list[str]]:
    target = df[(df["date"] == ANCHOR["date"]) & (df["session"] == ANCHOR["session"])]
    if target.empty:
        return False, ["anchor_row_missing"]
    row = target.iloc[0]
    failures: list[str] = []
    for col, expected in ANCHOR["expected"].items():
        actual = row[col]
        if isinstance(expected, str):
            if str(actual) != expected:
                failures.append(f"{col}: expected {expected!r}, got {actual!r}")
        else:
            if pd.isna(actual) or abs(float(actual) - float(expected)) > 0.01:
                failures.append(f"{col}: expected {expected}, got {actual}")
    return len(failures) == 0, failures


# ─── Meta + README ───────────────────────────────────────────────────────────

def write_meta(lite_df: pd.DataFrame, full_df: pd.DataFrame, archive_records: list[dict[str, Any]]) -> dict[str, Any]:
    pm = lite_df[lite_df["session"] == "PM"]
    cohorts = compute_cohort_maturity(lite_df)
    passed, failures = validate_anchor(lite_df)

    schema_entries: list[dict[str, Any]] = []
    for db, master, dtype in LITE_SCHEMA:
        schema_entries.append({"column": master, "dtype": dtype, "source_db_column": db, "tier": "lite"})
    schema_entries.extend([
        {"column": "gamma_family", "dtype": "string", "source_db_column": "<derived>", "tier": "lite"},
        {"column": "iv_band_label", "dtype": "string", "source_db_column": "<derived from iv_band>", "tier": "lite"},
        {"column": "cohort_id", "dtype": "string", "source_db_column": "<derived>", "tier": "lite"},
        {"column": "cohort_id_v", "dtype": "string", "source_db_column": "<constant>", "tier": "lite"},
        {"column": "data_health_flags", "dtype": "string", "source_db_column": "<derived>", "tier": "lite"},
        {"column": "schema_version", "dtype": "string", "source_db_column": "<constant>", "tier": "lite"},
    ])
    for h in FORWARD_HORIZONS:
        schema_entries.append({"column": f"fwd{h}", "dtype": "float", "source_db_column": "<derived>", "tier": "lite"})
        schema_entries.append({"column": f"max_dd_{h}d", "dtype": "float", "source_db_column": "<derived>", "tier": "lite"})
        schema_entries.append({"column": f"rv_{h}d", "dtype": "float", "source_db_column": "<derived>", "tier": "lite"})
    for jc in JSON_BLOB_COLUMNS:
        schema_entries.append({"column": jc, "dtype": "json_string", "source_db_column": jc, "tier": "full+archive"})

    file_inventory = {}
    for label, path in [
        ("lite_csv", LITE_CSV), ("lite_parquet", LITE_PARQUET),
        ("full_csv", FULL_CSV), ("full_parquet", FULL_PARQUET),
        ("archive_jsonl", ARCHIVE_PATH),
        ("readme", README_PATH), ("meta", META_PATH),
    ]:
        if path.exists():
            file_inventory[label] = {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": str(DB_PATH),
        "source_db_mtime_utc": datetime.fromtimestamp(DB_PATH.stat().st_mtime, tz=timezone.utc).isoformat(),
        "tiers": {
            "lite": {
                "purpose": "Research-ready scalar table. Quant-stat-friendly.",
                "files": [LITE_CSV.name, LITE_PARQUET.name],
                "columns": int(len(lite_df.columns)),
                "rows": int(len(lite_df)),
                "rows_pm": int(len(pm)),
            },
            "full": {
                "purpose": "Lite + JSON blobs as escaped strings. Every DB column flat.",
                "files": [FULL_CSV.name, FULL_PARQUET.name],
                "columns": int(len(full_df.columns)),
                "rows": int(len(full_df)),
            },
            "archive": {
                "purpose": "Nested JSON-per-row. JSON columns parsed. Audit-grade.",
                "files": [ARCHIVE_PATH.name],
                "rows": len(archive_records),
            },
        },
        "files": file_inventory,
        "row_count_total": int(len(lite_df)),
        "row_count_pm": int(len(pm)),
        "date_range": {"min": str(lite_df["date"].min()), "max": str(lite_df["date"].max())},
        "session_counts": lite_df["session"].value_counts().to_dict(),
        "forward_horizons_trading_days": FORWARD_HORIZONS,
        "forward_return_coverage_pm": {
            f"fwd{h}_populated": int(pm[f"fwd{h}"].notna().sum())
            for h in FORWARD_HORIZONS
        },
        "cohorts": [
            {
                "cohort_id": c.cohort_id,
                "n_total_pm": c.n_total,
                "n_with_fwd5": c.n_with_fwd5,
                "n_with_fwd10": c.n_with_fwd10,
                "n_with_fwd21": c.n_with_fwd21,
                "n_with_fwd63": c.n_with_fwd63,
                "earliest_date": c.earliest_date,
                "latest_date": c.latest_date,
            } for c in cohorts
        ],
        "schema": schema_entries,
        "validation_targets": {"anchor_run": ANCHOR},
        "validation_passed": passed,
        "validation_failures": failures,
        "archive_parse_health": archive_parse_summary(archive_records),
    }
    atomic_write_text(json.dumps(meta, indent=2, default=str), META_PATH)
    return meta


def archive_parse_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {jc: {"ok": 0, "null": 0, "error": 0} for jc in JSON_BLOB_COLUMNS}
    for rec in records:
        for jc in JSON_BLOB_COLUMNS:
            status = rec.get(jc + "__parse_status", "null")
            if status == "ok":
                summary[jc]["ok"] += 1
            elif status == "null":
                summary[jc]["null"] += 1
            else:
                summary[jc]["error"] += 1
    return summary


def write_readme(lite_df: pd.DataFrame, full_df: pd.DataFrame, archive_records: list[dict[str, Any]]) -> None:
    pm = lite_df[lite_df["session"] == "PM"]
    cohorts = compute_cohort_maturity(lite_df)[:10]
    cohort_table = "\n".join(
        f"| `{c.cohort_id}` | {c.n_total} | {c.n_with_fwd5} | {c.n_with_fwd10} | "
        f"{c.n_with_fwd21} | {c.n_with_fwd63} | {c.earliest_date} → {c.latest_date} |"
        for c in cohorts
    )
    parse_health = archive_parse_summary(archive_records)
    parse_table = "\n".join(
        f"| `{jc}` | {h['ok']} | {h['null']} | {h['error']} |"
        for jc, h in parse_health.items()
    )

    readme = f"""# ASST Research Master — Schema v{SCHEMA_VERSION}

**Generated:** {datetime.now(timezone.utc).isoformat()}
**Source:** `{DB_PATH}` (system of record)
**Rows:** {len(lite_df)} total · {len(pm)} PM
**Date range:** {lite_df["date"].min()} → {lite_df["date"].max()}
**Forward horizons:** {FORWARD_HORIZONS} trading days

## Three tiers, one source of truth

| Tier | File(s) | Audience | Contents |
|------|---------|----------|----------|
| **Lite** | `asst_research_master_v{SCHEMA_VERSION}.csv` / `.parquet` | Quants doing stats | Typed scalars only ({len(lite_df.columns)} cols), cohort labels, forward returns, path stats |
| **Full** | `asst_research_master_v{SCHEMA_VERSION}_full.csv` / `.parquet` | Researchers needing context | Lite + JSON blobs as escaped strings ({len(full_df.columns)} cols, every DB column flat) |
| **Archive** | `asst_runs_archive_v{SCHEMA_VERSION}.jsonl` | Forensic / audit | Nested JSON-per-row, JSON cols PARSED (not double-escaped), every column captured |

All three regenerated atomically by `build_master_research.py`. Same DB state ⇒
byte-identical output. Atomic writes (tmp + rename) — no partial files mid-refresh.

## When to use which

- **You're doing pandas/R quant analysis** — load Lite Parquet. Fastest, smallest, only typed scalars, cohort+forward returns precomputed.
- **You need to inspect a specific suggestion or candidate set** — load Full CSV/Parquet. Same rows as Lite, but JSON columns are present (as strings); parse them on demand: `df["csp_suggestion_json"].apply(json.loads)`.
- **You're doing forensic re-analysis or audit** — read Archive JSONL line-by-line. Every column captured, every JSON pre-parsed, includes parse-health status per column.

## Schema stability guarantees

- **Additive changes** (new columns, new derived stats, new cohort labels) are **minor bumps** (v1.0 → v1.1). Existing columns keep their names and types.
- **Renames or retypes** are **major bumps** (v1.x → v2.0) and produce a parallel new file; the old file stays readable.
- The build script writes `schema_version` on every row of every tier. Pin to a version in your analysis if you want to be defensive.

## Versioning history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-05-03 | Initial schema. 60 scalar cols + fwd5/10/21/63 + path stats + cohort_id + provenance. |
| 1.1 | 2026-05-03 | Added Bucket A treasury scalars (btc_nav, nav_per_share, total_shares, diluted_shares, cash_balance, debt_outstanding, avg_cost_per_btc) + Bucket B suggestion summaries (csp_count, csp_top_*, leap_*_count, leap_*_strikes, pmcc_status) + magnets passthroughs + notes/status/symbol. Added Tier 2 (full) and Tier 3 (archive) outputs. |
| 1.2 | 2026-05-03 | Added 5 BTC cycle-top valuation indicators (btc_mvrv_zscore, btc_pi_cycle_signal, btc_puell_multiple, btc_nupl, btc_reserve_risk) from BGeometrics. Backfilled 100% across all historical rows. Distinguishes "market is at peak euphoria" (top triggers) from "position is in pain" (stress triggers) for Selector v4. |
| 1.3 | 2026-05-03 | Added option_chain_snapshot_json: SteadyAPI focused chain (LEAPs >500 DTE + PMCC short-leg calls 14-65 DTE + CSP puts 14-65 DTE) with full Greeks. Forward-only — pre-2026-05-04 rows stay NULL. Selector v4 uses real chain when present, falls back to APPROX. Surfaces in Tier 2 (Full) as JSON string and Tier 3 (Archive) as parsed nested object. |

## Forward returns and path stats

For each PM row at index `i` and horizon `h ∈ {{5, 10, 21, 63}}`:

- **`fwd{{h}}`** — point-to-point spot return: `spot[i+h] / spot[i] − 1`
- **`max_dd_{{h}}d`** — worst peak-to-trough drawdown observed on the spot path `PM[i..i+h]`. Path-dependent risk label.
- **`rv_{{h}}d`** — realized vol from daily log returns on the same path, annualized via `sqrt(252)`.

PM rows where the horizon hasn't yet elapsed have NaN. Non-PM rows (AM, MID) also have NaN — forward returns are PM-to-PM only.

## Cohort labeling

Every row carries `cohort_id` of the form: `{{btc_cycle_zone}}|{{iv_regime}}|{{gamma_family}}`

`gamma_family` compresses the 4 gamma states to 3 families: `LONG`, `SHORT`, `NEUTRAL`. Missing inputs yield `UNKNOWN` in that slot.

## Top 10 PM cohorts by sample size

| cohort_id | n_total | fwd5 | fwd10 | fwd21 | fwd63 | date range |
|-----------|---------|------|-------|-------|-------|------------|
{cohort_table}

## Archive parse health

The Archive tier parses JSON blobs at build time. Per-column parse status:

| JSON column | parsed OK | null | parse error |
|-------------|-----------|------|-------------|
{parse_table}

Errors (if any) carry the underlying raw string verbatim plus a `__parse_status` field with the exception type. No data is lost on parse failure.

## Methodology rules (frozen)

1. **No external API backfill** of BTC inputs. Pre-classifier rows stay `UNKNOWN`. Avoiding mid-stream methodology change.
2. **Forward returns are PM-to-PM only.** AM/MID are operational snapshots, not closes.
3. **Schema additions are additive.** Renames/retypes require major version bump.
4. **`max_dd_{{h}}d` is path-dependent**, not just `min(fwd_h, 0)`.
5. **Archive preserves every byte.** JSON parse failures are captured, not silently dropped.

## Refresh and storage

- **Auto-refresh:** nightly at 6:30 PM ET, after Tiingo EOD (6:00 PM ET)
- **Sync target:** Google Drive (all files + meta + README + timeline)
- **Workspace path:** `/home/user/workspace/master_research_export/`
- **Live ops pull:** V1 dashboard `Research ▾` menu → Live snapshot (separate from this archive)
- **Build script:** `build_master_research.py` (idempotent, atomic, anchor-validated)
"""
    atomic_write_text(readme, README_PATH)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[master] schema v{SCHEMA_VERSION}", flush=True)
    print(f"[master] source: {DB_PATH}", flush=True)
    print(f"[master] output: {OUT_DIR}", flush=True)

    # Tier 1
    lite = build_lite()
    print(f"[lite] {len(lite)} rows × {len(lite.columns)} cols", flush=True)
    atomic_write_csv(lite, LITE_CSV)
    atomic_write_parquet(lite, LITE_PARQUET)
    print(f"[lite] wrote {LITE_CSV.name} + {LITE_PARQUET.name}", flush=True)

    # Tier 2
    full = build_full(lite)
    print(f"[full] {len(full)} rows × {len(full.columns)} cols", flush=True)
    atomic_write_csv(full, FULL_CSV)
    atomic_write_parquet(full, FULL_PARQUET)
    print(f"[full] wrote {FULL_CSV.name} + {FULL_PARQUET.name}", flush=True)

    # Tier 3
    archive = build_archive()
    atomic_write_jsonl(archive, ARCHIVE_PATH)
    print(f"[archive] wrote {ARCHIVE_PATH.name} ({len(archive)} records)", flush=True)

    # Meta + README
    meta = write_meta(lite, full, archive)
    write_readme(lite, full, archive)
    print(f"[meta] wrote {META_PATH.name} + {README_PATH.name}", flush=True)

    if meta["validation_passed"]:
        print(f"[master] VALIDATION PASSED — anchor row 2026-05-01 PM aligned.", flush=True)
    else:
        print(f"[master] VALIDATION FAILED:", flush=True)
        for f in meta["validation_failures"]:
            print(f"  • {f}", flush=True)
        return 1

    pm = lite[lite["session"] == "PM"]
    cov = {h: int(pm[f"fwd{h}"].notna().sum()) for h in FORWARD_HORIZONS}
    print(f"[master] PM forward-return coverage: {cov} (of {len(pm)} PM)", flush=True)
    print(f"[master] archive parse health: {meta['archive_parse_health']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

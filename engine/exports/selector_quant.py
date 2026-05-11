#!/usr/bin/env python3
"""
ASST Precision Contract Selector — quantitative export pipeline.

Five tiers, generated atomically from the Selector log + data.db:

  Tier 1: SNAPSHOTS (Lite, scalar-only)
    Use: time-series quant, joining to master research export by (date, session).
    - asst_selector_snapshots_v{V}.csv
    - asst_selector_snapshots_v{V}.parquet

  Tier 2: RECOMMENDATIONS (flat, vehicle×contract grain)
    Use: contract-level analysis — strikes/deltas/edges/confidence over time.
    - asst_selector_recommendations_v{V}.csv
    - asst_selector_recommendations_v{V}.parquet

  Tier 3: TRIGGERS (flat, trigger event grain)
    Use: stress + cycle-top trigger firing patterns and their forward returns.
    - asst_selector_triggers_v{V}.csv
    - asst_selector_triggers_v{V}.parquet

  Tier 4: COHORTS (flat, cohort×horizon grain)
    Use: cohort edge timeline — how (cohort, horizon) edge stats evolve.
    - asst_selector_cohorts_v{V}.csv
    - asst_selector_cohorts_v{V}.parquet

  Tier 5: ARCHIVE (raw JSONL — every snapshot verbatim)
    Use: full-fidelity replay/audit; every recommendation, contract, edge.
    - asst_selector_archive_v{V}.jsonl

Plus:
    - asst_selector_export_v{V}_meta.json
    - asst_selector_export_v{V}_README.md

Schema version: 1.0 (initial release)

Idempotent: same log+DB → byte-identical output. Atomic writes (tmp+rename).
Read-only with respect to the Selector log and data.db.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"
import os as _os
LOG_PATH = Path(_os.environ.get(
    "ASST_SELECTOR_LOG",
    "/home/user/workspace/asst-gamma-dashboard/selector_state/asst_precision_selector_log_v1.1.jsonl",
))
DB_PATH = Path(_os.environ.get(
    "ASST_DB_PATH", "/home/user/workspace/asst-gamma-dashboard/data.db"
))
OUT_DIR = Path(_os.environ.get(
    "ASST_SELECTOR_EXPORT_DIR", "/home/user/workspace/selector_export"
))
OUT_DIR.mkdir(exist_ok=True)

V = SCHEMA_VERSION
SNAPSHOTS_CSV     = OUT_DIR / f"asst_selector_snapshots_v{V}.csv"
SNAPSHOTS_PARQUET = OUT_DIR / f"asst_selector_snapshots_v{V}.parquet"
RECS_CSV          = OUT_DIR / f"asst_selector_recommendations_v{V}.csv"
RECS_PARQUET      = OUT_DIR / f"asst_selector_recommendations_v{V}.parquet"
TRIGGERS_CSV      = OUT_DIR / f"asst_selector_triggers_v{V}.csv"
TRIGGERS_PARQUET  = OUT_DIR / f"asst_selector_triggers_v{V}.parquet"
COHORTS_CSV       = OUT_DIR / f"asst_selector_cohorts_v{V}.csv"
COHORTS_PARQUET   = OUT_DIR / f"asst_selector_cohorts_v{V}.parquet"
ARCHIVE_PATH      = OUT_DIR / f"asst_selector_archive_v{V}.jsonl"
META_PATH         = OUT_DIR / f"asst_selector_export_v{V}_meta.json"
README_PATH       = OUT_DIR / f"asst_selector_export_v{V}_README.md"


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


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_log() -> list[dict[str, Any]]:
    """Read the Selector log JSONL into memory."""
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[selector-export] WARN: skipping malformed line {i}: {e}", file=sys.stderr)
    return records


MASTER_RESEARCH_PARQUET = Path(_os.environ.get(
    "ASST_MASTER_PARQUET",
    "/home/user/workspace/master_research_export/asst_research_master_v1.5.parquet",
))


def load_forward_returns_index() -> dict[tuple[str, str], dict[str, Any]]:
    """Pull forward-return columns from the master research export keyed by
    (date, session) so we can attach them to snapshots/triggers for backtest
    analysis. Forward returns live in the master export, not data.db."""
    if not MASTER_RESEARCH_PARQUET.exists():
        return {}
    try:
        df = pd.read_parquet(MASTER_RESEARCH_PARQUET)
    except Exception as e:
        print(f"[selector-export] WARN: cannot read master research export: {e}", file=sys.stderr)
        return {}
    wanted = ["fwd5", "fwd10", "fwd21", "fwd63"]
    cols = [c for c in wanted if c in df.columns]
    if not cols or "date" not in df.columns or "session" not in df.columns:
        return {}
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in df[["date", "session"] + cols].to_dict(orient="records"):
        date = str(rec["date"])
        session = str(rec["session"])
        idx[(date, session)] = {c: rec[c] for c in cols if pd.notna(rec[c])}
    return idx


# ─── Tier 1: SNAPSHOTS ───────────────────────────────────────────────────────

def build_snapshots(records: list[dict[str, Any]],
                    fwd_index: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    """One row per Selector evaluation — scalar fields + top-vehicle summary."""
    rows = []
    for rec in records:
        # Aggregate per-vehicle summary so the lite tier is single-row-friendly
        rec_list = rec.get("recommendations") or []
        by_vehicle: dict[str, dict[str, Any]] = {}
        for r in rec_list:
            v = r.get("vehicle")
            if not v:
                continue
            edge = r.get("historical_edge") or {}
            by_vehicle[v] = {
                "recommended": bool(r.get("recommended")),
                "confidence": r.get("confidence"),
                "n_contracts": len(r.get("contracts") or []),
                "edge_horizon": edge.get("horizon"),
                "edge_n_with_fwd": edge.get("n_with_fwd"),
                "edge_fwd_median": edge.get("fwd_median"),
                "edge_fwd_p10": edge.get("fwd_p10"),
                "edge_fwd_p90": edge.get("fwd_p90"),
                "edge_max_dd_median": edge.get("max_dd_median"),
                "edge_rv_median": edge.get("rv_median"),
            }

        stress = rec.get("defensive_yield_stress") or {}
        top = rec.get("defensive_yield_top") or {}
        top_metrics = top.get("metrics") or []
        top_metric_lookup = {m.get("key"): m for m in top_metrics}

        date = rec.get("date")
        session = rec.get("session")
        fwd = fwd_index.get((date, session), {}) if date and session else {}

        row = {
            # Provenance / keys
            "snapshot_id": rec.get("snapshot_id"),
            "schema_version": rec.get("schema_version"),
            "date": date,
            "session": session,
            "timestamp_utc": rec.get("timestamp_utc"),
            "server_received_at": rec.get("server_received_at"),

            # Regime fingerprint
            "cohort_id": rec.get("cohort_id"),
            "gamma_regime": rec.get("gamma_regime"),
            "iv_regime": rec.get("iv_regime"),
            "btc_cycle_zone": rec.get("btc_cycle_zone"),
            "risk_zone": rec.get("risk_zone"),
            "gex_percentile": rec.get("gex_percentile"),
            "overall_posture": rec.get("overall_posture"),
            "ratchet_streak_length": rec.get("ratchet_streak_length"),

            # Stress trigger summary
            "stress_triggered": bool(stress.get("triggered")),
            "stress_n_met": len(stress.get("triggers_met") or []),
            "stress_triggers_met": "; ".join(stress.get("triggers_met") or []) or None,
            "stress_rotation_pct": stress.get("rotation_pct_suggested"),

            # Cycle-top trigger summary (5 BGeometrics indicators)
            "top_triggered": bool(top.get("triggered")),
            "top_n_fired": top.get("n_fired"),
            "top_rotation_pct": top.get("rotation_pct_suggested"),
            "top_mvrv_z_value": (top_metric_lookup.get("mvrv_zscore") or {}).get("value"),
            "top_mvrv_z_pct_threshold": (top_metric_lookup.get("mvrv_zscore") or {}).get("pct_of_threshold"),
            "top_pi_cycle_value": (top_metric_lookup.get("pi_cycle") or {}).get("value"),
            "top_puell_value": (top_metric_lookup.get("puell") or {}).get("value"),
            "top_puell_pct_threshold": (top_metric_lookup.get("puell") or {}).get("pct_of_threshold"),
            "top_nupl_value": (top_metric_lookup.get("nupl") or {}).get("value"),
            "top_nupl_pct_threshold": (top_metric_lookup.get("nupl") or {}).get("pct_of_threshold"),
            "top_reserve_risk_value": (top_metric_lookup.get("reserve_risk") or {}).get("value"),
            "top_reserve_risk_pct_threshold": (top_metric_lookup.get("reserve_risk") or {}).get("pct_of_threshold"),

            # Vehicle summary (one column per vehicle for fast pivots)
            "leap_core_recommended": (by_vehicle.get("LEAP_CORE") or {}).get("recommended"),
            "leap_core_confidence": (by_vehicle.get("LEAP_CORE") or {}).get("confidence"),
            "leap_core_n_contracts": (by_vehicle.get("LEAP_CORE") or {}).get("n_contracts"),

            "leap_mid_recommended": (by_vehicle.get("LEAP_MID_TAIL") or {}).get("recommended"),
            "leap_mid_confidence": (by_vehicle.get("LEAP_MID_TAIL") or {}).get("confidence"),
            "leap_mid_n_contracts": (by_vehicle.get("LEAP_MID_TAIL") or {}).get("n_contracts"),

            "csp_recommended": (by_vehicle.get("CSP") or {}).get("recommended"),
            "csp_confidence": (by_vehicle.get("CSP") or {}).get("confidence"),
            "csp_n_contracts": (by_vehicle.get("CSP") or {}).get("n_contracts"),

            "pmcc_recommended": (by_vehicle.get("PMCC") or {}).get("recommended"),
            "pmcc_confidence": (by_vehicle.get("PMCC") or {}).get("confidence"),
            "pmcc_n_contracts": (by_vehicle.get("PMCC") or {}).get("n_contracts"),

            "def_stress_recommended": (by_vehicle.get("DEFENSIVE_YIELD_STRESS") or {}).get("recommended"),
            "def_stress_confidence": (by_vehicle.get("DEFENSIVE_YIELD_STRESS") or {}).get("confidence"),

            "def_top_recommended": (by_vehicle.get("DEFENSIVE_YIELD_TOP") or {}).get("recommended"),
            "def_top_confidence": (by_vehicle.get("DEFENSIVE_YIELD_TOP") or {}).get("confidence"),

            # Cohort edge (carried from LEAP_CORE — the primary vehicle)
            "primary_cohort_horizon": (by_vehicle.get("LEAP_CORE") or {}).get("edge_horizon"),
            "primary_cohort_n_with_fwd": (by_vehicle.get("LEAP_CORE") or {}).get("edge_n_with_fwd"),
            "primary_cohort_fwd_median": (by_vehicle.get("LEAP_CORE") or {}).get("edge_fwd_median"),
            "primary_cohort_fwd_p10": (by_vehicle.get("LEAP_CORE") or {}).get("edge_fwd_p10"),
            "primary_cohort_fwd_p90": (by_vehicle.get("LEAP_CORE") or {}).get("edge_fwd_p90"),
            "primary_cohort_max_dd_median": (by_vehicle.get("LEAP_CORE") or {}).get("edge_max_dd_median"),
            "primary_cohort_rv_median": (by_vehicle.get("LEAP_CORE") or {}).get("edge_rv_median"),

            # Forward returns (joined from master research export by date+session)
            "fwd_5d": fwd.get("fwd5"),
            "fwd_10d": fwd.get("fwd10"),
            "fwd_21d": fwd.get("fwd21"),
            "fwd_63d": fwd.get("fwd63"),

            # Health
            "data_health_flags": "; ".join(rec.get("data_health_flags") or []) or None,
            "isolation_note": rec.get("isolation_note"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "session", "snapshot_id"], kind="stable").reset_index(drop=True)
    return df


# ─── Tier 2: RECOMMENDATIONS ─────────────────────────────────────────────────

def build_recommendations(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (snapshot × vehicle × contract leg).

    For PMCC, the composite row carries long_leg/short_leg sub-objects: we
    flatten each leg into its own row with `leg_role` ∈ {LONG, SHORT, SINGLE}.
    """
    rows = []
    for rec in records:
        date = rec.get("date")
        session = rec.get("session")
        snapshot_id = rec.get("snapshot_id")
        cohort_id = rec.get("cohort_id")
        rec_list = rec.get("recommendations") or []

        for r in rec_list:
            vehicle = r.get("vehicle")
            recommended = bool(r.get("recommended"))
            confidence = r.get("confidence")
            confidence_detail = r.get("confidence_detail")
            delta_from_prior = r.get("delta_from_prior")
            edge = r.get("historical_edge") or {}
            contracts = r.get("contracts") or []

            base = {
                "snapshot_id": snapshot_id,
                "date": date,
                "session": session,
                "cohort_id": cohort_id,
                "vehicle": vehicle,
                "recommended": recommended,
                "confidence": confidence,
                "confidence_detail": confidence_detail,
                "delta_from_prior": delta_from_prior,
                "rationale": r.get("rationale"),
                "edge_horizon": edge.get("horizon"),
                "edge_n_total_pm": edge.get("n_total_pm"),
                "edge_n_with_fwd": edge.get("n_with_fwd"),
                "edge_fwd_median": edge.get("fwd_median"),
                "edge_fwd_p10": edge.get("fwd_p10"),
                "edge_fwd_p25": edge.get("fwd_p25"),
                "edge_fwd_p75": edge.get("fwd_p75"),
                "edge_fwd_p90": edge.get("fwd_p90"),
                "edge_max_dd_median": edge.get("max_dd_median"),
                "edge_rv_median": edge.get("rv_median"),
                "edge_sample_sufficient": edge.get("sample_sufficient"),
            }

            if not contracts:
                # Vehicle was evaluated but no contracts (CSP=OFF, etc.)
                rows.append({
                    **base,
                    "contract_idx": 0,
                    "leg_role": "NONE",
                    "underlying": None,
                    "type": None,
                    "direction": None,
                    "strike": None,
                    "expiry": None,
                    "dte": None,
                    "approx_delta": None,
                    "quantity": None,
                    "chain_data": None,
                    "est_annualized_yield": None,
                })
                continue

            for idx, c in enumerate(contracts):
                # PMCC composite row carries a `bci` annotation (engine-side
                # initialization check). We extract its scalar fields once per
                # composite and copy them onto BOTH leg rows so that any quant
                # `groupby(snapshot_id)` finds BCI on either leg without having
                # to reconstitute the composite. None when chain insufficient.
                bci = c.get("bci") or {}
                bci_fields = {
                    "bci_passes": bci.get("passes") if bci else None,
                    "bci_buffer": bci.get("buffer") if bci else None,
                    "bci_breakeven_short_premium": bci.get("breakeven_short_premium") if bci else None,
                    "bci_short_mid": bci.get("short_mid") if bci else None,
                    "bci_leap_mid": bci.get("leap_mid") if bci else None,
                    "bci_source": bci.get("source") if bci else None,
                    "bci_leap_strike_distance": bci.get("leap_strike_distance") if bci else None,
                }
                # Vintage BCI — parallel verdicts against engine's own LEAP_CORE
                # recommendations from N PM-trading-days ago. One column set per
                # configured lookback day; all are emitted (None when the vintage
                # didn't resolve) so the schema is stable across snapshots.
                vintage_lookbacks = (30, 60, 90, 180)
                vintages = c.get("bci_vintages") or []
                vintage_by_days = {v.get("lookback_days"): v for v in vintages if isinstance(v, dict)}
                for days in vintage_lookbacks:
                    v = vintage_by_days.get(days) or {}
                    v_bci = v.get("bci") or {}
                    bci_fields[f"bci_vintage_{days}d_recommendation_date"] = v.get("recommendation_date")
                    bci_fields[f"bci_vintage_{days}d_anchor_strike"] = v.get("strike")
                    bci_fields[f"bci_vintage_{days}d_anchor_expiry"] = v.get("expiry")
                    bci_fields[f"bci_vintage_{days}d_cost_basis"] = v.get("cost_basis")
                    bci_fields[f"bci_vintage_{days}d_current_mark"] = v.get("current_mark")
                    bci_fields[f"bci_vintage_{days}d_passes"] = v_bci.get("passes") if v_bci else None
                    bci_fields[f"bci_vintage_{days}d_buffer"] = v_bci.get("buffer") if v_bci else None
                    bci_fields[f"bci_vintage_{days}d_breakeven_short_premium"] = v_bci.get("breakeven_short_premium") if v_bci else None
                    bci_fields[f"bci_vintage_{days}d_source"] = v_bci.get("source") if v_bci else None
                # PMCC composite row uses long_leg / short_leg sub-objects
                long_leg = c.get("long_leg")
                short_leg = c.get("short_leg")
                if long_leg or short_leg:
                    for role, leg in [("LONG", long_leg), ("SHORT", short_leg)]:
                        if not leg:
                            continue
                        rows.append({
                            **base,
                            "contract_idx": idx,
                            "leg_role": role,
                            "underlying": leg.get("underlying"),
                            "type": leg.get("type"),
                            "direction": leg.get("direction"),
                            "strike": leg.get("strike"),
                            "expiry": leg.get("expiry"),
                            "dte": leg.get("dte"),
                            "approx_delta": leg.get("approx_delta"),
                            "quantity": leg.get("quantity"),
                            "chain_data": leg.get("chain_data"),
                            "est_annualized_yield": leg.get("est_annualized_yield"),
                            **bci_fields,
                        })
                else:
                    rows.append({
                        **base,
                        "contract_idx": idx,
                        "leg_role": "SINGLE",
                        "underlying": c.get("underlying"),
                        "type": c.get("type"),
                        "direction": c.get("direction"),
                        "strike": c.get("strike"),
                        "expiry": c.get("expiry"),
                        "dte": c.get("dte"),
                        "approx_delta": c.get("approx_delta"),
                        "quantity": c.get("quantity"),
                        "chain_data": c.get("chain_data"),
                        "est_annualized_yield": c.get("est_annualized_yield"),
                        **bci_fields,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "session", "snapshot_id", "vehicle", "contract_idx", "leg_role"],
                            kind="stable").reset_index(drop=True)
    return df


# ─── Tier 3: TRIGGERS ────────────────────────────────────────────────────────

def build_triggers(records: list[dict[str, Any]],
                   fwd_index: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    """One row per (snapshot × trigger_kind × metric).

    Stress triggers are stored as free-text strings, so we capture them as
    individual rows with metric_key='stress:<n>' and value=null.
    Top triggers come pre-structured with metric/value/threshold/fired.
    """
    rows = []
    for rec in records:
        date = rec.get("date")
        session = rec.get("session")
        snapshot_id = rec.get("snapshot_id")
        fwd = fwd_index.get((date, session), {}) if date and session else {}

        # Stress (free-text list)
        stress = rec.get("defensive_yield_stress") or {}
        triggers_met = stress.get("triggers_met") or []
        for i, t in enumerate(triggers_met):
            rows.append({
                "snapshot_id": snapshot_id,
                "date": date,
                "session": session,
                "trigger_kind": "STRESS",
                "metric_key": f"stress_{i+1}",
                "metric_label": t,
                "value": None,
                "threshold": None,
                "direction": None,
                "pct_of_threshold": None,
                "fired": True,  # only fired ones are listed in stress.triggers_met
                "category_triggered": bool(stress.get("triggered")),
                "category_n_fired": len(triggers_met),
                "category_rotation_pct": stress.get("rotation_pct_suggested"),
                "fwd_5d": fwd.get("fwd5"),
                "fwd_21d": fwd.get("fwd21"),
            })

        # Top (structured: 5 BGeometrics metrics, both fired and not)
        top = rec.get("defensive_yield_top") or {}
        for m in (top.get("metrics") or []):
            rows.append({
                "snapshot_id": snapshot_id,
                "date": date,
                "session": session,
                "trigger_kind": "TOP",
                "metric_key": m.get("key"),
                "metric_label": m.get("label"),
                "value": m.get("value"),
                "threshold": m.get("top_threshold"),
                "direction": m.get("direction"),
                "pct_of_threshold": m.get("pct_of_threshold"),
                "fired": bool(m.get("fired")),
                "category_triggered": bool(top.get("triggered")),
                "category_n_fired": top.get("n_fired"),
                "category_rotation_pct": top.get("rotation_pct_suggested"),
                "fwd_5d": fwd.get("fwd5"),
                "fwd_21d": fwd.get("fwd21"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "session", "trigger_kind", "metric_key"], kind="stable").reset_index(drop=True)
    return df


# ─── Tier 4: COHORTS ─────────────────────────────────────────────────────────

def build_cohorts(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (snapshot × cohort × horizon).

    Each Selector snapshot's recommendations carry the same cohort edge
    (computed from cohort_id + horizon). We dedup by (date, session, cohort_id,
    horizon) and keep the latest per snapshot.
    """
    rows = []
    seen: set[tuple[str, str, str, Any]] = set()
    for rec in records:
        date = rec.get("date")
        session = rec.get("session")
        snapshot_id = rec.get("snapshot_id")
        cohort_id = rec.get("cohort_id")
        for r in (rec.get("recommendations") or []):
            edge = r.get("historical_edge") or {}
            # Skip recommendations with no real cohort edge (e.g. defensive vehicles)
            if not edge or edge.get("horizon") is None:
                continue
            ec = edge.get("cohort_id") or cohort_id
            h = edge.get("horizon")
            key = (date, session, ec, h)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "snapshot_id": snapshot_id,
                "date": date,
                "session": session,
                "cohort_id": ec,
                "horizon": h,
                "n_total_pm": edge.get("n_total_pm"),
                "n_with_fwd": edge.get("n_with_fwd"),
                "fwd_median": edge.get("fwd_median"),
                "fwd_p10": edge.get("fwd_p10"),
                "fwd_p25": edge.get("fwd_p25"),
                "fwd_p75": edge.get("fwd_p75"),
                "fwd_p90": edge.get("fwd_p90"),
                "max_dd_median": edge.get("max_dd_median"),
                "rv_median": edge.get("rv_median"),
                "sample_sufficient": edge.get("sample_sufficient"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "session", "cohort_id", "horizon"], kind="stable").reset_index(drop=True)
    return df


# ─── Tier 5: ARCHIVE ─────────────────────────────────────────────────────────

def build_archive(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verbatim raw log records, sorted by (date, session, timestamp_utc) so the
    archive is order-stable across rebuilds."""
    out = list(records)
    out.sort(key=lambda r: (r.get("date") or "", r.get("session") or "", r.get("timestamp_utc") or ""))
    return out


# ─── README ──────────────────────────────────────────────────────────────────

def build_readme(meta: dict[str, Any]) -> str:
    return f"""# ASST Precision Contract Selector — Quantitative Export Suite

**Schema version:** {meta['schema_version']}
**Generated:** {meta['generated_at']}
**Source log:** `{LOG_PATH}`
**Total snapshots:** {meta['n_snapshots']}

## Tiers

### Tier 1 — Snapshots ({meta['n_snapshots']} rows × {meta['n_cols_snapshots']} cols)
One row per Selector evaluation. Scalar fields only — joinable to the master research export by `(date, session)`.

Use cases: time-series posture analysis, regime-conditioned forward-return studies, cycle-top trigger backtests.

Files:
- `asst_selector_snapshots_v{meta['schema_version']}.csv`
- `asst_selector_snapshots_v{meta['schema_version']}.parquet`

### Tier 2 — Recommendations ({meta['n_recommendations']} rows × {meta['n_cols_recommendations']} cols)
One row per (snapshot × vehicle × contract leg). PMCC composite rows are flattened into LONG and SHORT leg rows.

Use cases: contract-level backtest of recommended strikes/expiries, delta drift analysis, chain_data quality (`from_chain` vs `approximated`) tracking.

Files:
- `asst_selector_recommendations_v{meta['schema_version']}.csv`
- `asst_selector_recommendations_v{meta['schema_version']}.parquet`

### Tier 3 — Triggers ({meta['n_triggers']} rows × {meta['n_cols_triggers']} cols)
One row per trigger event. Stress (free-text, only fired) and Top (structured, all 5 BGeometrics indicators with values/thresholds/fired flag).

Use cases: trigger firing patterns over time, predictive value of cycle-top indicators, stress-trigger forward-return distributions.

Files:
- `asst_selector_triggers_v{meta['schema_version']}.csv`
- `asst_selector_triggers_v{meta['schema_version']}.parquet`

### Tier 4 — Cohorts ({meta['n_cohorts']} rows × {meta['n_cols_cohorts']} cols)
Distinct (date, session, cohort_id, horizon) combinations with edge stats. Useful for cohort-edge stability analysis.

Files:
- `asst_selector_cohorts_v{meta['schema_version']}.csv`
- `asst_selector_cohorts_v{meta['schema_version']}.parquet`

### Tier 5 — Archive ({meta['n_archive']} records)
Raw JSONL — every snapshot preserved verbatim including all nested objects (contracts, edges, triggers, recommendations).

Files:
- `asst_selector_archive_v{meta['schema_version']}.jsonl`

## Joining to the Master Research Export

All four flat tiers carry `(date, session)` as a stable foreign key into the
master research export. Recommended join:

```python
import pandas as pd
sel = pd.read_parquet("asst_selector_snapshots_v{meta['schema_version']}.parquet")
mr  = pd.read_parquet("asst_research_master_v1.5.parquet")
joined = sel.merge(mr, on=["date", "session"], how="left", suffixes=("_sel", "_mr"))
```

## Methodology Notes

- **Forward-only**: Selector log entries are written by user-driven "Save to log"
  events — historical backfill is intentionally not performed (preserves clean cohorts).
- **Edge stats**: percentiles (p10/p25/p75/p90) are computed from the cohort
  population — not bootstrap-sampled. `sample_sufficient` flags cohorts with n ≥ 8.
- **Forward returns**: joined from `daily_runs.fwd_return_*` columns when present.
  Coverage depends on lookback availability.

## Provenance

- Selector log: append-only JSONL (`asst_precision_selector_log_v1.1.jsonl`)
- Daily runs: SQLite (`/home/user/workspace/asst-gamma-dashboard/data.db`)
- Build script: `build_selector_export.py` (atomic writes, idempotent)
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[selector-export] schema v{SCHEMA_VERSION}")
    print(f"[selector-export] log: {LOG_PATH}")
    print(f"[selector-export] db:  {DB_PATH}")
    print(f"[selector-export] out: {OUT_DIR}")

    records = load_log()
    fwd_index = load_forward_returns_index()
    print(f"[selector-export] loaded {len(records)} log records, {len(fwd_index)} fwd-return rows")

    snapshots = build_snapshots(records, fwd_index)
    recs = build_recommendations(records)
    triggers = build_triggers(records, fwd_index)
    cohorts = build_cohorts(records)
    archive = build_archive(records)

    print(f"[snapshots] {len(snapshots)} rows × {len(snapshots.columns) if not snapshots.empty else 0} cols")
    print(f"[recommendations] {len(recs)} rows × {len(recs.columns) if not recs.empty else 0} cols")
    print(f"[triggers] {len(triggers)} rows × {len(triggers.columns) if not triggers.empty else 0} cols")
    print(f"[cohorts] {len(cohorts)} rows × {len(cohorts.columns) if not cohorts.empty else 0} cols")
    print(f"[archive] {len(archive)} records")

    # Validation: archive count must equal snapshot count
    if len(archive) != len(snapshots):
        print(f"[selector-export] VALIDATION FAILED: archive ({len(archive)}) != snapshots ({len(snapshots)})", file=sys.stderr)
        return 2

    # Atomic writes
    if not snapshots.empty:
        atomic_write_csv(snapshots, SNAPSHOTS_CSV)
        atomic_write_parquet(snapshots, SNAPSHOTS_PARQUET)
    else:
        # Still write empty files so consumers can detect "no data"
        atomic_write_csv(pd.DataFrame(), SNAPSHOTS_CSV)
        atomic_write_parquet(pd.DataFrame({"_empty": []}), SNAPSHOTS_PARQUET)

    if not recs.empty:
        atomic_write_csv(recs, RECS_CSV)
        atomic_write_parquet(recs, RECS_PARQUET)
    else:
        atomic_write_csv(pd.DataFrame(), RECS_CSV)
        atomic_write_parquet(pd.DataFrame({"_empty": []}), RECS_PARQUET)

    if not triggers.empty:
        atomic_write_csv(triggers, TRIGGERS_CSV)
        atomic_write_parquet(triggers, TRIGGERS_PARQUET)
    else:
        atomic_write_csv(pd.DataFrame(), TRIGGERS_CSV)
        atomic_write_parquet(pd.DataFrame({"_empty": []}), TRIGGERS_PARQUET)

    if not cohorts.empty:
        atomic_write_csv(cohorts, COHORTS_CSV)
        atomic_write_parquet(cohorts, COHORTS_PARQUET)
    else:
        atomic_write_csv(pd.DataFrame(), COHORTS_CSV)
        atomic_write_parquet(pd.DataFrame({"_empty": []}), COHORTS_PARQUET)

    atomic_write_jsonl(archive, ARCHIVE_PATH)

    # Meta
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_log": str(LOG_PATH),
        "source_db": str(DB_PATH),
        "n_snapshots": len(snapshots),
        "n_recommendations": len(recs),
        "n_triggers": len(triggers),
        "n_cohorts": len(cohorts),
        "n_archive": len(archive),
        "n_cols_snapshots": len(snapshots.columns) if not snapshots.empty else 0,
        "n_cols_recommendations": len(recs.columns) if not recs.empty else 0,
        "n_cols_triggers": len(triggers.columns) if not triggers.empty else 0,
        "n_cols_cohorts": len(cohorts.columns) if not cohorts.empty else 0,
        "fwd_return_index_size": len(fwd_index),
        "validation_passed": True,
        "files": {
            "snapshots_csv": str(SNAPSHOTS_CSV.name),
            "snapshots_parquet": str(SNAPSHOTS_PARQUET.name),
            "recommendations_csv": str(RECS_CSV.name),
            "recommendations_parquet": str(RECS_PARQUET.name),
            "triggers_csv": str(TRIGGERS_CSV.name),
            "triggers_parquet": str(TRIGGERS_PARQUET.name),
            "cohorts_csv": str(COHORTS_CSV.name),
            "cohorts_parquet": str(COHORTS_PARQUET.name),
            "archive_jsonl": str(ARCHIVE_PATH.name),
            "meta_json": str(META_PATH.name),
            "readme_md": str(README_PATH.name),
        },
    }
    atomic_write_text(json.dumps(meta, indent=2, default=str), META_PATH)
    atomic_write_text(build_readme(meta), README_PATH)

    print(f"[selector-export] DONE — {len(snapshots)} snapshots, {len(recs)} recs, {len(triggers)} triggers, {len(cohorts)} cohorts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

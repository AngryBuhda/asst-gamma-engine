"""
Stochastic Layer Phase 1 — Parallel Interpretation Engine.

Reads current snapshot + historical runs, buckets state, finds nearest comparable
cases, and ranks broad vehicle families in dual modes (gated/ungated).

Does NOT:
  - override deterministic permissions
  - pick specific contracts (that's suggestions.py)
  - run Monte Carlo or optimizer logic
  - allocate capital

Only provides probabilistic advisory comparison alongside the doctrine layer.
"""

import json
import hashlib
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Versioning & Configuration
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_VERSION = "stoch-phase1"
BUCKET_VERSION = "phase1-buckets-v2"  # v2: canonicalized 5th bucket as drawdown; valuation_bucket is optional refinement only

VEHICLE_UNIVERSE = ["LEAP_ONLY", "PMCC", "CSP", "SHARES", "DEFENSIVE_YIELD", "NONE"]

# Evidence thresholds (see spec)
EVIDENCE_HIGH = 15
EVIDENCE_MEDIUM = 5
EVIDENCE_LOW = 1

# Weights for nearest-state matching
DIMENSION_WEIGHTS = {
    "regime_bucket": 3.0,   # Most stable, most important
    "gex_bucket": 2.5,
    "iv_bucket": 2.0,
    "cycle_bucket": 2.0,
    "action_banner": 1.5,
    "drawdown_bucket": 1.0, # Refines match, shouldn't dominate
}

# ═══════════════════════════════════════════════════════════════════════════════
# State Bucketing
# ═══════════════════════════════════════════════════════════════════════════════

def bucket_state(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a run's raw fields into a bucketed state representation.
    Most buckets are strings or None; iv_bucket is an int (band index 0..4)
    or None after the v1.5 cutover.
    """
    result: Dict[str, Any] = {}

    # Regime: use raw value
    result["regime_bucket"] = run.get("regime")

    # GEX percentile: LOW / MID / HIGH
    gex_pct = run.get("gex_percentile")
    if gex_pct is None:
        result["gex_bucket"] = None
    elif gex_pct < 15:
        result["gex_bucket"] = "LOW"
    elif gex_pct > 85:
        result["gex_bucket"] = "HIGH"
    else:
        result["gex_bucket"] = "MID"

    # IV bucket: integer band (0..4) over iv_percentile.
    # Switched from iv_regime (3-state string) to iv_band (5-state int) in
    # the v1.5 cohort cutover. The integer makes adjacent-band soft-matching
    # computable (see _bucket_match_score), which the string version couldn't do.
    iv_band_val = run.get("iv_band")
    if iv_band_val is None:
        result["iv_bucket"] = None
    else:
        try:
            result["iv_bucket"] = int(iv_band_val)
        except (TypeError, ValueError):
            result["iv_bucket"] = None

    # BTC cycle zone: raw
    result["cycle_bucket"] = run.get("btc_cycle_zone")

    # Action banner: raw (for refinement)
    result["action_banner"] = run.get("action_banner")

    # Drawdown: SHALLOW / MODERATE / DEEP
    dd = run.get("asst_drawdown_90d")
    if dd is None:
        result["drawdown_bucket"] = None
    elif dd > -0.15:
        result["drawdown_bucket"] = "SHALLOW"
    elif dd > -0.30:
        result["drawdown_bucket"] = "MODERATE"
    else:
        result["drawdown_bucket"] = "DEEP"

    # Optional buckets (when data present)
    vanna_regime = run.get("vanna_regime")
    if vanna_regime:
        result["vanna_bucket"] = vanna_regime

    mnav_discount = run.get("mnav_discount")
    if mnav_discount is not None:
        if mnav_discount > 0.05:
            result["valuation_bucket"] = "DISCOUNT"
        elif mnav_discount < -0.05:
            result["valuation_bucket"] = "PREMIUM"
        else:
            result["valuation_bucket"] = "FAIR"

    btc_conf = run.get("btc_gex_secondary_confirm")
    if btc_conf:
        result["btc_deriv_bucket"] = btc_conf

    return result


# Canonical 5-bucket state key dimensions (frozen for Phase 1)
# Order is fixed and documented in stochastic_methodology_v2.3.md.
STATE_KEY_DIMENSIONS = [
    "regime_bucket",   # 1. regime
    "gex_bucket",      # 2. GEX percentile band
    "iv_bucket",       # 3. IV regime
    "cycle_bucket",    # 4. BTC cycle zone
    "drawdown_bucket", # 5. ASST 90-day drawdown band (CANONICAL — not mNAV)
]

def state_key(buckets: Dict[str, Any]) -> str:
    """Produce a compact string key for a bucketed state.

    The 5 dimensions are (in order): regime, GEX, IV, BTC cycle, drawdown.
    valuation_bucket (EV/mNAV discount) is captured separately as an
    optional refinement field — it is NOT part of the primary state key.

    iv_bucket is an int (band index 0..4) post-v1.5 — we coerce to string
    explicitly so that band 0 doesn't get treated as "missing" by a falsy
    `or "?"` shortcut.
    """
    parts = []
    for d in STATE_KEY_DIMENSIONS:
        v = buckets.get(d)
        if v is None:
            parts.append("?")
        else:
            parts.append(str(v))
    return "|".join(parts).lower()


def compute_run_hash(current_run: Dict[str, Any], state_key_str: str) -> str:
    """Deterministic short hash for provenance.

    Derived from the canonical state key plus the doctrine gates active
    at computation time. Same (state, gates) → same hash. Used to
    distinguish freshly recomputed stochastic rows from genuine historical
    cases and to detect drift.
    """
    payload = "|".join([
        state_key_str,
        str(current_run.get("csp_allowed") or ""),
        str(current_run.get("pmcc_allowed") or ""),
        str(current_run.get("leap_add_allowed") or ""),
        str(current_run.get("risk_zone") or ""),
        str(current_run.get("action_banner") or ""),
        METHOD_VERSION,
        BUCKET_VERSION,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# Nearest-State Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def _bucket_match_score(dim: str, cur_val: Any, hist_val: Any) -> float:
    """Per-dimension partial-credit match score in [0, 1].

    Most dimensions are categorical strings: exact match -> 1.0, mismatch -> 0.0.

    iv_bucket is an integer band index (0..4) over iv_percentile. Adjacent
    bands represent similar IV states (band 0 EXTREME_CHEAP and band 1 CHEAP_VOL
    are closer to each other than to band 4 EXTREME_RICH). We give partial
    credit on band-distance:
        distance 0 (exact)      -> 1.0
        distance 1 (adjacent)   -> 0.5
        distance >= 2           -> 0.0
    This is a soft-match the string version couldn't express — it's why we
    moved iv_bucket from string to integer in the v1.5 cutover.
    """
    if dim == "iv_bucket":
        try:
            d = abs(int(cur_val) - int(hist_val))
        except (TypeError, ValueError):
            return 1.0 if cur_val == hist_val else 0.0
        if d == 0:
            return 1.0
        if d == 1:
            return 0.5
        return 0.0
    # Default: exact-match string comparison.
    return 1.0 if cur_val == hist_val else 0.0


def match_score(current: Dict[str, Any], historical: Dict[str, Any]) -> float:
    """
    Compute weighted match score between current state and a historical state.
    Returns 0-1 (1 = identical on all weighted dimensions).

    Per-dimension scoring lives in _bucket_match_score so dimensions with
    natural ordering (iv_bucket as band index) can give partial credit on
    adjacent buckets, while categorical dimensions remain exact-match.
    """
    total_weight = 0.0
    matched_weight = 0.0

    for dim, weight in DIMENSION_WEIGHTS.items():
        cur_val = current.get(dim)
        hist_val = historical.get(dim)
        if cur_val is None or hist_val is None:
            continue  # Skip missing dimensions (don't penalize)
        total_weight += weight
        matched_weight += weight * _bucket_match_score(dim, cur_val, hist_val)

    if total_weight == 0:
        return 0.0
    return matched_weight / total_weight


def find_nearest_cases(
    current_state: Dict[str, Any],
    history: List[Dict[str, Any]],
    min_score: float = 0.5,
    max_cases: int = 50,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Find historical cases most similar to current state.
    Returns (cases, used_fallback). Each case has 'score' and 'buckets' added.
    """
    scored = []
    for run in history:
        h_buckets = bucket_state(run)
        score = match_score(current_state, h_buckets)
        if score >= min_score:
            scored.append({
                **run,
                "_buckets": h_buckets,
                "_score": score,
            })

    # If not enough exact matches, lower the bar (fallback)
    used_fallback = False
    if len(scored) < EVIDENCE_MEDIUM:
        used_fallback = True
        scored = []
        for run in history:
            h_buckets = bucket_state(run)
            score = match_score(current_state, h_buckets)
            if score >= 0.3:  # Looser partial match
                scored.append({
                    **run,
                    "_buckets": h_buckets,
                    "_score": score,
                })

    scored.sort(key=lambda c: c["_score"], reverse=True)
    return scored[:max_cases], used_fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Vehicle Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def infer_active_vehicle(run: Dict[str, Any]) -> str:
    """
    Given a historical run, infer which vehicle family would have been
    the top recommendation at that time. This uses the deterministic
    permissions that were already active.

    Priority:
      1. If action_banner says "BUILD: LEAP ONLY" → LEAP_ONLY
      2. If pmcc_allowed=ALLOWED and long gamma → PMCC
      3. If csp_allowed=ON → CSP
      4. If leap_add_allowed=ALLOWED (any size) → LEAP_ONLY
      5. If risk_zone=RED or regime=short_gamma_strong → DEFENSIVE_YIELD or NONE
      6. Otherwise → SHARES (accumulate passively)
    """
    banner = (run.get("action_banner") or "").upper()
    csp = (run.get("csp_allowed") or "").upper()
    pmcc = (run.get("pmcc_allowed") or "").upper()
    leap = (run.get("leap_add_allowed") or "").upper()
    regime = (run.get("regime") or "").lower()
    risk_zone = (run.get("risk_zone") or "").upper()

    if "LEAP ONLY" in banner or "LEAP_ONLY" in banner:
        return "LEAP_ONLY"
    if pmcc == "ALLOWED" and regime.startswith("long"):
        return "PMCC"
    if csp == "ON":
        return "CSP"
    if leap == "ALLOWED":
        return "LEAP_ONLY"
    if risk_zone == "RED" and regime.startswith("short"):
        return "NONE"
    if risk_zone == "RED":
        return "DEFENSIVE_YIELD"
    return "SHARES"


def rank_vehicles(
    cases: List[Dict[str, Any]],
    eligible: List[str],
) -> Tuple[List[str], Dict[str, float]]:
    """
    Rank eligible vehicles by frequency they were "active" in comparable cases,
    weighted by match score.

    Returns (ranking, scores_by_vehicle).
    """
    if not cases:
        # No evidence — return universe in arbitrary order, all zero
        return list(eligible), {v: 0.0 for v in eligible}

    weighted_counts: Dict[str, float] = {v: 0.0 for v in eligible}
    total_weight = 0.0

    for case in cases:
        score = case.get("_score", 0)
        active = infer_active_vehicle(case)
        if active in eligible:
            weighted_counts[active] += score
        total_weight += score

    # Normalize to 0-1
    if total_weight > 0:
        vehicle_scores = {v: round(w / total_weight, 4) for v, w in weighted_counts.items()}
    else:
        vehicle_scores = weighted_counts

    # Rank descending
    ranking = sorted(eligible, key=lambda v: vehicle_scores.get(v, 0), reverse=True)
    return ranking, vehicle_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════════════════

def confidence_label(
    evidence_count: int,
    used_fallback: bool,
    missing_optional: int,
) -> str:
    """
    Assign a confidence label based on evidence count and quality.
    Fallback matching caps at MEDIUM even with many cases.
    """
    if evidence_count == 0:
        return "INSUFFICIENT_DATA"
    if evidence_count < EVIDENCE_MEDIUM:
        return "LOW"

    # Cap at MEDIUM if fallback was used (unless evidence is overwhelming)
    if used_fallback and evidence_count < EVIDENCE_HIGH * 2:
        return "MEDIUM"

    # Penalize heavily if lots of optional fields are missing
    if missing_optional >= 3 and evidence_count < EVIDENCE_HIGH:
        return "LOW"

    if evidence_count >= EVIDENCE_HIGH:
        return "HIGH"
    return "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stochastic_output(
    current_run: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Main entry point. Given the current snapshot + historical runs,
    produce a structured stochastic output with gated + ungated rankings.

    Args:
        current_run: The just-computed run (not yet in history).
        history: All prior runs from the DB (list of dicts).

    Returns:
        Structured dict matching the spec's output contract.
    """

    # 1. Bucket current state
    current_buckets = bucket_state(current_run)
    skey = state_key(current_buckets)

    # 2. Find nearest cases
    nearest, used_fallback = find_nearest_cases(current_buckets, history)
    evidence_count = len(nearest)

    # 3. Determine eligibility for gated mode (respect doctrine)
    csp_ok = (current_run.get("csp_allowed") or "").upper() == "ON"
    pmcc_ok = (current_run.get("pmcc_allowed") or "").upper() == "ALLOWED"
    leap_ok = (current_run.get("leap_add_allowed") or "").upper() == "ALLOWED"

    gated_eligible = ["SHARES", "DEFENSIVE_YIELD", "NONE"]  # Always permitted
    if csp_ok:
        gated_eligible.append("CSP")
    if pmcc_ok:
        gated_eligible.append("PMCC")
    if leap_ok:
        gated_eligible.append("LEAP_ONLY")

    ungated_eligible = list(VEHICLE_UNIVERSE)

    # 4. Rank vehicles
    gated_ranking, gated_scores = rank_vehicles(nearest, gated_eligible)
    ungated_ranking, ungated_scores = rank_vehicles(nearest, ungated_eligible)

    # 5. Compute confidence
    optional_missing = sum(
        1 for f in ["vanna_regime", "mnav_discount", "btc_gex_secondary_confirm", "iv_skew_25d", "put_call_oi_ratio"]
        if current_run.get(f) is None
    )
    conf_label = confidence_label(evidence_count, used_fallback, optional_missing)

    # 6. Nearest case IDs (compact format) + richer per-case detail
    nearest_ids = [
        f"{c.get('date')}_{c.get('session', '?')}"
        for c in nearest[:10]
    ]
    nearest_detailed = [
        {
            "case_id": f"{c.get('date')}_{c.get('session', '?')}",
            "date": c.get("date"),
            "session": c.get("session"),
            "match_score": round(c.get("_score", 0.0), 4),
            "active_vehicle": infer_active_vehicle(c),
            "regime": c.get("regime"),
            "spot": c.get("spot"),
        }
        for c in nearest[:15]
    ]

    # 6b. Per-dimension match fingerprint (exact vs mismatched vs missing) across nearest cases
    dim_match_rates: Dict[str, float] = {}
    if nearest:
        for dim in DIMENSION_WEIGHTS.keys():
            cur_val = current_buckets.get(dim)
            if cur_val is None:
                dim_match_rates[dim] = -1.0  # sentinel: missing in current
                continue
            hits = 0
            valid = 0
            for c in nearest:
                hval = c.get("_buckets", {}).get(dim)
                if hval is None:
                    continue
                valid += 1
                if hval == cur_val:
                    hits += 1
            dim_match_rates[dim] = round(hits / valid, 4) if valid > 0 else 0.0

    # 6c. Score distribution metrics (concentration, rank gap, dispersion)
    def _score_metrics(scores: Dict[str, float]) -> Dict[str, Any]:
        vals = sorted([v for v in scores.values() if v > 0], reverse=True)
        total = sum(vals)
        if total <= 0:
            return {"concentration": 0.0, "rank_gap": 0.0, "dispersion": 0, "top_share": 0.0}
        # Herfindahl on normalized shares
        shares = [v / total for v in vals]
        herfindahl = round(sum(s * s for s in shares), 4)
        rank_gap = round(vals[0] - (vals[1] if len(vals) > 1 else 0.0), 4)
        return {
            "concentration": herfindahl,   # 1.0 = all in one vehicle, 1/n = even
            "rank_gap": rank_gap,
            "dispersion": len(vals),       # # of vehicles with nonzero mass
            "top_share": round(shares[0], 4),
        }
    gated_metrics = _score_metrics(gated_scores)
    ungated_metrics = _score_metrics(ungated_scores)

    # 6d. Exclusion reasons for non-eligible vehicles (gated side)
    exclusion_reasons: Dict[str, str] = {}
    for v in VEHICLE_UNIVERSE:
        if v in gated_eligible:
            continue
        if v == "CSP":
            exclusion_reasons[v] = f"csp_allowed={current_run.get('csp_allowed') or 'null'}"
        elif v == "PMCC":
            exclusion_reasons[v] = f"pmcc_allowed={current_run.get('pmcc_allowed') or 'null'}"
        elif v == "LEAP_ONLY":
            exclusion_reasons[v] = f"leap_add_allowed={current_run.get('leap_add_allowed') or 'null'}"
        else:
            exclusion_reasons[v] = "doctrine_excluded"

    # 6e. Doctrine gates snapshot
    doctrine_gates = {
        "csp_allowed": current_run.get("csp_allowed"),
        "pmcc_allowed": current_run.get("pmcc_allowed"),
        "leap_add_allowed": current_run.get("leap_add_allowed"),
        "risk_zone": current_run.get("risk_zone"),
        "action_banner": current_run.get("action_banner"),
        "regime": current_run.get("regime"),
    }

    # 7. Conflict detection
    gated_top = gated_ranking[0] if gated_ranking else None
    ungated_top = ungated_ranking[0] if ungated_ranking else None
    has_conflict = gated_top != ungated_top and gated_top is not None and ungated_top is not None
    conflict_score_delta = 0.0
    if has_conflict:
        conflict_score_delta = round(
            (ungated_scores.get(ungated_top, 0.0) - gated_scores.get(gated_top, 0.0)),
            4,
        )

    # 8. Notes (document missing data and fallback)
    notes = []
    if used_fallback:
        notes.append("Exact bucket matches insufficient; used weighted partial matching (confidence capped at MEDIUM)")
    if optional_missing >= 3:
        notes.append(f"{optional_missing} optional refinement fields missing; confidence reduced")
    if has_conflict:
        notes.append(f"Gated top ({gated_top}) differs from ungated top ({ungated_top}) — doctrine constrains preference")
    notes.append("Ungated output is research only and does not alter deterministic permissions")

    snapshot_id = f"{current_run.get('date')}_{current_run.get('session', '?')}_v2.3"

    run_hash = compute_run_hash(current_run, skey)

    return {
        "snapshot_id": snapshot_id,
        "method_version": METHOD_VERSION,
        "bucket_version": BUCKET_VERSION,
        "run_hash": run_hash,
        "state_key": skey,
        "state_key_dimensions": STATE_KEY_DIMENSIONS,
        "current_buckets": current_buckets,
        "gated": {
            "eligible_vehicles": gated_eligible,
            "excluded_vehicles": [v for v in VEHICLE_UNIVERSE if v not in gated_eligible],
            "exclusion_reasons": exclusion_reasons,
            "top_vehicle": gated_top,
            "ranking": gated_ranking,
            "scores": gated_scores,
            "score_metrics": gated_metrics,
            "confidence_label": conf_label,
            "evidence_count": evidence_count,
            "nearest_cases": nearest_ids,
        },
        "ungated": {
            "eligible_vehicles": ungated_eligible,
            "excluded_vehicles": [],
            "top_vehicle": ungated_top,
            "ranking": ungated_ranking,
            "scores": ungated_scores,
            "score_metrics": ungated_metrics,
            "confidence_label": conf_label,
            "evidence_count": evidence_count,
            "nearest_cases": nearest_ids,
        },
        "doctrine_gates": doctrine_gates,
        "nearest_cases_detailed": nearest_detailed,
        "dimension_match_rates": dim_match_rates,
        "conflict": {
            "has_conflict": has_conflict,
            "gated_top_vehicle": gated_top,
            "ungated_top_vehicle": ungated_top,
            "score_delta": conflict_score_delta,
        },
        "notes": notes,
        "metadata": {
            "optional_missing_count": optional_missing,
            "used_fallback_matching": used_fallback,
            "history_size": len(history),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DB Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_history_from_db(db_path: str, exclude_run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load all prior runs from the DB, optionally excluding a specific run id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if exclude_run_id is not None:
        rows = cur.execute(
            "SELECT * FROM daily_runs WHERE id != ? ORDER BY id ASC",
            (exclude_run_id,),
        ).fetchall()
    else:
        rows = cur.execute("SELECT * FROM daily_runs ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Full column list for stochastic_log (v2 schema — review-surface equivalence with snapshot)
STOCHASTIC_LOG_COLUMNS = [
    "id", "run_id", "snapshot_id",
    "method_version", "bucket_version", "run_hash",
    "state_key",
    "regime_bucket", "gex_bucket", "iv_bucket", "cycle_bucket", "drawdown_bucket",
    "valuation_bucket", "vanna_bucket", "btc_deriv_bucket", "action_banner",
    "csp_allowed", "pmcc_allowed", "leap_add_allowed", "risk_zone",
    "gated_top", "gated_confidence", "gated_evidence",
    "gated_top_score", "gated_hhi", "gated_rank_gap", "gated_dispersion",
    "ungated_top", "ungated_confidence", "ungated_evidence",
    "ungated_top_score", "ungated_hhi", "ungated_rank_gap", "ungated_dispersion",
    "has_conflict", "conflict_score_delta",
    "used_fallback", "optional_missing", "history_size",
    "output_json", "created_at",
]


def _ensure_log_schema(conn: sqlite3.Connection) -> None:
    """Create the v2 log table if missing, then ADD any columns that don't exist
    (safe forward-migration without dropping existing rows)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stochastic_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            snapshot_id TEXT,
            method_version TEXT,
            bucket_version TEXT,
            state_key TEXT,
            gated_top TEXT,
            ungated_top TEXT,
            confidence_label TEXT,
            evidence_count INTEGER,
            has_conflict INTEGER,
            output_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES daily_runs(id)
        )
    """)
    # Discover existing columns
    existing = {r[1] for r in conn.execute("PRAGMA table_info(stochastic_log)").fetchall()}
    wanted_cols_defs = [
        ("run_hash", "TEXT"),
        ("regime_bucket", "TEXT"),
        ("gex_bucket", "TEXT"),
        ("iv_bucket", "TEXT"),
        ("cycle_bucket", "TEXT"),
        ("drawdown_bucket", "TEXT"),
        ("valuation_bucket", "TEXT"),
        ("vanna_bucket", "TEXT"),
        ("btc_deriv_bucket", "TEXT"),
        ("action_banner", "TEXT"),
        ("csp_allowed", "TEXT"),
        ("pmcc_allowed", "TEXT"),
        ("leap_add_allowed", "TEXT"),
        ("risk_zone", "TEXT"),
        ("gated_confidence", "TEXT"),
        ("gated_evidence", "INTEGER"),
        ("gated_top_score", "REAL"),
        ("gated_hhi", "REAL"),
        ("gated_rank_gap", "REAL"),
        ("gated_dispersion", "INTEGER"),
        ("ungated_confidence", "TEXT"),
        ("ungated_evidence", "INTEGER"),
        ("ungated_top_score", "REAL"),
        ("ungated_hhi", "REAL"),
        ("ungated_rank_gap", "REAL"),
        ("ungated_dispersion", "INTEGER"),
        ("conflict_score_delta", "REAL"),
        ("used_fallback", "INTEGER"),
        ("optional_missing", "INTEGER"),
        ("history_size", "INTEGER"),
    ]
    for name, typ in wanted_cols_defs:
        if name not in existing:
            conn.execute(f"ALTER TABLE stochastic_log ADD COLUMN {name} {typ}")


def log_stochastic_output(db_path: str, run_id: int, output: Dict[str, Any]) -> None:
    """Append (or replace) stochastic output for a run.

    UPSERT semantics on run_id: if a stochastic_log row already exists for this
    run_id, it is replaced with the freshest computation. This makes the
    fetch_data retry path idempotent.

    Backed by the unique index ix_stochastic_log_run_id.
    """
    conn = sqlite3.connect(db_path)
    _ensure_log_schema(conn)
    # Ensure the unique index is present (safe if it already exists)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_stochastic_log_run_id ON stochastic_log(run_id)"
    )

    g = output["gated"]
    u = output["ungated"]
    gm = g.get("score_metrics", {}) or {}
    um = u.get("score_metrics", {}) or {}
    buckets = output.get("current_buckets", {}) or {}
    doctrine = output.get("doctrine_gates", {}) or {}
    metadata = output.get("metadata", {}) or {}

    # Delete any existing row for this run_id so we always have a single,
    # current entry. This is simpler and more robust than ON CONFLICT in SQLite
    # given the wide column list and forward-migrated schema.
    conn.execute("DELETE FROM stochastic_log WHERE run_id = ?", (run_id,))

    conn.execute("""
        INSERT INTO stochastic_log (
            run_id, snapshot_id, method_version, bucket_version, run_hash, state_key,
            regime_bucket, gex_bucket, iv_bucket, cycle_bucket, drawdown_bucket,
            valuation_bucket, vanna_bucket, btc_deriv_bucket, action_banner,
            csp_allowed, pmcc_allowed, leap_add_allowed, risk_zone,
            gated_top, gated_confidence, gated_evidence,
            gated_top_score, gated_hhi, gated_rank_gap, gated_dispersion,
            ungated_top, ungated_confidence, ungated_evidence,
            ungated_top_score, ungated_hhi, ungated_rank_gap, ungated_dispersion,
            confidence_label, evidence_count, has_conflict, conflict_score_delta,
            used_fallback, optional_missing, history_size,
            output_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        output["snapshot_id"],
        output["method_version"],
        output["bucket_version"],
        output.get("run_hash"),
        output["state_key"],
        buckets.get("regime_bucket"),
        buckets.get("gex_bucket"),
        buckets.get("iv_bucket"),
        buckets.get("cycle_bucket"),
        buckets.get("drawdown_bucket"),
        buckets.get("valuation_bucket"),
        buckets.get("vanna_bucket"),
        buckets.get("btc_deriv_bucket"),
        buckets.get("action_banner") or doctrine.get("action_banner"),
        doctrine.get("csp_allowed"),
        doctrine.get("pmcc_allowed"),
        doctrine.get("leap_add_allowed"),
        doctrine.get("risk_zone"),
        g["top_vehicle"],
        g["confidence_label"],
        g["evidence_count"],
        (g.get("scores") or {}).get(g["top_vehicle"], 0.0),
        gm.get("concentration", 0.0),
        gm.get("rank_gap", 0.0),
        gm.get("dispersion", 0),
        u["top_vehicle"],
        u["confidence_label"],
        u["evidence_count"],
        (u.get("scores") or {}).get(u["top_vehicle"], 0.0),
        um.get("concentration", 0.0),
        um.get("rank_gap", 0.0),
        um.get("dispersion", 0),
        g["confidence_label"],            # legacy column kept populated
        g["evidence_count"],              # legacy column kept populated
        1 if output["conflict"]["has_conflict"] else 0,
        output["conflict"].get("score_delta", 0.0),
        1 if metadata.get("used_fallback_matching") else 0,
        metadata.get("optional_missing_count", 0),
        metadata.get("history_size", 0),
        json.dumps(output),
    ))
    conn.commit()
    conn.close()

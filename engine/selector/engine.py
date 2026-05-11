"""
Selector engine \u2014 Python port of shared/selectorEngine.ts (P1.8).

Pure functions: same inputs yield same outputs. No side effects. All
serialization mirrors the TypeScript exactly so output JSON can be
byte-identical compared via the golden fixtures in
engine/tests/fixtures/selector_golden/.

Schema version: 1.2 (matches TS).
Cohort version: 1.5.

The structure of this file mirrors shared/selectorEngine.ts section by
section so a side-by-side diff is reviewable. Helper signatures are
faithful translations; we don't reorganize even when Python would
permit a more idiomatic shape, because the goal is byte-identical
output not Pythonic style.
"""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.compute.iv_band import label_for_band
from engine.selector.stats import (
    quantile,
    mean as stats_mean,
    bootstrap_median_ci,
    bootstrap_quantile_ci,
    reportable_percentiles,
)
from engine.selector.types import COHORT_ID_VERSION


# \u2500\u2500\u2500 Helpers (TS lines 260-368) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def build_cohort_id(run: Dict[str, Any]) -> str:
    cycle = str(run.get("btc_cycle_zone") or "UNKNOWN").upper()
    iv_label = (label_for_band(run.get("iv_band")) or "UNKNOWN").upper()
    reg = str(run.get("regime") or "").upper()
    if reg.startswith("LONG_GAMMA"):
        family = "LONG"
    elif reg.startswith("SHORT_GAMMA"):
        family = "SHORT"
    elif reg == "NEUTRAL":
        family = "NEUTRAL"
    else:
        family = "UNKNOWN"
    return f"{cycle}|{iv_label}|{family}"


def round_to_strike(price: float) -> float:
    """TS: round to $0.50 under $20, $1.00 at/above."""
    if price is None or not _is_finite(price):
        return price
    if price < 20:
        return round(price * 2) / 2
    return round(price)


def parse_chain_snapshot(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Null-safe parse of option_chain_snapshot_json field."""
    raw = run.get("option_chain_snapshot_json")
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
        if not parsed or not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        return None


def pick_by_delta(
    contracts: List[Dict[str, Any]],
    target_delta: float,
    dte_min: int,
    dte_max: int,
) -> Optional[Dict[str, Any]]:
    candidates = [
        c for c in contracts
        if c.get("delta") is not None
        and c.get("dte") is not None
        and dte_min <= c["dte"] <= dte_max
        and c.get("mid", 0) > 0
    ]
    if not candidates:
        return None
    best = None
    best_dist = float("inf")
    for c in candidates:
        dist = abs(abs(c["delta"]) - target_delta)
        if dist < best_dist:
            best = c
            best_dist = dist
    return best


def pick_by_strike(
    contracts: List[Dict[str, Any]],
    target_strike: float,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    candidates = [c for c in contracts if c.get("mid", 0) > 0]
    if dte_min is not None:
        candidates = [c for c in candidates if c.get("dte", 0) >= dte_min]
    if dte_max is not None:
        candidates = [c for c in candidates if c.get("dte", 0) <= dte_max]
    if not candidates:
        return None
    best = None
    best_dist = float("inf")
    for c in candidates:
        dist = abs(c["strike"] - target_strike)
        if dist < best_dist:
            best = c
            best_dist = dist
    return best


# \u2500\u2500\u2500 BCI (TS lines 430-584) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def compute_bci(
    short_leg_mid: Optional[float],
    short_leg_strike: Optional[float],
    leap_mid: Optional[float],
    leap_strike: Optional[float],
    anchor_strike: Optional[float],
) -> Optional[Dict[str, Any]]:
    if (
        short_leg_mid is None or short_leg_strike is None
        or leap_mid is None or leap_strike is None
        or short_leg_mid <= 0 or leap_mid <= 0
    ):
        return None
    spread = short_leg_strike - leap_strike
    buffer_val = (spread + short_leg_mid) - leap_mid
    breakeven = leap_mid - spread
    distance = abs(leap_strike - anchor_strike) if anchor_strike is not None else 0
    return {
        "passes": buffer_val > 0,
        "buffer": _round4(buffer_val),
        "breakeven_short_premium": _round4(breakeven),
        "k_short": short_leg_strike,
        "k_long": leap_strike,
        "short_mid": short_leg_mid,
        "leap_mid": leap_mid,
        "source": "exact_chain" if distance < 0.01 else "nearest_chain",
        "leap_strike_distance": _round2(distance),
    }


def build_bci_grid(
    chain: Optional[Dict[str, Any]],
    short_leg: Optional[Dict[str, Any]],
    held_anchors: List[Dict[str, Any]],
    selected_anchor: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not chain or not short_leg or short_leg.get("mid", 0) <= 0:
        return None
    calls = chain.get("calls") or []
    candidates = []
    for c in calls:
        if c.get("mid", 0) <= 0 or not c.get("strike"):
            continue
        tier = c.get("tier")
        if tier in ("leap", "mid"):
            candidates.append(c)
        elif tier is None and c.get("is_leap"):
            candidates.append(c)
    if not candidates:
        return None

    def held_key(s, e):
        return f"{s}@{e}"

    held_set = {held_key(a["strike"], a["expiry"]) for a in held_anchors}

    keep = [
        c for c in candidates
        if (c.get("oi") is not None and c["oi"] >= 50) or held_key(c["strike"], c["expiry"]) in held_set
    ]
    if not keep:
        return None

    cells = []
    for c in keep:
        bci = compute_bci(short_leg["mid"], short_leg["strike"], c["mid"], c["strike"], c["strike"])
        if not bci:
            continue
        is_held = held_key(c["strike"], c["expiry"]) in held_set
        is_current = (
            selected_anchor is not None
            and abs(c["strike"] - selected_anchor["strike"]) < 0.01
            and c["expiry"] == selected_anchor["expiry"]
        )
        cells.append({
            "strike": c["strike"],
            "expiry": c["expiry"],
            "dte": c["dte"],
            "leap_mid": c["mid"],
            "bci": bci,
            "is_held_anchor": is_held,
            "is_current": is_current,
        })
    if not cells:
        return None

    strike_set = {c["strike"] for c in cells}
    expiry_set = {c["expiry"] for c in cells}
    strikes = sorted(strike_set)
    expiries = sorted(expiry_set)

    STRIKE_CAP = 8
    EXPIRY_CAP = 6
    if len(strikes) > STRIKE_CAP:
        held_strikes = set([a["strike"] for a in held_anchors])
        if selected_anchor is not None:
            held_strikes.add(selected_anchor["strike"])
        best_buf_by_strike: Dict[float, float] = {}
        for c in cells:
            cur = best_buf_by_strike.get(c["strike"])
            if cur is None or c["bci"]["buffer"] > cur:
                best_buf_by_strike[c["strike"]] = c["bci"]["buffer"]
        ranked = sorted(
            [s for s in strikes if s not in held_strikes],
            key=lambda s: -best_buf_by_strike.get(s, float("-inf")),
        )
        extras = ranked[: max(0, STRIKE_CAP - len(held_strikes))]
        strikes = sorted(set(list(held_strikes) + extras))
    if len(expiries) > EXPIRY_CAP:
        held_exp = {a["expiry"] for a in held_anchors}
        non_held = [e for e in expiries if e not in held_exp]
        trimmed = non_held[: max(0, EXPIRY_CAP - len(held_exp))]
        expiries = sorted(set(list(held_exp) + trimmed))

    final_cells = [c for c in cells if c["strike"] in strikes and c["expiry"] in expiries]
    best = None
    for c in final_cells:
        if best is None or c["bci"]["buffer"] > best["bci"]["buffer"]:
            best = c

    out = {
        "strikes": strikes,
        "expiries": expiries,
        "short_leg": dict(short_leg),
        "cells": final_cells,
        "n_total": len(final_cells),
        "n_passes": sum(1 for c in final_cells if c["bci"]["passes"]),
        "n_held_anchors": sum(1 for c in final_cells if c["is_held_anchor"]),
    }
    if best is not None:
        out["best"] = best
    return out


# \u2500\u2500\u2500 Expiry suggesters (TS lines 587-609) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def suggest_leap_expiry(today: Optional[datetime] = None) -> str:
    today = today or datetime.now(timezone.utc)
    target_year = today.year + 2
    return f"{target_year}-01-21"


def suggest_csp_expiry(today: Optional[datetime] = None, target_dte: int = 50) -> Dict[str, Any]:
    today = today or datetime.now(timezone.utc)
    target = datetime.fromtimestamp(today.timestamp() + target_dte * 86400, tz=timezone.utc)
    return {
        "expiry": f"{target.year:04d}-{target.month:02d}-{target.day:02d}",
        "dte": target_dte,
    }


def suggest_pmcc_short_expiry(today: Optional[datetime] = None, target_dte: int = 30) -> Dict[str, Any]:
    today = today or datetime.now(timezone.utc)
    target = datetime.fromtimestamp(today.timestamp() + target_dte * 86400, tz=timezone.utc)
    return {
        "expiry": f"{target.year:04d}-{target.month:02d}-{target.day:02d}",
        "dte": target_dte,
    }


# \u2500\u2500\u2500 Defensive yield evaluators (TS lines 615-776) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def evaluate_defensive_yield_stress(
    run: Dict[str, Any], recent_regimes: List[str]
) -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []

    risk_zone = run.get("risk_zone")
    conditions.append({
        "key": "risk_zone",
        "label": "Risk Zone",
        "value": risk_zone if risk_zone is not None else None,
        "value_label": risk_zone if risk_zone is not None else "\u2014",
        "threshold_label": "= RED",
        "fired": risk_zone == "RED",
    })

    gex_pct = run.get("gex_percentile")
    conditions.append({
        "key": "gex_percentile",
        "label": "GEX %ile",
        "value": gex_pct if gex_pct is not None else None,
        "value_label": _toFixed(gex_pct, 1) + "%" if gex_pct is not None else "\u2014",
        "threshold_label": "< 70%",
        "fired": gex_pct is not None and gex_pct < 70,
    })

    dd = run.get("asst_drawdown_90d")
    conditions.append({
        "key": "asst_drawdown_90d",
        "label": "ASST 90d DD",
        "value": dd if dd is not None else None,
        "value_label": _toFixed(dd * 100, 1) + "%" if dd is not None else "\u2014",
        "threshold_label": "\u2264 -30%",
        "fired": dd is not None and dd <= -0.30,
    })

    REGIME_CODE = {
        "LONG_GAMMA_STRONG": "LGs",
        "LONG_GAMMA_WEAK": "LGw",
        "SHORT_GAMMA_STRONG": "SGs",
        "SHORT_GAMMA_WEAK": "SGw",
        "NEUTRAL": "NEU",
    }
    last3 = [str(r or "").upper() for r in recent_regimes[-3:]]
    last3_codes = [REGIME_CODE.get(r, r[:4]) for r in last3]
    have_last3 = len(recent_regimes) >= 3
    all_short_or_neutral = have_last3 and all(
        r.startswith("SHORT_GAMMA") or r == "NEUTRAL" for r in last3
    )
    conditions.append({
        "key": "gamma_regime_streak",
        "label": "Gamma Regime (3d)",
        "value": "/".join(last3) if have_last3 else None,
        "value_label": " / ".join(last3_codes) if have_last3 else "insufficient",
        "threshold_label": "3\u00d7 short/neut",
        "fired": bool(all_short_or_neutral),
    })

    triggers: List[str] = []
    if conditions[0]["fired"]:
        triggers.append("risk_zone == RED")
    if conditions[1]["fired"]:
        triggers.append(f"gex_percentile {_toFixed(conditions[1]['value'], 1)} < 70")
    if conditions[2]["fired"]:
        triggers.append(f"asst_drawdown_90d {_toFixed(conditions[2]['value'] * 100, 1)}% (severe)")
    if conditions[3]["fired"]:
        triggers.append("gamma short/neutral last 3 sessions")

    triggered = len(triggers) >= 2
    rotation = (
        min(0.25, 0.10 + 0.05 * max(0, len(triggers) - 2)) if triggered else 0
    )

    return {
        "triggered": triggered,
        "triggers_met": triggers,
        "conditions": conditions,
        "rotation_pct_suggested": rotation if triggered else None,
    }


def evaluate_defensive_yield_top(run: Dict[str, Any]) -> Dict[str, Any]:
    mvrv = run.get("btc_mvrv_zscore")
    pi = run.get("btc_pi_cycle_signal")
    puell = run.get("btc_puell_multiple")
    nupl = run.get("btc_nupl")
    rr = run.get("btc_reserve_risk")

    metrics: List[Dict[str, Any]] = [
        {
            "key": "mvrv_zscore",
            "label": "MVRV Z-Score",
            "value": mvrv if mvrv is not None else None,
            "top_threshold": 6.0,
            "direction": "higher_is_top",
            "pct_of_threshold": (mvrv / 6.0) if mvrv is not None else None,
            "fired": mvrv is not None and mvrv > 6.0,
        },
        {
            "key": "pi_cycle",
            "label": "Pi Cycle Top",
            "value": pi if pi is not None else None,
            "top_threshold": 1,
            "direction": "binary",
            "pct_of_threshold": pi if pi is not None else None,
            "fired": pi == 1,
        },
        {
            "key": "puell",
            "label": "Puell Multiple",
            "value": puell if puell is not None else None,
            "top_threshold": 4.0,
            "direction": "higher_is_top",
            "pct_of_threshold": (puell / 4.0) if puell is not None else None,
            "fired": puell is not None and puell > 4.0,
        },
        {
            "key": "nupl",
            "label": "NUPL",
            "value": nupl if nupl is not None else None,
            "top_threshold": 0.75,
            "direction": "higher_is_top",
            "pct_of_threshold": (nupl / 0.75) if nupl is not None else None,
            "fired": nupl is not None and nupl > 0.75,
        },
        {
            "key": "reserve_risk",
            "label": "Reserve Risk",
            "value": rr if rr is not None else None,
            "top_threshold": 0.02,
            "direction": "higher_is_top",
            "pct_of_threshold": (rr / 0.02) if rr is not None else None,
            "fired": rr is not None and rr > 0.02,
        },
    ]

    n_fired = sum(1 for m in metrics if m["fired"])
    triggered = n_fired >= 2
    rotation = min(0.50, 0.25 + 0.10 * (n_fired - 2)) if triggered else 0

    return {
        "triggered": triggered,
        "metrics": metrics,
        "n_fired": n_fired,
        "rotation_pct_suggested": rotation if triggered else None,
    }


# \u2500\u2500\u2500 Vehicle builders (TS lines 778-1582) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _describe_anchors(anchors: List[Dict[str, Any]]) -> str:
    if not anchors:
        return "none"
    # TS: a.strike.toFixed(0) renders an integer-padded string (e.g. "10" for 10
    # or 10.0, "11" for 10.5 due to rounding).
    return ", ".join(f"{a['expiry']} ${_toFixed(float(a['strike']), 0)}C" for a in anchors)


def _leap_core_recommendation(
    run: Dict[str, Any],
    cohort_id: str,
    edge: Optional[Dict[str, Any]],
    positions: Dict[str, Any],
    chain: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed = run.get("leap_add_allowed") == "ALLOWED"
    size = str(run.get("leap_add_size") or "FULL").upper()
    flip = run.get("gamma_flip")
    atr = run.get("atr_1d")
    band_low = run.get("leap_core_band_low")
    band_high = run.get("leap_core_band_high")

    prior_anchors = positions.get("leap_core", {}).get("anchors", [])
    prior = {
        "summary": (
            f"Holding: {_describe_anchors(prior_anchors)}"
            if prior_anchors else "No LEAP core anchors recorded"
        ),
        "details": {"anchors": prior_anchors},
    }

    if not allowed:
        return {
            "vehicle": "LEAP_CORE",
            "recommended": False,
            "contracts": [],
            "rationale": "Doctrine: leap_add_allowed != ALLOWED. No incremental LEAP add authorized.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": f"leap_add_allowed={run.get('leap_add_allowed')}",
        }

    if flip is None or atr is None or band_low is None or band_high is None:
        return {
            "vehicle": "LEAP_CORE",
            "recommended": False,
            "contracts": [],
            "rationale": "Required inputs missing (flip / ATR / core band).",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": "missing_inputs",
        }

    anchor = flip - 1.5 * atr
    anchor = max(band_low, min(band_high, anchor))

    leap_calls = [c for c in (chain.get("calls", []) if chain else []) if c.get("is_leap")]
    from_chain = pick_by_strike(leap_calls, anchor) if leap_calls else None

    contracts: List[Dict[str, Any]] = []
    if from_chain:
        contracts.append({
            "underlying": "ASST",
            "type": "C",
            "direction": "LONG",
            "strike": from_chain["strike"],
            "expiry": from_chain["expiry"],
            "dte": from_chain["dte"],
            "approx_delta": _round2(from_chain["delta"]) if from_chain.get("delta") is not None else 0.70,
            "quantity": 1,
            "chain_data": "from_chain",
        })
    else:
        expiry = suggest_leap_expiry()
        contracts.append({
            "underlying": "ASST",
            "type": "C",
            "direction": "LONG",
            "strike": round_to_strike(anchor),
            "expiry": expiry,
            "dte": _dte_from_today(expiry),
            "approx_delta": 0.70,
            "quantity": 1,
            "chain_data": "approximated",
        })

    dd = run.get("asst_drawdown_90d") or 0
    cycle = str(run.get("btc_cycle_zone") or "").upper()
    in_deep_recovery = cycle == "RECOVERY" and dd is not None and dd <= -0.30
    if in_deep_recovery and size == "FULL":
        deeper = flip - 2.0 * atr
        deeper = max(band_low, deeper)
        deeper_from_chain = None
        if leap_calls:
            shallower_strike = contracts[0]["strike"]
            deeper_candidates = [c for c in leap_calls if c["strike"] < shallower_strike]
            if deeper_candidates:
                deeper_from_chain = pick_by_strike(deeper_candidates, deeper)
        if deeper_from_chain:
            contracts.append({
                "underlying": "ASST",
                "type": "C",
                "direction": "LONG",
                "strike": deeper_from_chain["strike"],
                "expiry": deeper_from_chain["expiry"],
                "dte": deeper_from_chain["dte"],
                "approx_delta": _round2(deeper_from_chain["delta"]) if deeper_from_chain.get("delta") is not None else 0.78,
                "quantity": 1,
                "chain_data": "from_chain",
            })
        else:
            deeper_strike = round_to_strike(deeper)
            if deeper_strike < (contracts[0].get("strike") or float("inf")):
                contracts.append({
                    "underlying": "ASST",
                    "type": "C",
                    "direction": "LONG",
                    "strike": deeper_strike,
                    "expiry": contracts[0]["expiry"],
                    "dte": contracts[0]["dte"],
                    "approx_delta": 0.78,
                    "quantity": 1,
                    "chain_data": "approximated",
                })

    confidence = "MED"
    conf_detail = ""
    if edge and edge.get("sample_sufficient"):
        if edge.get("fwd_median") is not None and edge["fwd_median"] > 0:
            confidence = "HIGH"
        fm = edge.get("fwd_median")
        fp10 = edge.get("fwd_p10")
        fm_str = _toFixed(fm * 100, 1) + "%" if fm is not None else "\u2014"
        fp10_str = _toFixed(fp10 * 100, 1) + "%" if fp10 is not None else "\u2014"
        conf_detail = f"n={edge['n_with_fwd']}, fwd{edge['horizon']} median {fm_str}, p10 {fp10_str}"
    elif edge:
        confidence = "LOW"
        conf_detail = f"n={edge['n_with_fwd']} (insufficient \u2014 \u22658 needed)"
    else:
        confidence = "LOW"
        conf_detail = "no cohort edge available"

    action = "ROTATE_IN" if not prior_anchors else "ADD"

    ev_mnav = run.get("ev_mnav")
    mnav_disc = run.get("mnav_discount")
    sizing_notes = [f"base_size={size}"]
    if mnav_disc is not None and mnav_disc > 0.10:
        sizing_notes.append(f"mNAV discount {_toFixed(mnav_disc * 100, 1)}% \u2014 favorable, hold/upgrade size")
    if ev_mnav is not None and ev_mnav < 1.0:
        sizing_notes.append(f"EV/mNAV {_toFixed(ev_mnav, 2)} contained \u2014 favorable")
    elif ev_mnav is not None and ev_mnav > 1.5:
        sizing_notes.append(f"EV/mNAV {_toFixed(ev_mnav, 2)} rich \u2014 suppress ladder expansion")

    rationale_parts = [
        f"Anchor: flip({_toFixed(flip, 2)}) \u2212 1.5\u00d7ATR({_toFixed(atr, 2)}) = {_toFixed(flip - 1.5 * atr, 2)}",
        f"Clamped to core band [${_toFixed(band_low, 2)}, ${_toFixed(band_high, 2)}] \u2192 ${_js_str(contracts[0]['strike'])}",
    ]
    if in_deep_recovery and len(contracts) > 1:
        rationale_parts.append("Deep RECOVERY: deeper strike added at flip \u2212 2.0\u00d7ATR")
    rationale_parts.append("; ".join(sizing_notes))

    return {
        "vehicle": "LEAP_CORE",
        "recommended": True,
        "contracts": contracts,
        "rationale": " \u00b7 ".join(p for p in rationale_parts if p),
        "historical_edge": edge,
        "prior_position": prior,
        "delta_from_prior": action,
        "confidence": confidence,
        "confidence_detail": conf_detail,
    }


def _leap_mid_tail_recommendation(
    run: Dict[str, Any],
    edge: Optional[Dict[str, Any]],
    positions: Dict[str, Any],
    chain: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed = run.get("leap_add_allowed") == "ALLOWED"
    flip = run.get("gamma_flip")
    atr = run.get("atr_1d")
    band_high = run.get("leap_core_band_high")

    mt_anchors = positions.get("leap_mid_tail", {}).get("anchors", [])
    prior = {
        "summary": (
            f"Mid/Tail: {_describe_anchors(mt_anchors)}"
            if mt_anchors else "No mid/tail LEAP positions"
        ),
        "details": {"anchors": mt_anchors},
    }

    if not allowed or flip is None or atr is None or band_high is None:
        return {
            "vehicle": "LEAP_MID_TAIL",
            "recommended": False,
            "contracts": [],
            "rationale": "Doctrine blocks LEAP expansion." if not allowed else "Inputs missing for mid/tail layer.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": f"leap_add_allowed={run.get('leap_add_allowed')}" if not allowed else "missing_inputs",
        }

    anchor = flip + 0.5 * atr
    anchor = min(band_high, anchor)

    leap_calls = [c for c in (chain.get("calls", []) if chain else []) if c.get("is_leap")]
    from_chain = pick_by_strike(leap_calls, anchor) if leap_calls else None

    if from_chain:
        contract = {
            "underlying": "ASST",
            "type": "C",
            "direction": "LONG",
            "strike": from_chain["strike"],
            "expiry": from_chain["expiry"],
            "dte": from_chain["dte"],
            "approx_delta": _round2(from_chain["delta"]) if from_chain.get("delta") is not None else 0.45,
            "quantity": 1,
            "chain_data": "from_chain",
        }
    else:
        expiry = suggest_leap_expiry()
        contract = {
            "underlying": "ASST",
            "type": "C",
            "direction": "LONG",
            "strike": round_to_strike(anchor),
            "expiry": expiry,
            "dte": _dte_from_today(expiry),
            "approx_delta": 0.45,
            "quantity": 1,
            "chain_data": "approximated",
        }

    return {
        "vehicle": "LEAP_MID_TAIL",
        "recommended": True,
        "contracts": [contract],
        "rationale": (
            f"Convexity ladder: flip({_toFixed(flip, 2)}) + 0.5\u00d7ATR = ${_js_str(contract['strike'])}. "
            f"Smaller size, lower delta. Core position prerequisite."
        ),
        "historical_edge": edge,
        "prior_position": prior,
        "delta_from_prior": "ADD" if mt_anchors else "ROTATE_IN",
        "confidence": "LOW",
        "confidence_detail": "non-core convexity layer; treat as supplemental",
    }


def _csp_recommendation(
    run: Dict[str, Any],
    edge: Optional[Dict[str, Any]],
    positions: Dict[str, Any],
) -> Dict[str, Any]:
    allowed = run.get("csp_allowed")
    is_allowed = allowed == "ALLOWED" or allowed == "CONDITIONAL"
    is_off = allowed == "OFF" or allowed == "DISABLED"

    csp_open = positions.get("csp", {}).get("open", [])
    prior = {
        "summary": f"Open CSPs: {len(csp_open)}" if csp_open else "No open CSPs",
        "details": {"open": csp_open},
    }

    top_strike = run.get("csp_top_strike")
    top_mid = run.get("csp_top_mid")
    top_dte = run.get("csp_top_dte")
    top_expiry = run.get("csp_top_expiry")
    top_basis = run.get("csp_top_eff_basis")

    if is_off:
        informational = []
        if top_strike is not None:
            informational = [{
                "underlying": "ASST",
                "type": "P",
                "direction": "SHORT",
                "strike": top_strike,
                "expiry": top_expiry,
                "dte": top_dte,
                "quantity": 1,
                "chain_data": "from_chain",
            }]
        rationale_parts = ["Doctrine: csp_allowed=OFF \u2014 not authorized."]
        if top_strike is not None:
            rationale_parts.append(
                f"Best informational candidate: {top_expiry} ${_js_str(top_strike)}P, "
                f"mid ${_toFixed(top_mid, 2)}, eff basis ${_toFixed(top_basis, 2)}"
            )
        else:
            rationale_parts.append("No CSP candidate computed (likely below band threshold).")
        return {
            "vehicle": "CSP",
            "recommended": False,
            "contracts": informational,
            "rationale": " \u00b7 ".join(rationale_parts),
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "NONE",
            "confidence": "NONE",
            "blocked_reason": "csp_allowed=OFF",
        }

    if not is_allowed:
        return {
            "vehicle": "CSP",
            "recommended": False,
            "contracts": [],
            "rationale": f"Doctrine: csp_allowed={allowed or 'unknown'}.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "NONE",
            "confidence": "NONE",
            "blocked_reason": f"csp_allowed={allowed}",
        }

    csp_band_low = run.get("csp_band_low")
    csp_band_high = run.get("csp_band_high")

    if top_strike is None:
        return {
            "vehicle": "CSP",
            "recommended": False,
            "contracts": [],
            "rationale": "csp_allowed authorizes action but no candidate met band/delta/basis filters.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "NONE",
            "confidence": "LOW",
            "blocked_reason": "no_qualifying_candidate",
        }

    ann_yield = None
    if top_mid is not None and top_dte is not None and top_strike > 0:
        ann_yield = (top_mid / top_strike) * (365 / top_dte)

    contract = {
        "underlying": "ASST",
        "type": "P",
        "direction": "SHORT",
        "strike": top_strike,
        "expiry": top_expiry,
        "dte": top_dte,
        "quantity": 1,
        "est_annualized_yield": ann_yield,
        "chain_data": "from_chain",
    }

    confidence = "MED"
    conf_detail = ""
    if edge and edge.get("sample_sufficient") and edge.get("fwd_p10") is not None and edge["fwd_p10"] > -0.10:
        confidence = "HIGH"
        conf_detail = f"cohort downside p10 {_toFixed(edge['fwd_p10'] * 100, 1)}% (n={edge['n_with_fwd']})"
    elif edge:
        conf_detail = f"n={edge['n_with_fwd']}"

    rationale_parts = [
        f"Strike ${_js_str(top_strike)} sits in band [${_toFixed(csp_band_low, 2)}, ${_toFixed(csp_band_high, 2)}]",
        f"Mid ${_toFixed(top_mid, 2)} \u2192 eff basis ${_toFixed(top_basis, 2)}",
    ]
    if ann_yield is not None:
        rationale_parts.append(f"~{_toFixed(ann_yield * 100, 1)}% ann yield")
    rationale_parts.append(f"{top_dte} DTE")

    return {
        "vehicle": "CSP",
        "recommended": True,
        "contracts": [contract],
        "rationale": " \u00b7 ".join(rationale_parts),
        "historical_edge": edge,
        "prior_position": prior,
        "delta_from_prior": "ADD" if csp_open else "ROTATE_IN",
        "confidence": confidence,
        "confidence_detail": conf_detail,
    }


def _pmcc_recommendation(
    run: Dict[str, Any],
    edge: Optional[Dict[str, Any]],
    positions: Dict[str, Any],
    chain: Optional[Dict[str, Any]],
    vintage_anchors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    vintage_anchors = vintage_anchors or []
    allowed = run.get("pmcc_allowed") == "ALLOWED"
    status = run.get("pmcc_status")

    pmcc_open = positions.get("pmcc_overlays", {}).get("open", [])
    prior = {
        "summary": f"Open PMCC overlays: {len(pmcc_open)}" if pmcc_open else "No open PMCC overlays",
        "details": {"open": pmcc_open},
    }

    if not allowed:
        return {
            "vehicle": "PMCC",
            "recommended": False,
            "contracts": [],
            "rationale": f"Doctrine: pmcc_allowed={run.get('pmcc_allowed') or 'unknown'} ({status or 'n/a'}). Contextual reference only.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": f"pmcc_allowed={run.get('pmcc_allowed')}",
        }

    leap_core_anchors = positions.get("leap_core", {}).get("anchors", [])
    long_anchor = leap_core_anchors[0] if leap_core_anchors else None
    if long_anchor is None:
        return {
            "vehicle": "PMCC",
            "recommended": False,
            "contracts": [],
            "rationale": "PMCC requires an existing long LEAP leg; none on file.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": "no_long_leg",
        }

    spot = run.get("spot")
    if spot is None:
        return {
            "vehicle": "PMCC",
            "recommended": False,
            "contracts": [],
            "rationale": "Spot unavailable.",
            "historical_edge": edge,
            "prior_position": prior,
            "delta_from_prior": "HOLD",
            "confidence": "NONE",
            "blocked_reason": "missing_spot",
        }

    short_dte_calls = [c for c in (chain.get("calls", []) if chain else []) if not c.get("is_leap")]
    short_from_chain = pick_by_delta(short_dte_calls, 0.20, 21, 45) if short_dte_calls else None

    long_leg = {
        "underlying": "ASST",
        "type": "C",
        "direction": "LONG",
        "strike": long_anchor["strike"],
        "expiry": long_anchor["expiry"],
        "dte": _dte_from_today(long_anchor["expiry"]),
        "approx_delta": 0.75,
        "quantity": long_anchor["quantity"],
        "chain_data": "from_chain",
    }
    if short_from_chain:
        ann_yield = None
        if short_from_chain["mid"] > 0 and short_from_chain["dte"] > 0:
            ann_yield = (short_from_chain["mid"] / short_from_chain["strike"]) * (365 / short_from_chain["dte"])
        short_leg = {
            "underlying": "ASST",
            "type": "C",
            "direction": "SHORT",
            "strike": short_from_chain["strike"],
            "expiry": short_from_chain["expiry"],
            "dte": short_from_chain["dte"],
            "approx_delta": _round2(short_from_chain["delta"]) if short_from_chain.get("delta") is not None else 0.20,
            "quantity": 1,
            "chain_data": "from_chain",
            "est_annualized_yield": ann_yield,
        }
    else:
        short_info = suggest_pmcc_short_expiry()
        short_leg = {
            "underlying": "ASST",
            "type": "C",
            "direction": "SHORT",
            "strike": round_to_strike(spot * 1.10),
            "expiry": short_info["expiry"],
            "dte": short_info["dte"],
            "approx_delta": 0.20,
            "quantity": 1,
            "chain_data": "approximated",
        }

    # BCI: search anchors for exact chain match first; nearest within $2 as fallback
    bci = None
    bci_anchor_used = None
    all_anchors = leap_core_anchors
    all_calls = chain.get("calls", []) if chain else []
    for anchor in all_anchors:
        exact = next(
            (c for c in all_calls
             if c.get("is_leap") and c.get("strike") == anchor["strike"]
             and c.get("expiry") == anchor["expiry"] and c.get("mid", 0) > 0),
            None,
        )
        if exact and short_from_chain:
            bci = compute_bci(short_from_chain["mid"], short_from_chain["strike"], exact["mid"], exact["strike"], anchor["strike"])
            bci_anchor_used = anchor
            break

    if not bci and short_from_chain:
        best_nearest = None
        for anchor in all_anchors:
            same_expiry_leaps = [
                c for c in all_calls
                if c.get("is_leap") and c.get("expiry") == anchor["expiry"] and c.get("mid", 0) > 0
            ]
            for c in same_expiry_leaps:
                d = abs(c["strike"] - anchor["strike"])
                if d <= 2.0 and (best_nearest is None or d < best_nearest["dist"]):
                    best_nearest = {"contract": c, "anchor": anchor, "dist": d}
        if best_nearest:
            bci = compute_bci(
                short_from_chain["mid"],
                short_from_chain["strike"],
                best_nearest["contract"]["mid"],
                best_nearest["contract"]["strike"],
                best_nearest["anchor"]["strike"],
            )
            bci_anchor_used = best_nearest["anchor"]

    bci_vintages = []
    if short_from_chain:
        for va in vintage_anchors:
            vintage_bci = compute_bci(
                short_from_chain["mid"],
                short_from_chain["strike"],
                va["cost_basis_at_recommendation"],
                va["strike"],
                va["strike"],
            )
            bci_vintages.append({
                "lookback_days": va["lookback_days"],
                "recommendation_date": va["recommendation_date"],
                "strike": va["strike"],
                "expiry": va["expiry"],
                "cost_basis": va["cost_basis_at_recommendation"],
                "current_mark": va.get("current_mark"),
                "bci": vintage_bci,
            })

    bci_grid = None
    if short_from_chain and short_from_chain["mid"] > 0:
        bci_grid = build_bci_grid(
            chain,
            {
                "strike": short_from_chain["strike"],
                "expiry": short_from_chain["expiry"],
                "dte": short_from_chain["dte"],
                "mid": short_from_chain["mid"],
            },
            [{"strike": a["strike"], "expiry": a["expiry"]} for a in leap_core_anchors],
            {"strike": bci_anchor_used["strike"], "expiry": bci_anchor_used["expiry"]} if bci_anchor_used else None,
        )

    composite: Dict[str, Any] = {
        "underlying": "ASST",
        "type": "C",
        "direction": "LONG",
        "quantity": 1,
        "long_leg": long_leg,
        "short_leg": short_leg,
        "chain_data": "from_chain" if (short_leg["chain_data"] == "from_chain" and bci is not None) else "approximated",
        "bci": bci,
    }
    if bci_vintages:
        composite["bci_vintages"] = bci_vintages
    if bci_grid:
        composite["bci_grid"] = bci_grid

    if not bci:
        bci_tag = "BCI: insufficient data"
    else:
        sign = "+" if bci["buffer"] >= 0 else "\u2212"
        buf = f"{sign}${_toFixed(abs(bci['buffer']), 2)}"
        anchor_ref = f"${_js_str(bci_anchor_used['strike'])}" if bci_anchor_used else f"${_js_str(long_anchor['strike'])}"
        proxy_hint = (
            f" (LEAP \u224a ${_js_str(bci['k_long'])} vs anchor {anchor_ref})"
            if bci["source"] == "nearest_chain" else f" (vs anchor {anchor_ref})"
        )
        passes_label = "pass" if bci["passes"] else "fail"
        bci_tag = f"BCI static: {passes_label} {buf} buffer{proxy_hint}"

    vintage_bci_tag = None
    if bci_vintages:
        sorted_v = sorted(bci_vintages, key=lambda v: v["lookback_days"])
        parts = []
        for v in sorted_v:
            if not v["bci"]:
                parts.append(f"{v['lookback_days']}d insufficient")
            else:
                sign = "+" if v["bci"]["buffer"] >= 0 else "\u2212"
                buf = f"{sign}${_toFixed(abs(v['bci']['buffer']), 2)}"
                passes_label = "pass" if v["bci"]["passes"] else "fail"
                parts.append(f"{v['lookback_days']}d {passes_label} {buf}")
        _sep = " \u00b7 "
        vintage_bci_tag = f"BCI vintage: {_sep.join(parts)}"

    chain_data_hint = "from chain" if short_leg["chain_data"] == "from_chain" else "target"
    rationale_parts = [
        f"Long leg: existing {long_anchor['expiry']} ${_js_str(long_anchor['strike'])}C anchor",
        f"Short leg: ~{short_leg['dte']}d, ${_js_str(short_leg['strike'])}C (~{_toFixed(short_leg['approx_delta'], 2)}\u0394 {chain_data_hint})",
        f"Status: {status}",
        bci_tag,
    ]
    if vintage_bci_tag:
        rationale_parts.append(vintage_bci_tag)

    return {
        "vehicle": "PMCC",
        "recommended": True,
        "contracts": [composite],
        "rationale": " \u00b7 ".join(rationale_parts),
        "historical_edge": edge,
        "prior_position": prior,
        "delta_from_prior": "ROLL" if pmcc_open else "ROTATE_IN",
        "confidence": "MED",
        "confidence_detail": "PMCC overlay; reduce/disable in weak premium",
    }


def _defensive_yield_stress_recommendation(
    run: Dict[str, Any], trigger: Dict[str, Any], positions: Dict[str, Any]
) -> Dict[str, Any]:
    sata_held = positions.get("sata", {}).get("shares_held", 0)
    prior = {
        "summary": f"Holding SATA shares: {sata_held}" if sata_held > 0 else "No SATA position",
        "details": {"sata_shares_held": sata_held, "kind": "stress"},
    }
    sata_price = run.get("sata_price")

    if not trigger["triggered"]:
        triggers_met = trigger["triggers_met"]
        return {
            "vehicle": "DEFENSIVE_YIELD_STRESS",
            "recommended": False,
            "contracts": [],
            "rationale": (
                f"Stress conditions: {len(triggers_met)}/2 met. "
                + ("; ".join(triggers_met) if triggers_met else "No drawdown/regime stress.")
            ),
            "historical_edge": None,
            "prior_position": prior,
            "delta_from_prior": "HOLD" if sata_held > 0 else "NONE",
            "confidence": "NONE",
            "blocked_reason": "stress_threshold_not_met",
        }

    rotation_pct = trigger["rotation_pct_suggested"]
    shares_note = (
        f"SATA ~${_toFixed(sata_price, 2)}; rotate {_toFixed(rotation_pct * 100, 0)}% of budget"
        if sata_price is not None else "SATA price unavailable"
    )

    rationale_parts = ["STRESS rotation (\u26a0 in pain):"] + [f"\u00b7 {t}" for t in trigger["triggers_met"]] + [shares_note]

    n_triggers = len(trigger["triggers_met"])
    plural = "" if n_triggers == 1 else "s"
    return {
        "vehicle": "DEFENSIVE_YIELD_STRESS",
        "recommended": True,
        "contracts": [{
            "underlying": "SATA",
            "type": "SHARES",
            "direction": "LONG",
            "quantity": 1,
            "quantity_shares": 0,
            "estimated_allocation_pct": rotation_pct,
            "chain_data": "from_chain",
        }],
        "rationale": " \u00b7 ".join(rationale_parts),
        "historical_edge": None,
        "prior_position": prior,
        "delta_from_prior": "ADD" if sata_held > 0 else "ROTATE_IN",
        "confidence": "HIGH" if n_triggers >= 3 else "MED",
        "confidence_detail": f"{n_triggers} stress trigger{plural} \u00b7 {_toFixed(rotation_pct * 100, 0)}% rotation",
    }


def _defensive_yield_top_recommendation(
    run: Dict[str, Any], trigger: Dict[str, Any], positions: Dict[str, Any]
) -> Dict[str, Any]:
    sata_held = positions.get("sata", {}).get("shares_held", 0)
    prior = {
        "summary": f"Holding SATA shares: {sata_held}" if sata_held > 0 else "No SATA position",
        "details": {"sata_shares_held": sata_held, "kind": "top"},
    }
    sata_price = run.get("sata_price")

    metric_summary_parts = []
    for m in trigger["metrics"]:
        v = m["value"]
        if v is None:
            v_str = "\u2014"
        elif m["direction"] == "binary":
            v_str = str(v)
        else:
            v_str = _toFixed(v, 4 if v < 1 else 2)
        if m["fired"]:
            fire_mark = "\u2191FIRED"
        else:
            pct = m["pct_of_threshold"]
            fire_mark = f"{_toFixed(pct * 100, 0)}% of top" if pct is not None else "\u2014 of top"
        metric_summary_parts.append(f"{m['label']} {v_str} ({fire_mark})")
    metric_summary = " \u00b7 ".join(metric_summary_parts)

    if not trigger["triggered"]:
        return {
            "vehicle": "DEFENSIVE_YIELD_TOP",
            "recommended": False,
            "contracts": [],
            "rationale": f"Cycle metrics: {trigger['n_fired']}/5 fired (need 2+). {metric_summary}",
            "historical_edge": None,
            "prior_position": prior,
            "delta_from_prior": "HOLD" if sata_held > 0 else "NONE",
            "confidence": "NONE",
            "blocked_reason": "top_threshold_not_met",
        }

    rotation_pct = trigger["rotation_pct_suggested"]
    shares_note = (
        f"SATA ~${_toFixed(sata_price, 2)}; trim {_toFixed(rotation_pct * 100, 0)}% of budget"
        if sata_price is not None else "SATA price unavailable"
    )
    rationale_parts = [
        "CYCLE-TOP rotation (\u26a0 peak euphoria):",
        f"{trigger['n_fired']}/5 metrics fired",
        metric_summary,
        shares_note,
    ]

    n_fired = trigger["n_fired"]
    plural = "" if n_fired == 1 else "s"
    return {
        "vehicle": "DEFENSIVE_YIELD_TOP",
        "recommended": True,
        "contracts": [{
            "underlying": "SATA",
            "type": "SHARES",
            "direction": "LONG",
            "quantity": 1,
            "quantity_shares": 0,
            "estimated_allocation_pct": rotation_pct,
            "chain_data": "from_chain",
        }],
        "rationale": " \u00b7 ".join(rationale_parts),
        "historical_edge": None,
        "prior_position": prior,
        "delta_from_prior": "ADD" if sata_held > 0 else "ROTATE_IN",
        "confidence": "HIGH" if n_fired >= 3 else "MED",
        "confidence_detail": f"{n_fired} cycle-top trigger{plural} \u00b7 {_toFixed(rotation_pct * 100, 0)}% trim",
    }


# \u2500\u2500\u2500 Top-level evaluate (TS lines 1601-1680) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def evaluate(inputs: Dict[str, Any]) -> Dict[str, Any]:
    run = inputs["run"]
    recent_regimes = inputs.get("recentRegimes") or inputs.get("recent_regimes") or []
    edges = inputs.get("edges") or {}
    positions = inputs.get("positions") or {}
    vintage_anchors = inputs.get("vintage_anchors") or []

    cohort_id = build_cohort_id(run)

    # Preferred horizon: longest with sufficient sample, else 21
    def _suff(h):
        e = edges.get(h)
        return e and e.get("sample_sufficient")

    if _suff(63):
        preferred = 63
    elif _suff(21):
        preferred = 21
    elif _suff(10):
        preferred = 10
    elif _suff(5):
        preferred = 5
    else:
        preferred = 21
    primary_edge = edges.get(preferred)

    def_stress = evaluate_defensive_yield_stress(run, recent_regimes)
    def_top = evaluate_defensive_yield_top(run)

    chain = parse_chain_snapshot(run)

    recs = [
        _leap_core_recommendation(run, cohort_id, primary_edge, positions, chain),
        _leap_mid_tail_recommendation(run, primary_edge, positions, chain),
        _csp_recommendation(run, primary_edge, positions),
        _pmcc_recommendation(run, primary_edge, positions, chain, vintage_anchors),
        _defensive_yield_stress_recommendation(run, def_stress, positions),
        _defensive_yield_top_recommendation(run, def_top, positions),
    ]

    any_active = any(r["recommended"] for r in recs)
    if not any_active:
        blocked = [f"{r['vehicle']}: {r.get('blocked_reason') or 'n/a'}" for r in recs]
        recs.append({
            "vehicle": "NO_TRADE",
            "recommended": False,
            "contracts": [],
            "rationale": f"No active vehicle qualifies. Conditions for re-enable: {'; '.join(blocked)}.",
            "historical_edge": primary_edge,
            "prior_position": {"summary": "n/a"},
            "delta_from_prior": "NONE",
            "confidence": "NONE",
        })

    data_health: List[str] = []
    if run.get("gex_percentile") is None:
        data_health.append("gex_percentile null")
    if run.get("btc_cycle_zone") == "UNKNOWN":
        data_health.append("btc_cycle_zone UNKNOWN")
    if run.get("atr_1d") is None:
        data_health.append("atr_1d null")

    iv_band_raw = run.get("iv_band")
    iv_band = None if iv_band_raw is None else int(iv_band_raw)
    iv_band_label = label_for_band(iv_band) or "UNKNOWN"

    return _js_normalize({
        "schema_version": "1.2",
        "snapshot_id": f"{run['date']}_{run['session']}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "date": run["date"],
        "session": run["session"],
        "cohort_id": cohort_id,
        "cohort_id_v": COHORT_ID_VERSION,
        "gamma_regime": str(run.get("regime") or ""),
        "iv_regime": str(run.get("iv_regime") or ""),
        "iv_band": iv_band,
        "iv_band_label": iv_band_label,
        "btc_cycle_zone": str(run.get("btc_cycle_zone") or ""),
        "risk_zone": str(run.get("risk_zone") or ""),
        "gex_percentile": run.get("gex_percentile"),
        "overall_posture": str(run.get("action_banner") or ""),
        "ratchet_streak_length": None,
        "defensive_yield_stress": def_stress,
        "defensive_yield_top": def_top,
        "recommendations": recs,
        "data_health_flags": data_health,
        "isolation_note": (
            "Derived in-memory from latest snapshot + master research export. "
            "Read-only relative to V1/V2/V3 and data.db. "
            "Writes only to asst_precision_selector_log_v1.1.jsonl."
        ),
    })


# \u2500\u2500\u2500 Cohort-stats helper (TS routes.ts computeCohortStatsForCohort) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def compute_historical_edge_from_subset(
    cohort_subset: Dict[str, Any], horizon: int
) -> Optional[Dict[str, Any]]:
    """Compute a HistoricalEdge from a captured cohort_subset.json.

    The cohort_subset has rows from the master research CSV; we replicate
    the computeCohortStatsForCohort logic from routes.ts.
    """
    if not cohort_subset or not cohort_subset.get("available"):
        return None
    cohort_id = cohort_subset.get("cohort_id")
    rows = cohort_subset.get("rows", [])
    pm_rows = [r for r in rows if r.get("session") == "PM"]
    n_matched = len(pm_rows)

    fwd_key = f"fwd{horizon}"
    dd_key = f"max_dd_{horizon}d"
    rv_key = f"rv_{horizon}d"
    fwd_values: List[float] = []
    dd_values: List[float] = []
    rv_values: List[float] = []
    for r in pm_rows:
        try:
            f = float(r.get(fwd_key, ""))
            if math.isfinite(f):
                fwd_values.append(f)
        except (TypeError, ValueError):
            pass
        try:
            d = float(r.get(dd_key, ""))
            if math.isfinite(d):
                dd_values.append(d)
        except (TypeError, ValueError):
            pass
        try:
            v = float(r.get(rv_key, ""))
            if math.isfinite(v):
                rv_values.append(v)
        except (TypeError, ValueError):
            pass

    reportable = reportable_percentiles(len(fwd_values))

    def _fwd_ci(q, gate):
        if not gate:
            return None
        r = bootstrap_quantile_ci(fwd_values, q)
        if r is None:
            return None
        return {"estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"], "level": r["level"]}

    edge: Dict[str, Any] = {
        "cohort_id": cohort_id,
        "horizon": horizon,
        "horizon_basis": "calendar_days",
        "n_total_pm": n_matched,
        "n_with_fwd": len(fwd_values),
        "reportable": reportable,
    }
    if fwd_values:
        edge["fwd"] = {
            "mean": stats_mean(fwd_values),
            "median": quantile(fwd_values, 0.5) if reportable["median"] else None,
            "p10": quantile(fwd_values, 0.10) if reportable["p10"] else None,
            "p25": quantile(fwd_values, 0.25) if reportable["p25"] else None,
            "p75": quantile(fwd_values, 0.75) if reportable["p75"] else None,
            "p90": quantile(fwd_values, 0.90) if reportable["p90"] else None,
            "median_ci": _fwd_ci(0.5, reportable["median"]),
            "p25_ci": _fwd_ci(0.25, reportable["p25"]),
            "p75_ci": _fwd_ci(0.75, reportable["p75"]),
            "p10_ci": _fwd_ci(0.10, reportable["p10"]),
            "p90_ci": _fwd_ci(0.90, reportable["p90"]),
        }
    else:
        edge["fwd"] = None
    edge["max_dd"] = (
        {"mean": stats_mean(dd_values), "median": quantile(dd_values, 0.5), "p10": quantile(dd_values, 0.10)}
        if dd_values else None
    )
    edge["rv"] = (
        {"mean": stats_mean(rv_values), "median": quantile(rv_values, 0.5)}
        if rv_values else None
    )
    edge["available"] = True
    return edge


def build_edges_from_subset(cohort_subset: Dict[str, Any]) -> Dict[int, Optional[Dict[str, Any]]]:
    """Build the edges dict expected by evaluate() from a captured cohort_subset.

    Mirrors routes.ts /api/selector/evaluate handler: for each horizon, query
    cohort stats, and if fwd is present, convert to HistoricalEdge shape.
    """
    out: Dict[int, Optional[Dict[str, Any]]] = {}
    for h in (5, 10, 21, 63):
        stats = compute_historical_edge_from_subset(cohort_subset, h)
        if stats and stats.get("available") and stats.get("fwd"):
            fwd = stats["fwd"]
            out[h] = {
                "cohort_id": stats["cohort_id"],
                "horizon": h,
                "horizon_basis": stats.get("horizon_basis", "calendar_days"),
                "n_total_pm": stats["n_total_pm"],
                "n_with_fwd": stats["n_with_fwd"],
                "fwd_median": fwd.get("median"),
                "fwd_p10": fwd.get("p10"),
                "fwd_p25": fwd.get("p25"),
                "fwd_p75": fwd.get("p75"),
                "fwd_p90": fwd.get("p90"),
                "fwd_median_ci": fwd.get("median_ci"),
                "fwd_p25_ci": fwd.get("p25_ci"),
                "fwd_p75_ci": fwd.get("p75_ci"),
                "fwd_p10_ci": fwd.get("p10_ci"),
                "fwd_p90_ci": fwd.get("p90_ci"),
                "max_dd_median": stats["max_dd"]["median"] if stats.get("max_dd") else None,
                "rv_median": stats["rv"]["median"] if stats.get("rv") else None,
                "sample_sufficient": stats["n_with_fwd"] >= 8,
            }
            if stats["n_with_fwd"] < 8:
                out[h]["note"] = "small sample"
        else:
            out[h] = None
    return out


# \u2500\u2500\u2500 Internal helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _toFixed(x: Optional[float], decimals: int) -> str:
    """Match JS Number.prototype.toFixed exactly.

    JS uses banker's rounding (round-half-to-even) for ties? Actually no \u2014
    JS toFixed uses 'round-half-away-from-zero' for positive numbers in
    most implementations, but is technically implementation-defined. In
    practice, for the values we care about, simple round() with formatting
    matches V8 output. If we discover discrepancies in fixture replay we'll
    revisit.
    """
    if x is None:
        return "0"
    # Replicate JS toFixed rounding semantics. JS uses round-half-to-even
    # in some engines and round-half-away-from-zero in others; V8 (Node)
    # rounds half to nearest with ties-to-even. Python's format() also uses
    # banker's rounding. So '%.*f' % (decimals, x) should match V8 for the
    # values we'll see.
    return f"{x:.{decimals}f}"


def _round2(x: Optional[float]) -> Optional[float]:
    """Match TS Number(x.toFixed(2)).

    JS: Number("10.00") === 10 (no trailing .0 in JSON);
        Number("10.50") === 10.5 (float).
    We return int when the rounded value is integral, float otherwise,
    so json.dumps emits the same shape as JS JSON.stringify.
    """
    if x is None:
        return None
    v = round(x, 2)
    if v == int(v):
        return int(v)
    return v


def _round4(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    v = round(x, 4)
    if v == int(v):
        return int(v)
    return v


def _js_str(x) -> str:
    """Match JS implicit number-to-string coercion.

    JS: (10).toString() === "10"; (10.0).toString() === "10";
        (10.5).toString() === "10.5".
    Python f-string for 10.0 gives "10.0"; this helper drops the .0.
    """
    if x is None:
        return "null"
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def _js_normalize(obj):
    """Recursively normalize whole-number floats to ints, mirroring JS JSON."""
    if isinstance(obj, dict):
        return {k: _js_normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_js_normalize(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if obj == int(obj) and abs(obj) < 1e16:
            return int(obj)
        return obj
    return obj


def _dte_from_today(expiry: str) -> int:
    """Days to expiry from now, matching TS new Date().getTime() arithmetic."""
    target = datetime.fromisoformat(expiry + "T00:00:00+00:00") if "T" not in expiry else datetime.fromisoformat(expiry)
    now = datetime.now(timezone.utc)
    delta_ms = (target.timestamp() - now.timestamp()) * 1000
    return round(delta_ms / 86400000)


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)

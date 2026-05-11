"""
Contract Suggestion Engine for Terminal v2.

Consumes raw Terminal 1 fields + option chain data to produce
structured CSP and LEAP suggestion objects per run.

All thresholds are configurable constants at the top of the file.
Does NOT modify Terminal 1 schema or data.

v2.4 — schema-and-semantics hardening:
  * 6-state typed status enum: unavailable | not_applicable | no_candidate |
    monitor | actionable | blocked
  * Suggestion envelope: { status, status_detail, blockers[], candidate, ... }
  * to_envelope() normalizes legacy {none, watch, recommend} suggestions to the
    new contract without changing internal engine logic.
"""

import math
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# Configurable Constants
# ═══════════════════════════════════════════════════════════════════════════════

# -- CSP Gates & Filters --
CSP_GEX_FLOOR_PERCENTILE = 15.0     # Below this = "deeply negative" → block CSPs
CSP_DELTA_MIN = -0.35               # Widest acceptable put delta
CSP_DELTA_MAX = -0.20               # Shallowest acceptable put delta
CSP_DELTA_TARGET_MIN = -0.30        # Preferred band
CSP_DELTA_TARGET_MAX = -0.25        # Preferred band
CSP_PER_LEG_RISK_CAP = 0.02        # Max 2% of portfolio per leg
CSP_COLLATERAL_CAP = 0.25          # Max 25% of equity in CSP collateral
CSP_DTE_CHEAP_VOL = (45, 60)       # DTE range when IV is cheap
CSP_DTE_DEFAULT = (30, 45)         # DTE range when IV is mid/unknown

# -- LEAP Gates & Filters --
LEAP_IV_RANK_MAX = 40.0             # IV rank must be below this to recommend
LEAP_IV_PERCENTILE_MAX = 40.0       # IV percentile must be below this to recommend
LEAP_PER_LEG_RISK_CAP = 0.02       # Max 2% of portfolio per leg
LEAP_SLEEVE_CAP = 0.07             # Max 7% of portfolio in total LEAP premium
LEAP_CORE_STRIKE = 10.0            # Core strike target
LEAP_MID_STRIKE = 12.5             # Mid strike target (expansion)
LEAP_TAIL_STRIKE = 15.0            # Tail strike target (reach)
LEAP_SLEEVE_CORE_THRESHOLD = 0.03  # Below 3% sleeve → favor core
LEAP_SLEEVE_MID_THRESHOLD = 0.04   # Above 4% → consider tail
LEAP_TARGET_EXPIRY_YEAR = 2028     # LEAP target year (Jan 2028)

# -- Black-Scholes --
RISK_FREE_RATE = 0.043             # ~4.3% 10Y treasury

# ═══════════════════════════════════════════════════════════════════════════════
# v2.4 Suggestion Status Enum
# ═══════════════════════════════════════════════════════════════════════════════

SUGGESTION_ENGINE_VERSION = "v2.4-suggestions"

# Canonical 6-state status enum (sorted by escalation):
#   unavailable    — required input missing; engine could not run
#   not_applicable — logic does not apply in this state by design
#   no_candidate   — logic ran but no candidate met all filters
#   monitor        — candidate exists; conditions warrant only watching
#   blocked        — candidate exists but a hard gate (permission/risk/cap) blocks action
#   actionable     — candidate exists and conditions allow action
SUGGESTION_STATUSES = (
    "unavailable", "not_applicable", "no_candidate",
    "monitor", "blocked", "actionable",
)

# Map legacy {none, watch, recommend} statuses to the v2.4 enum given context.
_LEGACY_STATUS_HINTS = {
    # Legacy reason fragments -> v2.4 status. Order matters: first match wins.
    "unavailable":     "unavailable",
    "no chain":        "unavailable",
    "no put chain":    "unavailable",
    "no call chain":   "unavailable",
    "missing":         "unavailable",
    "too low":         "not_applicable",      # GEX floor / IV regime gates
    "too high":        "not_applicable",
    "not aligned":     "not_applicable",
    "sleeve cap":      "blocked",
    "per-leg risk":    "blocked",
    "collateral cap":  "blocked",
    "no candidates":   "no_candidate",
    "no puts passed":  "no_candidate",
    "no calls passed": "no_candidate",
    "no candidate":    "no_candidate",
}

def _classify_legacy_status(legacy_status: str, reason: str = "") -> str:
    """Map legacy status + reason text to a v2.4 status."""
    if legacy_status == "recommend":
        return "actionable"
    if legacy_status == "watch":
        return "monitor"
    if legacy_status == "none":
        rl = (reason or "").lower()
        for hint, s in _LEGACY_STATUS_HINTS.items():
            if hint in rl:
                return s
        return "no_candidate"  # safe fallback
    if legacy_status in SUGGESTION_STATUSES:
        return legacy_status
    return "unavailable"


def to_envelope(
    sugg: Optional[Dict[str, Any]],
    *,
    vehicle: str,
    permissions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize any suggestion (legacy or new) to the v2.4 envelope shape.

    Adds the typed status, structured blockers list, and engine_version.
    Preserves all existing fields (contract, reason_primary, etc.) so
    downstream consumers that haven't migrated still see what they expect.
    Idempotent: calling on an already-normalized envelope is a no-op.
    """
    permissions = permissions or {}

    # Case 1: nothing came back at all — the engine couldn't run.
    if sugg is None:
        return {
            "status": "unavailable",
            "status_detail": f"{vehicle} suggestion engine produced no output",
            "blockers": [],
            "candidate": None,
            "reason_primary": None,
            "reason_secondary": [],
            "invalidates_if": [],
            "engine_version": SUGGESTION_ENGINE_VERSION,
            "_legacy_status": None,
        }

    legacy = sugg.get("status")
    reason = sugg.get("reason_primary", "") or ""

    # Case 2: already an envelope (status is one of the v2.4 enum values).
    if legacy in SUGGESTION_STATUSES:
        out = dict(sugg)
        out.setdefault("engine_version", SUGGESTION_ENGINE_VERSION)
        out.setdefault("blockers", [])
        out.setdefault("status_detail", reason)
        return out

    # Case 3: legacy {none, watch, recommend} — reclassify.
    new_status = _classify_legacy_status(legacy or "", reason)

    # Compute structured blockers from doctrine permissions for this vehicle.
    blockers: List[Dict[str, str]] = []
    perm_field = {"csp": "csp_allowed", "leap": "leap_add_allowed", "pmcc": "pmcc_allowed"}.get(vehicle.lower())
    if perm_field and permissions.get(perm_field) == "OFF":
        blockers.append({"kind": "doctrine_permission", "field": perm_field, "value": "OFF"})
    if permissions.get("risk_zone") == "RED":
        # RED risk zone is informational — doesn't always block, but worth surfacing.
        # Only mark as a blocker if the legacy reason text mentions risk.
        if any(w in reason.lower() for w in ("risk zone", "red", "drawdown")):
            blockers.append({"kind": "risk_zone", "value": "RED"})
    if any(w in reason.lower() for w in ("sleeve cap", "collateral cap", "per-leg")):
        blockers.append({"kind": "sizing_cap", "detail": reason})

    # If the legacy status was recommend/watch but doctrine forbids the vehicle,
    # the candidate exists but is BLOCKED, not actionable.
    if blockers and new_status in ("actionable", "monitor"):
        new_status = "blocked"

    # Determine which fields constitute a "candidate" depending on vehicle
    candidate = sugg.get("contract")
    if vehicle.lower() == "pmcc":
        # PMCC candidate is the spread (short_leg + long_leg)
        sl = sugg.get("short_leg")
        ll = sugg.get("long_leg")
        if sl or ll:
            candidate = {"short_leg": sl, "long_leg": ll}

    # If status indicates no candidate but a contract object slipped through,
    # null it out so consumers don't read stale data.
    if new_status in ("unavailable", "not_applicable", "no_candidate"):
        candidate = None

    return {
        "status": new_status,
        "status_detail": reason,
        "blockers": blockers,
        "candidate": candidate,
        "reason_primary": sugg.get("reason_primary"),
        "reason_secondary": sugg.get("reason_secondary") or [],
        # Preserve the rich detail blocks already produced by the engine
        "effective_basis":   sugg.get("effective_basis"),
        "yield_metrics":     sugg.get("yield_metrics"),
        "risk_fit":          sugg.get("risk_fit"),
        "management_rules":  sugg.get("management_rules"),
        "entry_checks":      sugg.get("entry_checks"),
        "context":           sugg.get("context"),
        "ladder_role":       sugg.get("ladder_role"),
        "ladder_plan_hint":  sugg.get("ladder_plan_hint"),
        "spread_metrics":    sugg.get("spread_metrics"),
        "regime_context":    sugg.get("regime_context"),
        "invalidates_if":    sugg.get("invalidates_if") or [],
        "engine_version":    SUGGESTION_ENGINE_VERSION,
        "_legacy_status":    legacy,    # for audit/debugging only
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Black-Scholes Delta Approximation
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(spot: float, strike: float, dte: int, iv: float, is_put: bool = False) -> float:
    """
    Compute Black-Scholes delta.
    iv = annualized implied volatility as decimal (e.g. 1.05 for 105%).
    """
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        return 0.0
    T = dte / 365.0
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    if is_put:
        return _norm_cdf(d1) - 1.0
    return _norm_cdf(d1)


# ═══════════════════════════════════════════════════════════════════════════════
# CSP Suggestion
# ═══════════════════════════════════════════════════════════════════════════════

def suggest_csp(
    spot: float,
    puts: List[Dict],  # [{strike, dte, mid, expiry, iv}]
    gamma_flip: float,
    net_gex: float,
    gex_percentile: float,
    regime: str,
    csp_band_low: float,
    csp_band_high: float,
    iv_regime: str,
    iv_rank: Optional[float],
    btc_price: Optional[float],
    btc_weekly_rsi: Optional[float],
    market_closed: bool,
    portfolio_value: Optional[float],
    csp_collateral_deployed: Optional[float],
    iv_band: Optional[int] = None,  # v1.5: 5-state band index, 0..4
) -> Dict[str, Any]:
    """
    Produce a structured CSP suggestion object.
    Returns dict with status, reason, contract, basis, yield, risk, management, invalidation.
    """

    # ── Global Hard Gates ───────────────────────────────────────────────
    if market_closed:
        return _csp_none("Market is closed")

    # Deeply negative GEX + BTC downtrend → block
    btc_downtrend = _is_btc_downtrend(btc_price, btc_weekly_rsi)
    if gex_percentile < CSP_GEX_FLOOR_PERCENTILE and btc_downtrend:
        return _csp_none(
            f"GEX deeply negative ({gex_percentile:.0f}th pct) + BTC downtrend",
        )

    # CSP permission check (from engine)
    # We don't re-check csp_allowed here because the spec says to use raw fields.
    # But if regime is strongly negative and GEX is weak, set to watch.

    # ── DTE window ──────────────────────────────────────────────────────
    # v1.5: prefer iv_band (5-state) when available; fall back to iv_regime
    # for callers that haven't migrated yet. "Cheap" covers bands 0 & 1
    # (EXTREME_CHEAP, CHEAP_VOL) under the new partition.
    is_cheap_iv = (
        (iv_band is not None and iv_band <= 1)
        if iv_band is not None
        else (iv_regime == "CHEAP_VOL")
    )
    if is_cheap_iv:
        dte_min, dte_max = CSP_DTE_CHEAP_VOL
    else:
        dte_min, dte_max = CSP_DTE_DEFAULT

    # ── Filter candidates ───────────────────────────────────────────────
    candidates = []
    for p in puts:
        dte = p.get("dte", 0)
        strike = p.get("strike", 0)
        mid = p.get("mid", 0)
        iv = p.get("iv", 0)
        expiry = p.get("expiry", "")

        if not (dte_min <= dte <= dte_max):
            continue
        if mid <= 0 or strike <= 0:
            continue

        # Compute delta
        delta = bs_delta(spot, strike, dte, iv, is_put=True) if iv > 0 else None
        if delta is None or not (CSP_DELTA_MIN <= delta <= CSP_DELTA_MAX):
            continue

        # Effective basis
        eff_basis = strike - mid

        # Gamma alignment: prefer basis inside CSP band and below flip
        basis_vs_csp_band = (
            "below" if eff_basis < csp_band_low else
            "above" if eff_basis > csp_band_high else
            "inside"
        )
        basis_vs_flip = (
            "below" if eff_basis < gamma_flip else
            "above" if eff_basis > gamma_flip else
            "inside"
        )

        # Yield metrics
        premium_per_contract = round(mid * 100, 2)
        collateral_per_contract = round(strike * 100, 2)
        raw_yield_pct = round(mid / strike * 100, 2) if strike > 0 else 0
        annualized_yield_pct = round(raw_yield_pct * (365 / dte), 2) if dte > 0 else 0

        # Risk checks
        per_leg_risk = collateral_per_contract - premium_per_contract
        per_leg_risk_pct = per_leg_risk / portfolio_value if portfolio_value and portfolio_value > 0 else None

        violates_per_leg = per_leg_risk_pct is not None and per_leg_risk_pct > CSP_PER_LEG_RISK_CAP

        new_collateral = (csp_collateral_deployed or 0) + collateral_per_contract
        collateral_after_pct = new_collateral / portfolio_value if portfolio_value and portfolio_value > 0 else None
        violates_collateral_cap = collateral_after_pct is not None and collateral_after_pct > CSP_COLLATERAL_CAP

        if violates_per_leg or violates_collateral_cap:
            continue

        # Score: annualized yield × gamma alignment bonus
        gamma_bonus = 1.2 if basis_vs_csp_band == "inside" and basis_vs_flip == "below" else 1.0
        score = annualized_yield_pct * gamma_bonus

        candidates.append({
            "strike": strike,
            "expiry": expiry,
            "dte": dte,
            "delta": round(delta, 4),
            "mid": mid,
            "eff_basis": round(eff_basis, 4),
            "basis_vs_csp_band": basis_vs_csp_band,
            "basis_vs_flip": basis_vs_flip,
            "premium_per_contract": premium_per_contract,
            "collateral_per_contract": collateral_per_contract,
            "raw_yield_pct": raw_yield_pct,
            "annualized_yield_pct": annualized_yield_pct,
            "per_leg_risk_pct": round(per_leg_risk_pct * 100, 2) if per_leg_risk_pct else None,
            "collateral_after_pct": round(collateral_after_pct * 100, 2) if collateral_after_pct else None,
            "score": round(score, 2),
        })

    # ── Rank and select ─────────────────────────────────────────────────
    if not candidates:
        # Check if we're in a "watch" state vs hard "none"
        if gex_percentile < CSP_GEX_FLOOR_PERCENTILE:
            return _csp_none(f"GEX too low ({gex_percentile:.0f}th pct) — no candidates")
        if not puts:
            return _csp_none("No put chain data available")
        return _csp_none("No puts passed delta/DTE/risk filters")

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    # ── Status classification ───────────────────────────────────────────
    # Borderline conditions → watch
    reasons_secondary = []
    is_watch = False

    if gex_percentile < 30:
        reasons_secondary.append(f"GEX weak ({gex_percentile:.0f}th pct)")
        is_watch = True
    if btc_downtrend:
        reasons_secondary.append("BTC showing downtrend pressure")
        is_watch = True
    if best["basis_vs_csp_band"] != "inside":
        reasons_secondary.append(f"Basis {best['basis_vs_csp_band']} CSP band")

    status = "watch" if is_watch else "recommend"

    reason_primary = (
        f"Sell {best['strike']}P {best['expiry']} — "
        f"basis ${best['eff_basis']:.2f} {best['basis_vs_csp_band']} band, "
        f"yield {best['annualized_yield_pct']:.0f}% ann"
    )

    return {
        "status": status,
        "reason_primary": reason_primary,
        "reason_secondary": reasons_secondary,
        "contract": {
            "underlying": "ASST",
            "type": "PUT",
            "expiry": best["expiry"],
            "strike": best["strike"],
            "dte": best["dte"],
            "approx_delta": best["delta"],
            "mid_price_est": best["mid"],
        },
        "effective_basis": {
            "basis_price": best["eff_basis"],
            "basis_vs_spot_pct": round((best["eff_basis"] - spot) / spot * 100, 2) if spot > 0 else None,
            "basis_vs_nav_pct": None,  # Needs NAV/share from caller — computed later
            "basis_vs_gamma_flip": best["basis_vs_flip"],
            "basis_vs_csp_band": best["basis_vs_csp_band"],
        },
        "yield_metrics": {
            "premium_per_contract": best["premium_per_contract"],
            "collateral_per_contract": best["collateral_per_contract"],
            "raw_yield_pct": best["raw_yield_pct"],
            "annualized_yield_pct": best["annualized_yield_pct"],
        },
        "risk_fit": {
            "csp_collateral_after_trade_pct": best["collateral_after_pct"],
            "portfolio_risk_per_leg_pct": best["per_leg_risk_pct"],
            "violates_csp_risk_cap": False,
        },
        "management_rules": {
            "take_profit": "50% premium captured OR 21 DTE, whichever first",
            "hard_stop": "BTC breaks key support or book drawdown ~-8%",
        },
        "invalidates_if": [
            f"GEX percentile < {CSP_GEX_FLOOR_PERCENTILE:.0f} AND BTC downtrend",
            "CSP collateral exceeds 25% of portfolio",
            "Per-leg risk exceeds 2% of portfolio",
        ],
    }


def _csp_none(reason: str) -> Dict[str, Any]:
    return {
        "status": "none",
        "reason_primary": reason,
        "reason_secondary": [],
        "contract": None,
        "effective_basis": None,
        "yield_metrics": None,
        "risk_fit": None,
        "management_rules": None,
        "invalidates_if": [
            f"GEX percentile < {CSP_GEX_FLOOR_PERCENTILE:.0f} AND BTC downtrend",
            "CSP collateral exceeds 25% of portfolio",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEAP Suggestion
# ═══════════════════════════════════════════════════════════════════════════════

def suggest_leap(
    spot: float,
    calls: List[Dict],  # [{strike, dte, mid, expiry, iv}]
    regime: str,
    iv_rank: Optional[float],
    iv_percentile: Optional[float],
    iv_regime: str,
    leap_band_low: float,
    leap_band_high: float,
    nav_per_share: Optional[float],
    btc_price: Optional[float],
    btc_weekly_rsi: Optional[float],
    market_closed: bool,
    portfolio_value: Optional[float],
    leap_premium_at_risk: Optional[float],
) -> Dict[str, Any]:
    """
    Produce a structured LEAP suggestion object.
    Returns dict with status, reason, contract, entry_checks, risk_fit, context, invalidation.
    """

    # ── Global Hard Gates ───────────────────────────────────────────────

    # Gamma regime gate: only negative regimes get recommendations
    # The spec says regime != "NEG" → no recommendation.
    # Our regimes: short_gamma_strong, short_gamma_weak = negative; long_gamma_* = positive
    is_negative_regime = regime.lower().startswith("short")
    gamma_ok = is_negative_regime

    # IV gate
    iv_rank_ok = iv_rank is not None and iv_rank < LEAP_IV_RANK_MAX
    iv_pct_ok = iv_percentile is not None and iv_percentile < LEAP_IV_PERCENTILE_MAX
    iv_gate_ok = iv_rank_ok and iv_pct_ok

    # If both IV metrics missing, gate fails
    if iv_rank is None and iv_percentile is None:
        iv_gate_ok = False

    # If gamma gate fails → watch at best
    if not gamma_ok:
        return _leap_watch_or_none(
            "watch",
            f"Gamma regime is {regime} (not short/negative) — LEAP adds not favored",
            regime=regime,
            gamma_ok=False,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_gate_ok=iv_gate_ok,
        )

    # If IV gate fails → watch
    if not iv_gate_ok:
        reason = "IV metrics missing" if (iv_rank is None and iv_percentile is None) else (
            f"IV rank={iv_rank:.1f} or pct={iv_percentile:.1f} too high (cap={LEAP_IV_RANK_MAX})"
        )
        return _leap_watch_or_none(
            "watch",
            reason,
            regime=regime,
            gamma_ok=True,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_gate_ok=False,
        )

    if market_closed:
        return _leap_watch_or_none(
            "watch",
            "Market closed — cannot price LEAP chain",
            regime=regime,
            gamma_ok=True,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_gate_ok=True,
        )

    # ── Filter LEAP calls ───────────────────────────────────────────────
    # Find Jan 2028 LEAPs (or closest far-dated chain)
    leap_calls = [c for c in calls if c.get("dte", 0) > 500]
    if not leap_calls:
        return _leap_watch_or_none(
            "none",
            "No LEAP chain data available (>500 DTE)",
            regime=regime,
            gamma_ok=True,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_gate_ok=True,
        )

    # Map strikes to ladder roles
    def find_nearest(target: float) -> Optional[Dict]:
        valid = [c for c in leap_calls if c["mid"] > 0]
        if not valid:
            return None
        return min(valid, key=lambda c: abs(c["strike"] - target))

    core = find_nearest(LEAP_CORE_STRIKE)
    mid = find_nearest(LEAP_MID_STRIKE)
    tail = find_nearest(LEAP_TAIL_STRIKE)

    # ── Sleeve sizing → determine ladder role ───────────────────────────
    pv = portfolio_value or 0
    current_sleeve_pct = (leap_premium_at_risk or 0) / pv if pv > 0 else 0

    if current_sleeve_pct < LEAP_SLEEVE_CORE_THRESHOLD:
        # Small sleeve → core first
        preferred_role = "core"
        preferred = core
    elif current_sleeve_pct < LEAP_SLEEVE_MID_THRESHOLD:
        # Mid expansion
        preferred_role = "mid"
        preferred = mid
    else:
        # Tail only if momentum strong
        btc_momentum_strong = btc_weekly_rsi is not None and btc_weekly_rsi > 50
        if btc_momentum_strong:
            preferred_role = "tail"
            preferred = tail
        else:
            preferred_role = "mid"
            preferred = mid

    # ── Risk cap checks for each candidate ──────────────────────────────
    candidates = []
    for role, opt in [("core", core), ("mid", mid), ("tail", tail)]:
        if opt is None or opt["mid"] <= 0:
            continue

        debit = round(opt["mid"] * 100, 2)
        debit_pct = debit / pv if pv > 0 else None

        leap_after = ((leap_premium_at_risk or 0) + debit) / pv if pv > 0 else None

        violates_per_leg = debit_pct is not None and debit_pct > LEAP_PER_LEG_RISK_CAP
        violates_sleeve = leap_after is not None and leap_after > LEAP_SLEEVE_CAP

        delta = bs_delta(spot, opt["strike"], opt["dte"], opt.get("iv", 0.5)) if opt.get("iv") else None

        # Utility score: delta per unit risk
        score = (delta / debit_pct) if delta and debit_pct and debit_pct > 0 else 0

        candidates.append({
            "role": role,
            "strike": opt["strike"],
            "expiry": opt["expiry"],
            "dte": opt["dte"],
            "mid": opt["mid"],
            "delta": round(delta, 4) if delta else None,
            "debit": debit,
            "debit_pct": round(debit_pct * 100, 2) if debit_pct else None,
            "leap_after_pct": round(leap_after * 100, 2) if leap_after else None,
            "violates_per_leg": violates_per_leg,
            "violates_sleeve": violates_sleeve,
            "score": round(score, 2),
            "preferred": role == preferred_role,
        })

    # Filter out those that violate risk caps
    valid = [c for c in candidates if not c["violates_per_leg"] and not c["violates_sleeve"]]

    if not valid:
        return _leap_watch_or_none(
            "none",
            "All LEAP candidates violate risk caps",
            regime=regime,
            gamma_ok=True,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_gate_ok=True,
        )

    # Prefer the designated role; fall back to highest score
    chosen = next((c for c in valid if c["preferred"]), None) or max(valid, key=lambda c: c["score"])

    # ── Build result ────────────────────────────────────────────────────
    spot_discount = round((spot - (nav_per_share or spot)) / (nav_per_share or spot) * 100, 2) if nav_per_share else None
    spot_vs_leap_band = (
        "below" if spot < leap_band_low else
        "above" if spot > leap_band_high else
        "inside"
    )

    return {
        "status": "recommend",
        "reason_primary": (
            f"Buy {chosen['strike']}C {chosen['expiry']} ({chosen['role']}) — "
            f"delta {chosen['delta']:.2f}, debit ${chosen['debit']:.0f}"
        ),
        "reason_secondary": [
            f"IV rank {iv_rank:.1f} / pct {iv_percentile:.1f} — cheap vol window",
            f"Gamma regime {regime} — short gamma supports accumulation",
            f"Sleeve at {current_sleeve_pct*100:.1f}% → {chosen['role']} role",
        ],
        "ladder_role": chosen["role"],
        "contract": {
            "underlying": "ASST",
            "type": "CALL",
            "expiry": chosen["expiry"],
            "strike": chosen["strike"],
            "dte": chosen["dte"],
            "approx_delta": chosen["delta"],
            "mid_price_est": chosen["mid"],
        },
        "entry_checks": {
            "gamma_regime": regime,
            "gamma_ok": True,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "iv_gate_ok": True,
        },
        "risk_fit": {
            "debit": chosen["debit"],
            "debit_pct_of_portfolio": chosen["debit_pct"],
            "leap_premium_after_trade_pct": chosen["leap_after_pct"],
            "violates_per_leg_cap": False,
            "violates_total_leap_cap": False,
        },
        "context": {
            "spot": spot,
            "navpershare": nav_per_share,
            "spot_discount_to_nav_pct": spot_discount,
            "spot_vs_leap_band": spot_vs_leap_band,
        },
        "ladder_plan_hint": (
            "core_first_then_mid_tail" if current_sleeve_pct < LEAP_SLEEVE_CORE_THRESHOLD
            else "expand_mid_then_tail"
        ),
        "invalidates_if": [
            f"IV rank >= {LEAP_IV_RANK_MAX:.0f} or IV percentile >= {LEAP_IV_PERCENTILE_MAX:.0f}",
            f"Gamma regime != short/negative",
            f"LEAP sleeve > {LEAP_SLEEVE_CAP*100:.0f}% of portfolio",
            f"Per-leg debit > {LEAP_PER_LEG_RISK_CAP*100:.0f}% of portfolio",
        ],
    }


def _leap_watch_or_none(
    status: str,
    reason: str,
    regime: str,
    gamma_ok: bool,
    iv_rank: Optional[float],
    iv_percentile: Optional[float],
    iv_gate_ok: bool,
) -> Dict[str, Any]:
    return {
        "status": status,
        "reason_primary": reason,
        "reason_secondary": [],
        "ladder_role": None,
        "contract": None,
        "entry_checks": {
            "gamma_regime": regime,
            "gamma_ok": gamma_ok,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "iv_gate_ok": iv_gate_ok,
        },
        "risk_fit": None,
        "context": None,
        "ladder_plan_hint": None,
        "invalidates_if": [
            f"IV rank >= {LEAP_IV_RANK_MAX:.0f} or IV percentile >= {LEAP_IV_PERCENTILE_MAX:.0f}",
            f"Gamma regime != short/negative",
            f"LEAP sleeve > {LEAP_SLEEVE_CAP*100:.0f}% of portfolio",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_btc_downtrend(btc_price: Optional[float], btc_weekly_rsi: Optional[float]) -> bool:
    """
    Approximate BTC downtrend using weekly RSI as proxy.
    RSI < 40 on weekly = downtrend signal.
    """
    if btc_weekly_rsi is None:
        return False  # Insufficient data → no block
    return btc_weekly_rsi < 40.0


def enrich_puts_with_iv(puts: List[Dict], spot: float) -> List[Dict]:
    """
    Add 'iv' field to put dicts from yfinance chain data.
    If iv is missing, use a default high-vol estimate.
    """
    for p in puts:
        if "iv" not in p or p["iv"] is None or p["iv"] <= 0:
            # Default: use 100% annualized for ASST (it's a high-vol stock)
            p["iv"] = 1.0
    return puts


def enrich_calls_with_iv(calls: List[Dict], spot: float) -> List[Dict]:
    """
    Add 'iv' field to call dicts from yfinance chain data.
    """
    for c in calls:
        if "iv" not in c or c["iv"] is None or c["iv"] <= 0:
            c["iv"] = 1.0
    return calls


# ═════════════════════════════════════════════════════════════════════════════
# Enrichment Derivations
# ═════════════════════════════════════════════════════════════════════════════

def compute_enrichment_fields(
    spot: float,
    ev_mnav: Optional[float],
    btc_holdings: Optional[float],
    total_shares: Optional[float],
    diluted_shares: Optional[float],
    csp_band_low: float,
    csp_band_high: float,
    gamma_flip: float,
    pos_magnets: Optional[List[Dict]],  # [{strike, net_gex}]
) -> Dict[str, Any]:
    """
    Compute trivial enrichment derivations from existing fields.
    Returns dict of new fields, any of which may be None.
    """
    result: Dict[str, Any] = {}

    # mNAV discount: how far EV/mNAV is below 1.0x (positive = discount)
    if ev_mnav is not None:
        result["mnav_discount"] = round(1.0 - ev_mnav, 4) if ev_mnav > 0 else None
    else:
        result["mnav_discount"] = None

    # BTC per share (basic and diluted)
    if btc_holdings is not None and total_shares is not None and total_shares > 0:
        result["btc_per_share_basic"] = round(btc_holdings / total_shares, 6)
    else:
        result["btc_per_share_basic"] = None

    if btc_holdings is not None and diluted_shares is not None and diluted_shares > 0:
        result["btc_per_share_diluted"] = round(btc_holdings / diluted_shares, 6)
    else:
        result["btc_per_share_diluted"] = None

    # CSP delta-to-band: distance from spot to nearest CSP band edge (normalized by band width)
    if csp_band_low > 0 and csp_band_high > csp_band_low:
        band_width = csp_band_high - csp_band_low
        if spot > csp_band_high:
            result["csp_delta_to_band"] = round((spot - csp_band_high) / band_width, 3)
        elif spot < csp_band_low:
            result["csp_delta_to_band"] = round((spot - csp_band_low) / band_width, 3)
        else:
            result["csp_delta_to_band"] = 0.0  # Inside band
    else:
        result["csp_delta_to_band"] = None

    # CSP magnet proximity: nearest positive GEX magnet strike within CSP band
    if pos_magnets and csp_band_low > 0:
        magnets_in_band = [
            m for m in pos_magnets
            if csp_band_low <= m.get("strike", 0) <= csp_band_high
        ]
        if magnets_in_band:
            nearest = min(magnets_in_band, key=lambda m: abs(m["strike"] - spot))
            result["csp_magnet_proximity"] = round(abs(spot - nearest["strike"]), 4)
            result["csp_magnet_strike"] = nearest["strike"]
            result["csp_magnet_gex"] = nearest.get("net_gex", 0)
        else:
            result["csp_magnet_proximity"] = None
            result["csp_magnet_strike"] = None
            result["csp_magnet_gex"] = None
    else:
        result["csp_magnet_proximity"] = None
        result["csp_magnet_strike"] = None
        result["csp_magnet_gex"] = None

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PMCC Suggestion
# ═════════════════════════════════════════════════════════════════════════════

# -- PMCC Constants --
PMCC_SHORT_DTE_MIN = 14
PMCC_SHORT_DTE_MAX = 45
PMCC_SHORT_DELTA_MIN = -0.30    # Widest acceptable short call delta
PMCC_SHORT_DELTA_MAX = -0.15    # Shallowest
PMCC_SHORT_DELTA_TARGET = -0.20 # Preferred
PMCC_MIN_CREDIT_PCT = 0.005     # Min premium as % of spot (0.5%)
PMCC_LONG_MIN_DTE = 500         # LEAP anchor must be >500 DTE


def suggest_pmcc(
    spot: float,
    calls: List[Dict],  # Short call candidates [{strike, dte, mid, expiry, delta, iv}]
    leap_calls: List[Dict],  # Long LEAP calls (the anchor leg)
    regime: str,
    pmcc_allowed: Optional[str],
    net_gex: float,
    gex_percentile: float,
    market_closed: bool,
) -> Dict[str, Any]:
    """
    Produce a structured PMCC (Poor Man's Covered Call) suggestion.
    PMCC = long deep-ITM LEAP call + short OTM near-term call.
    Returns dict with status, reason, short_leg, long_leg, invalidation.
    """

    # Gate: PMCC regime eligibility
    if pmcc_allowed not in ("ALLOWED", "allowed"):
        return _pmcc_watch_or_none(
            "none",
            f"PMCC not eligible — pmcc_allowed={pmcc_allowed}",
            regime=regime,
        )

    if market_closed:
        return _pmcc_watch_or_none(
            "watch",
            "Market closed — cannot price PMCC short leg",
            regime=regime,
        )

    # Need long gamma regime for PMCC (dealers damping = good for overwrite)
    is_long_gamma = regime.lower().startswith("long")
    if not is_long_gamma:
        return _pmcc_watch_or_none(
            "watch",
            f"Gamma regime {regime} not long — PMCC overwrite less favorable",
            regime=regime,
        )

    # Filter short call candidates: 14-45 DTE, delta -0.15 to -0.30, has premium
    short_candidates = []
    for c in calls:
        dte = c.get("dte", 0)
        delta = c.get("delta")
        mid = c.get("mid", 0)
        if not (PMCC_SHORT_DTE_MIN <= dte <= PMCC_SHORT_DTE_MAX):
            continue
        if mid <= 0 or mid < spot * PMCC_MIN_CREDIT_PCT:
            continue
        if delta is None:
            continue
        # Short call delta is negative in our system (selling calls)
        abs_delta = abs(delta)
        if not (0.15 <= abs_delta <= 0.30):
            continue

        short_candidates.append({
            "strike": c["strike"],
            "expiry": c["expiry"],
            "dte": dte,
            "delta": round(delta, 4),
            "mid": mid,
            "iv": c.get("iv"),
        })

    # Filter long LEAP anchor: >500 DTE, deep ITM (delta > 0.70)
    long_anchor = None
    for lc in leap_calls:
        dte = lc.get("dte", 0)
        delta = lc.get("delta")
        if dte < PMCC_LONG_MIN_DTE or delta is None:
            continue
        if delta > 0.70 and lc.get("mid", 0) > 0:
            if long_anchor is None or delta > long_anchor.get("delta", 0):
                long_anchor = {
                    "strike": lc["strike"],
                    "expiry": lc["expiry"],
                    "dte": dte,
                    "delta": round(delta, 4),
                    "mid": lc["mid"],
                }

    if not short_candidates:
        return _pmcc_watch_or_none(
            "none",
            "No short call candidates passed delta/DTE/premium filters",
            regime=regime,
        )

    if not long_anchor:
        return _pmcc_watch_or_none(
            "watch",
            "No deep-ITM LEAP call available as anchor (need delta > 0.70, >500 DTE)",
            regime=regime,
        )

    # Rank short candidates: prefer delta nearest to target, then highest premium
    short_candidates.sort(key=lambda c: (
        abs(abs(c["delta"]) - abs(PMCC_SHORT_DELTA_TARGET)),
        -c["mid"],
    ))
    best_short = short_candidates[0]

    # Net debit / credit
    net_debit = round(long_anchor["mid"] - best_short["mid"], 2)
    credit_per_contract = round(best_short["mid"] * 100, 2)
    max_risk = round(net_debit * 100, 2) if net_debit > 0 else 0

    # Annualized yield from short leg
    ann_yield = round(best_short["mid"] / long_anchor["mid"] * (365 / best_short["dte"]) * 100, 1) if best_short["dte"] > 0 and long_anchor["mid"] > 0 else 0

    # Status: recommend if GEX is strong (supportive pinning), watch if borderline
    is_watch = gex_percentile < 50
    reasons_secondary = []
    if gex_percentile < 50:
        reasons_secondary.append(f"GEX at {gex_percentile:.0f}th pct — pinning support moderate")

    return {
        "status": "watch" if is_watch else "recommend",
        "reason_primary": (
            f"Sell {best_short['strike']}C {best_short['expiry']} vs {long_anchor['strike']}C {long_anchor['expiry']} LEAP anchor — "
            f"credit ${credit_per_contract:.0f}, ann yield {ann_yield:.0f}%"
        ),
        "reason_secondary": reasons_secondary,
        "short_leg": {
            "underlying": "ASST",
            "type": "CALL",
            "direction": "sell",
            "strike": best_short["strike"],
            "expiry": best_short["expiry"],
            "dte": best_short["dte"],
            "approx_delta": best_short["delta"],
            "mid_price_est": best_short["mid"],
        },
        "long_leg": {
            "underlying": "ASST",
            "type": "CALL",
            "direction": "buy",
            "strike": long_anchor["strike"],
            "expiry": long_anchor["expiry"],
            "dte": long_anchor["dte"],
            "approx_delta": long_anchor["delta"],
            "mid_price_est": long_anchor["mid"],
        },
        "spread_metrics": {
            "credit_per_contract": credit_per_contract,
            "net_debit": round(net_debit, 2),
            "max_risk_per_contract": max_risk,
            "annualized_yield_pct": ann_yield,
        },
        "regime_context": {
            "gamma_regime": regime,
            "gex_percentile": gex_percentile,
            "long_gamma_ok": is_long_gamma,
        },
        "management_rules": {
            "take_profit": "50% of short leg premium captured",
            "roll": "Roll short leg at 21 DTE or 75% premium captured, whichever first",
            "hard_stop": "Close spread if short leg goes ITM by > 1 ATR",
        },
        "invalidates_if": [
            "PMCC regime eligibility revoked",
            "Gamma regime flips to short/negative",
            "Short call goes deep ITM (assignment risk)",
        ],
    }


def _pmcc_watch_or_none(
    status: str,
    reason: str,
    regime: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "reason_primary": reason,
        "reason_secondary": [],
        "short_leg": None,
        "long_leg": None,
        "spread_metrics": None,
        "regime_context": {
            "gamma_regime": regime,
            "long_gamma_ok": regime.lower().startswith("long"),
        },
        "management_rules": None,
        "invalidates_if": [
            "PMCC regime eligibility revoked",
            "Gamma regime flips to short/negative",
        ],
    }

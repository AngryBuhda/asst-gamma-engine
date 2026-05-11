"""
ASST Gamma-Aligned Flywheel Strategy Engine — v2.0
Last research update: April 1, 2026

Key empirical sources:
  - Barbon & Buraschi (2021) "Gamma Fragility": illiquidity multiplier, flash crash probability
  - tastylive DTE selection study: IV Rank-based DTE windows
  - tastylive 21 DTE management study: highest Sharpe exit rule
  - Galaxy Research (2025): mNAV flywheel mechanics, Darwinian bifurcation
  - Glassnode taker-flow GEX (Dec 2025): crypto GEX convention inversion

Three-factor LEAP entry gate: Gamma + Vega/IV + mNAV = 0-3.0 composite score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np

# LEAP allocation rule of thumb: 70% core, 20% mid, 10% tail tranches
# (handled upstream by sizing, not here).

# =========================
# Public API
# =========================

__all__ = [
    "Regime",
    "OptionQuote",
    "GexStrike",
    "MagnetBand",
    "StrategyConfig",
    "STRATEGY_DEFAULTS",
    "classify_regime_from_gex",
    "compute_negative_gamma_band",
    "compute_csp_band",
    "compute_leap_band",
    "find_gex_magnets",
    "filter_csp_candidates",
    "filter_leap_core_candidates",
    "filter_leap_mid_candidates",
    "filter_leap_tail_candidates",
    "can_gamma_scalp",
    "classify_risk_zone",
    "get_monthly_allocation",
    "classify_btc_cycle_zone",
    "compute_trade_permissions",
    "compute_action_banner",
    "generate_daily_recommendations",
    "daily_test_harness",
]

Regime = Literal[
    "short_gamma_strong",
    "short_gamma_weak",
    "long_gamma_weak",
    "long_gamma_strong",
    "neutral",
]


# =========================
# Strategy configuration
# =========================

@dataclass
class StrategyConfig:
    """
    Tunable parameters for the ASST gamma-aligned strategy.

    You can instantiate different configs per asset (e.g., ASST vs MSTR)
    if needed, but the defaults are designed for ASST.
    """

    # Regime classification
    pct_strong_neg: float = 25.0
    pct_strong_pos: float = 75.0
    pct_neutral_low: float = 40.0
    pct_neutral_high: float = 60.0

    # Flip band (legacy — kept for backward compat)
    band_below_flip_low: float = 0.30
    band_below_flip_high: float = 0.80

    # CSP income band (ATR multipliers relative to flip)
    csp_band_below_flip: float = 0.5   # lower bound = flip - 0.5*ATR
    csp_band_above_flip: float = 1.0   # upper bound = flip + 1.0*ATR

    # LEAP convexity band (ATR multipliers below flip)
    leap_band_below_flip_far: float = 2.5   # lower = flip - 2.5*ATR
    leap_band_below_flip_near: float = 0.5  # upper = flip - 0.5*ATR

    # GEX magnets
    gex_top_n: int = 5
    gex_multiple_of_median: float = 3.0
    default_strike_step: float = 0.5

    # Gamma scalp
    min_scalp_move_pct: float = 0.03      # 3%
    atr_multiplier: float = 1.25
    daily_trim_cap: float = 0.30          # 30% of starting delta per day
    weekly_trim_cap: float = 0.60         # 60% per week

    # LEAP tranche widths (relative to core band / flip)
    mid_band_extension_above_flip: float = 0.50   # up to flip + 0.5 for mid
    tail_band_extension_above_flip: float = 2.00  # up to flip + 2 for tail

    # ── Engine v2 constants (empirically grounded) ──

    # IV Rank-based DTE windows — tastylive DTE selection study
    CSP_DTE_MIN_LOW_IV: int = 45       # IV Rank < 30%: go farther out for premium
    CSP_DTE_MAX_LOW_IV: int = 60
    CSP_DTE_MIN_MID_IV: int = 30       # IV Rank 30-70%: standard window
    CSP_DTE_MAX_MID_IV: int = 45
    CSP_DTE_MIN_HIGH_IV: int = 21      # IV Rank > 70%: capture rich near-term vol
    CSP_DTE_MAX_HIGH_IV: int = 30

    # CSP exit rules — tastylive 21 DTE management study (highest Sharpe exit)
    CSP_EXIT_PROFIT_PCT: float = 0.50   # Close at 50% max profit
    CSP_EXIT_DTE: int = 21              # Or at 21 DTE, whichever comes first

    # LEAP illiquidity size adjustment — Barbon & Buraschi (2021) "Gamma Fragility"
    LEAP_ILLIQUIDITY_SIZE_ADJUSTMENT: float = 0.65

    # PMCC regime gate — only sell calls against LEAPs in long-gamma regimes
    PMCC_ALLOWED_REGIMES: list = field(default_factory=lambda: ["long_gamma_strong", "long_gamma_weak"])

    # ── Risk Zone thresholds (tunable defaults — research-derived) ──

    ZONE_AMBER_DRAWDOWN: float = -0.35   # ASST drawdown from 90d high
    ZONE_RED_DRAWDOWN: float = -0.60
    ZONE_AMBER_MVRV: float = 1.0         # BTC MVRV ratio
    ZONE_RED_MVRV: float = 0.80
    ZONE_AMBER_RSI: float = 50.0         # BTC weekly RSI
    ZONE_RED_RSI: float = 40.0

    # Half-Kelly position sizing caps (tunable defaults)
    KELLY_SPOT_SESSION_MAX: float = 0.10    # 10% of account per session
    KELLY_SPOT_TOTAL_MAX: float = 0.60      # 60% total
    KELLY_CSP_NOTIONAL_MAX: float = 0.25    # 25% single CSP notional
    KELLY_LEAP_PREMIUM_MAX: float = 0.15    # 15% single LEAP premium
    KELLY_COMBINED_OPTIONS_MAX: float = 0.40  # 40% combined CSPs + LEAPs

    # SATA buffer targets by zone
    SATA_GREEN_MIN: float = 0.10
    SATA_GREEN_MAX: float = 0.20
    SATA_AMBER_MIN: float = 0.20
    SATA_AMBER_MAX: float = 0.35
    SATA_RED_MIN: float = 0.35
    SATA_RED_MAX: float = 0.50

    # BTC cycle reference
    BTC_CYCLE_TROUGH_DATE: str = "2026-03-02"
    BTC_CYCLE_TROUGH_PRICE: float = 65993.0


STRATEGY_DEFAULTS = StrategyConfig()


# =========================
# Data structures
# =========================

@dataclass
class OptionQuote:
    """
    Minimal option quote representation used for strike selection.
    """
    strike: float
    dte: int
    mid: float
    expiry: str
    is_leap: bool = False


@dataclass
class GexStrike:
    """
    Per-strike gamma exposure snapshot.
    """
    strike: float
    net_gex: float


@dataclass
class MagnetBand:
    """
    A GEX-based magnet zone around a strike.
    """
    strike: float
    net_gex: float
    band_low: float
    band_high: float


# ==========================
# Regime & gamma flip bands
# ==========================

def classify_regime_from_gex(
    net_gex: Optional[float],
    percentile: Optional[float],
    spot: Optional[float] = None,
    flip: Optional[float] = None,
    atr_1d: Optional[float] = None,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Regime:
    """
    Classify dealer gamma regime for ASST.

    v2 logic (Apr 2026): Distance from flip in ATR units is the PRIMARY driver.
    GEX sign is a secondary confirmation. GEX percentile is a weak tie-breaker only.

    Rationale: ASST's 95%+ call OI from speculative buying means raw GEX sign can
    misclassify true dealer hedging conditions. Spot-vs-flip distance is the most
    reliable structural signal regardless of GEX convention issues.

    Regime bands (distance = (flip - spot) / ATR, positive = spot below flip):
      short_gamma_strong : distance >= 1.5 ATR, OR (net_gex < 0 AND spot below flip)
      short_gamma_weak   : distance in [0.5, 1.5) ATR, unless GEX is strongly positive
      neutral            : distance in (-0.5, 0.5) ATR
      long_gamma_weak    : distance in (-1.5, -0.5] ATR AND positive GEX
      long_gamma_strong  : distance <= -1.5 ATR AND positive GEX AND high percentile
    """
    if net_gex is None or percentile is None:
        return "neutral"

    if net_gex == 0:
        return "neutral"

    # If no spatial context, fall back to GEX sign + percentile
    if spot is None or flip is None or flip <= 0:
        p = percentile
        if cfg.pct_neutral_low <= p <= cfg.pct_neutral_high:
            return "neutral"
        elif net_gex < 0:
            return "short_gamma_strong" if p <= cfg.pct_strong_neg else "short_gamma_weak"
        elif net_gex > 0:
            return "long_gamma_strong" if p >= cfg.pct_strong_pos else "long_gamma_weak" if p >= 50 else "neutral"
        return "neutral"

    # Distance-first classification
    atr = atr_1d if atr_1d is not None and atr_1d > 0 else 0.80
    distance = (flip - spot) / atr  # positive = spot below flip

    # Hard structural condition: negative GEX while below flip = strong short gamma
    if net_gex < 0 and distance > 0:
        return "short_gamma_strong"

    if distance >= 1.5:
        # Spot materially below flip: short gamma regardless of positive GEX
        # (positive GEX may be speculative OI, not real dealer support)
        return "short_gamma_strong"

    elif 0.5 <= distance < 1.5:
        # Spot meaningfully below flip: short gamma weak
        # Exception: very strongly positive GEX (top quartile) can hold neutral
        if net_gex > 0 and percentile >= cfg.pct_strong_pos:
            return "neutral"
        return "short_gamma_weak"

    elif -0.5 < distance < 0.5:
        # Spot near flip: neutral zone
        return "neutral"

    elif -1.5 < distance <= -0.5:
        # Spot meaningfully above flip
        if net_gex > 0:
            return "long_gamma_weak"
        return "neutral"  # negative GEX above flip is unusual, stay neutral

    else:  # distance <= -1.5
        # Spot materially above flip
        if net_gex > 0 and percentile >= cfg.pct_strong_pos:
            return "long_gamma_strong"
        elif net_gex > 0:
            return "long_gamma_weak"
        return "neutral"


def compute_negative_gamma_band(
    flip: Optional[float],
    atr_1d: Optional[float] = None,
    force_wider_gap: bool = False,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Tuple[Optional[float], Optional[float]]:
    """
    DEPRECATED: Use compute_csp_band() or compute_leap_band() instead.

    Kept for backward compatibility. Computes a narrow band just below the flip.
    """
    if flip is None or flip <= 0:
        return None, None

    base_low = flip - cfg.band_below_flip_high
    base_high = flip - cfg.band_below_flip_low

    if force_wider_gap and atr_1d is not None and atr_1d > 0:
        required_gap = max(cfg.band_below_flip_low, atr_1d)
        band_high = min(base_high, flip - required_gap)
        band_low = base_low
    else:
        band_low, band_high = base_low, base_high

    return band_low, band_high


def compute_csp_band(
    flip: Optional[float],
    atr_1d: Optional[float] = None,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CSP income band: centered near the flip where dealer dampening provides support.

    Band: (flip - csp_band_below_flip * atr, flip + csp_band_above_flip * atr)

    Returns:
        (band_low, band_high) in price units, or (None, None) if flip is invalid.
    """
    if flip is None or flip <= 0:
        return None, None

    atr = atr_1d if atr_1d is not None and atr_1d > 0 else 0.80
    band_low = flip - cfg.csp_band_below_flip * atr
    band_high = flip + cfg.csp_band_above_flip * atr
    return round(band_low, 4), round(band_high, 4)


def compute_leap_band(
    flip: Optional[float],
    atr_1d: Optional[float] = None,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Tuple[Optional[float], Optional[float]]:
    """
    LEAP convexity band: below the flip where negative gamma creates opportunity.

    Band: (flip - leap_band_below_flip_far * atr, flip - leap_band_below_flip_near * atr)

    Returns:
        (band_low, band_high) in price units, or (None, None) if flip is invalid.
    """
    if flip is None or flip <= 0:
        return None, None

    atr = atr_1d if atr_1d is not None and atr_1d > 0 else 0.80
    band_low = flip - cfg.leap_band_below_flip_far * atr
    band_high = flip - cfg.leap_band_below_flip_near * atr
    return round(band_low, 4), round(band_high, 4)


# ======================
# GEX magnet detection
# ======================

def find_gex_magnets(
    strikes: List[GexStrike],
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
    include_negative: bool = False,
) -> Tuple[List[MagnetBand], List[MagnetBand]]:
    """
    Identify strong positive (and optionally negative) GEX magnet levels.

    When to call:
        - Once per day per ticker after you build a per-strike GEX table.

    Returns:
        (pos_magnets, neg_magnets), each a list of MagnetBand.

    Notes:
        - Current strike filters only use positive magnets; negative
          magnets are returned for future extensions (e.g., put walls).
    """
    if not strikes:
        return [], []

    abs_vals = [abs(s.net_gex) for s in strikes]
    med_abs = float(np.median(abs_vals)) if abs_vals else 0.0
    if med_abs == 0.0:
        return [], []

    ranked = sorted(
        strikes,
        key=lambda s: (abs(s.net_gex), s.strike),
        reverse=True,
    )
    primary = ranked[: cfg.gex_top_n]

    pos_magnets: List[MagnetBand] = []
    neg_magnets: List[MagnetBand] = []

    for s in primary:
        if abs(s.net_gex) < cfg.gex_multiple_of_median * med_abs:
            continue
        band_low = s.strike - cfg.default_strike_step
        band_high = s.strike + cfg.default_strike_step
        magnet = MagnetBand(
            strike=s.strike,
            net_gex=s.net_gex,
            band_low=band_low,
            band_high=band_high,
        )
        if s.net_gex > 0:
            pos_magnets.append(magnet)
        elif include_negative and s.net_gex < 0:
            neg_magnets.append(magnet)

    return pos_magnets, neg_magnets


# ==========================
# CSP & LEAP strike filters
# ==========================

def filter_csp_candidates(
    puts: List[OptionQuote],
    flip: float,
    atr_1d: Optional[float],
    regime: Regime,
    pos_magnets: List[MagnetBand],
    dte_min: int = 30,
    dte_max: int = 45,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
    iv_rank: float = 25.0,
) -> Tuple[List[Dict], Tuple[float, float], Tuple[int, int]]:
    """
    Filter CSPs whose effective basis is in the CSP income band and
    which respect GEX magnet guardrails (positive GEX only for now).

    DTE window is selected based on IV Rank (tastylive study):
      - Low IV (<30%):  45-60 DTE — go farther out for premium
      - Mid IV (30-70%): 30-45 DTE — standard window
      - High IV (>70%): 21-30 DTE — capture rich near-term vol

    Returns: (candidates, band, dte_window)
    """
    # IV Rank-based DTE window selection
    if iv_rank < 30:
        dte_min, dte_max = cfg.CSP_DTE_MIN_LOW_IV, cfg.CSP_DTE_MAX_LOW_IV
    elif iv_rank < 70:
        dte_min, dte_max = cfg.CSP_DTE_MIN_MID_IV, cfg.CSP_DTE_MAX_MID_IV
    else:
        dte_min, dte_max = cfg.CSP_DTE_MIN_HIGH_IV, cfg.CSP_DTE_MAX_HIGH_IV

    band_low, band_high = compute_csp_band(flip, atr_1d, cfg=cfg)
    if band_low is None or band_high is None:
        return [], (0.0, 0.0), (dte_min, dte_max)

    def is_pinned_by_magnet(strike: float) -> bool:
        return any(m.band_low <= strike <= m.band_high for m in pos_magnets)

    raw: List[Dict] = []
    for opt in puts:
        if not (dte_min <= opt.dte <= dte_max):
            continue

        eff_basis = opt.strike - opt.mid
        if not (band_low <= eff_basis <= band_high):
            continue

        # Guardrail: in long-gamma regimes, avoid CSPs at strong
        # positive-GEX magnets materially above the flip.
        if regime in {"long_gamma_strong", "long_gamma_weak"} and opt.strike > flip + 0.10:
            if is_pinned_by_magnet(opt.strike):
                continue

        raw.append(
            {
                "strike": opt.strike,
                "dte": opt.dte,
                "mid": opt.mid,
                "expiry": opt.expiry,
                "eff_basis": eff_basis,
            }
        )

    if not raw:
        return [], (band_low, band_high), (dte_min, dte_max)

    target_mid = (band_low + band_high) / 2.0
    raw.sort(key=lambda c: abs(c["eff_basis"] - target_mid))
    return raw, (band_low, band_high), (dte_min, dte_max)


def filter_leap_core_candidates(
    calls: List[OptionQuote],
    flip: float,
    atr_1d: Optional[float],
    regime: Regime,
    pos_magnets: List[MagnetBand],
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Tuple[List[OptionQuote], Tuple[float, float]]:
    """
    Core LEAP ladder strikes in the negative-gamma band, avoiding
    exact magnet strikes in long-gamma regimes.

    When to call:
        - Once per day when recomputing core LEAP ladder candidates.
    """
    band_low, band_high = compute_leap_band(flip, atr_1d, cfg=cfg)
    if band_low is None or band_high is None:
        return [], (0.0, 0.0)

    magnet_strikes = {m.strike for m in pos_magnets}
    core: List[OptionQuote] = []

    for c in calls:
        if not c.is_leap:
            continue
        if not (band_low <= c.strike <= band_high):
            continue
        if regime in {"long_gamma_strong", "long_gamma_weak"} and c.strike in magnet_strikes:
            continue
        core.append(c)

    return core, (band_low, band_high)


def filter_leap_mid_candidates(
    calls: List[OptionQuote],
    flip: float,
    regime: Regime,
    pos_magnets: List[MagnetBand],
    atr_1d: Optional[float] = None,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> List[OptionQuote]:
    """
    Mid LEAP tranche (20% of LEAP bucket).

    When to call:
        - After selecting core ladder, to identify slightly higher strikes
          from leap_band_high up to flip + mid_band_extension_above_flip.

    Rules:
        - Accept LEAP calls with strikes in [leap_band_high, flip + mid_ext].
        - Prefer strikes not exactly at primary positive-GEX magnets in
          long-gamma regimes.
    """
    _, band_high = compute_leap_band(flip, atr_1d, cfg=cfg)
    if band_high is None:
        return []

    upper = flip + cfg.mid_band_extension_above_flip
    magnet_strikes = {m.strike for m in pos_magnets}
    mids: List[OptionQuote] = []

    for c in calls:
        if not c.is_leap:
            continue
        if not (band_high <= c.strike <= upper):
            continue
        if regime in {"long_gamma_strong", "long_gamma_weak"} and c.strike in magnet_strikes:
            continue
        mids.append(c)

    return mids


def filter_leap_tail_candidates(
    calls: List[OptionQuote],
    flip: float,
    regime: Regime,
    pos_magnets: List[MagnetBand],
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> List[OptionQuote]:
    """
    Tail LEAP tranche (10% of LEAP bucket).

    When to call:
        - After core and mid tranches, to identify far OTM spike-capture
          strikes.

    Rules:
        - Accept LEAP calls with strikes in
          [flip + mid_ext, flip + tail_ext].
        - Magnet rejection is relaxed; tail can be at or above magnets,
          but size should be small at portfolio level (enforced upstream).
    """
    upper_mid = flip + cfg.mid_band_extension_above_flip
    upper_tail = flip + cfg.tail_band_extension_above_flip

    tails: List[OptionQuote] = []
    for c in calls:
        if not c.is_leap:
            continue
        if not (upper_mid <= c.strike <= upper_tail):
            continue
        tails.append(c)

    return tails


# ======================
# Gamma scalp decision
# ======================

def can_gamma_scalp(
    regime: Regime,
    intraday_return: float,
    atr_1d: Optional[float],
    spot: float,
    flip: float,
    pos_magnets: List[MagnetBand],
    started_below_flip: bool,
    daily_trim_used: float,
    weekly_trim_used: float,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Tuple[bool, str]:
    """
    Decide whether gamma-scaling (delta trim) is permitted.

    When to call:
        - Intraday, at most a few times on strong move days, after you
          know spot, intraday_return, magnets, and whether price opened
          below the flip.

    Conditions:
        - Regime strong short gamma.
        - Move >= max(min_scalp_move_pct, atr_multiplier * ATR_1d / spot).
        - Level near a positive-GEX magnet OR has crossed flip from below.
        - Daily and weekly trim caps not exceeded.

    Returns:
        (allowed, reason)
    """
    if regime != "short_gamma_strong":
        return False, "Regime not strong short gamma."

    # Move condition
    min_move = cfg.min_scalp_move_pct
    if atr_1d is not None and spot > 0:
        atr_based = cfg.atr_multiplier * atr_1d / spot
        move_threshold = max(min_move, atr_based)
    else:
        move_threshold = min_move

    if intraday_return < move_threshold:
        return (
            False,
            f"Move {intraday_return:.2%} below threshold {move_threshold:.2%}.",
        )

    # Level condition
    near_magnet = any(m.band_low <= spot <= m.band_high for m in pos_magnets)
    crossed_flip = started_below_flip and spot >= flip

    if not (near_magnet or crossed_flip):
        return False, "Not near magnet band or crossing flip."

    # Trim quotas
    if daily_trim_used >= cfg.daily_trim_cap:
        return False, "Daily trim quota reached."
    if weekly_trim_used >= cfg.weekly_trim_cap:
        return False, "Weekly trim quota reached."

    return True, "Gamma scalp allowed."


# ======================
# Risk zone classification
# ======================

def classify_risk_zone(
    asst_drawdown_90d: float,
    btc_mvrv: Optional[float] = None,
    btc_weekly_rsi: Optional[float] = None,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> str:
    """
    Three-zone classification using most-conservative-wins logic.
    When inputs conflict, use highest-risk zone.

    Inputs:
        - asst_drawdown_90d: negative pct, e.g. -0.25 = -25% from 90d high
        - btc_mvrv: BTC MVRV ratio (optional, pending Glassnode)
        - btc_weekly_rsi: BTC 14-period weekly RSI (optional)

    Returns: "GREEN" | "AMBER" | "RED"
    """
    zones = []

    # ASST drawdown zone
    if asst_drawdown_90d <= cfg.ZONE_RED_DRAWDOWN:
        zones.append("RED")
    elif asst_drawdown_90d <= cfg.ZONE_AMBER_DRAWDOWN:
        zones.append("AMBER")
    else:
        zones.append("GREEN")

    # BTC MVRV zone (if available)
    if btc_mvrv is not None:
        if btc_mvrv < cfg.ZONE_RED_MVRV:
            zones.append("RED")
        elif btc_mvrv < cfg.ZONE_AMBER_MVRV:
            zones.append("AMBER")
        else:
            zones.append("GREEN")

    # BTC weekly RSI zone (if available)
    if btc_weekly_rsi is not None:
        if btc_weekly_rsi < cfg.ZONE_RED_RSI:
            zones.append("RED")
        elif btc_weekly_rsi < cfg.ZONE_AMBER_RSI:
            zones.append("AMBER")
        else:
            zones.append("GREEN")

    # Most conservative wins
    if "RED" in zones:
        return "RED"
    elif "AMBER" in zones:
        return "AMBER"
    return "GREEN"


def get_monthly_allocation(
    zone: str,
    leap_entry_score: float,
    iv_regime: str,
    regime: str,
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
) -> Dict:
    """
    Monthly capital allocation flow based on zone + scoring.
    Returns dict with csp_pct, leap_pct, sata_pct, csp_allowed, note.
    Monthly capital goes to CSPs (income/accumulation) and LEAPs (convexity).
    Remainder held in SATA buffer. Spot exposure comes from CSP assignment, not direct allocation.
    """
    if zone == "GREEN":
        if leap_entry_score >= 3.0:
            return {"csp_pct": 30, "leap_pct": 60, "sata_pct": 10, "csp_allowed": True,
                    "note": "Green + perfect score → 60% LEAP, 30% CSP, 10% SATA. Full deployment."}
        elif leap_entry_score >= 2.0:
            return {"csp_pct": 30, "leap_pct": 60, "sata_pct": 10, "csp_allowed": True,
                    "note": "Green + score 2.0 → 60% LEAP, 30% CSP, 10% SATA. Accumulation window."}
        else:
            return {"csp_pct": 40, "leap_pct": 30, "sata_pct": 30, "csp_allowed": True,
                    "note": "Green + low score → 40% CSP, 30% LEAP, 30% SATA. Score weak, lean income."}
    elif zone == "AMBER":
        csp_ok = regime == "short_gamma_strong"
        return {"csp_pct": 0, "leap_pct": 30 if leap_entry_score >= 3.0 else 0,
                "sata_pct": 70 if leap_entry_score >= 3.0 else 100, "csp_allowed": csp_ok,
                "note": f"Amber → {'70% SATA, 30% LEAP (score qualifies)' if leap_entry_score >= 3.0 else '100% SATA'}. CSP {'only on short_gamma_strong' if csp_ok else 'paused'}."}
    else:  # RED
        return {"csp_pct": 0, "leap_pct": 0, "sata_pct": 100, "csp_allowed": False,
                "note": "Red → CSP moratorium. 100% to SATA. Deploy buffer to LEAPs on confirmed deep signal."}


# ===================================
# BTC Cycle Zone & Trade Permissions
# ===================================

def classify_btc_cycle_zone(btc_weekly_rsi=None, btc_mvrv=None):
    """BTC macro cycle position. Informational only — does NOT gate trades."""
    if btc_weekly_rsi is None and btc_mvrv is None:
        return "UNKNOWN"
    rsi = btc_weekly_rsi or 50
    mvrv = btc_mvrv or 1.5
    if rsi > 70 or mvrv > 3.0:
        return "EUPHORIA"
    if rsi > 55 or mvrv > 1.5:
        return "EXPANSION"
    if rsi > 43 or mvrv > 1.0:
        return "RECOVERY"
    return "DEEP_ACCUM"


def compute_trade_permissions(
    regime, iv_rank, iv_percentile, basic_mnav, ev_mnav,
    leap_entry_score, csp_count, spot, flip, atr,
    btc_cycle_zone="UNKNOWN", data_health_ok=True
):
    """
    Three independent permission flags driven by ASST structural data.
    BTC cycle zone is context (sizing), not a gate.
    """
    # CSP: follows ASST gamma regime
    flip_dist_atr = abs(flip - spot) / atr if atr and atr > 0 else 99
    if not data_health_ok:
        csp, csp_r = "OFF", "data health flag — stale or missing inputs"
    elif regime in ("short_gamma_strong", "short_gamma_weak"):
        if csp_count > 0:
            csp, csp_r = "ALLOWED", f"{regime.replace('_', ' ')}, {csp_count} candidates in band"
        else:
            csp, csp_r = "CONDITIONAL", f"{regime.replace('_', ' ')}, no candidates at current strikes"
    elif regime == "neutral" and flip_dist_atr < 0.3 and csp_count > 0:
        csp, csp_r = "CONDITIONAL", f"neutral near flip ({flip_dist_atr:.1f}σ) with {csp_count} candidates"
    elif regime in ("long_gamma_weak", "long_gamma_strong"):
        csp, csp_r = "OFF", f"{regime.replace('_', ' ')} — dealers dampen moves"
    else:
        csp, csp_r = "OFF", f"{regime.replace('_', ' ')}"

    # LEAP: Permission = IV + score (mNAV does NOT gate permission)
    # Sizing = mNAV bands × EV/mNAV modifier × BTC zone overlay
    score = leap_entry_score or 0
    ir = iv_rank if iv_rank is not None else 50
    ip = iv_percentile if iv_percentile is not None else 50

    if not data_health_ok:
        leap, leap_r = "OFF", "data health flag — stale or missing inputs"
    elif ir < 40 and ip < 40 and score >= 1.0:
        leap, leap_r = "ALLOWED", f"cheap IV (rank {ir:.0f}, pct {ip:.0f}) + score {score:.1f}"
    elif ir > 70 or ip > 70:
        leap, leap_r = "OFF", f"IV expensive (rank {ir:.0f}, pct {ip:.0f})"
    else:
        leap, leap_r = "OFF", f"IV not cheap enough (rank {ir:.0f}, pct {ip:.0f}) or score {score:.1f} < 1"

    # LEAP sizing: mNAV bands (monotonic: higher mNAV → smaller size)
    mnav_val = basic_mnav if basic_mnav is not None else (ev_mnav or 1.0)
    ev_val = ev_mnav if ev_mnav is not None else mnav_val

    if leap == "ALLOWED":
        # Step 1: mNAV band
        if mnav_val <= 0.9:
            leap_size = "FULL_PLUS"
        elif mnav_val <= 1.1:
            leap_size = "FULL"
        elif mnav_val <= 1.4:
            leap_size = "REDUCED"
        else:
            leap_size = "MINIMAL"

        # Step 2: EV/mNAV caution modifier (one-step downgrade if rich)
        if ev_val > 1.6:
            downgrade = {"FULL_PLUS": "FULL", "FULL": "REDUCED", "REDUCED": "MINIMAL", "MINIMAL": "MINIMAL"}
            leap_size = downgrade.get(leap_size, leap_size)
            leap_r += f" | EV/mNAV {ev_val:.2f}x caution → size downgraded"

        # Step 3: BTC zone overlay (one-step downgrade in late cycle)
        if btc_cycle_zone in ("EXPANSION", "EUPHORIA"):
            downgrade = {"FULL_PLUS": "FULL", "FULL": "REDUCED", "REDUCED": "MINIMAL", "MINIMAL": "MINIMAL"}
            leap_size = downgrade.get(leap_size, leap_size)
            if btc_cycle_zone == "EUPHORIA":
                # Double downgrade in euphoria
                leap_size = downgrade.get(leap_size, leap_size)

        leap_r += f" | mNAV {mnav_val:.3f}x, {btc_cycle_zone.replace('_',' ').lower()}"
    else:
        leap_size = "—"

    # PMCC: follows long-gamma regimes only.
    # No IV floor for ASST — absolute IV is 100%+ even at rank 0, ample premium for PMCC.
    # The regime gate (long gamma = dealers dampen moves) is the real protection.
    if regime in ("long_gamma_weak", "long_gamma_strong"):
        pmcc, pmcc_r = "ALLOWED", f"{regime.replace('_', ' ')} (IV {ir:.0f}%, premium available)"
    else:
        pmcc, pmcc_r = "OFF", f"{regime.replace('_', ' ')} — don't cap upside"

    return {
        "csp_allowed": csp, "csp_reason": csp_r,
        "leap_add_allowed": leap, "leap_add_size": leap_size, "leap_add_reason": leap_r,
        "pmcc_allowed": pmcc, "pmcc_reason": pmcc_r,
    }


def compute_action_banner(btc_cycle_zone, csp_allowed, leap_add_allowed, pmcc_allowed):
    """Merge BTC context + ASST permissions into one headline."""
    if btc_cycle_zone in ("DEEP_ACCUM", "RECOVERY"):
        if csp_allowed != "OFF" and leap_add_allowed == "ALLOWED":
            return "BUILD: CSP + LEAP"
        if leap_add_allowed == "ALLOWED":
            return "BUILD: LEAP ONLY"
        if csp_allowed != "OFF":
            return "BUILD: CSP ONLY"
        return "ACCUMULATE: WATCH"
    elif btc_cycle_zone == "EXPANSION":
        if pmcc_allowed == "ALLOWED":
            return "HARVEST: PMCC ON"
        if csp_allowed != "OFF":
            return "HARVEST: CSP INCOME"
        return "HARVEST: HOLD"
    elif btc_cycle_zone == "EUPHORIA":
        return "DEFEND: STAND DOWN"
    return "WATCH"


# ======================
# Top-level daily API
# ======================

def generate_daily_recommendations(
    spot: float,
    flip: float,
    atr_1d: Optional[float],
    net_gex: float,
    gex_percentile: float,
    gex_strikes: List[GexStrike],
    puts: List[OptionQuote],
    calls: List[OptionQuote],
    cfg: StrategyConfig = STRATEGY_DEFAULTS,
    # Engine v2 parameters (all with safe defaults for backward compat)
    iv_rank: float = 25.0,
    iv_percentile: float = 25.0,
    basic_mnav: float = 0.72,
    bitcoin_yield_pct: float = 13.8,
    # Risk management parameters (all with safe defaults)
    asst_drawdown_90d: float = 0.0,
    btc_mvrv: Optional[float] = None,
    btc_weekly_rsi: Optional[float] = None,
) -> Dict:
    """
    High-level wrapper for the daily co-pilot.

    Inputs per ticker:
        - spot: current underlying price.
        - flip: current gamma flip level.
        - atr_1d: 1-day ATR (optional but recommended).
        - net_gex: total net GEX.
        - gex_percentile: percentile of net_gex vs history.
        - gex_strikes: per-strike GEX snapshot.
        - puts: short-dated puts (OptionQuote).
        - calls: long-dated calls incl. LEAPs (OptionQuote, is_leap=True).
        - iv_rank: IV Rank (0-100), default 25.0
        - iv_percentile: IV Percentile (0-100), default 25.0
        - basic_mnav: Market Cap / BTC NAV, default 0.72
        - bitcoin_yield_pct: BTC Yield YTD %, default 13.8

    Returns:
        dict with regime, magnets, CSP/LEAP candidates, scores, and notes.
    """
    regime = classify_regime_from_gex(
        net_gex, gex_percentile, spot=spot, flip=flip, atr_1d=atr_1d, cfg=cfg
    )
    pos_magnets, neg_magnets = find_gex_magnets(gex_strikes, cfg=cfg, include_negative=True)

    csp_candidates, csp_band, dte_window = filter_csp_candidates(
        puts=puts,
        flip=flip,
        atr_1d=atr_1d,
        regime=regime,
        pos_magnets=pos_magnets,
        cfg=cfg,
        iv_rank=iv_rank,
    )
    dte_min, dte_max = dte_window

    leap_core, core_band = filter_leap_core_candidates(
        calls=calls,
        flip=flip,
        atr_1d=atr_1d,
        regime=regime,
        pos_magnets=pos_magnets,
        cfg=cfg,
    )

    leap_mid = filter_leap_mid_candidates(
        calls=calls,
        flip=flip,
        regime=regime,
        pos_magnets=pos_magnets,
        atr_1d=atr_1d,
        cfg=cfg,
    )

    leap_tail = filter_leap_tail_candidates(
        calls=calls,
        flip=flip,
        regime=regime,
        pos_magnets=pos_magnets,
        cfg=cfg,
    )

    # ── Three-factor LEAP entry gate (empirically grounded) ──
    gamma_score = 1.0 if (net_gex < 0 and gex_percentile < 50) else 0.0
    vega_score = 1.0 if (iv_rank < 40 and iv_percentile < 40) else 0.5 if (iv_rank < 40) else 0.0
    mnav_score = 1.0 if (basic_mnav < 1.0 and bitcoin_yield_pct > 0) else 0.5 if basic_mnav < 1.0 else 0.0
    leap_entry_score = gamma_score + vega_score + mnav_score  # max 3.0

    # ── IV Regime classification ──
    iv_regime = "CHEAP_VOL" if iv_rank < 30 else "EXPENSIVE_VOL" if iv_rank > 70 else "NEUTRAL_VOL"

    # ── PMCC status (only sell calls against LEAPs in long-gamma regimes) ──
    pmcc_status = "AVAILABLE" if regime in cfg.PMCC_ALLOWED_REGIMES else "PROHIBITED"

    # ── Notes ──
    notes: List[str] = []
    notes.append(
        f"Regime {regime}; spot {spot:.2f}, flip {flip:.2f}, "
        f"ATR1d {atr_1d if atr_1d is not None else 'n/a'}."
    )
    notes.append(
        f"CSP income band: [{csp_band[0]:.2f}, {csp_band[1]:.2f}]."
    )
    notes.append(
        f"LEAP convexity band: [{core_band[0]:.2f}, {core_band[1]:.2f}]."
    )
    if pos_magnets:
        notes.append(
            "Positive GEX magnets: "
            + ", ".join(f"{m.strike} (GEX {m.net_gex:.1e})" for m in pos_magnets)
        )
    else:
        notes.append("Positive GEX magnets: none.")

    if not csp_candidates:
        notes.append("CSP: no gamma-aligned candidates today.")
    else:
        top = csp_candidates[0]
        notes.append(
            "CSP: top candidate "
            f"{top['strike']}P {top['expiry']} DTE {top['dte']} "
            f"mid {top['mid']:.2f}, eff basis {top['eff_basis']:.2f}."
        )

    if leap_core:
        core_strikes = sorted({float(o.strike) for o in leap_core})
        notes.append(f"LEAP core strikes: {core_strikes}.")
    else:
        notes.append("LEAP core: no strikes in band.")

    if leap_mid:
        mid_strikes = sorted({float(o.strike) for o in leap_mid})
        notes.append(f"LEAP mid strikes: {mid_strikes}.")
    if leap_tail:
        tail_strikes = sorted({float(o.strike) for o in leap_tail})
        notes.append(f"LEAP tail strikes: {tail_strikes}.")

    # ── Engine v2 notes ──

    # GEX convention warning — ASST 95%+ call OI is speculative buying
    notes.append(
        "GEX CONVENTION: ASST 95%+ call OI from speculative buying. "
        "Standard equity GEX tools may understate dealer short-gamma."
    )

    # PMCC regime eligibility (mechanical regime check only — pmcc_allowed is the actual trade permission)
    notes.append(f"PMCC regime eligible: {pmcc_status} ({regime}). Permission: see pmcc_allowed.")

    # CSP management rule — tastylive 21 DTE study
    notes.append(
        f"CSP management: exit at 50% profit OR 21 DTE, whichever first. "
        f"DTE window: {dte_min}-{dte_max} (IV regime: {iv_regime})."
    )

    # LEAP entry score
    notes.append(
        f"LEAP entry score: {leap_entry_score:.1f}/3.0 "
        f"(gamma={gamma_score:.0f}, vega={vega_score:.1f}, mNAV={mnav_score:.1f})."
    )

    # TODO: Compute numerical GammaIB threshold for ASST (Barbon & Buraschi eq. 12)
    # TODO: IV Percentile automated calculation from historical IV data
    # TODO: Historical regime flip frequency — track regime transitions per 30-day window
    # TODO: Bitcoin Yield breakeven BTC price — compute from avg_cost_per_btc + debt
    # TODO: ASST beta to BTC by regime — compute rolling 30d beta conditioned on regime

    # ── Risk zone classification & monthly allocation ──
    zone = classify_risk_zone(
        asst_drawdown_90d=asst_drawdown_90d,
        btc_mvrv=btc_mvrv,
        btc_weekly_rsi=btc_weekly_rsi,
        cfg=cfg,
    )
    allocation = get_monthly_allocation(
        zone=zone,
        leap_entry_score=leap_entry_score,
        iv_regime=iv_regime,
        regime=regime,
        cfg=cfg,
    )

    # ── BTC Cycle Zone & Trade Permissions (replaces old risk zone / monthly allocation) ──
    btc_zone = classify_btc_cycle_zone(btc_weekly_rsi=btc_weekly_rsi, btc_mvrv=btc_mvrv)
    permissions = compute_trade_permissions(
        regime=regime, iv_rank=iv_rank, iv_percentile=iv_percentile,
        basic_mnav=basic_mnav, ev_mnav=basic_mnav,
        leap_entry_score=leap_entry_score, csp_count=len(csp_candidates),
        spot=spot, flip=flip, atr=atr_1d, btc_cycle_zone=btc_zone,
    )
    action = compute_action_banner(
        btc_zone, permissions["csp_allowed"],
        permissions["leap_add_allowed"], permissions["pmcc_allowed"],
    )

    # Structured notes (no zone-gated allocation language)
    notes.append(f"BTC cycle: {btc_zone} (RSI={btc_weekly_rsi if btc_weekly_rsi is not None else 'N/A'}, MVRV={'N/A' if btc_mvrv is None else f'{btc_mvrv:.2f}'}). ASST drawdown: {asst_drawdown_90d:.1%}.")
    notes.append(f"Action: {action}. CSP: {permissions['csp_allowed']} ({permissions['csp_reason']}). LEAP: {permissions['leap_add_allowed']} {permissions['leap_add_size']} ({permissions['leap_add_reason']}). PMCC: {permissions['pmcc_allowed']} ({permissions['pmcc_reason']}).")

    # Reconcile PMCC regime eligibility vs trade permission
    if pmcc_status == "AVAILABLE" and permissions['pmcc_allowed'] == "OFF":
        notes.append(f"PMCC note: regime eligible ({pmcc_status}) but permission OFF — check trade permissions for reason.")

    return {
        "regime": regime,
        "posmagnets": pos_magnets,
        "negmagnets": neg_magnets,
        "cspcandidates": csp_candidates,
        "csp_band": csp_band,
        "leapcore": leap_core,
        "leapcore_band": core_band,
        "leapmid": leap_mid,
        "leaptail": leap_tail,
        "notes": notes,
        # Engine v2 fields
        "leap_entry_score": leap_entry_score,
        "pmcc_status": pmcc_status,
        "iv_regime": iv_regime,
        "gex_convention_warning": True,
        "management_rules": {
            "exit": f"50% profit OR {cfg.CSP_EXIT_DTE} DTE",
            "dte_window": f"{dte_min}-{dte_max} DTE ({iv_regime.lower().replace('_', ' ')})",
        },
        # Risk management fields
        "risk_zone": zone,
        "monthly_allocation": allocation,
        # Trade permissions
        "btc_cycle_zone": btc_zone,
        "action_banner": action,
        **permissions,
    }


# ======================
# Simple test harness
# ======================

def daily_test_harness() -> None:
    """
    Minimal test harness using example ASST-like numbers.

    Tests three states of the three-factor LEAP entry gate:
      1. Full score (3.0): short gamma + cheap vol + mNAV discount
      2. Partial score (1.5): mixed signals
      3. Zero score (0.0): long gamma + expensive vol + mNAV premium

    This does NOT touch any live APIs. It is for local / Computer dry-runs
    to sanity-check that the pipeline wires correctly.
    """

    gex_strikes = [
        GexStrike(strike=10.0, net_gex=1.5e5),
        GexStrike(strike=11.0, net_gex=3.2e5),
        GexStrike(strike=12.0, net_gex=0.8e5),
    ]

    puts = [
        OptionQuote(strike=10.5, dte=32, mid=1.90, expiry="2026-04-25"),
        OptionQuote(strike=11.0, dte=32, mid=2.00, expiry="2026-04-25"),
        OptionQuote(strike=10.5, dte=50, mid=2.10, expiry="2026-05-15"),
        OptionQuote(strike=11.0, dte=50, mid=2.30, expiry="2026-05-15"),
    ]

    calls = [
        OptionQuote(strike=9.0,  dte=660, mid=2.70, expiry="2028-01-21", is_leap=True),
        OptionQuote(strike=9.5,  dte=660, mid=2.40, expiry="2028-01-21", is_leap=True),
        OptionQuote(strike=10.5, dte=660, mid=2.00, expiry="2028-01-21", is_leap=True),
    ]

    scenarios = [
        {
            "name": "FULL SCORE (3.0) — Short gamma + Cheap vol + mNAV discount + GREEN zone",
            "spot": 10.02, "flip": 9.47, "atr_1d": 0.40,
            "net_gex": -3.0e5, "gex_percentile": 20.0,
            "iv_rank": 20.0, "iv_percentile": 15.0,
            "basic_mnav": 0.72, "bitcoin_yield_pct": 13.8,
            "asst_drawdown_90d": -0.10, "btc_weekly_rsi": 55.0, "btc_mvrv": None,
        },
        {
            "name": "PARTIAL SCORE (1.5) — Mixed signals + AMBER zone",
            "spot": 10.02, "flip": 9.47, "atr_1d": 0.40,
            "net_gex": 1.0e5, "gex_percentile": 55.0,
            "iv_rank": 35.0, "iv_percentile": 50.0,
            "basic_mnav": 0.85, "bitcoin_yield_pct": 5.0,
            "asst_drawdown_90d": -0.40, "btc_weekly_rsi": 48.0, "btc_mvrv": None,
        },
        {
            "name": "ZERO SCORE (0.0) — Long gamma + Expensive vol + mNAV premium + RED zone",
            "spot": 10.02, "flip": 9.47, "atr_1d": 0.40,
            "net_gex": 5.0e5, "gex_percentile": 80.0,
            "iv_rank": 75.0, "iv_percentile": 80.0,
            "basic_mnav": 1.20, "bitcoin_yield_pct": -2.0,
            "asst_drawdown_90d": -0.65, "btc_weekly_rsi": 35.0, "btc_mvrv": None,
        },
    ]

    for i, sc in enumerate(scenarios):
        recs = generate_daily_recommendations(
            spot=sc["spot"],
            flip=sc["flip"],
            atr_1d=sc["atr_1d"],
            net_gex=sc["net_gex"],
            gex_percentile=sc["gex_percentile"],
            gex_strikes=gex_strikes,
            puts=puts,
            calls=calls,
            iv_rank=sc["iv_rank"],
            iv_percentile=sc["iv_percentile"],
            basic_mnav=sc["basic_mnav"],
            bitcoin_yield_pct=sc["bitcoin_yield_pct"],
            asst_drawdown_90d=sc["asst_drawdown_90d"],
            btc_weekly_rsi=sc["btc_weekly_rsi"],
            btc_mvrv=sc["btc_mvrv"],
        )

        print(f"\n{'='*60}")
        print(f"=== Scenario {i+1}: {sc['name']} ===")
        print(f"{'='*60}")
        print(f"  Regime: {recs['regime']}")
        print(f"  CSP band (income): {recs['csp_band']}")
        print(f"  LEAP band (convexity): {recs['leapcore_band']}")
        print(f"  LEAP entry score: {recs['leap_entry_score']:.1f}/3.0")
        print(f"  IV regime: {recs['iv_regime']}")
        print(f"  PMCC status: {recs['pmcc_status']}")
        print(f"  Management rules: {recs['management_rules']}")
        print(f"  Risk zone: {recs['risk_zone']}")
        print(f"  Monthly allocation: {recs['monthly_allocation']}")
        print(f"  CSP candidates: {len(recs['cspcandidates'])}")
        print(f"  LEAP core: {len(recs['leapcore'])}")
        print("  Notes:")
        for line in recs["notes"]:
            print(f"    - {line}")

    # Verify regime distance override directly
    print(f"\n{'='*60}")
    print("=== Regime Distance Override Test ===")
    regime_no_dist = classify_regime_from_gex(-3.0e5, 20.0)
    regime_with_dist = classify_regime_from_gex(
        -3.0e5, 20.0, spot=10.02, flip=9.47, atr_1d=0.40
    )
    print(f"  Regime without distance override: {regime_no_dist}")
    print(f"  Regime with distance override:    {regime_with_dist}")

    # Risk zone classification tests
    print(f"\n{'='*60}")
    print("=== Risk Zone Classification Tests ===")
    print(f"{'='*60}")

    zone_tests = [
        {"dd": -0.10, "rsi": 55.0, "mvrv": None, "expected": "GREEN"},
        {"dd": -0.40, "rsi": 55.0, "mvrv": None, "expected": "AMBER"},
        {"dd": -0.10, "rsi": 45.0, "mvrv": None, "expected": "AMBER"},
        {"dd": -0.65, "rsi": 55.0, "mvrv": None, "expected": "RED"},
        {"dd": -0.10, "rsi": 35.0, "mvrv": None, "expected": "RED"},
        {"dd": -0.10, "rsi": 55.0, "mvrv": 0.75, "expected": "RED"},
        {"dd": -0.10, "rsi": 55.0, "mvrv": 0.90, "expected": "AMBER"},
        {"dd": -0.40, "rsi": 35.0, "mvrv": 0.75, "expected": "RED"},
    ]
    for t in zone_tests:
        zone = classify_risk_zone(t["dd"], btc_mvrv=t["mvrv"], btc_weekly_rsi=t["rsi"])
        status = "PASS" if zone == t["expected"] else "FAIL"
        print(f"  [{status}] dd={t['dd']:.0%} rsi={t['rsi']} mvrv={t['mvrv']} → {zone} (expected {t['expected']})")


if __name__ == "__main__":
    daily_test_harness()
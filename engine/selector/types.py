"""
Type definitions for the selector engine port (P1.8).

These mirror the TypeScript types in shared/selectorEngine.ts (v1.5).
Where the TS used union string literals, Python uses string constants
with type aliases (Literal where strict checking matters).

Rendering convention: types in this module are NOT used as runtime
constraints during evaluate() \u2014 instead we work with plain dicts and
lists to match JSON serialization exactly. The types exist for IDE
hints and documentation only.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# Schema versioning
COHORT_ID_VERSION = "1.5"
SCHEMA_VERSION: Literal["1.2"] = "1.2"

# String literal types
DeltaAction = Literal["ADD", "REDUCE", "ROLL", "ROTATE_IN", "ROTATE_OUT", "HOLD", "NONE"]
Confidence = Literal["HIGH", "MED", "LOW", "NONE"]
ChainDataStatus = Literal["from_chain", "approximated", "unavailable"]
Vehicle = Literal[
    "LEAP_CORE", "LEAP_MID_TAIL", "PMCC", "CSP",
    "DEFENSIVE_YIELD_STRESS", "DEFENSIVE_YIELD_TOP", "NO_TRADE",
]
Horizon = Literal[5, 10, 21, 63]


class CIBand(TypedDict, total=False):
    estimate: float
    lo: float
    hi: float
    level: float


class HistoricalEdge(TypedDict, total=False):
    cohort_id: str
    horizon: int
    n_total_pm: int
    n_with_fwd: int
    horizon_basis: str
    fwd_median: Optional[float]
    fwd_p10: Optional[float]
    fwd_p25: Optional[float]
    fwd_p75: Optional[float]
    fwd_p90: Optional[float]
    fwd_median_ci: Optional[CIBand]
    fwd_p25_ci: Optional[CIBand]
    fwd_p75_ci: Optional[CIBand]
    fwd_p10_ci: Optional[CIBand]
    fwd_p90_ci: Optional[CIBand]
    max_dd_median: Optional[float]
    max_dd_median_ci: Optional[CIBand]
    rv_median: Optional[float]
    rv_median_ci: Optional[CIBand]
    sample_sufficient: bool
    note: Optional[str]


class ContractRow(TypedDict, total=False):
    underlying: str
    type: str
    direction: str
    strike: Optional[float]
    expiry: Optional[str]
    dte: Optional[int]
    approx_delta: Optional[float]
    quantity: float
    est_annualized_yield: Optional[float]
    long_leg: "ContractRow"
    short_leg: "ContractRow"
    quantity_shares: float
    estimated_allocation_pct: float
    chain_data: str
    bci: Optional[Dict[str, Any]]
    bci_vintages: List[Dict[str, Any]]
    bci_grid: Dict[str, Any]


class PriorPosition(TypedDict, total=False):
    summary: str
    details: Dict[str, Any]


class Recommendation(TypedDict, total=False):
    vehicle: str
    recommended: bool
    contracts: List[ContractRow]
    rationale: str
    historical_edge: Optional[HistoricalEdge]
    prior_position: PriorPosition
    delta_from_prior: str
    confidence: str
    confidence_detail: Optional[str]
    blocked_reason: Optional[str]


class CycleMetric(TypedDict, total=False):
    key: str
    label: str
    value: Optional[float]
    top_threshold: float
    direction: str
    pct_of_threshold: Optional[float]
    fired: bool


class DefensiveYieldStress(TypedDict, total=False):
    triggered: bool
    triggers_met: List[str]
    conditions: List[Dict[str, Any]]
    rotation_pct_suggested: Optional[float]


class DefensiveYieldTop(TypedDict, total=False):
    triggered: bool
    metrics: List[CycleMetric]
    n_fired: int
    rotation_pct_suggested: Optional[float]


class SelectorOutput(TypedDict, total=False):
    schema_version: str
    snapshot_id: str
    timestamp_utc: str
    date: str
    session: str
    cohort_id: str
    cohort_id_v: str
    gamma_regime: str
    iv_regime: str
    iv_band: Optional[int]
    iv_band_label: str
    btc_cycle_zone: str
    risk_zone: str
    gex_percentile: Optional[float]
    overall_posture: str
    ratchet_streak_length: Optional[int]
    defensive_yield_stress: DefensiveYieldStress
    defensive_yield_top: DefensiveYieldTop
    recommendations: List[Recommendation]
    data_health_flags: List[str]
    isolation_note: str


__all__ = [
    "COHORT_ID_VERSION", "SCHEMA_VERSION",
    "CIBand", "HistoricalEdge", "ContractRow", "PriorPosition",
    "Recommendation", "CycleMetric", "DefensiveYieldStress", "DefensiveYieldTop",
    "SelectorOutput",
]

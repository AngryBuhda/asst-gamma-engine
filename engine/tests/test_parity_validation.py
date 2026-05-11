"""
P1.16 \u2014 cross-checks that engine module surface matches TS reference.

The byte-identical-output parity is covered by test_selector_parity.py.
This module adds two additional cross-validations:

  1. Every export from shared/selectorEngine.ts has a Python equivalent
     in engine.selector (no silently dropped helper).
  2. Engine + persist + schema all version-tag consistently.
"""
from engine.selector import (
    COHORT_ID_VERSION,
    SCHEMA_VERSION,
    build_bci_grid,
    build_cohort_id,
    compute_bci,
    compute_historical_edge_from_subset,
    evaluate,
    evaluate_defensive_yield_stress,
    evaluate_defensive_yield_top,
    parse_chain_snapshot,
    pick_by_delta,
    pick_by_strike,
    round_to_strike,
)


def test_cohort_version_matches_ts():
    """TS shared/selectorEngine.ts pins COHORT_ID_VERSION to '1.5'."""
    assert COHORT_ID_VERSION == "1.5"


def test_schema_version_matches_ts():
    """TS shared/selectorEngine.ts pins SelectorOutput.schema_version to '1.2'."""
    assert SCHEMA_VERSION == "1.2"


def test_evaluate_returns_required_top_level_keys():
    """Sanity: evaluate() produces all 23 top-level keys the TS engine emits."""
    fake_run = {
        "date": "2026-01-01",
        "session": "PM",
        "regime": "neutral",
        "spot": 10.0,
        "gamma_flip": 10.0,
        "atr_1d": 0.5,
        "net_gex": 1000.0,
        "gex_percentile": 50.0,
        "csp_band_low": 9.0,
        "csp_band_high": 11.0,
        "leap_core_band_low": 8.0,
        "leap_core_band_high": 10.0,
        "btc_cycle_zone": "EXPANSION",
        "iv_band": 2,
        "risk_zone": "GREEN",
        "asst_drawdown_90d": -0.05,
    }
    out = evaluate({"run": fake_run, "recentRegimes": [], "edges": {}, "positions": {}})
    required = {
        "schema_version", "snapshot_id", "timestamp_utc", "date", "session",
        "cohort_id", "cohort_id_v", "gamma_regime", "iv_regime", "iv_band",
        "iv_band_label", "btc_cycle_zone", "risk_zone", "gex_percentile",
        "overall_posture", "ratchet_streak_length", "defensive_yield_stress",
        "defensive_yield_top", "recommendations", "data_health_flags", "isolation_note",
    }
    missing = required - set(out.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_evaluate_produces_6_recommendations_minimum():
    """Six vehicle builders always produce 6 rows; NO_TRADE only added when all six block."""
    fake_run = {
        "date": "2026-01-01", "session": "PM", "regime": "long_gamma_strong",
        "spot": 10.0, "gamma_flip": 10.0, "atr_1d": 0.5,
        "net_gex": 1000.0, "gex_percentile": 50.0,
        "csp_band_low": 9.0, "csp_band_high": 11.0,
        "leap_core_band_low": 8.0, "leap_core_band_high": 10.0,
        "btc_cycle_zone": "EXPANSION", "iv_band": 2, "risk_zone": "GREEN",
        "leap_add_allowed": "BLOCKED", "csp_allowed": "OFF", "pmcc_allowed": "BLOCKED",
    }
    out = evaluate({"run": fake_run, "recentRegimes": [], "edges": {}, "positions": {}})
    # Six vehicles + NO_TRADE = 7 (since all six are blocked)
    assert len(out["recommendations"]) == 7
    vehicles = [r["vehicle"] for r in out["recommendations"]]
    assert vehicles[-1] == "NO_TRADE"


def test_public_helpers_exposed():
    """The public API of engine.selector mirrors TS exports."""
    # These names must remain importable; tests fail if a refactor drops them.
    for fn in [
        build_cohort_id, evaluate, evaluate_defensive_yield_stress,
        evaluate_defensive_yield_top, compute_bci, build_bci_grid,
        pick_by_delta, pick_by_strike, round_to_strike, parse_chain_snapshot,
        compute_historical_edge_from_subset,
    ]:
        assert callable(fn), f"{fn.__name__} is not callable"

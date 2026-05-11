"""Unit tests for defensive yield trigger evaluators \u2014 stress and top.

Stress fires when 2+ of 4 conditions met (risk_zone=RED, gex < 70,
drawdown <= -30%, 3-session short/neutral streak).

Top fires when 2+ of 5 cycle metrics exceed thresholds (MVRV-Z > 6,
Pi-Cycle == 1, Puell > 4, NUPL > 0.75, Reserve Risk > 0.02).
"""
from engine.selector.engine import (
    evaluate_defensive_yield_stress,
    evaluate_defensive_yield_top,
)


def _base_run():
    return {
        "risk_zone": "GREEN",
        "gex_percentile": 80.0,
        "asst_drawdown_90d": -0.10,
        "btc_mvrv_zscore": 3.0,
        "btc_pi_cycle_signal": 0,
        "btc_puell_multiple": 2.0,
        "btc_nupl": 0.50,
        "btc_reserve_risk": 0.005,
    }


def test_stress_no_triggers():
    result = evaluate_defensive_yield_stress(_base_run(), ["LONG_GAMMA_STRONG"] * 3)
    assert result["triggered"] is False
    assert result["triggers_met"] == []
    assert result["rotation_pct_suggested"] is None


def test_stress_one_trigger_not_enough():
    run = _base_run()
    run["risk_zone"] = "RED"
    result = evaluate_defensive_yield_stress(run, ["LONG_GAMMA_STRONG"] * 3)
    assert result["triggered"] is False
    assert len(result["triggers_met"]) == 1


def test_stress_two_triggers_fires():
    run = _base_run()
    run["risk_zone"] = "RED"
    run["asst_drawdown_90d"] = -0.35
    result = evaluate_defensive_yield_stress(run, ["LONG_GAMMA_STRONG"] * 3)
    assert result["triggered"] is True
    assert len(result["triggers_met"]) == 2
    assert result["rotation_pct_suggested"] == 0.10


def test_stress_four_triggers_caps_at_25pct():
    run = _base_run()
    run["risk_zone"] = "RED"
    run["gex_percentile"] = 30.0
    run["asst_drawdown_90d"] = -0.40
    result = evaluate_defensive_yield_stress(
        run, ["SHORT_GAMMA_WEAK", "NEUTRAL", "SHORT_GAMMA_STRONG"]
    )
    assert result["triggered"] is True
    assert len(result["triggers_met"]) == 4
    # 0.10 + 0.05 * (4 - 2) = 0.20, well below the 0.25 cap
    assert result["rotation_pct_suggested"] == 0.20


def test_stress_streak_short_circuit_needs_3_sessions():
    run = _base_run()
    run["risk_zone"] = "RED"
    # Only 2 recent regimes \u2014 streak condition can't fire
    result = evaluate_defensive_yield_stress(run, ["SHORT_GAMMA_WEAK", "NEUTRAL"])
    assert "gamma short/neutral last 3 sessions" not in result["triggers_met"]


def test_stress_conditions_always_present():
    """All 4 conditions are in the structured array even when not fired."""
    result = evaluate_defensive_yield_stress(_base_run(), [])
    assert len(result["conditions"]) == 4
    for c in result["conditions"]:
        assert "key" in c
        assert "label" in c
        assert "fired" in c


def test_top_no_triggers():
    result = evaluate_defensive_yield_top(_base_run())
    assert result["triggered"] is False
    assert result["n_fired"] == 0
    assert len(result["metrics"]) == 5


def test_top_two_triggers_fires():
    run = _base_run()
    run["btc_mvrv_zscore"] = 7.0  # > 6
    run["btc_nupl"] = 0.80  # > 0.75
    result = evaluate_defensive_yield_top(run)
    assert result["triggered"] is True
    assert result["n_fired"] == 2
    assert result["rotation_pct_suggested"] == 0.25


def test_top_four_triggers_partial_cap():
    run = _base_run()
    run["btc_mvrv_zscore"] = 7.0
    run["btc_pi_cycle_signal"] = 1
    run["btc_puell_multiple"] = 5.0
    run["btc_nupl"] = 0.80
    result = evaluate_defensive_yield_top(run)
    assert result["n_fired"] == 4
    # 0.25 + 0.10 * (4 - 2) = 0.45
    assert result["rotation_pct_suggested"] == 0.45


def test_top_all_five_triggers_caps_at_50pct():
    run = _base_run()
    run["btc_mvrv_zscore"] = 7.0
    run["btc_pi_cycle_signal"] = 1
    run["btc_puell_multiple"] = 5.0
    run["btc_nupl"] = 0.80
    run["btc_reserve_risk"] = 0.05
    result = evaluate_defensive_yield_top(run)
    assert result["n_fired"] == 5
    # 0.25 + 0.10 * (5 - 2) = 0.55 > 0.50 cap
    assert result["rotation_pct_suggested"] == 0.50


def test_top_null_metrics_dont_fire():
    run = _base_run()
    run["btc_mvrv_zscore"] = None
    run["btc_nupl"] = None
    result = evaluate_defensive_yield_top(run)
    # null values must not count as fired
    for m in result["metrics"]:
        if m["key"] in ("mvrv_zscore", "nupl"):
            assert m["fired"] is False

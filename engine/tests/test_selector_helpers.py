"""Unit tests for selector engine pure helpers \u2014 round_to_strike,
build_cohort_id, compute_bci, pick_by_delta, pick_by_strike,
parse_chain_snapshot.

These functions are deterministic and have no external dependencies,
so we can exercise them with synthetic inputs covering the edge cases
the golden fixtures don't reach (e.g., iv_band=2 NORMAL_VOL which has
zero rows in the v1 dataset).
"""
from engine.selector.engine import (
    build_cohort_id,
    compute_bci,
    parse_chain_snapshot,
    pick_by_delta,
    pick_by_strike,
    round_to_strike,
)


def test_round_to_strike_below_20():
    # $0.50 increments under $20
    assert round_to_strike(8.0) == 8.0
    assert round_to_strike(8.2) == 8.0
    assert round_to_strike(8.3) == 8.5
    assert round_to_strike(8.7) == 8.5
    assert round_to_strike(8.8) == 9.0


def test_round_to_strike_at_or_above_20():
    # $1.00 increments at $20+
    assert round_to_strike(20.0) == 20
    assert round_to_strike(22.4) == 22
    assert round_to_strike(22.5) == 22  # banker's rounding to even
    assert round_to_strike(22.7) == 23


def test_build_cohort_id_long_gamma_neutral_vol():
    # iv_band=2 (NORMAL_VOL) is the gap-coverage case
    run = {"btc_cycle_zone": "EXPANSION", "iv_band": 2, "regime": "LONG_GAMMA_STRONG"}
    assert build_cohort_id(run) == "EXPANSION|NORMAL_VOL|LONG"


def test_build_cohort_id_short_gamma():
    run = {"btc_cycle_zone": "RECOVERY", "iv_band": 0, "regime": "SHORT_GAMMA_WEAK"}
    assert build_cohort_id(run) == "RECOVERY|EXTREME_CHEAP|SHORT"


def test_build_cohort_id_neutral_regime():
    run = {"btc_cycle_zone": "EXPANSION", "iv_band": 3, "regime": "NEUTRAL"}
    assert build_cohort_id(run) == "EXPANSION|RICH_VOL|NEUTRAL"


def test_build_cohort_id_missing_fields_fallback():
    run = {"btc_cycle_zone": None, "iv_band": None, "regime": None}
    assert build_cohort_id(run) == "UNKNOWN|UNKNOWN|UNKNOWN"


def test_compute_bci_pass():
    # spread = K_short(20) - K_long(10) = 10
    # buffer = (10 + 0.50) - 5.00 = 5.50 > 0 \u2014 pass
    result = compute_bci(0.50, 20, 5.00, 10, 10)
    assert result is not None
    assert result["passes"] is True
    assert result["buffer"] == 5.5
    assert result["source"] == "exact_chain"


def test_compute_bci_fail():
    # spread = K_short(15) - K_long(10) = 5
    # buffer = (5 + 0.10) - 6.00 = -0.90 < 0 \u2014 fail
    result = compute_bci(0.10, 15, 6.00, 10, 10)
    assert result is not None
    assert result["passes"] is False
    assert result["buffer"] == -0.9


def test_compute_bci_nearest_chain_source():
    # leap_strike != anchor_strike by 1.0 \u2014 source is nearest_chain
    result = compute_bci(0.50, 20, 5.00, 11, 10)
    assert result is not None
    assert result["source"] == "nearest_chain"
    assert result["leap_strike_distance"] == 1.0


def test_compute_bci_null_inputs_return_none():
    assert compute_bci(None, 20, 5.0, 10, 10) is None
    assert compute_bci(0.5, None, 5.0, 10, 10) is None
    assert compute_bci(0.5, 20, None, 10, 10) is None
    assert compute_bci(0.5, 20, 5.0, None, 10) is None
    assert compute_bci(0, 20, 5.0, 10, 10) is None  # short_mid <= 0
    assert compute_bci(0.5, 20, 0, 10, 10) is None  # leap_mid <= 0


def test_parse_chain_snapshot_null():
    assert parse_chain_snapshot({"option_chain_snapshot_json": None}) is None
    assert parse_chain_snapshot({"option_chain_snapshot_json": ""}) is None
    assert parse_chain_snapshot({}) is None


def test_parse_chain_snapshot_valid_json():
    run = {"option_chain_snapshot_json": '{"calls": [{"strike": 10}], "puts": []}'}
    result = parse_chain_snapshot(run)
    assert result is not None
    assert result["calls"][0]["strike"] == 10


def test_parse_chain_snapshot_invalid_json_returns_none():
    run = {"option_chain_snapshot_json": "not json"}
    assert parse_chain_snapshot(run) is None


def test_pick_by_delta_picks_closest():
    contracts = [
        {"delta": 0.10, "dte": 30, "mid": 0.5, "strike": 15},
        {"delta": 0.22, "dte": 30, "mid": 0.8, "strike": 12},
        {"delta": 0.35, "dte": 30, "mid": 1.2, "strike": 10},
    ]
    best = pick_by_delta(contracts, 0.20, 21, 45)
    assert best is not None
    assert best["strike"] == 12  # 0.22 closest to 0.20 target


def test_pick_by_delta_filters_dte_window():
    contracts = [
        {"delta": 0.20, "dte": 10, "mid": 0.5, "strike": 15},  # out of window
        {"delta": 0.50, "dte": 30, "mid": 0.8, "strike": 12},  # in window, wrong delta
    ]
    best = pick_by_delta(contracts, 0.20, 21, 45)
    assert best is not None
    assert best["strike"] == 12


def test_pick_by_strike_picks_closest():
    contracts = [
        {"strike": 10, "mid": 1.0},
        {"strike": 12, "mid": 0.8},
        {"strike": 15, "mid": 0.5},
    ]
    assert pick_by_strike(contracts, 11)["strike"] == 10
    assert pick_by_strike(contracts, 11.5)["strike"] == 12


def test_pick_returns_none_when_no_candidates():
    assert pick_by_delta([], 0.20, 21, 45) is None
    assert pick_by_strike([], 10) is None
    # Contracts with mid=0 are filtered out
    assert pick_by_strike([{"strike": 10, "mid": 0}], 10) is None

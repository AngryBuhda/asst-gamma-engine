"""Unit tests for engine.compute.iv_band \u2014 the 5-band IV partition.

Verifies edge cases at the threshold boundaries since the band edges
are FROZEN per shared/iv_band_spec.json (changing them requires a
version bump).
"""
from engine.compute.iv_band import (
    BAND_VERSION,
    compute_iv_band,
    label_for_band,
    legacy_iv_regime_for_band,
    band_edges,
)


def test_band_version_is_v1():
    assert BAND_VERSION == "v1"


def test_null_in_null_out():
    assert compute_iv_band(None) is None
    assert label_for_band(None) is None
    assert legacy_iv_regime_for_band(None) is None


def test_boundary_at_zero():
    assert compute_iv_band(0.0) == 0
    assert label_for_band(0) == "EXTREME_CHEAP"


def test_below_zero_clamps_to_extreme_cheap():
    assert compute_iv_band(-5.0) == 0


def test_above_hundred_clamps_to_extreme_rich():
    # We don't know the exact upper-band index, but it must equal the
    # last band in the spec.
    edges = band_edges()
    assert compute_iv_band(150.0) == edges[-1]["band"]


def test_midpoint_normal_vol():
    assert compute_iv_band(35.0) == 2
    assert compute_iv_band(50.0) == 2
    assert label_for_band(2) == "NORMAL_VOL"


def test_cheap_vol_band():
    assert compute_iv_band(15.0) == 1
    assert label_for_band(1) == "CHEAP_VOL"


def test_rich_vol_band():
    assert compute_iv_band(75.0) == 3
    assert label_for_band(3) == "RICH_VOL"


def test_extreme_rich_band():
    assert compute_iv_band(95.0) == 4
    assert label_for_band(4) == "EXTREME_RICH"


def test_legacy_iv_regime_mapping():
    assert legacy_iv_regime_for_band(0) == "CHEAP_VOL"
    assert legacy_iv_regime_for_band(1) == "CHEAP_VOL"
    assert legacy_iv_regime_for_band(2) == "NEUTRAL_VOL"
    assert legacy_iv_regime_for_band(3) == "EXPENSIVE_VOL"
    assert legacy_iv_regime_for_band(4) == "EXPENSIVE_VOL"


def test_nan_returns_none():
    nan = float("nan")
    assert compute_iv_band(nan) is None


def test_string_input_is_coerced_via_float():
    # The function tolerates string input via float() coercion.
    assert compute_iv_band("35.0") == 2
    # Invalid strings return None instead of raising.
    assert compute_iv_band("abc") is None

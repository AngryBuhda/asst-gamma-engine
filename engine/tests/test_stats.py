"""Unit tests for engine.selector.stats \u2014 the byte-identical-with-TS
statistical layer. The PRNG and bootstrap CI were already verified against
JS reference output in P1.8a; these tests catch regressions if anyone
edits the math later.
"""
import math

from engine.selector.stats import (
    bootstrap_median_ci,
    bootstrap_quantile_ci,
    mean,
    mulberry32,
    quantile,
    reportable_percentiles,
    seed_from_values,
    stdev,
)


def test_quantile_linear_interpolation():
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert quantile(arr, 0.5) == 3.0
    # 25th percentile: position = (5-1)*0.25 = 1; arr[1] = 2.0
    assert quantile(arr, 0.25) == 2.0
    # 75th percentile: position = (5-1)*0.75 = 3; arr[3] = 4.0
    assert quantile(arr, 0.75) == 4.0


def test_quantile_interpolated():
    arr = [1.0, 2.0, 3.0, 4.0]
    # median: position = (4-1)*0.5 = 1.5; arr[1]+arr[2]*0.5 = 2.5
    assert quantile(arr, 0.5) == 2.5


def test_quantile_empty():
    assert quantile([], 0.5) is None


def test_mean_basic():
    assert mean([1, 2, 3, 4, 5]) == 3.0
    assert mean([]) is None


def test_stdev_sample():
    # Sample stdev of [1,2,3,4,5] with n-1 denominator = sqrt(2.5)
    assert abs(stdev([1.0, 2.0, 3.0, 4.0, 5.0]) - math.sqrt(2.5)) < 1e-10
    assert stdev([1.0]) is None  # n < 2


def test_mulberry32_deterministic():
    # Same seed always produces same sequence
    rng_a = mulberry32(42)
    rng_b = mulberry32(42)
    for _ in range(10):
        assert rng_a() == rng_b()


def test_mulberry32_seed_42_first_value():
    # Verified against TS reference (P1.8a)
    rng = mulberry32(42)
    assert abs(rng() - 0.6011037519) < 1e-9


def test_seed_from_values_is_deterministic():
    a = seed_from_values([1.0, 2.0, 3.0])
    b = seed_from_values([1.0, 2.0, 3.0])
    assert a == b


def test_seed_from_values_changes_with_input():
    a = seed_from_values([1.0, 2.0])
    b = seed_from_values([2.0, 1.0])  # different order = different seed
    assert a != b


def test_reportable_percentiles_thresholds():
    # n=7: nothing reportable (below 8)
    r = reportable_percentiles(7)
    assert r == {"median": False, "p25": False, "p75": False, "p10": False, "p90": False}
    # n=8: median only
    r = reportable_percentiles(8)
    assert r["median"] is True
    assert r["p25"] is False
    # n=15: median + p25 + p75
    r = reportable_percentiles(15)
    assert r["median"] is True and r["p25"] is True and r["p75"] is True
    assert r["p10"] is False
    # n=30: all reportable
    r = reportable_percentiles(30)
    assert all(r.values())


def test_bootstrap_median_ci_below_threshold():
    # n < 8 returns None
    assert bootstrap_median_ci([1.0, 2.0, 3.0, 4.0]) is None
    assert bootstrap_median_ci([1.0] * 7) is None


def test_bootstrap_median_ci_returns_band():
    vals = [-0.05, 0.012, 0.08, -0.022, 0.034, 0.0, 0.067, -0.011]
    result = bootstrap_median_ci(vals)
    assert result is not None
    assert "estimate" in result
    assert "lo" in result and "hi" in result
    assert result["n"] == 8
    assert result["B"] == 1000
    assert result["level"] == 0.95
    # lo <= estimate <= hi
    assert result["lo"] <= result["estimate"] <= result["hi"]


def test_bootstrap_quantile_ci_deterministic():
    vals = [-0.05, 0.012, 0.08, -0.022, 0.034, 0.0, 0.067, -0.011, 0.05, 0.02]
    a = bootstrap_quantile_ci(vals, 0.5)
    b = bootstrap_quantile_ci(vals, 0.5)
    assert a == b  # same input \u2192 same seed \u2192 same output

"""
Stats helpers \u2014 Python port of shared/stats.ts (P1.8).

Bootstrap CIs must match the TypeScript output byte-for-byte. The
TypeScript code uses:
  - Mulberry32 PRNG seeded deterministically from rounded input values
  - JavaScript bitwise/arithmetic semantics on uint32
  - Quantile by linear interpolation on sorted values

We replicate JS semantics exactly using numpy's uint32 type for the PRNG
and the same indexing pattern. Any drift here propagates into HistoricalEdge
CIs and fails the golden-fixture parity gate.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, TypedDict


def quantile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile matching TS implementation.

    TS: pos = (n-1)*q; lo=floor; hi=ceil; return v[lo] + (v[hi]-v[lo])*(pos-lo)
    """
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    pos = (n - 1) * q
    import math
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values: List[float]) -> Optional[float]:
    """Sample stdev (n-1 denominator). None when n < 2."""
    if len(values) < 2:
        return None
    m = mean(values)
    assert m is not None
    ss = sum((x - m) ** 2 for x in values)
    return (ss / (len(values) - 1)) ** 0.5


# \u2500\u2500\u2500 Mulberry32 PRNG (matching TS semantics) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#
# The TS PRNG uses JavaScript's bitwise semantics:
#   - `| 0` truncates to int32
#   - `>>> 0` reinterprets as uint32
#   - `Math.imul(a, b)` is 32-bit integer multiply
# We implement these by masking with 0xFFFFFFFF.

_UINT32_MASK = 0xFFFFFFFF


def _to_uint32(x: int) -> int:
    return x & _UINT32_MASK


def _to_int32(x: int) -> int:
    """JS `| 0` semantics: truncate to int32, sign-extending."""
    x = x & _UINT32_MASK
    if x >= 0x80000000:
        x -= 0x100000000
    return x


def _imul(a: int, b: int) -> int:
    """JS Math.imul: 32-bit integer multiply, signed int32 result."""
    # Multiply low 32 bits of each, keep low 32, sign-extend
    a32 = a & _UINT32_MASK
    b32 = b & _UINT32_MASK
    prod = (a32 * b32) & _UINT32_MASK
    if prod >= 0x80000000:
        prod -= 0x100000000
    return prod


def mulberry32(seed: int) -> Callable[[], float]:
    """Mulberry32 PRNG. Same output sequence as the TS implementation."""
    t = _to_uint32(seed)

    def _next() -> float:
        nonlocal t
        t = _to_int32(t + 0x6D2B79F5)  # (t + ...) | 0  in TS
        # r = Math.imul(t ^ (t >>> 15), 1 | t)
        x = _to_uint32(t) ^ (_to_uint32(t) >> 15)
        r = _imul(x, 1 | _to_uint32(t))
        # r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r
        x2 = _to_uint32(r) ^ (_to_uint32(r) >> 7)
        r = _to_int32(_imul(x2, 61 | _to_uint32(r)) + r) ^ r
        # return ((r ^ (r >>> 14)) >>> 0) / 4294967296
        out = (_to_uint32(r) ^ (_to_uint32(r) >> 14)) & _UINT32_MASK
        return out / 4294967296
    return _next


def seed_from_values(values: List[float]) -> int:
    """Deterministic seed from input values; matches TS seedFromValues exactly.

    h = 2166136261; for v in values: rounded = round(v*1e6); h ^= rounded; h = imul(h, 16777619)
    """
    h = 2166136261
    for v in values:
        rounded = round(v * 1e6)
        h = _to_int32(h ^ rounded)
        h = _imul(h, 16777619)
    return _to_uint32(h)


# \u2500\u2500\u2500 Bootstrap CI \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class BootstrapCIResult(TypedDict):
    estimate: float
    lo: float
    hi: float
    n: int
    B: int
    level: float


def bootstrap_ci(
    values: List[float],
    stat: Callable[[List[float]], Optional[float]],
    level: float = 0.95,
    B: int = 1000,
    seed: Optional[int] = None,
) -> Optional[BootstrapCIResult]:
    """Percentile bootstrap CI matching TS bootstrapCI.

    Returns None when n < 8 or when the statistic fails on >50% of resamples.
    """
    n = len(values)
    if n < 8:
        return None
    if seed is None:
        seed = seed_from_values(values)

    point = stat(values)
    if point is None or not _is_finite(point):
        return None

    rng = mulberry32(seed)
    replicates: List[float] = []
    import math
    for _b in range(B):
        resample = [values[math.floor(rng() * n)] for _ in range(n)]
        r = stat(resample)
        replicates.append(r if (r is not None and _is_finite(r)) else float("nan"))

    valid = sorted(x for x in replicates if _is_finite(x))
    if len(valid) < B * 0.5:
        return None
    alpha = 1 - level
    lo = quantile(valid, alpha / 2)
    hi = quantile(valid, 1 - alpha / 2)
    assert lo is not None and hi is not None
    return {"estimate": point, "lo": lo, "hi": hi, "n": n, "B": B, "level": level}


def bootstrap_median_ci(values: List[float], **opts) -> Optional[BootstrapCIResult]:
    return bootstrap_ci(values, lambda s: quantile(s, 0.5), **opts)


def bootstrap_quantile_ci(values: List[float], q: float, **opts) -> Optional[BootstrapCIResult]:
    return bootstrap_ci(values, lambda s: quantile(s, q), **opts)


def reportable_percentiles(n: int) -> Dict[str, bool]:
    """Sample-size policy mirroring TS reportablePercentiles."""
    return {
        "median": n >= 8,
        "p25": n >= 15,
        "p75": n >= 15,
        "p10": n >= 30,
        "p90": n >= 30,
    }


def _is_finite(x: float) -> bool:
    import math
    return isinstance(x, (int, float)) and math.isfinite(x)

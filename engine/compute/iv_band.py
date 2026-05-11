"""
iv_band — canonical cohort dimension for IV state.

Replaces iv_regime as the partition key going forward. Reads its edge
definition from shared/iv_band_spec.json so the TypeScript and Python
sides cannot drift apart.

Forward-only doctrine: NULL iv_percentile -> NULL iv_band. Never impute,
never backfill from a vendor reconstruction. Backfill is only allowed
when re-deriving from an iv_percentile that was already persisted at
fetch time (a pure recomputation, not new information).

Usage:
    from iv_band import compute_iv_band, label_for_band, BAND_SPEC
    band = compute_iv_band(iv_pct)            # int 0..4 or None
    label = label_for_band(band)              # "EXTREME_CHEAP" / ... / None
    legacy = legacy_iv_regime_for_band(band)  # backward-compat label
"""
from __future__ import annotations
import json
import os
from typing import Optional

# v2 path: iv_band_spec.json lives at engine/shared/iv_band_spec.json,
# i.e., one directory up from this file (engine/compute/) then into shared/.
_SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shared",
    "iv_band_spec.json",
)

with open(_SPEC_PATH, "r") as _f:
    BAND_SPEC = json.load(_f)

_EDGES = BAND_SPEC["edges_lo_inclusive_hi_exclusive"]
BAND_VERSION = BAND_SPEC["version"]


def compute_iv_band(iv_percentile: Optional[float]) -> Optional[int]:
    """Return band index 0..4 for an iv_percentile in [0, 100], or None.

    NULL in -> NULL out. Out-of-range values clamp to the nearest edge
    (defensive: shouldn't happen, but if iv_percentile=100.5 sneaks in
    we'd rather classify it as EXTREME_RICH than crash).
    """
    if iv_percentile is None:
        return None
    try:
        v = float(iv_percentile)
    except (TypeError, ValueError):
        return None
    # NaN check
    if v != v:
        return None
    # Clamp defensively
    if v < 0:
        v = 0.0
    if v > 100:
        v = 100.0
    for edge in _EDGES:
        if edge["lo"] <= v < edge["hi"]:
            return int(edge["band"])
    # Should be unreachable given clamping + the [0, 100.01) coverage,
    # but if it happens fall through to the top band.
    return int(_EDGES[-1]["band"])


def label_for_band(band: Optional[int]) -> Optional[str]:
    """Return the canonical display label for a band index, or None."""
    if band is None:
        return None
    for edge in _EDGES:
        if edge["band"] == band:
            return edge["label"]
    return None


def legacy_iv_regime_for_band(band: Optional[int]) -> Optional[str]:
    """Map band -> legacy 3-state iv_regime label for backward compat.

    bands 0-1 -> "CHEAP_VOL", band 2 -> "NEUTRAL_VOL", bands 3-4 -> "EXPENSIVE_VOL".
    """
    if band is None:
        return None
    if band <= 1:
        return "CHEAP_VOL"
    if band == 2:
        return "NEUTRAL_VOL"
    return "EXPENSIVE_VOL"


def band_edges() -> list[dict]:
    """Return a copy of the edges (read-only callers only)."""
    return [dict(e) for e in _EDGES]

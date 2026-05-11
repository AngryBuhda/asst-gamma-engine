"""Derived signals computed from fetched market data.

Modules:
- banding: gamma/banding logic, regime classification
- stochastics: stochastic state machine (state_key, buckets, evidence)
- iv_band: 5-state IV regime derivation (EXTREME_CHEAP \u2026 EXTREME_RICH)
- legacy_suggestions: v1.0-v1.3 suggestion stream (CSP/LEAP/PMCC); preserved
  for parallel-track comparison against the v1.5+ engine recommendations.
"""

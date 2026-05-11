"""Pure helpers for parsing SteadyAPI option-chain responses.

Lifted out of fetch_data.py so they can be tested in isolation. Every helper
here takes raw vendor data (or already-typed values) and returns either a
validated dict or None on parse failure. No network, no DB, no side effects.

Test surface (server/tests/test_chain_parser.py):
  - classify_tier: DTE → tier mapping with explicit boundaries
  - parse_contract: vendor row → typed dict, including the comma-OI fix
  - serialize_chain_snapshot: focus filter + tier counts + format version

The comma-OI bug (see fetch_data.py history): SteadyAPI returns openInterest
as a comma-formatted string for high-OI contracts (e.g. "1,340"). int("1,340")
raises ValueError; the original LEAP parser silently caught it and dropped
every high-OI strike, including the user's $10C anchor, for months. The
golden test below regression-armors that fix.
"""
from __future__ import annotations
from typing import Optional


def classify_tier(dte: int) -> Optional[str]:
    """Map DTE to one of {short, mid, leap}, or None if outside the kept window.

    Boundaries (inclusive on the lower end, inclusive on the upper end):
      short:  14 ≤ dte ≤ 65   — CSP candidates + PMCC short legs
      mid:    65 < dte ≤ 365  — diagonal anchors, calendars, mid-term hedges
      leap:   365 < dte ≤ 800 — true LEAPs, PMCC long legs, LEAP_CORE
      None:   everything else (0–13 too noisy, 800+ too speculative)
    """
    if 14 <= dte <= 65:
        return "short"
    if 65 < dte <= 365:
        return "mid"
    if 365 < dte <= 800:
        return "leap"
    return None


def parse_contract(raw: dict, dte: int, exp_str: str, tier: str) -> Optional[dict]:
    """Parse one SteadyAPI contract row into our internal dict.

    Returns None on:
      - Missing/zero midpoint (uninvestable)
      - Missing strikePrice (unparseable)
      - Any ValueError/KeyError/TypeError during conversion (silently dropped
        — caller logs raw vs parsed counts so silent-drop bugs are visible)

    Critical: openInterest may be a comma-formatted string ("1,340"). Always
    strip commas. The bid/ask/midpoint fields are also strings.
    """
    try:
        mid_str = raw.get("midpoint", "0")
        mid = float(mid_str) if mid_str else 0.0
        if mid <= 0:
            return None

        delta_str = raw.get("delta", "0")
        iv_str = str(raw.get("volatility", "0")).replace("%", "").replace(",", "")

        return {
            "strike": float(raw["strikePrice"]),
            "dte": dte,
            "mid": mid,
            "expiry": exp_str,
            "delta": float(delta_str) if delta_str else None,
            "vega": float(raw.get("vega", 0) or 0),
            "theta": float(raw.get("theta", 0) or 0),
            "iv": float(iv_str) / 100.0 if iv_str else None,
            "oi": int(str(raw.get("openInterest", 0) or 0).replace(",", "")),
            "bid": float(raw.get("bidPrice", 0) or 0),
            "ask": float(raw.get("askPrice", 0) or 0),
            "tier": tier,
            "is_leap": tier == "leap",
        }
    except (ValueError, KeyError, TypeError):
        return None


def serialize_chain_snapshot_payload(puts_all: list, calls_all: list, as_of_utc: str) -> Optional[dict]:
    """Build the focused snapshot dict (caller serializes to JSON).

    Lifts the focus-filter logic from fetch_data.serialize_chain_snapshot()
    so it can be unit-tested without touching the timestamp generation.
    Returns None if both input lists are empty.

    Filter rules (format_version 1.2):
      - Puts: keep DTE 14-70 (CSP band + 5d buffer)
      - Calls: keep all tiers (short/mid/leap) within 14-800 DTE
      - Legacy fallback: contracts without `tier` use is_leap + DTE band
    """
    if not puts_all and not calls_all:
        return None

    puts_focused = [p for p in puts_all if 14 <= (p.get("dte") or 0) <= 70]

    def _call_in_window(c: dict) -> bool:
        dte_v = c.get("dte") or 0
        if not (14 <= dte_v <= 800):
            return False
        return c.get("tier") is not None or c.get("is_leap") or 14 <= dte_v <= 70

    calls_focused = [c for c in calls_all if _call_in_window(c)]

    n_short = sum(1 for c in calls_focused if c.get("tier") == "short")
    n_mid = sum(1 for c in calls_focused if c.get("tier") == "mid")
    n_leap = sum(1 for c in calls_focused if c.get("tier") == "leap")
    distinct_expiries = sorted({c.get("expiry") for c in calls_focused if c.get("expiry")})

    return {
        "puts": puts_focused,
        "calls": calls_focused,
        "as_of_utc": as_of_utc,
        "counts": {
            "puts_total": len(puts_all),
            "calls_total": len(calls_all),
            "puts_persisted": len(puts_focused),
            "calls_persisted": len(calls_focused),
            "calls_leap": n_leap,
            "calls_short_dte": n_short,
            "calls_by_tier": {"short": n_short, "mid": n_mid, "leap": n_leap},
            "distinct_call_expiries": len(distinct_expiries),
        },
        "format_version": "1.2",
    }

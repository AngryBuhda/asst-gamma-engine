"""Regime transition alert helper.

Why this exists (2026-05-09):
  Across 120 PM sessions, every observation has been iv_regime=CHEAP_VOL.
  Every claim about "doctrine works in CHEAP_VOL" is therefore unfalsifiable
  until we observe a non-CHEAP_VOL session. The first such session is the
  highest-information observation we'll have for the rest of the cohort year:
  it tests every doctrine claim conditional on the volatility regime.

  We do NOT want it to pass silently. This helper produces a notification
  prefix that the cron-task notification path can prepend on regime
  transitions, so the operator (and the project log) sees it loudly.

Usage from the AM/MID/PM cron:
  prefix = check_regime_transition(date='2026-05-12', session='AM')
  if prefix:
      notification_body = f"{prefix}\\n\\n{notification_body}"

Detection logic:
  Look at the latest persisted iv_regime + gamma_regime pair, compare against
  the prior session's pair. Emit a prefix only when the regime actually
  changed. Special-case: the first non-CHEAP_VOL session ever (across
  full history) gets a louder banner, regardless of what came before.

This is read-only relative to data.db. It never blocks fetches. It does not
write any state — the cron's notification path is the persistence layer.
"""
from __future__ import annotations
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from engine.compute.iv_band import label_for_band  # noqa: E402

# v2: env-var override with v1 fallback.
DB_PATH = os.environ.get(
    "ASST_DB_PATH",
    "/home/user/workspace/asst-gamma-dashboard/data.db",
)


def _connect():
    return sqlite3.connect(DB_PATH)


def get_latest_two_iv_regimes() -> list[tuple[str, str, str | None, str | None, int | None]]:
    """Return [(date, session, iv_regime, gamma_regime, iv_band), ...] for
    the most recent 2 sessions. Newest first.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT date, session, iv_regime, regime, iv_band
            FROM daily_runs
            ORDER BY date DESC, session DESC
            LIMIT 2
        """)
        return cur.fetchall()
    finally:
        conn.close()


def count_historical_iv_regime(iv_regime: str) -> int:
    """How many rows in history have this iv_regime label?"""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_runs WHERE iv_regime = ?", (iv_regime,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def count_historical_iv_band(band: int) -> int:
    """How many rows in history have this iv_band index?"""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_runs WHERE iv_band = ?", (band,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def count_total_rows() -> int:
    """How many daily_runs rows total?"""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_runs")
        return cur.fetchone()[0]
    finally:
        conn.close()


def check_regime_transition() -> Optional[dict]:
    """Inspect the latest two sessions; return a dict describing the alert
    (or None if no transition / not noteworthy).

    Return shape:
      {
        "kind": "first_rich_vol" | "iv_transition" | "gamma_family_flip",
        "severity": "high" | "medium",
        "prefix": "...",        # ready-to-prepend banner string
        "details": {...},       # for logs
      }
    """
    rows = get_latest_two_iv_regimes()
    if len(rows) < 1:
        return None

    latest = rows[0]
    # Defensive unpacking: legacy callers / mocks may return 4-tuples without
    # iv_band. Treat missing band as None (no band-aware alerts fire).
    if len(latest) >= 5:
        latest_date, latest_session, latest_iv, latest_regime, latest_band = latest[:5]
    else:
        latest_date, latest_session, latest_iv, latest_regime = latest[:4]
        latest_band = None
    latest_band_label = label_for_band(latest_band) if latest_band is not None else None

    # ── Highest-priority signal: first observation ever in this iv_band ────
    # The band partition resolves states the legacy 3-state iv_regime conflated.
    # First-ever observation of a band is the highest-information row for that
    # band: every cohort statistic conditional on it gets its first data point.
    if latest_band is not None:
        prior_with_same_band = count_historical_iv_band(latest_band) - 1  # subtract self
        if prior_with_same_band <= 0:
            total = count_total_rows()
            return {
                "kind": f"first_band_{latest_band}",
                "severity": "high",
                "prefix": (
                    f"[!!] FIRST iv_band={latest_band} ({latest_band_label}) SESSION EVER  ·  "
                    f"this band had zero observations across {total} prior rows. "
                    f"Highest-information observation — cohort statistics conditional on "
                    f"band {latest_band} are now seeded for the first time."
                ),
                "details": {
                    "date": latest_date,
                    "session": latest_session,
                    "iv_band": latest_band,
                    "iv_band_label": latest_band_label,
                    "iv_regime_legacy": latest_iv,
                    "gamma_regime": latest_regime,
                    "total_rows": total,
                },
            }

    # ── Second-priority: iv_band crossing vs prior session ────────────
    # distance>=2 (skipped a band) = high; distance=1 (adjacent) = medium.
    if len(rows) >= 2:
        prior = rows[1]
        prior_iv = prior[2] if len(prior) > 2 else None
        prior_band = prior[4] if len(prior) > 4 else None
        if (latest_band is not None and prior_band is not None
                and latest_band != prior_band):
            distance = abs(int(latest_band) - int(prior_band))
            prior_band_label = label_for_band(prior_band)
            severity = "high" if distance >= 2 else "medium"
            marker = "!!" if severity == "high" else "!"
            return {
                "kind": "iv_band_crossing",
                "severity": severity,
                "prefix": (
                    f"[{marker}] iv_band crossing  ·  "
                    f"{prior_band} ({prior_band_label}) -> {latest_band} ({latest_band_label})  "
                    f"distance={distance}  "
                    f"({prior[0]} {prior[1]} -> {latest_date} {latest_session})"
                ),
                "details": {
                    "from_band": prior_band,
                    "from_band_label": prior_band_label,
                    "to_band": latest_band,
                    "to_band_label": latest_band_label,
                    "distance": distance,
                    "from_date": prior[0],
                    "from_session": prior[1],
                    "to_date": latest_date,
                    "to_session": latest_session,
                },
            }

        # ── Third-priority (legacy): iv_regime transition ──────────────
        if prior_iv and latest_iv and prior_iv != latest_iv:
            return {
                "kind": "iv_transition",
                "severity": "medium",
                "prefix": (
                    f"[!] IV regime transition (legacy)  ·  {prior_iv} -> {latest_iv}  "
                    f"({prior[0]} {prior[1]} -> {latest_date} {latest_session})"
                ),
                "details": {
                    "from": prior_iv,
                    "to": latest_iv,
                    "from_date": prior[0],
                    "from_session": prior[1],
                    "to_date": latest_date,
                    "to_session": latest_session,
                },
            }

        # ── Fourth-priority: gamma family flip (long_* <-> short_*) ───────────
        prior_regime = prior[3] or ""
        latest_regime_str = latest_regime or ""
        prior_family = (
            "long" if prior_regime.startswith("long")
            else "short" if prior_regime.startswith("short")
            else "neutral"
        )
        latest_family = (
            "long" if latest_regime_str.startswith("long")
            else "short" if latest_regime_str.startswith("short")
            else "neutral"
        )
        if prior_family != latest_family and prior_family != "neutral" and latest_family != "neutral":
            return {
                "kind": "gamma_family_flip",
                "severity": "medium",
                "prefix": (
                    f"[*] Gamma family flip  ·  {prior_family} -> {latest_family}  "
                    f"({prior_regime} -> {latest_regime_str})"
                ),
                "details": {
                    "from_family": prior_family,
                    "to_family": latest_family,
                    "from_regime": prior_regime,
                    "to_regime": latest_regime_str,
                },
            }

    return None


# ── CLI for cron use ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    result = check_regime_transition()
    if result is None:
        # No transition — exit silently with code 0
        sys.exit(0)
    # Print JSON to stdout so the cron can parse it
    print(json.dumps(result, indent=2))
    sys.exit(0)

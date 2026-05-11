#!/usr/bin/env python3
"""
asst-gamma-engine — Daily Data Fetcher (v2)

Fetches GEX from FlashAlpha, market data from SteadyAPI + Tiingo + BGeometrics,
computes derived signals, persists the run directly to data.db via
`engine.persist`, and runs the Python selector (P1.8).

Usage:
    python -m engine.fetch [--session AM|MID|PM] [--persist-only]

Flags:
    --session {AM,MID,PM}  Which session this run belongs to (required for crons)
    --persist-only         Skip selector evaluation; persist run row only (for tests)

Migration notes (P1.2 — 2026-05-11):
- The legacy `--api-url` flag and POST-to-Express fallback are removed.
  direct_persist is now the only persistence path.
- API keys read from environment variables. Fallback to legacy hardcoded
  values during v1→v2 transition; P1.19 rotates and removes fallbacks.
- The legacy `/api/selector/evaluate` and `/api/selector/log` HTTP calls
  remain as a TEMPORARY shim. P1.8 ports the TypeScript selector engine
  to Python and wires it in directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests

# Local: canonical iv_band derivation (shared with TS via iv_band_spec.json).
from iv_band import compute_iv_band

# ── Configuration ─────────────────────────────────────────────────────────────

# ── API keys ─────────────────────────────────────────────────────────────────
# Read from environment with v1 hardcoded fallbacks during transition.
# After P1.19, remove the fallbacks so a missing env var fails fast.

_LEGACY_FLASHALPHA_KEY = "i0Bdxt7SPzGkpiNacBecC51wODlec3zqkD7BUmcB"  # noqa: S105
_LEGACY_STEADY_KEY = "2092|eS5j7ePdb6Z1wlkGgGTnnlIiO3IteBQOW06Dj1SV"  # noqa: S105
_LEGACY_TIINGO_KEY = "e4e2c2f069952c774c9716909f6f4dba000707e7"  # noqa: S105

FLASHALPHA_API_BASE = "https://lab.flashalpha.com/v1/exposure/gex/ASST"
FLASHALPHA_API_KEY = os.environ.get("FLASHALPHA_API_KEY", _LEGACY_FLASHALPHA_KEY)
STEADYAPI_KEY = os.environ.get("STEADYAPI_KEY", _LEGACY_STEADY_KEY)
STEADYAPI_BASE = "https://api.steadyapi.com"
TIINGO_TOKEN = os.environ.get("TIINGO_API_KEY", _LEGACY_TIINGO_KEY)
DEFAULT_API_BASE = "http://localhost:5000"  # retained only for the temp selector shim; removed in P1.8
SYMBOL = "ASST"
GEX_HISTORY_PATH = Path(__file__).parent / "gex_history.json"
MIN_HISTORY_FOR_PERCENTILE = 30

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_gex_history() -> list:
    """
    Load saved historical data for percentile and flip smoothing.
    Returns List[Dict] with keys: date, net_gex, flip.
    Handles backward compat with old format (plain float array).
    """
    if GEX_HISTORY_PATH.exists():
        try:
            data = json.loads(GEX_HISTORY_PATH.read_text())
            if not data:
                return []
            # Backward compat: old format was [float, float, ...]
            if isinstance(data[0], (int, float)):
                return [{"date": None, "net_gex": float(x), "flip": None} for x in data if x is not None]
            return data
        except Exception:
            pass
    return []


def save_gex_history(history: list) -> None:
    """Persist updated gex history, keeping last 365 entries."""
    GEX_HISTORY_PATH.write_text(json.dumps(history[-365:]))


def compute_gex_percentile(net_gex: float, history: list) -> float:
    """
    Return the percentile (0–100) of net_gex vs. history.
    Returns 25.0 as default if history has fewer than MIN_HISTORY_FOR_PERCENTILE entries.
    """
    gex_vals = []
    for entry in history:
        if isinstance(entry, dict):
            v = entry.get("net_gex")
        else:
            v = entry
        if v is not None:
            gex_vals.append(float(v))
    if len(gex_vals) < MIN_HISTORY_FOR_PERCENTILE:
        return 25.0
    below = sum(1 for h in gex_vals if h <= net_gex)
    return round(100.0 * below / len(gex_vals), 2)


def compute_vanna_percentile(net_vanna: float, history: list) -> float:
    """Return the percentile (0-100) of net_vanna vs. history."""
    vanna_vals = []
    for entry in history:
        v = entry.get("net_vanna")
        if v is not None:
            try:
                vanna_vals.append(float(v))
            except (TypeError, ValueError):
                continue
    if len(vanna_vals) < 5:
        return 50.0
    below = sum(1 for h in vanna_vals if h <= net_vanna)
    return round(below / len(vanna_vals) * 100, 2)


def compute_smoothed_flip(
    raw_flip: float,
    history: list,
    window: int = 3,
    current_regime: Optional[str] = None,
    prior_regime: Optional[str] = None,
) -> Tuple[float, bool]:
    """
    3-period EMA of the gamma flip with regime-rotation reset.

    When `current_regime` differs from `prior_regime`, the EMA is reset:
    the new session's smoothed flip equals the raw flip directly. Subsequent
    sessions resume normal smoothing from this new anchor.

    Rationale: a regime rotation (long↔short or weak↔strong) is a structural
    change, not noise. The 3-EMA dampens noise within a regime but lags reality
    on rotations, producing a smoothed flip that reflects neither the old nor the
    new dealer book. Resetting honors the rotation while preserving smoothing
    within a stable regime.

    Falls back to raw_flip when fewer than 2 historical data points exist.

    Returns:
        (smoothed_flip, was_reset) — the second element is True iff a regime
        reset triggered. Caller can surface this in logs/snapshot warnings.
    """
    # Detect regime rotation. Both regimes must be known and different.
    regime_rotated = (
        current_regime is not None
        and prior_regime is not None
        and current_regime != prior_regime
    )
    if regime_rotated:
        return round(raw_flip, 4), True

    flips = []
    for entry in history:
        if isinstance(entry, dict) and entry.get("flip") is not None:
            flips.append(float(entry["flip"]))
    flips.append(raw_flip)
    if len(flips) < 2:
        return round(raw_flip, 4), False
    # EMA with span=window
    alpha = 2.0 / (window + 1)
    ema = flips[0]
    for val in flips[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return round(ema, 4), False


def compute_leap_entry_percentile(spot: float, leap_band_low: float, leap_band_high: float, history: list) -> Optional[float]:
    """
    Compute how attractive the current LEAP entry is as a percentile.
    Scale: 100% = most attractive (deepest in band), 0% = least attractive (above band).
    This is intuitive: higher = better entry.
    Returns None if insufficient data.
    """
    if leap_band_low is None or leap_band_high is None or leap_band_high <= leap_band_low:
        return None
    
    band_width = leap_band_high - leap_band_low
    if band_width <= 0:
        return None
    
    # Raw position: 0 = at band_low, 1 = at band_high
    if spot <= leap_band_low:
        # Below band — most attractive, cap at 100%
        attractiveness = 100.0
    elif spot >= leap_band_high:
        # Above band — not in convexity zone, 0%
        attractiveness = 0.0
    else:
        # Invert: closer to band_low = higher score
        attractiveness = (1.0 - (spot - leap_band_low) / band_width) * 100.0
    
    return round(attractiveness, 1)


def compute_flip_migration(raw_flip: float, history: list) -> dict:
    """
    Compute flip migration vs yesterday and vs 5-day ago.
    Returns {"vs_yesterday": float|None, "vs_5d": float|None}
    """
    flips = []
    for entry in history:
        if isinstance(entry, dict) and entry.get("flip") is not None:
            flips.append(float(entry["flip"]))
    
    vs_yesterday = round(raw_flip - flips[-1], 4) if len(flips) >= 1 else None
    vs_5d = round(raw_flip - flips[-5], 4) if len(flips) >= 5 else None
    return {"vs_yesterday": vs_yesterday, "vs_5d": vs_5d}


# ── Risk Management Computations ──────────────────────────────────────────────


def compute_btc_weekly_rsi(period=14):
    """
    Compute 14-period RSI on BTC weekly closes using Tiingo crypto API.
    Returns float (0-100) or None on failure.
    """
    try:
        resp = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            params={
                "tickers": "btcusd",
                "startDate": (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d"),
                "resampleFreq": "1day",
                "token": TIINGO_TOKEN,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[fetch_data] Tiingo BTC crypto: HTTP {resp.status_code}", file=sys.stderr)
            return None
        data = resp.json()
        if not data or not data[0].get("priceData"):
            return None
        bars = data[0]["priceData"]
        # Sample every 7th close for weekly
        daily_closes = [b["close"] for b in bars]
        weekly_closes = daily_closes[::7]
        if len(weekly_closes) < period + 1:
            print(f"[fetch_data] BTC weekly RSI: insufficient data ({len(weekly_closes)} weekly bars)", file=sys.stderr)
            return None
        gains, losses = [], []
        for i in range(1, len(weekly_closes)):
            diff = weekly_closes[i] - weekly_closes[i - 1]
            gains.append(diff if diff > 0 else 0)
            losses.append(-diff if diff < 0 else 0)
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        print(f"[fetch_data] BTC weekly RSI (Tiingo): {rsi:.1f}")
        return round(rsi, 1)
    except Exception as e:
        print(f"[fetch_data] BTC weekly RSI computation failed (non-fatal): {e}", file=sys.stderr)
        return None


def compute_asst_drawdown_90d(current_spot):
    """
    Compute ASST drawdown from 90-day high using Tiingo nightly cache.
    Falls back to Tiingo API if cache missing.
    Returns (drawdown, high_90d) or (None, None) on failure.
    """
    try:
        # Primary: Tiingo nightly cache (computed by tiingo_eod.py)
        cache_path = Path(__file__).parent.parent / "tiingo_cache" / "asst_stats.json"
        if cache_path.exists():
            import json as _json
            with open(cache_path) as f:
                stats = _json.load(f)
            high_90d = stats.get("high_90d")
            dd = stats.get("drawdown_90d")
            if high_90d is not None and dd is not None:
                print(f"[fetch_data] ASST 90d drawdown (Tiingo cache): {dd:.4f}, high={high_90d:.2f}")
                return round(float(dd), 4), round(float(high_90d), 2)

        # Fallback: Tiingo API direct
        start = (datetime.now() - timedelta(days=95)).strftime("%Y-%m-%d")
        resp = requests.get(
            f"https://api.tiingo.com/tiingo/daily/ASST/prices",
            params={"startDate": start, "token": TIINGO_TOKEN},
            timeout=15,
        )
        if resp.status_code == 200:
            bars = resp.json()
            if len(bars) >= 5:
                high_90d = max(float(b["adjHigh"]) for b in bars)
                drawdown = (current_spot - high_90d) / high_90d
                print(f"[fetch_data] ASST 90d drawdown (Tiingo API): {drawdown:.4f}, high={high_90d:.2f}")
                return round(drawdown, 4), round(high_90d, 2)
        return None, None
    except Exception as e:
        print(f"[fetch_data] ASST 90d drawdown computation failed (non-fatal): {e}", file=sys.stderr)
        return None, None




# -- IV Metrics (from yfinance options chain + stored/RV history) -------------------


def compute_iv_metrics(spot: float, chain_data: Optional[dict] = None) -> Optional[dict]:
    """
    Compute current 30-day ATM IV, IV Rank, and IV Percentile for ASST.

    Data source: SteadyAPI (options chain avg_iv + iv_rank_1y) + DB history.
    Method:
      - Current IV: averageVolatility from SteadyAPI chain response
      - IV Rank: impliedVolatilityRank1y from SteadyAPI (exchange-computed 1yr rank)
      - IV Percentile: computed from stored DB history when 60+ readings exist
      - iv_rank_method: "IV" when using SteadyAPI rank, "RV" as fallback from Tiingo
    """
    try:
        import sqlite3 as _sq

        current_iv = None
        iv_rank = None
        iv_percentile = None
        method = "IV"

        # 1. Current IV + IV rank from SteadyAPI chain data (already fetched)
        if chain_data:
            avg_iv = chain_data.get("avg_iv")
            if avg_iv is not None:
                current_iv = float(avg_iv)  # Already in %
            iv_rank_1y = chain_data.get("iv_rank_1y")
            if iv_rank_1y is not None:
                iv_rank = float(iv_rank_1y)
            hv30d = chain_data.get("hv30d")

        # 2. If SteadyAPI didn't provide IV, try Tiingo RV as fallback
        if current_iv is None:
            try:
                cache_path = Path(__file__).parent.parent / "tiingo_cache" / "asst_stats.json"
                if cache_path.exists():
                    import json as _json
                    with open(cache_path) as f:
                        stats = _json.load(f)
                    rv = stats.get("rv_current_30d")
                    if rv is not None:
                        current_iv = float(rv)
                        method = "RV"
                        print(f"[fetch_data] IV: using Tiingo RV as current_iv fallback: {current_iv:.1f}%")
            except Exception:
                pass

        if current_iv is None:
            print("[fetch_data] IV: no IV source available", file=sys.stderr)
            return None

        # 3. IV Percentile from stored DB history
        iv_history = []
        try:
            db_path = os.path.join(os.path.dirname(__file__), "..", "data.db")
            if os.path.exists(db_path):
                conn = _sq.connect(db_path)
                rows = conn.execute("""
                    SELECT date, current_iv FROM daily_runs
                    WHERE current_iv IS NOT NULL AND current_iv > 0
                    GROUP BY date
                    ORDER BY date DESC
                    LIMIT 252
                """).fetchall()
                conn.close()
                iv_history = [r[1] for r in rows if r[1] is not None and r[1] > 0]
        except Exception:
            pass

        if len(iv_history) >= 20:
            iv_percentile = sum(1 for v in iv_history if v < current_iv) / len(iv_history) * 100
            # If SteadyAPI didn't give us iv_rank, compute from history too
            if iv_rank is None:
                iv_min = min(iv_history)
                iv_max = max(iv_history)
                iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100 if iv_max > iv_min else 50.0
                iv_rank = max(0.0, min(100.0, iv_rank))
                method = "IV" if len(iv_history) >= 60 else "RV"

        rank_str = f"{iv_rank:.1f}" if iv_rank is not None else "?"
        pct_str = f"{iv_percentile:.1f}" if iv_percentile is not None else "?"
        print(f"[fetch_data] IV (method={method}): current={current_iv:.1f}%  rank={rank_str}  pct={pct_str}  ({len(iv_history)} daily readings)")

        return {
            "current_iv": round(current_iv, 1),
            "iv_rank": round(iv_rank, 1) if iv_rank is not None else None,
            "iv_percentile": round(iv_percentile, 1) if iv_percentile is not None else None,
            "iv_rank_method": method,
        }

    except Exception as e:
        print(f"[fetch_data] IV metrics failed (non-fatal): {e}", file=sys.stderr)
        return None


_LEGACY_BGEO_KEY = "gLkYn1qRsa"  # noqa: S105
BGEOMETRICS_TOKEN = os.environ.get("BGEOMETRICS_API_KEY", _LEGACY_BGEO_KEY)
BGEOMETRICS_API = "https://api.bitcoin-data.com"


def _bg_get(endpoint: str, timeout: int = 10, retries: int = 2, retry_wait: float = 5.0):
    """Authenticated GET to BGeometrics API.

    Rate-limit aware: 429 responses trigger a brief sleep + retry instead of
    silently returning None (which had been silently nulling out cycle metrics
    in the DB on cron-heavy days). Critical messages go to stdout so they're
    visible in the standard fetch logs.

    Returns parsed JSON or None.
    """
    url = f"{BGEOMETRICS_API}{endpoint}"
    sep = "&" if "?" in endpoint else "?"
    url = f"{url}{sep}token={BGEOMETRICS_TOKEN}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited — print to stdout (captured in cron logs) so the
                # user can see if BGeometrics quota is the cause of NULL metrics.
                if attempt < retries:
                    print(f"[fetch_data] BGeometrics 429 on {endpoint} (attempt {attempt+1}/{retries+1}) — sleeping {retry_wait}s", flush=True)
                    time.sleep(retry_wait)
                    continue
                print(f"[fetch_data] BGeometrics RATE LIMITED on {endpoint} after {retries+1} attempts — returning None", flush=True)
            else:
                print(f"[fetch_data] BGeometrics {endpoint} HTTP {resp.status_code}", flush=True)
        except Exception as e:
            print(f"[fetch_data] BGeometrics {endpoint} failed: {e}", flush=True)
        return None
    return None


def _bg_latest(data):
    """Extract latest entry from BGeometrics response (single obj or list)."""
    if isinstance(data, list) and len(data) > 0:
        return data[-1]
    if isinstance(data, dict) and "error" not in data:
        return data
    return None


def fetch_btc_onchain():
    """
    Fetch BTC on-chain data from BGeometrics authenticated API.
    Source: https://api.bitcoin-data.com (token-authenticated)
    Advanced plan: 200 req/hr, 400 req/day.
    Returns: dict with btc_mvrv, btc_realized_price, or None
    """
    result = {}

    # MVRV
    data = _bg_get("/v1/mvrv/1")
    entry = _bg_latest(data)
    if entry:
        result["btc_mvrv"] = float(entry.get("mvrv", 0))
        print(f"[fetch_data] BTC MVRV: {result['btc_mvrv']:.4f} (date: {entry.get('d', '?')})")

    # Realized Price (full dataset — /1 sometimes returns empty, fetch last few entries)
    data = _bg_get("/v1/realized-price")
    if isinstance(data, list) and len(data) > 0:
        latest = data[-1]
        rp = latest.get("realizedPrice")
        if rp is not None:
            result["btc_realized_price"] = float(rp)
            print(f"[fetch_data] BTC Realized Price: ${result['btc_realized_price']:,.2f} (date: {latest.get('theDay') or latest.get('d', '?')})")
    elif data is None:
        pass  # Already logged by _bg_get

    return result if result else None


def fetch_btc_derivatives():
    """
    Fetch BTC derivatives data from BGeometrics for secondary confirmation.
    Endpoints: taker buy/sell ratio, funding rate, daily liquidations.
    Returns: dict with raw values + computed btc_gex_secondary_confirm.
    """
    result = {}

    # 1. Taker Buy/Sell Ratio (1h resolution, latest)
    data = _bg_get("/v1/taker-vol-1h/1")
    entry = _bg_latest(data)
    if entry:
        result["btc_taker_buy_sell_ratio"] = float(entry.get("taker_buy_sell_ratio", 1.0))
        result["btc_taker_buy_vol"] = float(entry.get("taker_buy_vol", 0))
        result["btc_taker_sell_vol"] = float(entry.get("taker_sell_vol", 0))
        print(f"[fetch_data] BTC Taker B/S ratio: {result['btc_taker_buy_sell_ratio']:.4f}")

    # 2. Funding Rate
    data = _bg_get("/v1/funding-rate/1")
    entry = _bg_latest(data)
    if entry:
        result["btc_funding_rate"] = float(entry.get("fundingRate", 0))
        print(f"[fetch_data] BTC Funding Rate: {result['btc_funding_rate']:.8f}")

    # 3. Daily Liquidations
    data = _bg_get("/v1/btc-liquidations-1d/1")
    entry = _bg_latest(data)
    if entry:
        total = float(entry.get("totalLiquidationsUsd", 0))
        longs = float(entry.get("longLiquidationsUsd", 0))
        shorts = float(entry.get("shortLiquidationsUsd", 0))
        result["btc_liq_total_usd"] = total
        result["btc_liq_long_usd"] = longs
        result["btc_liq_short_usd"] = shorts
        result["btc_liq_long_pct"] = (longs / total * 100) if total > 0 else 50.0
        print(f"[fetch_data] BTC Liquidations: ${total/1e6:.1f}M (longs {result['btc_liq_long_pct']:.0f}%)")

    # 4. Fear & Greed (cheap call, useful context)
    data = _bg_get("/v1/fear-greed/1")
    entry = _bg_latest(data)
    if entry:
        result["btc_fear_greed"] = int(entry.get("fearGreed", 50))
        print(f"[fetch_data] BTC Fear & Greed: {result['btc_fear_greed']}")

    # 5. Cycle-top valuation indicators (added 2026-05-03 for Selector v4).
    # These distinguish "market is at peak euphoria" (top triggers) from
    # "position is in pain" (stress triggers). Each is a single scalar from
    # BGeometrics' authenticated daily endpoints.
    cycle_endpoints = [
        ("/v1/mvrv-zscore/1", "mvrvZscore", "btc_mvrv_zscore", "MVRV Z"),
        ("/v1/puell-multiple/1", "puellMultiple", "btc_puell_multiple", "Puell"),
        ("/v1/nupl/1", "nupl", "btc_nupl", "NUPL"),
        ("/v1/reserve-risk/1", "reserveRisk", "btc_reserve_risk", "ReserveRisk"),
    ]
    for endpoint, src_field, db_field, label in cycle_endpoints:
        data = _bg_get(endpoint)
        entry = _bg_latest(data)
        if entry and entry.get(src_field) is not None:
            try:
                result[db_field] = float(entry[src_field])
                print(f"[fetch_data] BTC {label}: {result[db_field]:.4f}")
            except (TypeError, ValueError):
                pass
    # Pi Cycle is a separate-shape endpoint (returns piSignal binary)
    data = _bg_get("/v1/pi-cycle/1")
    entry = _bg_latest(data)
    if entry and entry.get("piSignal") is not None:
        try:
            result["btc_pi_cycle_signal"] = int(entry["piSignal"])
            print(f"[fetch_data] BTC Pi Cycle signal: {result['btc_pi_cycle_signal']}")
        except (TypeError, ValueError):
            pass

    # Compute btc_gex_secondary_confirm composite
    # Methodology: BTC doesn't have equity-style GEX, so we use futures positioning
    # as a proxy for dealer/market-maker directional pressure:
    #   - Taker B/S < 1.0 = sellers dominate (dealers absorbing buys → short gamma analog)
    #   - Negative funding = shorts paying longs (market bearish / short-heavy)
    #   - Long liquidations > 60% = leveraged longs getting flushed (negative pressure)
    # When 2+ of 3 signals align → "negative" (bullish for LEAP entry, max conviction)
    # When 2+ of 3 are opposite → "positive" (reduce LEAP sizing)
    # Otherwise → "neutral"
    neg_signals = 0
    pos_signals = 0
    total_signals = 0

    taker_ratio = result.get("btc_taker_buy_sell_ratio")
    if taker_ratio is not None:
        total_signals += 1
        if taker_ratio < 0.97:     # Sellers dominating
            neg_signals += 1
        elif taker_ratio > 1.03:   # Buyers dominating
            pos_signals += 1

    funding = result.get("btc_funding_rate")
    if funding is not None:
        total_signals += 1
        if funding < -0.0001:      # Significantly negative funding
            neg_signals += 1
        elif funding > 0.0001:     # Significantly positive funding
            pos_signals += 1

    long_pct = result.get("btc_liq_long_pct")
    if long_pct is not None:
        total_signals += 1
        if long_pct > 65:          # Long-heavy liquidations (bearish flush)
            neg_signals += 1
        elif long_pct < 35:        # Short-heavy liquidations (short squeeze)
            pos_signals += 1

    if total_signals >= 2:
        if neg_signals >= 2:
            result["btc_gex_secondary_confirm"] = "negative"
        elif pos_signals >= 2:
            result["btc_gex_secondary_confirm"] = "positive"
        else:
            result["btc_gex_secondary_confirm"] = "neutral"
    else:
        result["btc_gex_secondary_confirm"] = None  # Insufficient data

    confirm = result.get("btc_gex_secondary_confirm")
    print(f"[fetch_data] BTC GEX secondary confirm: {confirm} (neg={neg_signals} pos={pos_signals} of {total_signals} signals)")

    return result if result else None


# ── SteadyAPI (Options chain with Greeks) ────────────────────────────────────


def serialize_chain_snapshot(chain_data: Optional[dict]) -> Optional[str]:
    """
    Serialize a *focused* option chain snapshot to a JSON string for DB storage.

    The Selector v4 tab needs real chain data to replace the APPROX flag on
    LEAP / PMCC short-leg recommendations. We persist a tier-aware subset:
      - Puts: only the short tier (14-65 DTE) — CSP candidates
      - Calls: ALL tiers (short / mid / leap) — PMCC shorts + diagonal mids +
        LEAP_CORE longs. The expanded mid tier (66–365 DTE) means the BCI
        heatmap can show 3 expiry columns instead of 1.

    Format bumped to v1.2 to signal the expanded coverage. Schema is
    backward-compatible — readers that key on `is_leap` continue to work,
    new readers can branch on `tier` instead.

    Per-contract fields are already trimmed in fetch_steadyapi_chain() (no waste).
    With 12 expirations now passing the filter (vs. 6 before), the snapshot
    grows from ~12KB to ~25–40KB — still well under the run-payload budget.

    Returns: JSON string of {puts, calls, as_of_utc, counts} or None if chain unavailable.
    """
    if not chain_data or not isinstance(chain_data, dict):
        return None
    from chain_parser import serialize_chain_snapshot_payload
    payload = serialize_chain_snapshot_payload(
        chain_data.get("puts") or [],
        chain_data.get("calls") or [],
        datetime.now(timezone.utc).isoformat(),
    )
    if payload is None:
        return None
    return json.dumps(payload)


def fetch_steadyapi_chain(symbol: str = SYMBOL) -> Optional[dict]:
    """
    Fetch options chain with exchange-computed Greeks from SteadyAPI v3.
    Returns: {puts: [...], calls: [...], iv_rank_1y: float, hv30d: float, avg_iv: float}

    Each put/call: {strike, dte, mid, expiry, delta, vega, theta, iv, oi, bid, ask}
    Puts: 28-65 DTE range (CSP candidates)
    Calls: >500 DTE (LEAP candidates)
    """
    headers = {"Authorization": f"Bearer {STEADYAPI_KEY}"}
    puts = []
    calls = []
    iv_rank_1y = None
    hv30d = None
    avg_iv = None

    try:
        # Step 1: Get available expirations
        resp = requests.get(
            f"{STEADYAPI_BASE}/v3/markets/options",
            headers=headers,
            params={"ticker": symbol},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[fetch_data] SteadyAPI expirations: HTTP {resp.status_code}", file=sys.stderr)
            return None

        data = resp.json()
        meta = data.get("meta", {})
        exps = meta.get("expirations", {})
        weekly = exps.get("weekly", [])
        monthly = exps.get("monthly", [])
        all_exps = sorted(set(weekly + monthly))

        # Extract IV metadata from first response body (same for all contracts).
        # SteadyAPI v3 returns body as a dict during market hours, but as an
        # empty list when markets are closed (e.g. Saturday afternoon). Treat
        # any non-dict shape as "no body" and skip the metadata extraction
        # rather than crashing on .get("Call").
        body = data.get("body")
        if not isinstance(body, dict):
            if body:  # only log non-empty unexpected shapes; empty list is normal off-hours
                print(f"[fetch_data] SteadyAPI body is {type(body).__name__} (expected dict) — skipping IV metadata", file=sys.stderr)
            body = {}
        first_call = (body.get("Call") or [None])[0]
        if first_call:
            try:
                iv_rank_str = first_call.get("impliedVolatilityRank1y", "")
                iv_rank_1y = float(iv_rank_str.replace("%", "")) if iv_rank_str else None
            except (ValueError, TypeError):
                pass
            try:
                hv30d_str = first_call.get("historicVolatility30d", "")
                hv30d = float(hv30d_str.replace("%", "")) if hv30d_str else None
            except (ValueError, TypeError):
                pass
            try:
                avg_iv_str = first_call.get("averageVolatility", "")
                avg_iv = float(avg_iv_str.replace("%", "")) if avg_iv_str else None
            except (ValueError, TypeError):
                pass

        print(f"[fetch_data] SteadyAPI: {len(all_exps)} expirations, IV rank 1y={iv_rank_1y}, HV30d={hv30d}")

        # Step 2: Fetch chains for relevant expirations
        from datetime import date as _date
        today = _date.today()

        # Tier classification + per-contract parse logic lives in chain_parser.py
        # so it can be unit-tested in isolation. Imported here to avoid pulling
        # in network/DB code from tests.
        from chain_parser import classify_tier, parse_contract

        for exp_str in all_exps:
            try:
                exp_date = _date.fromisoformat(exp_str)
                dte = (exp_date - today).days
            except ValueError:
                continue

            tier = classify_tier(dte)
            if tier is None:
                continue

            resp = requests.get(
                f"{STEADYAPI_BASE}/v3/markets/options",
                headers=headers,
                params={"ticker": symbol, "expiration": exp_str},
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"[fetch_data] SteadyAPI chain {exp_str}: HTTP {resp.status_code}", file=sys.stderr)
                continue

            chain = resp.json().get("body", {})

            # Parse puts — only meaningful for the short tier (CSP candidates).
            # Mid/leap puts are deep cycle, low signal for our strategies and
            # would balloon the snapshot for no benefit.
            if tier == "short":
                for p in chain.get("Put", []):
                    parsed = parse_contract(p, dte, exp_str, tier)
                    if parsed is not None:
                        puts.append(parsed)

            # Parse calls — always (all tiers carry usable calls).
            for c in chain.get("Call", []):
                parsed = parse_contract(c, dte, exp_str, tier)
                if parsed is not None:
                    calls.append(parsed)

            # Log raw vs parsed strike counts so silent-drop bugs surface in logs.
            raw_calls = len(chain.get("Call", []))
            raw_puts = len(chain.get("Put", []))
            print(f"[fetch_data] SteadyAPI chain {exp_str} (DTE {dte}, tier {tier}): {raw_puts}P / {raw_calls}C")

    except Exception as e:
        print(f"[fetch_data] SteadyAPI chain fetch failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    # Compute IV skew (25-delta put IV minus ATM call IV) and put/call OI ratio
    # across ALL fetched expirations in the CSP DTE range
    iv_skew_25d = None
    put_call_oi_ratio = None
    total_put_oi = 0
    total_call_oi = 0
    skew_samples = []

    for p in puts:
        total_put_oi += p.get("oi", 0)
        d = p.get("delta")
        iv = p.get("iv")
        if d is not None and iv is not None and -0.30 <= d <= -0.20:
            skew_samples.append(("put_25d", iv, p["strike"], p["dte"]))

    # Need ATM calls from the CSP DTE range too — re-fetch if not already in calls
    # For now, use the avg_iv as ATM proxy (it's the chain average)
    atm_iv_proxy = avg_iv / 100.0 if avg_iv else None

    if skew_samples and atm_iv_proxy:
        avg_put_25d_iv = sum(s[1] for s in skew_samples) / len(skew_samples)
        iv_skew_25d = round((avg_put_25d_iv - atm_iv_proxy) * 100, 2)  # In percentage points
        print(f"[fetch_data] IV skew (25d put - ATM): {iv_skew_25d:+.2f}%")

    # Put/call OI uses ALL chains (puts from CSP range + calls from LEAP range isn't apples-to-apples)
    # So we track put OI from CSP-range expirations only, and we need call OI from same expirations.
    # For now, store put OI / total as a directional indicator.
    for c in calls:
        total_call_oi += c.get("oi", 0)

    if total_call_oi > 0:
        put_call_oi_ratio = round(total_put_oi / total_call_oi, 3)
        print(f"[fetch_data] Put/Call OI ratio: {put_call_oi_ratio:.3f} (put={total_put_oi}, call={total_call_oi})")

    print(f"[fetch_data] SteadyAPI total: {len(puts)} puts, {len(calls)} calls")
    return {
        "puts": puts,
        "calls": calls,
        "iv_rank_1y": iv_rank_1y,
        "hv30d": hv30d,
        "avg_iv": avg_iv,
        "iv_skew_25d": iv_skew_25d,
        "put_call_oi_ratio": put_call_oi_ratio,
    }


# ── StrategyTracker ───────────────────────────────────────────────────────────


def fetch_strategytracker():
    """
    Fetch mNAV, BTC Yield, and treasury data from Strive's StrategyTracker dashboard API.
    Source: https://treasury.strive.com/ (powered by StrategyTracker.com)
    """
    try:
        # Step 1: Get current data version
        manifest = requests.get("https://data.strategytracker.com/latest.json", timeout=10).json()
        version = manifest["version"]

        # Step 2: Fetch ASST-specific dataset
        url = f"https://data.strategytracker.com/ASST.v{version}.json"
        data = requests.get(url, timeout=10).json()

        # Data is nested: { companies: { ASST: { processedMetrics: {...} } } }
        metrics = data.get("companies", {}).get("ASST", {}).get("processedMetrics", data)

        # Compute EV-based mNAV = (Market Cap + Debt + Preferred - Cash) / BTC NAV
        # This is how Strive calculates it -- includes the preferred equity obligation
        market_cap = float(metrics.get("marketCapBasic") or 0)
        btc_nav = float(metrics.get("holdingsValue") or 0)
        cash = float(metrics.get("latestCashBalance") or 0)
        debt = float(metrics.get("latestDebt") or 0)
        preferred_stocks = data.get("companies", {}).get("ASST", {}).get(
            "preferredStocks", []
        ) or metrics.get("preferredStocks") or []
        preferred_notional = sum(
            float(ps.get("notionalUSD") or 0) for ps in preferred_stocks
        )
        ev = market_cap + debt + preferred_notional - cash
        ev_mnav = ev / btc_nav if btc_nav > 0 else 0.0

        # basic_mnav = market cap only (research/comparison multiple, navPremiumBasic from API)
        basic_mnav_val = float(metrics.get("navPremiumBasic") or 0) or (market_cap / btc_nav if btc_nav > 0 else 0)

        result = {
            "basic_mnav": round(basic_mnav_val, 4),  # Market Cap / BTC NAV (research multiple)
            "ev_mnav": round(ev_mnav, 4),              # (EV including preferred) / BTC NAV (primary multiple)
            "btc_yield_ytd": float(metrics.get("btcYieldYtd") or 0),
            "btc_holdings": float(metrics.get("latestBtcBalance") or 0),
            "btc_nav": float(metrics.get("holdingsValue") or 0),
            "nav_per_share": float(metrics.get("btcValuePerShare") or 0),
            "avg_cost_per_btc": float(metrics.get("avgCostPerBtc") or 0),
            "total_shares": int(metrics.get("latestTotalShares") or 0),
            "diluted_shares": int(metrics.get("latestDilutedShares") or 0),
            "cash_balance": float(metrics.get("latestCashBalance") or 0),
            "debt_outstanding": float(metrics.get("latestDebt") or 0),
        }

        print(f"[fetch_data] StrategyTracker: basic mNAV={result['basic_mnav']:.4f}x  EV mNAV={result['ev_mnav']:.4f}x  (mcap=${market_cap/1e6:.0f}M + debt=${debt/1e6:.0f}M + pref=${preferred_notional/1e6:.0f}M - cash=${cash/1e6:.0f}M) / BTC NAV=${btc_nav/1e6:.0f}M")
        print(f"[fetch_data] BTC Yield YTD={result['btc_yield_ytd']:.1f}%  BTC holdings={result['btc_holdings']:.1f}")
        return result

    except Exception as e:
        print(f"[fetch_data] StrategyTracker fetch failed (non-fatal): {e}", file=sys.stderr)
        return None


# ── FlashAlpha ────────────────────────────────────────────────────────────────


def _fetch_single_expiry(expiry: str) -> Optional[dict]:
    """Fetch GEX for a single expiration date from FlashAlpha Basic plan."""
    url = f"{FLASHALPHA_API_BASE}?expiration={expiry}"
    try:
        resp = requests.get(url, headers={"X-Api-Key": FLASHALPHA_API_KEY}, timeout=15)
        if resp.status_code == 404:
            # No data for this expiry — not an error, just skip
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"[fetch_data]   Expiry {expiry}: HTTP {e.response.status_code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[fetch_data]   Expiry {expiry}: {e}", file=sys.stderr)
        return None


def _fetch_single_expiry_vex(expiry: str) -> Optional[dict]:
    """Fetch VEX (vanna exposure) for a single expiration date from FlashAlpha."""
    url = f"https://lab.flashalpha.com/v1/exposure/vex/{SYMBOL}?expiration={expiry}"
    try:
        resp = requests.get(url, headers={"X-Api-Key": FLASHALPHA_API_KEY}, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"[fetch_data]   VEX {expiry}: HTTP {e.response.status_code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[fetch_data]   VEX {expiry}: {e}", file=sys.stderr)
        return None


def fetch_flashalpha_gex(symbol: str = SYMBOL):
    """
    Fetch GEX data from FlashAlpha (Basic plan — one expiry per request).
    Loops through all expirations, aggregates per-strike GEX,
    and computes the overall gamma flip from the merged strike data.
    Returns a dict with keys: gamma_flip, net_gex, gex_strikes (list of {strike, net_gex}).
    """
    import time as _time

    # Primary: SteadyAPI expirations (no yfinance dependency)
    expirations = []
    try:
        resp = requests.get(
            f"{STEADYAPI_BASE}/v3/markets/options",
            headers={"Authorization": f"Bearer {STEADYAPI_KEY}"},
            params={"ticker": symbol},
            timeout=15,
        )
        if resp.status_code == 200:
            meta = resp.json().get("meta", {})
            exps = meta.get("expirations", {})
            weekly = exps.get("weekly", [])
            monthly = exps.get("monthly", [])
            expirations = sorted(set(weekly + monthly))
            print(f"[fetch_data] Expirations from SteadyAPI: {len(expirations)}")
    except Exception as e:
        print(f"[fetch_data] SteadyAPI expirations failed: {e}", file=sys.stderr)

    # Fallback: yfinance if SteadyAPI returned nothing
    if not expirations:
        try:
            import yfinance as yf
            expirations = list(yf.Ticker(symbol).options)
            print(f"[fetch_data] Expirations from yfinance fallback: {len(expirations)}")
        except Exception as e:
            print(f"[fetch_data] yfinance expirations fallback also failed: {e}", file=sys.stderr)
            return None

    if not expirations:
        print(f"[fetch_data] No expirations found for {symbol}", file=sys.stderr)
        return None

    print(f"[fetch_data] Querying FlashAlpha for {len(expirations)} expirations...")

    # Aggregate per-strike net_gex across all expirations
    strike_gex: dict[float, float] = {}  # strike -> total net_gex
    underlying_price = None
    success_count = 0
    total_vex = 0.0
    vex_success = 0
    # Collect per-expiry gamma flips for weighted average
    expiry_flips: list[tuple[float, float]] = []  # (gamma_flip, abs_net_gex) per expiry

    for exp in expirations:
        data = _fetch_single_expiry(exp)
        if data is None:
            continue

        success_count += 1
        if underlying_price is None:
            underlying_price = float(data.get("underlying_price", 0))

        # Collect this expiry's gamma_flip (computed by FlashAlpha)
        exp_flip = float(data.get("gamma_flip") or 0)
        exp_net = float(data.get("net_gex") or 0)
        if exp_flip > 0:
            expiry_flips.append((exp_flip, abs(exp_net)))

        for s in data.get("strikes", []):
            try:
                k = float(s["strike"])
                gex = float(s["net_gex"])
                strike_gex[k] = strike_gex.get(k, 0.0) + gex
            except (KeyError, TypeError, ValueError):
                continue

        # Fetch VEX (vanna exposure) for same expiry
        vex_data = _fetch_single_expiry_vex(exp)
        if vex_data is not None:
            vex_success += 1
            exp_vex = float(vex_data.get("net_vex") or 0)
            total_vex += exp_vex

        # Small delay to be polite to the API
        _time.sleep(0.15)

    if success_count == 0:
        print("[fetch_data] No FlashAlpha data returned for any expiry.", file=sys.stderr)
        return None

    print(f"[fetch_data] Aggregated {success_count}/{len(expirations)} expirations, {len(strike_gex)} strikes.")

    # Compute aggregated net_gex
    net_gex = sum(strike_gex.values())

    # Compute gamma flip via two methods and pick the best:
    # Method 1: Interpolate from aggregated cumulative GEX crossover
    sorted_strikes = sorted(strike_gex.keys())
    running = 0.0
    prev_k = None
    prev_running = 0.0
    interpolated_flip = None
    for k in sorted_strikes:
        prev_running = running
        running += strike_gex[k]
        if running > 0 and prev_running <= 0 and prev_k is not None:
            # Linear interpolation between prev_k and k
            denom = running - prev_running
            if denom != 0:
                frac = (0 - prev_running) / denom
                interpolated_flip = prev_k + frac * (k - prev_k)
            else:
                interpolated_flip = k
            break
        prev_k = k

    # Method 2: Weighted average of per-expiry gamma flips (weighted by |net_gex|)
    weighted_flip = None
    if expiry_flips:
        total_weight = sum(w for _, w in expiry_flips)
        if total_weight > 0:
            weighted_flip = sum(gf * w for gf, w in expiry_flips) / total_weight

    # Choose: prefer interpolated (from aggregated data), fall back to weighted per-expiry
    gamma_flip = interpolated_flip or weighted_flip or underlying_price or 0.0
    gamma_flip = round(gamma_flip, 4)

    print(f"[fetch_data] Gamma flip: interpolated={interpolated_flip}, weighted={(f'{weighted_flip:.4f}' if weighted_flip else 'N/A')}, chosen={gamma_flip}")

    # Build gex_strikes list
    gex_strikes = [
        {"strike": k, "net_gex": strike_gex[k]}
        for k in sorted_strikes
        if abs(strike_gex[k]) > 0.01  # filter out zero strikes
    ]

    net_vanna = total_vex if vex_success > 0 else None
    if net_vanna is not None:
        print(f"[fetch_data] Net Vanna (VEX): {net_vanna:,.0f} ({vex_success} expirations)")

    return {"gamma_flip": gamma_flip, "net_gex": net_gex, "gex_strikes": gex_strikes, "net_vanna": net_vanna}


# ── Market Data (SteadyAPI + Tiingo, replaces yfinance) ────────────────────


def fetch_market_data(symbol: str = SYMBOL):
    """
    Fetch spot price, ATR, BTC price, SATA price, volume from SteadyAPI + Tiingo.
    Replaces the former yfinance-based fetch_yfinance_data().
    Returns dict: {spot, atr_1d, btc_price, sata_price, sata_volume, asst_volume, asst_market_cap, ...}
    """
    result = {}
    headers = {"Authorization": f"Bearer {STEADYAPI_KEY}"}

    # 1. ASST spot price from SteadyAPI quote
    try:
        resp = requests.get(
            f"{STEADYAPI_BASE}/v1/markets/quote",
            headers=headers,
            params={"ticker": symbol, "type": "STOCKS"},
            timeout=10,
        )
        if resp.status_code == 200:
            body = resp.json().get("body", {})
            primary = body.get("primaryData", {})
            price_str = primary.get("lastSalePrice", "").replace("$", "").replace(",", "")
            if price_str:
                result["spot"] = float(price_str)
                print(f"[fetch_data] ASST spot (SteadyAPI): ${result['spot']:.4f}")
    except Exception as e:
        print(f"[fetch_data] SteadyAPI ASST quote failed: {e}", file=sys.stderr)

    if "spot" not in result:
        print("[fetch_data] Could not get ASST spot price. Aborting.", file=sys.stderr)
        return None

    # 2. ATR from Tiingo cache (computed nightly by tiingo_eod.py) or Tiingo API
    atr_1d = None
    try:
        cache_path = Path(__file__).parent.parent / "tiingo_cache" / "asst_stats.json"
        if cache_path.exists():
            import json as _json
            with open(cache_path) as f:
                stats = _json.load(f)
            if "atr_14d" in stats:
                atr_1d = float(stats["atr_14d"])
                print(f"[fetch_data] ATR (Tiingo cache): {atr_1d:.4f}")
    except Exception:
        pass

    if atr_1d is None:
        # Fallback: compute from Tiingo API
        try:
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            resp = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
                params={"startDate": start, "token": TIINGO_TOKEN},
                timeout=15,
            )
            if resp.status_code == 200:
                bars = resp.json()
                if len(bars) >= 15:
                    trs = []
                    for i in range(1, len(bars)):
                        tr = max(
                            float(bars[i]["adjHigh"]) - float(bars[i]["adjLow"]),
                            abs(float(bars[i]["adjHigh"]) - float(bars[i - 1]["adjClose"])),
                            abs(float(bars[i]["adjLow"]) - float(bars[i - 1]["adjClose"])),
                        )
                        trs.append(tr)
                    atr_1d = sum(trs[-14:]) / 14
                    print(f"[fetch_data] ATR (Tiingo API): {atr_1d:.4f}")
        except Exception as e:
            print(f"[fetch_data] Tiingo ATR fallback failed: {e}", file=sys.stderr)

    result["atr_1d"] = atr_1d or 0

    # 3. Volume + market cap from Tiingo EOD
    volume = None
    market_cap = None
    try:
        resp = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
            params={"startDate": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), "token": TIINGO_TOKEN},
            timeout=10,
        )
        if resp.status_code == 200:
            bars = resp.json()
            if bars:
                volume = int(bars[-1].get("adjVolume", 0))
                print(f"[fetch_data] ASST volume (Tiingo): {volume:,}")
    except Exception:
        pass
    result["volume"] = volume
    result["market_cap"] = market_cap  # Will be None; derive from spot × shares if needed

    # 4. BTC price from Tiingo crypto
    btc_price = None
    try:
        resp = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            params={"tickers": "btcusd", "token": TIINGO_TOKEN},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and data[0].get("priceData"):
                btc_price = float(data[0]["priceData"][0]["close"])
                print(f"[fetch_data] BTC (Tiingo crypto): ${btc_price:,.2f}")
    except Exception as e:
        print(f"[fetch_data] Tiingo BTC fetch failed: {e}", file=sys.stderr)
    result["btc_price"] = btc_price

    # 5. SATA price + volume from SteadyAPI quote + Tiingo EOD
    sata_price = None
    sata_volume = None
    try:
        resp = requests.get(
            f"{STEADYAPI_BASE}/v1/markets/quote",
            headers=headers,
            params={"ticker": "SATA", "type": "STOCKS"},
            timeout=10,
        )
        if resp.status_code == 200:
            body = resp.json().get("body", {})
            price_str = body.get("primaryData", {}).get("lastSalePrice", "").replace("$", "").replace(",", "")
            if price_str:
                sata_price = float(price_str)
                print(f"[fetch_data] SATA (SteadyAPI): ${sata_price:.2f}")
    except Exception:
        pass

    if sata_price is None:
        try:
            resp = requests.get(
                f"https://api.tiingo.com/tiingo/daily/SATA/prices",
                params={"startDate": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), "token": TIINGO_TOKEN},
                timeout=10,
            )
            if resp.status_code == 200:
                bars = resp.json()
                if bars:
                    sata_price = float(bars[-1]["adjClose"])
                    sata_volume = int(bars[-1].get("adjVolume", 0))
                    print(f"[fetch_data] SATA (Tiingo): ${sata_price:.2f}  vol={sata_volume}")
        except Exception:
            pass

    result["sata_price"] = sata_price
    result["sata_volume"] = sata_volume
    result["sata_options_oi"] = None  # No longer fetched (was yfinance-only)

    # 6. Empty puts/calls (chain data now comes from SteadyAPI in fetch_steadyapi_chain)
    result["puts"] = []
    result["calls"] = []

    print(f"[fetch_data] Market data: spot=${result['spot']:.2f}, atr={result['atr_1d']:.4f}, btc=${result.get('btc_price') or 'N/A'}")
    return result


# ── Engine integration ────────────────────────────────────────────────────────


def run_engine(
    spot: float,
    gamma_flip: float,
    atr_1d: Optional[float],
    net_gex: float,
    gex_percentile: float,
    gex_strikes_raw: list,
    puts_raw: list,
    calls_raw: list,
    iv_rank: float = 25.0,
    iv_percentile: float = 25.0,
    basic_mnav: float = 0.72,
    bitcoin_yield_pct: float = 13.8,
    asst_drawdown_90d: float = 0.0,
    btc_mvrv: Optional[float] = None,
    btc_weekly_rsi: Optional[float] = None,
):
    """Import engine.py and call generate_daily_recommendations."""
    engine_dir = Path(__file__).parent
    sys.path.insert(0, str(engine_dir))

    from engine import (
        generate_daily_recommendations,
        GexStrike,
        OptionQuote,
    )

    gex_strikes = [GexStrike(strike=s["strike"], net_gex=s["net_gex"]) for s in gex_strikes_raw]
    puts = [
        OptionQuote(strike=p["strike"], dte=p["dte"], mid=p["mid"], expiry=p["expiry"])
        for p in puts_raw
    ]
    calls = [
        OptionQuote(
            strike=c["strike"],
            dte=c["dte"],
            mid=c["mid"],
            expiry=c["expiry"],
            is_leap=c.get("is_leap", False),
        )
        for c in calls_raw
    ]

    return generate_daily_recommendations(
        spot=spot,
        flip=gamma_flip,
        atr_1d=atr_1d,
        net_gex=net_gex,
        gex_percentile=gex_percentile,
        gex_strikes=gex_strikes,
        puts=puts,
        calls=calls,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        basic_mnav=basic_mnav,
        bitcoin_yield_pct=bitcoin_yield_pct,
        asst_drawdown_90d=asst_drawdown_90d,
        btc_mvrv=btc_mvrv,
        btc_weekly_rsi=btc_weekly_rsi,
    )


# ── Build API payload ─────────────────────────────────────────────────────────


def build_run_payload(
    spot: float,
    gamma_flip: float,
    atr_1d: Optional[float],
    net_gex: float,
    gex_percentile: float,
    recs: dict,
    status: str = "ok",
    session: str = "PM",
    btc_price: Optional[float] = None,
    asst_volume: Optional[int] = None,
    asst_market_cap: Optional[float] = None,
    sata_price: Optional[float] = None,
    sata_volume: Optional[int] = None,
    sata_options_oi: Optional[int] = None,
    leap_entry_percentile: Optional[float] = None,
    # StrategyTracker fields
    basic_mnav: Optional[float] = None,
    btc_yield_ytd: Optional[float] = None,
    btc_holdings: Optional[float] = None,
    btc_nav: Optional[float] = None,
    nav_per_share: Optional[float] = None,
    avg_cost_per_btc: Optional[float] = None,
    total_shares: Optional[int] = None,
    diluted_shares: Optional[int] = None,
    cash_balance: Optional[float] = None,
    debt_outstanding: Optional[float] = None,
    # Risk management fields
    btc_weekly_rsi: Optional[float] = None,
    asst_drawdown_90d: Optional[float] = None,
    asst_90d_high: Optional[float] = None,
    risk_zone: Optional[str] = None,
    raw_flip: Optional[float] = None,  # v2.4.1: raw unsmoothed flip persisted alongside smoothed gamma_flip
    current_iv: Optional[float] = None,
    iv_rank: Optional[float] = None,
    iv_percentile: Optional[float] = None,
    iv_rank_method: Optional[str] = None,
    ev_mnav: Optional[float] = None,
    btc_mvrv: Optional[float] = None,
    btc_realized_price: Optional[float] = None,
    market_closed: bool = False,
    # BTC derivatives (BGeometrics)
    btc_gex_secondary_confirm: Optional[str] = None,
    btc_taker_buy_sell_ratio: Optional[float] = None,
    btc_funding_rate: Optional[float] = None,
    btc_liq_total_usd: Optional[float] = None,
    btc_liq_long_pct: Optional[float] = None,
    btc_fear_greed: Optional[int] = None,
    # BTC cycle-top valuation indicators (BGeometrics) — used by Selector v4
    # cycle-top trigger panel and DEFENSIVE_YIELD_TOP rotation logic.
    btc_mvrv_zscore: Optional[float] = None,
    btc_puell_multiple: Optional[float] = None,
    btc_nupl: Optional[float] = None,
    btc_reserve_risk: Optional[float] = None,
    btc_pi_cycle_signal: Optional[int] = None,
    # Trade permissions
    btc_cycle_zone: Optional[str] = None,
    csp_allowed: Optional[str] = None,
    leap_add_allowed: Optional[str] = None,
    leap_add_size: Optional[str] = None,
    pmcc_allowed: Optional[str] = None,
    action_banner: Optional[str] = None,
    # Vanna (VEX) fields
    net_vanna: Optional[float] = None,
    vanna_percentile: Optional[float] = None,
    vanna_regime: Optional[str] = None,
    # Contract suggestions
    csp_suggestion_json: Optional[str] = None,
    leap_suggestion_json: Optional[str] = None,
    pmcc_suggestion_json: Optional[str] = None,
    stochastic_output_json: Optional[str] = None,
    # Enrichment fields
    mnav_discount: Optional[float] = None,
    btc_per_share_basic: Optional[float] = None,
    btc_per_share_diluted: Optional[float] = None,
    csp_delta_to_band: Optional[float] = None,
    csp_magnet_proximity: Optional[float] = None,
    iv_skew_25d: Optional[float] = None,
    put_call_oi_ratio: Optional[float] = None,
    # Option chain snapshot — raw SteadyAPI chain dict (or None if fetch failed)
    chain_data: Optional[dict] = None,
) -> dict:
    now = datetime.now(timezone.utc)

    csp_candidates = recs.get("cspcandidates") or []
    top_csp = csp_candidates[0] if csp_candidates else None

    leap_core = recs.get("leapcore") or []
    leap_mid = recs.get("leapmid") or []
    leap_tail = recs.get("leaptail") or []
    csp_band = recs.get("csp_band") or (0.0, 0.0)
    core_band = recs.get("leapcore_band") or (0.0, 0.0)

    pos_magnets = recs.get("posmagnets") or []
    neg_magnets = recs.get("negmagnets") or []

    def magnets_to_json(mags):
        try:
            return json.dumps([{"strike": m.strike, "net_gex": m.net_gex} for m in mags])
        except Exception:
            return "[]"

    def strikes_to_json(opts):
        try:
            return json.dumps(sorted({float(o.strike) for o in opts}))
        except Exception:
            return "[]"

    csp_candidates_dicts = []
    for c in csp_candidates:
        if isinstance(c, dict):
            csp_candidates_dicts.append(c)
        else:
            csp_candidates_dicts.append({
                "strike": float(getattr(c, "strike", 0)),
                "dte": int(getattr(c, "dte", 0)),
                "mid": float(getattr(c, "mid", 0)),
                "expiry": str(getattr(c, "expiry", "")),
                "eff_basis": float(getattr(c, "eff_basis", 0)),
            })

    return {
        "date": now.strftime("%Y-%m-%d"),
        "runtime_utc": now.isoformat(),
        "symbol": SYMBOL,
        "spot": spot,
        "gamma_flip": gamma_flip,                # SMOOTHED flip (used by bands)
        "raw_flip": raw_flip if raw_flip is not None else gamma_flip,  # RAW flip (audit/transparency)
        "atr_1d": atr_1d,
        "net_gex": net_gex,
        "gex_percentile": gex_percentile,
        "net_vanna": net_vanna,
        "vanna_percentile": vanna_percentile,
        "vanna_regime": vanna_regime,
        "regime": recs.get("regime", "neutral"),
        "csp_band_low": float(csp_band[0]),
        "csp_band_high": float(csp_band[1]),
        "leap_core_band_low": float(core_band[0]),
        "leap_core_band_high": float(core_band[1]),
        "pos_magnets": magnets_to_json(pos_magnets),
        "neg_magnets": magnets_to_json(neg_magnets),
        "csp_count": len(csp_candidates),
        "csp_top_strike": float(top_csp["strike"]) if top_csp else None,
        "csp_top_expiry": top_csp["expiry"] if top_csp else None,
        "csp_top_dte": int(top_csp["dte"]) if top_csp else None,
        "csp_top_mid": float(top_csp["mid"]) if top_csp else None,
        "csp_top_eff_basis": float(top_csp["eff_basis"]) if top_csp else None,
        "leap_core_count": len(leap_core),
        "leap_mid_count": len(leap_mid),
        "leap_tail_count": len(leap_tail),
        "leap_core_strikes": strikes_to_json(leap_core),
        "leap_mid_strikes": strikes_to_json(leap_mid),
        "leap_tail_strikes": strikes_to_json(leap_tail),
        "notes": json.dumps(recs.get("notes") or []),
        "csp_candidates_json": json.dumps(csp_candidates_dicts),
        "option_chain_snapshot_json": serialize_chain_snapshot(chain_data),
        "status": status,
        "session": session,
        "btc_price": btc_price,
        "asst_volume": asst_volume,
        "asst_market_cap": asst_market_cap,
        "sata_price": sata_price,
        "sata_volume": sata_volume,
        "sata_options_oi": sata_options_oi,
        "leap_entry_percentile": leap_entry_percentile,
        # StrategyTracker (mNAV/treasury) fields
        "basic_mnav": basic_mnav,
        "btc_yield_ytd": btc_yield_ytd,
        "btc_holdings": btc_holdings,
        "btc_nav": btc_nav,
        "nav_per_share": nav_per_share,
        "avg_cost_per_btc": avg_cost_per_btc,
        "total_shares": total_shares,
        "diluted_shares": diluted_shares,
        "cash_balance": cash_balance,
        "debt_outstanding": debt_outstanding,
        # Engine v2 fields
        "iv_regime": recs.get("iv_regime"),
        "leap_entry_score": recs.get("leap_entry_score"),
        "pmcc_status": recs.get("pmcc_status"),
        # Risk management fields
        "btc_weekly_rsi": btc_weekly_rsi,
        "asst_drawdown_90d": asst_drawdown_90d,
        "asst_90d_high": asst_90d_high,
        "risk_zone": risk_zone,
        "current_iv": current_iv,
        "iv_rank": iv_rank,
        "iv_percentile": iv_percentile,
        # iv_band: canonical cohort dimension derived from iv_percentile.
        # NULL iv_percentile -> NULL iv_band (forward-only doctrine).
        "iv_band": compute_iv_band(iv_percentile),
        "iv_rank_method": iv_rank_method,
        "ev_mnav": ev_mnav,
        "btc_mvrv": btc_mvrv,
        "btc_realized_price": btc_realized_price,
        "market_closed": 1 if market_closed else 0,
        # Trade permissions
        "btc_cycle_zone": btc_cycle_zone,
        "csp_allowed": csp_allowed,
        "leap_add_allowed": leap_add_allowed,
        "leap_add_size": leap_add_size,
        "pmcc_allowed": pmcc_allowed,
        "action_banner": action_banner,
        # BTC derivatives
        "btc_gex_secondary_confirm": btc_gex_secondary_confirm,
        "btc_taker_buy_sell_ratio": btc_taker_buy_sell_ratio,
        "btc_funding_rate": btc_funding_rate,
        "btc_liq_total_usd": btc_liq_total_usd,
        "btc_liq_long_pct": btc_liq_long_pct,
        "btc_fear_greed": btc_fear_greed,
        # BTC cycle-top valuation indicators
        "btc_mvrv_zscore": btc_mvrv_zscore,
        "btc_puell_multiple": btc_puell_multiple,
        "btc_nupl": btc_nupl,
        "btc_reserve_risk": btc_reserve_risk,
        "btc_pi_cycle_signal": btc_pi_cycle_signal,
        # Contract suggestions
        "csp_suggestion_json": csp_suggestion_json,
        "leap_suggestion_json": leap_suggestion_json,
        "pmcc_suggestion_json": pmcc_suggestion_json,
        "stochastic_output_json": stochastic_output_json,
        # Enrichment fields
        "mnav_discount": mnav_discount,
        "btc_per_share_basic": btc_per_share_basic,
        "btc_per_share_diluted": btc_per_share_diluted,
        "csp_delta_to_band": csp_delta_to_band,
        "csp_magnet_proximity": csp_magnet_proximity,
        "iv_skew_25d": iv_skew_25d,
        "put_call_oi_ratio": put_call_oi_ratio,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="asst-gamma-engine Daily Data Fetcher (v2)")
    parser.add_argument("--session", default="PM", choices=["AM", "MID", "PM"], help="Session tag: AM (morning), MID (midday), or PM (afternoon)")
    parser.add_argument("--persist-only", action="store_true", help="Persist run row only; skip selector evaluation. Used by tests.")
    # NOTE (P1.2): the legacy --api-url flag has been removed. direct_persist
    # is the only persistence path. The selector evaluate/log HTTP calls
    # below still go to localhost:5000 as a TEMP shim until P1.8.
    parser.add_argument("--legacy-api-url", default=DEFAULT_API_BASE, help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Maintain compatibility with the existing code body that still references
    # args.api_url for the selector HTTP shim. Removed entirely in P1.8.
    args.api_url = args.legacy_api_url

    print(f"[fetch_data] Starting ASST data fetch — {datetime.now(timezone.utc).isoformat()}")

    # 1. Fetch FlashAlpha GEX
    print("[fetch_data] Fetching GEX from FlashAlpha...")
    gex_data = fetch_flashalpha_gex()
    if not gex_data:
        print("[fetch_data] FlashAlpha fetch failed. Aborting.", file=sys.stderr)
        sys.exit(1)

    gamma_flip = gex_data["gamma_flip"]
    net_gex = gex_data["net_gex"]
    gex_strikes_raw = gex_data["gex_strikes"]
    net_vanna = gex_data.get("net_vanna")
    print(f"[fetch_data] gamma_flip={gamma_flip:.4f}  net_gex={net_gex:.2e}")

    # 2. Fetch market data (SteadyAPI + Tiingo, replaces yfinance)
    print("[fetch_data] Fetching market data (SteadyAPI + Tiingo)...")
    mkt_data = fetch_market_data()
    if not mkt_data:
        print("[fetch_data] Market data fetch failed. Aborting.", file=sys.stderr)
        sys.exit(1)

    spot = mkt_data["spot"]
    atr_1d = mkt_data["atr_1d"]
    puts_raw = mkt_data["puts"]
    calls_raw = mkt_data["calls"]
    btc_price = mkt_data.get("btc_price")
    asst_volume = mkt_data.get("volume")
    asst_market_cap = mkt_data.get("market_cap")
    sata_price = mkt_data.get("sata_price")
    sata_volume = mkt_data.get("sata_volume")
    sata_options_oi = mkt_data.get("sata_options_oi")
    print(f"[fetch_data] spot={spot:.4f}  atr_1d={atr_1d}")
    print(f"[fetch_data] Session={args.session}  BTC=${btc_price:,.0f}" if btc_price else f"[fetch_data] Session={args.session}  BTC=N/A")

    # 2b. Load Tiingo-cleaned stats if available (nightly EOD backbone)
    tiingo_stats = None
    try:
        tiingo_stats_path = os.path.join(os.path.dirname(__file__), '..', 'tiingo_cache', 'asst_stats.json')
        if os.path.exists(tiingo_stats_path):
            with open(tiingo_stats_path) as _f:
                tiingo_stats = json.load(_f)
            if tiingo_stats.get('atr_14d'):
                tiingo_atr = tiingo_stats['atr_14d']
                print(f"[fetch_data] Tiingo ATR: {tiingo_atr:.4f} (market data: {atr_1d:.4f}) — using Tiingo")
                atr_1d = tiingo_atr
            if tiingo_stats.get('high_90d') is not None:
                print(f"[fetch_data] Tiingo 90d stats available: high={tiingo_stats['high_90d']:.2f}, DD={tiingo_stats['drawdown_90d']:.1%}")
    except Exception as e:
        print(f"[fetch_data] Tiingo stats load failed (non-fatal): {e}", file=sys.stderr)

    # 2b-cont. Risk management computations (BTC weekly RSI + ASST 90d drawdown)
    print("[fetch_data] Computing risk metrics...")
    btc_weekly_rsi = compute_btc_weekly_rsi()
    if btc_weekly_rsi is not None:
        print(f"[fetch_data] BTC weekly RSI: {btc_weekly_rsi:.1f}")
    else:
        print("[fetch_data] BTC weekly RSI: N/A")

    asst_drawdown_90d, asst_90d_high = compute_asst_drawdown_90d(spot)
    # Prefer Tiingo-cleaned values if available
    if tiingo_stats and tiingo_stats.get('drawdown_90d') is not None:
        asst_drawdown_90d = tiingo_stats['drawdown_90d']
        asst_90d_high = tiingo_stats['high_90d']
        print(f"[fetch_data] ASST 90d drawdown: {asst_drawdown_90d:.1%} from ${asst_90d_high:.2f} (Tiingo)")
    elif asst_drawdown_90d is not None:
        print(f"[fetch_data] ASST 90d drawdown: {asst_drawdown_90d:.1%} from ${asst_90d_high:.2f}")
    else:
        print("[fetch_data] ASST 90d drawdown: N/A")

    # 2c. Fetch options chain with Greeks from SteadyAPI (needed for IV metrics + suggestions)
    print("[fetch_data] Fetching options chain with Greeks (SteadyAPI)...")
    chain_data = fetch_steadyapi_chain()
    steady_puts = chain_data.get("puts", []) if chain_data else []
    steady_calls = chain_data.get("calls", []) if chain_data else []
    if chain_data:
        print(f"[fetch_data] SteadyAPI chain: {len(steady_puts)} puts, {len(steady_calls)} calls")

    # 2d. Compute IV metrics (uses SteadyAPI avg_iv + iv_rank_1y)
    print("[fetch_data] Computing IV metrics...")
    iv_data = compute_iv_metrics(spot, chain_data=chain_data)
    current_iv = iv_data["current_iv"] if iv_data else None
    iv_rank_val = iv_data["iv_rank"] if iv_data else None
    iv_percentile_val = iv_data["iv_percentile"] if iv_data else None

    # 2d. Fetch BTC on-chain data (MVRV, Realized Price) from BGeometrics authenticated API
    print("[fetch_data] Fetching BTC on-chain data (BGeometrics)...")
    onchain = fetch_btc_onchain()
    btc_mvrv = onchain.get("btc_mvrv") if onchain else None
    btc_realized_price = onchain.get("btc_realized_price") if onchain else None

    # 2e. Fetch BTC derivatives for secondary confirmation (all sessions)
    # Advanced plan: 200 req/hr, 400 req/day. 6 calls/session × 3 = 18/day. No issue.
    print("[fetch_data] Fetching BTC derivatives (BGeometrics)...")
    btc_deriv = fetch_btc_derivatives()
    btc_gex_secondary_confirm = btc_deriv.get("btc_gex_secondary_confirm") if btc_deriv else None
    btc_taker_buy_sell_ratio = btc_deriv.get("btc_taker_buy_sell_ratio") if btc_deriv else None
    btc_funding_rate = btc_deriv.get("btc_funding_rate") if btc_deriv else None
    btc_liq_total_usd = btc_deriv.get("btc_liq_total_usd") if btc_deriv else None
    btc_liq_long_pct = btc_deriv.get("btc_liq_long_pct") if btc_deriv else None
    btc_fear_greed = btc_deriv.get("btc_fear_greed") if btc_deriv else None
    # Cycle-top valuation indicators (computed inside fetch_btc_derivatives).
    # These were being computed and printed but never extracted into the run
    # payload — every new row since 5/04 had NULL cycle metrics. Selector v4's
    # cycle-top panel rendered blank as a result.
    btc_mvrv_zscore = btc_deriv.get("btc_mvrv_zscore") if btc_deriv else None
    btc_puell_multiple = btc_deriv.get("btc_puell_multiple") if btc_deriv else None
    btc_nupl = btc_deriv.get("btc_nupl") if btc_deriv else None
    btc_reserve_risk = btc_deriv.get("btc_reserve_risk") if btc_deriv else None
    btc_pi_cycle_signal = btc_deriv.get("btc_pi_cycle_signal") if btc_deriv else None

    # 2f. Fetch StrategyTracker treasury data (mNAV, BTC Yield, etc.)
    print("[fetch_data] Fetching StrategyTracker treasury data...")
    st_data = fetch_strategytracker()
    basic_mnav = st_data["basic_mnav"] if st_data else None
    ev_mnav_val = st_data["ev_mnav"] if st_data else None
    btc_yield_ytd = st_data["btc_yield_ytd"] if st_data else None
    btc_holdings = st_data["btc_holdings"] if st_data else None
    btc_nav = st_data["btc_nav"] if st_data else None
    nav_per_share = st_data["nav_per_share"] if st_data else None
    avg_cost_per_btc = st_data["avg_cost_per_btc"] if st_data else None
    total_shares = st_data["total_shares"] if st_data else None
    diluted_shares = st_data["diluted_shares"] if st_data else None
    cash_balance = st_data["cash_balance"] if st_data else None
    debt_outstanding = st_data["debt_outstanding"] if st_data else None

    # 2e. Detect market closed (holiday/weekend)
    # When market is closed, volume stays flat and spot doesn't move.
    # GEX data can drift as expiring OI rolls off — flip becomes unreliable.
    #
    # IMPORTANT: this check must compare to a *different* session, not the
    # current session. With UPSERT semantics, re-running MID (gap-check +
    # scheduled MID, or any retry) reads the prior MID row and matches itself
    # nearly perfectly — historical false positive that flagged a live market
    # as closed (e.g. 2026-04-29 MID). We now (a) explicitly exclude the
    # current session from the candidate prev rows and (b) loosen the spot
    # threshold to a more realistic floor (1 cent) since real intraday moves
    # of less than a penny across hours are themselves a closed-market signal.
    market_closed = False
    if asst_volume is not None and asst_volume == 0:
        market_closed = True
    elif args.session != "AM":
        # Check if spot is identical across sessions (no trading)
        try:
            import sqlite3 as _sq
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data.db')
            if os.path.exists(db_path):
                conn = _sq.connect(db_path)
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                # Pick the most recent row from a *different* session today.
                # AM is the canonical reference for MID/PM. PM also falls back
                # to MID. We never compare a session to itself.
                rows = conn.execute(
                    "SELECT spot, asst_volume, session FROM daily_runs "
                    "WHERE date = ? AND session != ? ORDER BY id",
                    (today, args.session),
                ).fetchall()
                conn.close()
                if len(rows) >= 1:
                    prev_spot = rows[-1][0]
                    prev_vol = rows[-1][1]
                    prev_sess = rows[-1][2]
                    # Spot identical to prior session AND volume unchanged = market closed.
                    # Spot threshold 1c is well below typical intraday tick;
                    # ASST routinely moves >$0.05 between sessions when open.
                    spot_match = (
                        prev_spot is not None
                        and abs(spot - prev_spot) < 0.01
                    )
                    vol_match = (
                        prev_vol is not None
                        and asst_volume is not None
                        and abs(asst_volume - prev_vol) < 100
                    )
                    if spot_match and vol_match:
                        market_closed = True
                        print(f"[fetch_data] Market-closed heuristic tripped: "
                              f"spot={spot:.4f} matches {prev_sess}={prev_spot:.4f} "
                              f"(\u0394={abs(spot - prev_spot):.4f}); "
                              f"vol={asst_volume} matches {prev_sess}={prev_vol}.")
        except Exception as e:
            print(f"[fetch_data] Market closed detection error (non-fatal): {e}", file=sys.stderr)

    if market_closed:
        print(f"[fetch_data] \u26a0\ufe0f MARKET CLOSED detected. GEX flip may reflect expiry distortion.")
        print(f"[fetch_data] Raw flip {gamma_flip:.4f} will be replaced with last clean flip from history.")

    # 3. Load GEX history, compute percentile, smooth flip
    history = load_gex_history()
    gex_percentile = compute_gex_percentile(net_gex, history)

    # If market is closed, use last clean flip instead of drifting raw flip
    if market_closed and len(history) > 0:
        last_clean_flip = history[-1].get("flip", gamma_flip)
        print(f"[fetch_data] Using last clean flip: {last_clean_flip:.4f} (overriding raw {gamma_flip:.4f})")
        gamma_flip = last_clean_flip

    # Preserve the raw (unsmoothed) flip for persistence and diagnostics.
    raw_flip = gamma_flip

    # Pre-classify the regime using the RAW flip so we can decide whether to
    # reset the EMA. This breaks the circular dependency between flip smoothing
    # and regime classification: regime is determined by the raw GEX integral,
    # not by our smoothing presentation choice.
    from engine import classify_regime_from_gex
    pre_regime = classify_regime_from_gex(
        net_gex=net_gex,
        percentile=gex_percentile,
        spot=spot,
        flip=raw_flip,
        atr_1d=atr_1d,
    )

    # Find the immediately prior session's regime from daily_runs.
    prior_regime = None
    try:
        import sqlite3 as _sq2
        _conn = _sq2.connect(str(Path(__file__).parent.parent / "data.db"))
        _row = _conn.execute(
            "SELECT regime FROM daily_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        _conn.close()
        if _row:
            prior_regime = _row[0]
    except Exception as _e:
        print(f"[fetch_data] Prior regime lookup failed (non-fatal): {_e}", file=sys.stderr)

    smoothed_flip, ema_was_reset = compute_smoothed_flip(
        gamma_flip, history,
        current_regime=pre_regime,
        prior_regime=prior_regime,
    )
    if ema_was_reset:
        print(
            f"[fetch_data] EMA RESET on regime rotation: "
            f"prior={prior_regime} → current={pre_regime}. "
            f"smoothed_flip set to raw_flip={raw_flip:.4f}."
        )
    flip_mig = compute_flip_migration(gamma_flip, history)

    if market_closed:
        # Suppress flip migration — it's noise from expiry roll-off
        flip_mig = {"vs_yesterday": None, "vs_5d": None}
        print(f"[fetch_data] Flip migration suppressed (market closed)")

    vanna_percentile = compute_vanna_percentile(net_vanna, history) if net_vanna is not None else None
    vanna_regime = ("positive" if net_vanna > 0 else "negative" if net_vanna < 0 else "neutral") if net_vanna is not None else None
    if net_vanna is not None:
        print(f"[fetch_data] Vanna percentile={vanna_percentile:.1f}%, regime={vanna_regime}")

    print(f"[fetch_data] gex_percentile={gex_percentile:.1f}% (history len={len(history)})")
    print(f"[fetch_data] Raw flip: {gamma_flip:.4f}, Smoothed flip: {smoothed_flip:.4f}")
    if flip_mig["vs_yesterday"] is not None:
        print(f"[fetch_data] Flip migration: {flip_mig['vs_yesterday']:+.4f} vs yesterday")
    if flip_mig["vs_5d"] is not None:
        print(f"[fetch_data] Flip migration: {flip_mig['vs_5d']:+.4f} vs 5-day")

    # 4. Run engine (using smoothed flip for band computation)
    print("[fetch_data] Running engine...")
    try:
        recs = run_engine(
            spot=spot,
            gamma_flip=smoothed_flip,  # Use smoothed flip for bands
            atr_1d=atr_1d,
            net_gex=net_gex,
            gex_percentile=gex_percentile,
            gex_strikes_raw=gex_strikes_raw,
            puts_raw=puts_raw,
            calls_raw=calls_raw,
            iv_rank=iv_rank_val if iv_rank_val is not None else 25.0,
            iv_percentile=iv_percentile_val if iv_percentile_val is not None else 25.0,
            basic_mnav=basic_mnav or 0.72,
            bitcoin_yield_pct=btc_yield_ytd or 13.8,
            asst_drawdown_90d=asst_drawdown_90d or 0.0,
            btc_mvrv=btc_mvrv,
            btc_weekly_rsi=btc_weekly_rsi,
        )
        # Flag if IV Rank is using placeholder
        if iv_rank_val is None:
            recs.setdefault("notes", []).append("IV_RANK_MISSING: using default 25.0. IV metrics unavailable this session.")
        else:
            iv_method = iv_data.get("iv_rank_method", "RV") if iv_data else "RV"
            recs.setdefault("notes", []).append(f"IV metrics (method={iv_method}): current={current_iv:.1f}% rank={iv_rank_val:.1f} pct={iv_percentile_val:.1f}")
        status = "ok"
    except Exception as e:
        print(f"[fetch_data] Engine error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Append flip migration to engine notes
    if flip_mig["vs_yesterday"] is not None:
        recs.setdefault("notes", []).append(f"Flip migration: {flip_mig['vs_yesterday']:+.4f} vs yesterday.")
    if flip_mig["vs_5d"] is not None:
        recs.setdefault("notes", []).append(f"Flip migration: {flip_mig['vs_5d']:+.4f} vs 5-day.")

    # Compute LEAP entry percentile
    leap_band = recs.get("leapcore_band") or (0.0, 0.0)
    leap_pct = compute_leap_entry_percentile(spot, leap_band[0], leap_band[1], history)
    if leap_pct is not None:
        zone_desc = "above band (not in zone)" if leap_pct == 0 else "below band (deep convexity)" if leap_pct >= 100 else f"{leap_pct:.0f}% attractiveness"
        recs.setdefault("notes", []).append(f"LEAP entry percentile: {leap_pct:.1f}% — {zone_desc}.")
        print(f"[fetch_data] LEAP entry percentile: {leap_pct:.1f}%")

    risk_zone = recs.get("risk_zone", "GREEN")
    btc_cycle_zone = recs.get("btc_cycle_zone")
    action_banner = recs.get("action_banner")
    csp_allowed = recs.get("csp_allowed")
    leap_add_allowed = recs.get("leap_add_allowed")
    leap_add_size = recs.get("leap_add_size")
    pmcc_allowed = recs.get("pmcc_allowed")
    print(f"[fetch_data] Regime: {recs.get('regime')}  CSPs: {len(recs.get('cspcandidates', []))}  LEAP core: {len(recs.get('leapcore', []))}  Risk zone: {risk_zone}")
    print(f"[fetch_data] BTC cycle: {btc_cycle_zone}  Action: {action_banner}  CSP={csp_allowed}  LEAP={leap_add_allowed}({leap_add_size})  PMCC={pmcc_allowed}")

    # 4b. Generate contract suggestions (uses SteadyAPI chain data from step 2c)
    csp_suggestion = None
    leap_suggestion = None
    try:
        from suggestions import suggest_csp, suggest_leap
        csp_suggestion = suggest_csp(
            spot=spot,
            puts=steady_puts,
            gamma_flip=smoothed_flip,
            net_gex=net_gex,
            gex_percentile=gex_percentile,
            regime=recs.get("regime", "unknown"),
            csp_band_low=recs.get("csp_band", (0, 0))[0] if isinstance(recs.get("csp_band"), tuple) else recs.get("csp_band_low", 0),
            csp_band_high=recs.get("csp_band", (0, 0))[1] if isinstance(recs.get("csp_band"), tuple) else recs.get("csp_band_high", 0),
            iv_regime=recs.get("iv_regime", ""),
            iv_rank=iv_rank_val,
            btc_price=btc_price,
            btc_weekly_rsi=btc_weekly_rsi,
            market_closed=market_closed,
            portfolio_value=None,
            csp_collateral_deployed=None,
            iv_band=compute_iv_band(iv_percentile_val),
        )
        print(f"[fetch_data] CSP suggestion: {csp_suggestion.get('status', 'error')}")
    except Exception as e:
        print(f"[fetch_data] CSP suggestion failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    try:
        from suggestions import suggest_leap as _suggest_leap
        leap_suggestion = _suggest_leap(
            spot=spot,
            calls=steady_calls,
            regime=recs.get("regime", "unknown"),
            iv_rank=iv_rank_val,
            iv_percentile=iv_percentile_val,
            iv_regime=recs.get("iv_regime", ""),
            leap_band_low=recs.get("leapcore_band", (0, 0))[0] if isinstance(recs.get("leapcore_band"), tuple) else 0,
            leap_band_high=recs.get("leapcore_band", (0, 0))[1] if isinstance(recs.get("leapcore_band"), tuple) else 0,
            nav_per_share=nav_per_share,
            btc_price=btc_price,
            btc_weekly_rsi=btc_weekly_rsi,
            market_closed=market_closed,
            portfolio_value=None,
            leap_premium_at_risk=None,
        )
        print(f"[fetch_data] LEAP suggestion: {leap_suggestion.get('status', 'error')}")
    except Exception as e:
        print(f"[fetch_data] LEAP suggestion failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # PMCC suggestion
    pmcc_suggestion = None
    try:
        from suggestions import suggest_pmcc
        # Short calls = near-term calls (14-45 DTE), Long = LEAP calls (>500 DTE)
        short_calls = [c for c in steady_calls if not c.get("is_leap", True)]
        leap_only = [c for c in steady_calls if c.get("is_leap", False)]
        pmcc_suggestion = suggest_pmcc(
            spot=spot,
            calls=short_calls,
            leap_calls=leap_only,
            regime=recs.get("regime", "unknown"),
            pmcc_allowed=pmcc_allowed,
            net_gex=net_gex,
            gex_percentile=gex_percentile,
            market_closed=market_closed,
        )
        print(f"[fetch_data] PMCC suggestion: {pmcc_suggestion.get('status', 'error')}")
    except Exception as e:
        print(f"[fetch_data] PMCC suggestion failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Enrichment fields (trivial derivations)
    enrichment = {}
    try:
        from suggestions import compute_enrichment_fields
        pos_magnets_parsed = json.loads(recs.get("pos_magnets", "[]")) if isinstance(recs.get("pos_magnets"), str) else recs.get("pos_magnets", [])
        enrichment = compute_enrichment_fields(
            spot=spot,
            ev_mnav=ev_mnav_val,
            btc_holdings=st_data.get("btc_holdings") if st_data else None,
            total_shares=st_data.get("total_shares") if st_data else None,
            diluted_shares=st_data.get("diluted_shares") if st_data else None,
            csp_band_low=recs.get("csp_band", (0, 0))[0] if isinstance(recs.get("csp_band"), tuple) else recs.get("csp_band_low", 0),
            csp_band_high=recs.get("csp_band", (0, 0))[1] if isinstance(recs.get("csp_band"), tuple) else recs.get("csp_band_high", 0),
            gamma_flip=smoothed_flip,
            pos_magnets=pos_magnets_parsed,
        )
        print(f"[fetch_data] Enrichment: mnav_discount={enrichment.get('mnav_discount')}, btc/sh={enrichment.get('btc_per_share_basic')}, csp_d2b={enrichment.get('csp_delta_to_band')}")
    except Exception as e:
        print(f"[fetch_data] Enrichment computation failed: {e}", file=sys.stderr)

    # SteadyAPI-derived fields
    iv_skew_25d = chain_data.get("iv_skew_25d") if chain_data else None
    put_call_oi_ratio = chain_data.get("put_call_oi_ratio") if chain_data else None

    # 4c. Stochastic Layer Phase 1 — dual-mode vehicle ranking
    stochastic_output = None
    stochastic_error = None
    try:
        from stochastics import compute_stochastic_output, load_history_from_db
        db_path = str(Path(__file__).parent.parent / "data.db")
        # Always use UTC date as a date object — avoids two prior bugs:
        #   (a) AM path never reaches the market-closed branch where `today`
        #       was conditionally assigned (UnboundLocalError).
        #   (b) MID/PM path assigned `today` as a string via strftime, then
        #       this code calls .isoformat() on it (AttributeError).
        stoch_today = datetime.now(timezone.utc).date()
        # Compute iv_band here so the stochastic layer receives the canonical
        # 5-state index (0..4), not the legacy iv_regime string. Post-Step-3
        # cutover (v1.5), stochastics.bucket_state reads iv_band as the IV
        # dimension; passing only iv_regime leaves iv_bucket null and the
        # state key collapses to '...|?|...' which under-matches against history.
        # Use iv_percentile_val (the resolved IV percentile from this fetch),
        # not recs.get("iv_percentile") — the engine doesn't echo iv_percentile
        # back into its recommendation dict, so the .get() returns None and the
        # band collapses to NULL. iv_percentile_val is the canonical source.
        _iv_pct_for_stoch = iv_percentile_val
        _iv_band_for_stoch = compute_iv_band(_iv_pct_for_stoch)
        current_run_for_stoch = {
            "date": stoch_today.isoformat(),
            "session": args.session,
            "regime": recs.get("regime"),
            "gex_percentile": gex_percentile,
            "iv_regime": recs.get("iv_regime"),
            "iv_band": _iv_band_for_stoch,
            "iv_percentile": _iv_pct_for_stoch,
            "btc_cycle_zone": btc_cycle_zone,
            "action_banner": action_banner,
            "csp_allowed": csp_allowed,
            "pmcc_allowed": pmcc_allowed,
            "leap_add_allowed": leap_add_allowed,
            "asst_drawdown_90d": asst_drawdown_90d,
            "risk_zone": risk_zone,
            "vanna_regime": vanna_regime,
            "mnav_discount": enrichment.get("mnav_discount"),
            "btc_gex_secondary_confirm": btc_gex_secondary_confirm,
            "iv_skew_25d": iv_skew_25d,
            "put_call_oi_ratio": put_call_oi_ratio,
        }
        history = load_history_from_db(db_path)
        stochastic_output = compute_stochastic_output(current_run_for_stoch, history)
        sg = stochastic_output["gated"]
        su = stochastic_output["ungated"]
        print(f"[fetch_data] Stochastic: gated={sg['top_vehicle']} ungated={su['top_vehicle']} conf={sg['confidence_label']} evidence={sg['evidence_count']}")
    except Exception as e:
        stochastic_error = f"{type(e).__name__}: {e}"
        # Persist the failure loudly — stdout + stderr + a tracking file
        err_line = f"[fetch_data] Stochastic computation FAILED (non-fatal): {stochastic_error}"
        print(err_line)                     # stdout so cron sees it
        print(err_line, file=sys.stderr)    # stderr as well
        traceback.print_exc(file=sys.stderr)
        # Append to a persistent failure log so we can diagnose after the fact
        try:
            err_log = Path(__file__).parent.parent / "stochastic_failures.log"
            with err_log.open("a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} session={args.session} "
                        f"error={stochastic_error}\n")
                f.write(traceback.format_exc())
                f.write("\n---\n")
        except Exception:
            pass

    # 5. Build and POST payload (store smoothed flip as the gamma_flip value;
    #    raw_flip is also persisted for audit transparency, v2.4.1+)
    payload = build_run_payload(
        spot=spot,
        gamma_flip=smoothed_flip,
        raw_flip=raw_flip,
        atr_1d=atr_1d,
        net_gex=net_gex,
        gex_percentile=gex_percentile,
        recs=recs,
        status=status,
        session=args.session,
        btc_price=btc_price,
        asst_volume=asst_volume,
        asst_market_cap=asst_market_cap,
        sata_price=sata_price,
        sata_volume=sata_volume,
        sata_options_oi=sata_options_oi,
        leap_entry_percentile=leap_pct,
        basic_mnav=basic_mnav,
        btc_yield_ytd=btc_yield_ytd,
        btc_holdings=btc_holdings,
        btc_nav=btc_nav,
        nav_per_share=nav_per_share,
        avg_cost_per_btc=avg_cost_per_btc,
        total_shares=total_shares,
        diluted_shares=diluted_shares,
        cash_balance=cash_balance,
        debt_outstanding=debt_outstanding,
        btc_weekly_rsi=btc_weekly_rsi,
        asst_drawdown_90d=asst_drawdown_90d,
        asst_90d_high=asst_90d_high,
        risk_zone=risk_zone,
        current_iv=current_iv,
        iv_rank=iv_rank_val,
        iv_percentile=iv_percentile_val,
        iv_rank_method=iv_data.get("iv_rank_method") if iv_data else None,
        ev_mnav=ev_mnav_val,
        btc_mvrv=btc_mvrv,
        btc_realized_price=btc_realized_price,
        market_closed=market_closed,
        btc_gex_secondary_confirm=btc_gex_secondary_confirm,
        btc_taker_buy_sell_ratio=btc_taker_buy_sell_ratio,
        btc_funding_rate=btc_funding_rate,
        btc_mvrv_zscore=btc_mvrv_zscore,
        btc_puell_multiple=btc_puell_multiple,
        btc_nupl=btc_nupl,
        btc_reserve_risk=btc_reserve_risk,
        btc_pi_cycle_signal=btc_pi_cycle_signal,
        btc_liq_total_usd=btc_liq_total_usd,
        btc_liq_long_pct=btc_liq_long_pct,
        btc_fear_greed=btc_fear_greed,
        btc_cycle_zone=btc_cycle_zone,
        csp_allowed=csp_allowed,
        leap_add_allowed=leap_add_allowed,
        leap_add_size=leap_add_size,
        pmcc_allowed=pmcc_allowed,
        action_banner=action_banner,
        net_vanna=net_vanna,
        vanna_percentile=vanna_percentile,
        vanna_regime=vanna_regime,
        csp_suggestion_json=json.dumps(csp_suggestion) if csp_suggestion else None,
        leap_suggestion_json=json.dumps(leap_suggestion) if leap_suggestion else None,
        pmcc_suggestion_json=json.dumps(pmcc_suggestion) if pmcc_suggestion else None,
        stochastic_output_json=json.dumps(stochastic_output) if stochastic_output else None,
        mnav_discount=enrichment.get("mnav_discount"),
        btc_per_share_basic=enrichment.get("btc_per_share_basic"),
        btc_per_share_diluted=enrichment.get("btc_per_share_diluted"),
        csp_delta_to_band=enrichment.get("csp_delta_to_band"),
        csp_magnet_proximity=enrichment.get("csp_magnet_proximity"),
        iv_skew_25d=iv_skew_25d,
        put_call_oi_ratio=put_call_oi_ratio,
        chain_data=chain_data,
    )

    # ── Persistence ─────────────────────────────────────────────
    # v2: direct_persist is the only persistence path. No Express fallback.
    # The (date, session) unique index makes this idempotent on retry.
    run_id: Optional[int] = None
    try:
        from engine.persist import persist_run
        run_id = persist_run(payload)
        print(f"[fetch_data] Success — direct DB UPSERT, run id={run_id}")
    except Exception as e_direct:
        print(f"[fetch_data] FATAL: direct persist failed: {e_direct}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise  # hard fail; no silent fallback

    # Sanity: run_id must be set if we reach here.
    if run_id is None:
        raise RuntimeError("persistence succeeded but run_id is None")

    # Persist-only mode for unit tests: short-circuit before selector logging.
    if getattr(args, "persist_only", False):
        print("[fetch_data] --persist-only set; skipping selector evaluate/log.")
        print("[fetch_data] Done.")
        return

    # Stochastic logging + retry. Same logic as the legacy POST path, just now
    # branching off a successfully-persisted run_id from either write path.
    try:
        if stochastic_output and run_id:
            try:
                from stochastics import log_stochastic_output
                db_path = str(Path(__file__).parent.parent / "data.db")
                log_stochastic_output(db_path, run_id, stochastic_output)
                print(f"[fetch_data] Stochastic output logged for run {run_id}")
            except Exception as e:
                print(f"[fetch_data] Stochastic logging failed (non-fatal): {e}", file=sys.stderr)

        # Safety net: if stochastic computation failed in step 4c, retry ONCE now.
        # The run row is already persisted; we read it back, recompute from the DB
        # record, and patch both daily_runs.stochastic_output_json and stochastic_log.
        elif run_id and stochastic_error is not None:
            print(f"[fetch_data] Retrying stochastic computation for run {run_id} after primary failure...")
            try:
                import sqlite3
                from stochastics import (
                    compute_stochastic_output,
                    log_stochastic_output,
                    load_history_from_db,
                )
                db_path = str(Path(__file__).parent.parent / "data.db")
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                persisted = dict(conn.execute("SELECT * FROM daily_runs WHERE id = ?", (run_id,)).fetchone())
                conn.close()
                history = load_history_from_db(db_path)
                # Exclude current run from history for matching purposes
                history = [h for h in history if h.get("id") != run_id]
                retry_output = compute_stochastic_output(persisted, history)
                # Patch daily_runs
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "UPDATE daily_runs SET stochastic_output_json = ? WHERE id = ?",
                    (json.dumps(retry_output), run_id),
                )
                conn.commit()
                conn.close()
                # Log it
                log_stochastic_output(db_path, run_id, retry_output)
                print(f"[fetch_data] Stochastic retry SUCCESS for run {run_id}: "
                      f"gated={retry_output['gated']['top_vehicle']} "
                      f"ungated={retry_output['ungated']['top_vehicle']} "
                      f"conf={retry_output['gated']['confidence_label']} "
                      f"evidence={retry_output['gated']['evidence_count']}")
            except Exception as e:
                print(f"[fetch_data] Stochastic retry ALSO FAILED: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
    except Exception as e:
        # Stochastic logging is non-fatal — the run row is already persisted.
        # Original code path exited 1 here because it conflated POST and
        # stochastic-logging failures; the new direct-persist path separates
        # them, so logging failure no longer kills the cron.
        print(f"[fetch_data] Stochastic logging block raised: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # 6. Update GEX history (enriched format with date, net_gex, flip)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history.append({"date": date_str, "net_gex": net_gex, "flip": gamma_flip, "net_vanna": net_vanna})
    save_gex_history(history)
    print(f"[fetch_data] GEX history updated ({len(history)} entries).")
    print(f"[fetch_data] Raw flip stored: {gamma_flip:.4f}, Smoothed used: {smoothed_flip:.4f}")

    # 7. Auto-log Selector evaluation. The Selector engine runs deterministically
    #    from the just-persisted run + known_positions + master export, so we can
    #    invoke /api/selector/evaluate and append the output to the log. This
    #    closes the historical gap where decisions were only saved on manual
    #    "Save to log" clicks. Non-fatal if it fails — we never want a Selector
    #    issue to block the core fetch from succeeding.
    if run_id:
        try:
            print("[fetch_data] Auto-logging Selector evaluation...")
            # Idempotency: skip if an auto-logged entry already exists for this
            # (date, session). Manual "Save to log" entries are preserved
            # separately. This makes cron retries safe.
            existing = requests.get(
                f"{args.api_url}/api/selector/log",
                params={"limit": 500},
                timeout=10,
            )
            already_logged = False
            if existing.ok:
                for entry in (existing.json() or {}).get("entries", []):
                    if (
                        entry.get("date") == payload.get("date")
                        and entry.get("session") == payload.get("session")
                        and entry.get("auto_logged") is True
                    ):
                        already_logged = True
                        break
            if already_logged:
                print(f"[fetch_data] Selector already auto-logged for {payload.get('date')}/{payload.get('session')} — skipping")
            else:
                eval_resp = requests.post(
                    f"{args.api_url}/api/selector/evaluate",
                    json={"date": payload.get("date"), "session": payload.get("session")},
                    timeout=15,
                )
                eval_resp.raise_for_status()
                evaluation = eval_resp.json()
                # Stamp auto-log flags so quant downstream can distinguish from
                # backfill rows (different prior_position semantics) and from
                # manual "Save to log" rows (different intent).
                log_payload = {
                    **evaluation,
                    "backfill": False,
                    "auto_logged": True,
                }
                log_resp = requests.post(
                    f"{args.api_url}/api/selector/log",
                    json=log_payload,
                    timeout=15,
                )
                log_resp.raise_for_status()
                print(f"[fetch_data] Selector auto-log SUCCESS: "
                      f"cohort={evaluation.get('cohort_id')} "
                      f"posture={evaluation.get('overall_posture')}")
        except Exception as e:
            print(f"[fetch_data] Selector auto-log failed (non-fatal): {e}", file=sys.stderr)

    # 8. Regenerate static snapshot.json so the deployed dashboard can render
    #    last-good data even when the Express server (or sandbox) is dead.
    #    The frontend reads /snapshot.json on boot before any API call, seeds
    #    React Query cache, and shows the StaleBanner with the snapshot age.
    #    Non-fatal — a snapshot failure must never kill an otherwise-successful
    #    fetch. See server/generate_snapshot.py for the schema and the
    #    queryClient bootstrap code that consumes this file.
    if run_id:
        try:
            from generate_snapshot import main as generate_snapshot_main
            rc = generate_snapshot_main()
            if rc == 0:
                print("[fetch_data] Snapshot regenerated for static-bundle delivery.")
            else:
                print(f"[fetch_data] Snapshot generator returned rc={rc} (non-fatal)", file=sys.stderr)
        except Exception as e:
            print(f"[fetch_data] Snapshot regeneration failed (non-fatal): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print("[fetch_data] Done.")


if __name__ == "__main__":
    main()

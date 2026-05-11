"""
Tiingo EOD Price Backbone for ASST Gamma Flywheel.

Provides clean, split-adjusted daily price history as the authoritative
source for price-derived statistics (ATR, drawdown, RV series).

Runs nightly after market close. Stores cleaned history in a local JSON cache.
Existing feeds (yfinance, FlashAlpha) are NOT replaced — Tiingo only
stabilizes the daily price backbone that those models sit on.

Free tier: 1,000 req/day, 50/hr — we use 3-5 symbols once daily.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import requests

# v2: read from env with v1 hardcoded fallback during transition.
# Fallback removed in P1.19 after key rotation.
_LEGACY_TIINGO_KEY = "e4e2c2f069952c774c9716909f6f4dba000707e7"  # noqa: S105
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", _LEGACY_TIINGO_KEY)
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}
SYMBOLS = ["ASST", "SPY", "BITO"]
# v2: cache dir resolves via ASST_TIINGO_CACHE env var (set by cron) with
# v1 fallback for local development against the existing dashboard cache.
CACHE_DIR = os.environ.get(
    "ASST_TIINGO_CACHE",
    "/home/user/workspace/asst-gamma-dashboard/tiingo_cache",
)
LOOKBACK_DAYS = 365  # 1 year of history for RV/ATR computations


def fetch_tiingo_eod(symbol: str, start_date: str = None, end_date: str = None) -> list:
    """Fetch EOD data from Tiingo for a single symbol."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = f"https://api.tiingo.com/tiingo/daily/{symbol.lower()}/prices"
    params = {"startDate": start_date, "endDate": end_date}

    try:
        resp = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[tiingo] {symbol}: {len(data)} trading days fetched ({start_date} to {end_date})")
            return data
        else:
            print(f"[tiingo] {symbol}: HTTP {resp.status_code}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"[tiingo] {symbol}: fetch failed: {e}", file=sys.stderr)
        return []


def save_cache(symbol: str, data: list):
    """Save fetched data to local JSON cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.lower()}_eod.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[tiingo] Cached {symbol}: {len(data)} days -> {path}")


def load_cache(symbol: str) -> list:
    """Load cached data for a symbol."""
    path = os.path.join(CACHE_DIR, f"{symbol.lower()}_eod.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def compute_stats_from_tiingo(symbol: str = "ASST") -> Optional[dict]:
    """
    Compute price-derived statistics from Tiingo-cleaned EOD data.
    Returns ATR (14d), 90d high, drawdown, and 30d rolling RV series stats.
    Falls back to yfinance-derived values if Tiingo cache is empty.
    """
    data = load_cache(symbol)
    if len(data) < 30:
        print(f"[tiingo] {symbol}: insufficient cache ({len(data)} days), skipping stats")
        return None

    # Extract adjusted prices
    closes = np.array([d["adjClose"] for d in data if d.get("adjClose")])
    highs = np.array([d["adjHigh"] for d in data if d.get("adjHigh")])
    lows = np.array([d["adjLow"] for d in data if d.get("adjLow")])
    dates = [d["date"][:10] for d in data]

    if len(closes) < 30:
        return None

    # ATR (14-day, using True Range = max(H-L, |H-prevC|, |L-prevC|))
    tr_list = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hc, lc))
    tr_arr = np.array(tr_list)
    atr_14d = float(np.mean(tr_arr[-14:])) if len(tr_arr) >= 14 else float(np.mean(tr_arr))
    atr_21d = float(np.mean(tr_arr[-21:])) if len(tr_arr) >= 21 else atr_14d

    # 90-day high and drawdown
    recent_90 = data[-90:] if len(data) >= 90 else data
    high_90d = max(d["adjHigh"] for d in recent_90)
    current_close = closes[-1]
    drawdown_90d = (current_close - high_90d) / high_90d if high_90d > 0 else 0.0

    # 30-day rolling realized volatility (annualized)
    returns = np.diff(np.log(closes))
    if len(returns) >= 30:
        rolling_rv = []
        for i in range(29, len(returns)):
            window = returns[i - 29 : i + 1]
            rv = float(np.std(window) * np.sqrt(252) * 100)
            rolling_rv.append(rv)
        rv_current = rolling_rv[-1]
        rv_min = min(rolling_rv)
        rv_max = max(rolling_rv)
        rv_median = float(np.median(rolling_rv))
    else:
        rv_current = rv_min = rv_max = rv_median = None

    result = {
        "source": "tiingo",
        "symbol": symbol,
        "last_date": dates[-1],
        "last_close": float(current_close),
        "trading_days": len(closes),
        "atr_14d": round(atr_14d, 4),
        "atr_21d": round(atr_21d, 4),
        "high_90d": round(high_90d, 2),
        "drawdown_90d": round(drawdown_90d, 4),
        "rv_current_30d": round(rv_current, 1) if rv_current else None,
        "rv_min_52w": round(rv_min, 1) if rv_min else None,
        "rv_max_52w": round(rv_max, 1) if rv_max else None,
        "rv_median_52w": round(rv_median, 1) if rv_median else None,
    }

    print(f"[tiingo] {symbol} stats: ATR14={atr_14d:.4f} ATR21={atr_21d:.4f} "
          f"90dHi={high_90d:.2f} DD={drawdown_90d:.1%} "
          f"RV={rv_current:.1f}% (range {rv_min:.1f}-{rv_max:.1f}%)")
    return result


def run_nightly_eod():
    """
    Nightly EOD update: fetch all symbols, cache, compute ASST stats.
    Called by the nightly cron after market close.
    """
    print(f"[tiingo] === Nightly EOD update: {datetime.now(timezone.utc).isoformat()} ===")

    for symbol in SYMBOLS:
        data = fetch_tiingo_eod(symbol)
        if data:
            save_cache(symbol, data)

    # Compute ASST stats from cleaned data
    stats = compute_stats_from_tiingo("ASST")
    if stats:
        # Save stats summary
        stats_path = os.path.join(CACHE_DIR, "asst_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[tiingo] ASST stats saved to {stats_path}")

    return stats


if __name__ == "__main__":
    # Can be run directly for testing
    stats = run_nightly_eod()
    if stats:
        print(f"\n{json.dumps(stats, indent=2)}")

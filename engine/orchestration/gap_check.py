#!/usr/bin/env python3
"""
Session-gap detection / self-healing helper.

Run before any of the regular fetch crons (MID, PM, keep-alive). For today's
date, checks which sessions are *required* by the current time and which are
already present in daily_runs. For each missing session, invokes fetch_data.py
to backfill it.

Idempotent because the (date, session) UNIQUE INDEX on daily_runs forces an
UPSERT; running this twice does not create duplicates.

Exit code 0 always (even if backfills fail). The point is best-effort
self-healing without ever blocking the caller.

Usage:
    python3 check_session_gaps.py [--api-url http://localhost:5000]

Output: emits stdout lines like:
    [gap-check] AM present, MID missing, PM not yet required
    [gap-check] Backfilling MID...
    [gap-check] Backfill MID completed (run id=129)

Schedule context:
    AM   should be present after 14:00 UTC (10:00 ET)
    MID  should be present after 16:00 UTC (12:00 ET)
    PM   should be present after 19:30 UTC (15:30 ET)
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# v2: env-var override with v1 fallback.
DB_PATH = os.environ.get(
    "ASST_DB_PATH",
    "/home/user/workspace/asst-gamma-dashboard/data.db",
)
# v2: fetch is invoked via python -m engine.fetch, not as a script file.
FETCH_SCRIPT = None  # legacy field; gap_check now calls engine.fetch.main()

# Each session: (name, threshold_hour_utc, threshold_minute_utc)
# A session is "required" iff current UTC time is at or past the threshold.
SESSION_THRESHOLDS = [
    ("AM",  14, 0),   # 10:00 ET
    ("MID", 16, 0),   # 12:00 ET
    ("PM",  19, 30),  # 15:30 ET
]


def get_today_sessions_present(db_path: Path, today_iso: str) -> set:
    """Return the set of sessions already in daily_runs for today."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT DISTINCT session FROM daily_runs WHERE date = ?",
        (today_iso,),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def required_sessions_now(now_utc: datetime) -> list:
    """Return the list of sessions that should be present given current UTC time."""
    out = []
    for name, hh, mm in SESSION_THRESHOLDS:
        threshold = now_utc.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if now_utc >= threshold:
            out.append(name)
    return out


def is_weekday(dt: datetime) -> bool:
    """Mon=0..Sun=6. Skip weekends entirely \u2014 nothing to backfill."""
    return dt.weekday() < 5


def run_backfill(session: str, api_url: str) -> tuple:
    """Invoke fetch_data.py for the given session. Returns (exit_code, tail_of_output)."""
    cmd = [
        "python3", str(FETCH_SCRIPT),
        "--api-url", api_url,
        "--session", session,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent),
        )
        tail = (result.stdout or "").strip().split("\n")[-3:]
        return result.returncode, " | ".join(tail)
    except subprocess.TimeoutExpired:
        return 124, "timeout after 120s"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:5000")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps but don't actually run backfills")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    today_iso = now.strftime("%Y-%m-%d")

    if not is_weekday(now):
        print(f"[gap-check] {today_iso} is a weekend; skipping.")
        return 0

    required = required_sessions_now(now)
    if not required:
        print(f"[gap-check] No sessions required yet at {now.strftime('%H:%M UTC')}.")
        return 0

    present = get_today_sessions_present(DB_PATH, today_iso)
    missing = [s for s in required if s not in present]

    summary = ", ".join(
        f"{s}={'present' if s in present else 'MISSING'}"
        for s in [name for name, *_ in SESSION_THRESHOLDS]
        if s in required
    )
    print(f"[gap-check] {today_iso} status: {summary}")

    if not missing:
        print(f"[gap-check] All required sessions present.")
        return 0

    if args.dry_run:
        print(f"[gap-check] DRY RUN: would backfill {missing}")
        return 0

    for session in missing:
        print(f"[gap-check] Backfilling {session}...")
        rc, tail = run_backfill(session, args.api_url)
        if rc == 0:
            print(f"[gap-check] Backfill {session} OK \u2014 {tail}")
        else:
            print(f"[gap-check] Backfill {session} FAILED (exit={rc}) \u2014 {tail}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

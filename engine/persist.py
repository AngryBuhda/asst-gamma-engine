"""
engine.persist — write a fetch payload directly to data.db.

v2 origin:
  This module is the v2 incarnation of v1's `direct_persist`. In v1 it was
  introduced as a fallback when the Express server's POST path was
  unreliable; in v2 it is the ONLY persistence path — there is no Express
  server. The semantics are unchanged: an idempotent UPSERT on
  (date, session).

Semantics:
  - INSERT...ON CONFLICT(date,session) DO UPDATE on daily_runs
  - The unique index ix_daily_runs_date_session makes retries idempotent
  - dict/list fields are JSON-serialized; bools become 0/1; None stays NULL
  - Returns the row id so stochastic logging can attach to it

Database path resolution (v2):
  Read from the ASST_DB_PATH environment variable. Falls back to the
  v1-compatible path (../../data.db relative to this file) for backward
  compatibility during the transition. After P1.18 (data.db git-push from
  fetch cron), the env var becomes the authoritative source.
"""
from __future__ import annotations
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

# v2 path resolution: env var first, then v1-compatible default.
# v1 default assumed server/direct_persist.py → ../data.db. In the engine
# repo layout, this module lives at engine/engine/persist.py and there is
# no data.db inside the repo. For local development against the v1 file,
# the operator can set ASST_DB_PATH to point at the existing dashboard's
# data.db. In production cron, ASST_DB_PATH is always set explicitly.
_DEFAULT_V1_PATH = Path("/home/user/workspace/asst-gamma-dashboard/data.db")
DB_PATH = os.environ.get("ASST_DB_PATH", str(_DEFAULT_V1_PATH))


def _serialize_value(v: Any) -> Any:
    """Match Drizzle's serialization for TEXT-typed columns that hold JSON.

    Drizzle stores dicts/lists as JSON text. SQLite has no native JSON type
    (it's just TEXT). We replicate by JSON-encoding any dict/list value;
    primitives pass through. None stays None (NULL in SQL).
    """
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    if isinstance(v, bool):
        # SQLite has no bool — store as 0/1 INTEGER, matching Drizzle's boolean cast.
        return 1 if v else 0
    return v


def persist_run(
    payload: Dict[str, Any],
    db_path: str = DB_PATH,
) -> int:
    """Insert (or UPSERT update) a daily_runs row from a fetch_data payload.

    Returns the row id. Raises if the schema can't accept the payload's keys.

    The payload keys are intersected with the actual daily_runs columns;
    unknown keys are silently dropped (defensive — same forgiving behavior
    as the previous Zod path: extra keys are ignored, not an error).
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    conn = sqlite3.connect(db_path)
    try:
        # Discover the current schema. We intersect with payload keys so a
        # column rename / addition can never silently misfile data.
        cur = conn.cursor()
        cols_info = cur.execute("PRAGMA table_info(daily_runs)").fetchall()
        all_cols = [c[1] for c in cols_info]
        col_set = set(all_cols)

        # Build the column list & matching values from the payload, preserving
        # order so the placeholders line up.
        cols: list[str] = []
        vals: list[Any] = []
        for k, v in payload.items():
            if k in col_set and k != "id":  # never insert an explicit id
                cols.append(k)
                vals.append(_serialize_value(v))

        if not cols:
            raise ValueError(
                "payload had no keys matching daily_runs columns. "
                f"payload keys (first 10): {list(payload.keys())[:10]}"
            )

        # Defensive: enforce that (date, session) are present — the UPSERT
        # target requires them, and missing them would silently create
        # NULL-keyed rows.
        if "date" not in cols or "session" not in cols:
            raise ValueError(
                f"payload missing date or session (have: date={'date' in cols}, "
                f"session={'session' in cols})"
            )

        placeholders = ",".join("?" * len(cols))
        # UPDATE clause: every non-key column gets updated to the new value
        # via excluded.<col>. Exclude date+session from the SET clause itself
        # (those are the conflict key — never updated).
        update_cols = [c for c in cols if c not in ("date", "session")]
        update_clause = ", ".join(
            f"{c} = excluded.{c}" for c in update_cols
        )
        # If only date+session were provided (impossible in practice but
        # guard anyway), do nothing on conflict instead of writing an empty
        # SET clause.
        on_conflict = (
            f"ON CONFLICT(date, session) DO UPDATE SET {update_clause}"
            if update_clause
            else "ON CONFLICT(date, session) DO NOTHING"
        )
        sql = (
            f"INSERT INTO daily_runs ({', '.join(cols)}) VALUES ({placeholders}) "
            f"{on_conflict} "
            f"RETURNING id"
        )

        row = cur.execute(sql, vals).fetchone()
        if row is None:
            # DO NOTHING branch hit — fetch the existing row's id.
            row = cur.execute(
                "SELECT id FROM daily_runs WHERE date = ? AND session = ?",
                (payload["date"], payload["session"]),
            ).fetchone()
        conn.commit()
        if row is None or row[0] is None:
            raise RuntimeError(
                "persist_run failed to obtain a row id after UPSERT — "
                "schema or unique-constraint may be misconfigured"
            )
        return int(row[0])
    finally:
        conn.close()


def ping(db_path: str = DB_PATH) -> bool:
    """Quick check that the DB exists and daily_runs is present."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            n = conn.execute("SELECT COUNT(*) FROM daily_runs").fetchone()[0]
            return n >= 0
        finally:
            conn.close()
    except Exception:
        return False


if __name__ == "__main__":
    # Smoke test: print column count.
    conn = sqlite3.connect(DB_PATH)
    cols = conn.execute("PRAGMA table_info(daily_runs)").fetchall()
    conn.close()
    print(f"daily_runs columns: {len(cols)}")
    print(f"db_path: {DB_PATH}")

"""Unit tests for engine.persist \u2014 idempotent UPSERT semantics.

Uses an in-memory SQLite + the migrations/0001_initial.sql to validate
that persist_run produces the same row on retry (matching the v1
ix_daily_runs_date_session unique-index UPSERT behavior).
"""
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

MIGRATION_SQL = (
    Path(__file__).resolve().parent.parent / "migrations" / "0001_initial.sql"
).read_text()


@pytest.fixture
def fresh_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(MIGRATION_SQL)
    con.commit()
    con.close()
    monkeypatch.setenv("ASST_DB_PATH", path)
    # Force engine.persist to re-resolve DB_PATH from the env var
    import importlib

    import engine.persist
    importlib.reload(engine.persist)
    yield path
    os.unlink(path)


def _sample_payload():
    return {
        "date": "2026-05-11",
        "session": "PM",
        "symbol": "ASST",
        "runtime_utc": "2026-05-11T20:00:00+00:00",
        "spot": 17.93,
        "gamma_flip": 17.89,
        "atr_1d": 1.21,
        "net_gex": 427669.76,
        "gex_percentile": 54.47,
        "regime": "neutral",
        "csp_band_low": 17.29,
        "csp_band_high": 19.10,
        "leap_core_band_low": 14.86,
        "leap_core_band_high": 17.29,
        "pos_magnets": [{"strike": 18.0, "net_gex": 1000}],
        "neg_magnets": [],
    }


def test_persist_insert(fresh_db):
    from engine.persist import persist_run

    run_id = persist_run(_sample_payload())
    assert run_id is not None and run_id > 0
    # Verify row landed
    con = sqlite3.connect(fresh_db)
    row = con.execute(
        "SELECT date, session, spot FROM daily_runs WHERE id = ?", (run_id,)
    ).fetchone()
    con.close()
    assert row == ("2026-05-11", "PM", 17.93)


def test_persist_upsert_idempotent(fresh_db):
    """Retrying the same (date, session) writes UPDATE, not duplicate row."""
    from engine.persist import persist_run

    id1 = persist_run(_sample_payload())
    id2 = persist_run(_sample_payload())
    assert id1 == id2

    con = sqlite3.connect(fresh_db)
    n = con.execute("SELECT COUNT(*) FROM daily_runs").fetchone()[0]
    con.close()
    assert n == 1


def test_persist_upsert_updates_fields(fresh_db):
    """UPSERT should overwrite changed fields, not error."""
    from engine.persist import persist_run

    p = _sample_payload()
    id1 = persist_run(p)
    p["spot"] = 99.0  # later session correction
    id2 = persist_run(p)
    assert id1 == id2

    con = sqlite3.connect(fresh_db)
    spot = con.execute("SELECT spot FROM daily_runs WHERE id = ?", (id1,)).fetchone()[0]
    con.close()
    assert spot == 99.0


def test_persist_dict_serialization(fresh_db):
    """dict/list fields should be JSON-serialized into TEXT columns."""
    from engine.persist import persist_run

    p = _sample_payload()
    p["pos_magnets"] = [{"strike": 20.0}, {"strike": 18.0}]
    run_id = persist_run(p)

    con = sqlite3.connect(fresh_db)
    raw = con.execute("SELECT pos_magnets FROM daily_runs WHERE id = ?", (run_id,)).fetchone()[0]
    con.close()
    assert isinstance(raw, str)
    import json
    parsed = json.loads(raw)
    assert parsed == [{"strike": 20.0}, {"strike": 18.0}]


def test_persist_two_sessions_same_day(fresh_db):
    """(date, session) is the unique key \u2014 AM and PM same day are two rows."""
    from engine.persist import persist_run

    p_am = _sample_payload()
    p_am["session"] = "AM"
    p_pm = _sample_payload()
    p_pm["session"] = "PM"
    id_am = persist_run(p_am)
    id_pm = persist_run(p_pm)
    assert id_am != id_pm

    con = sqlite3.connect(fresh_db)
    n = con.execute("SELECT COUNT(*) FROM daily_runs WHERE date = ?", ("2026-05-11",)).fetchone()[0]
    con.close()
    assert n == 2

#!/usr/bin/env python3
"""
P1.8 golden-fixture input capture.

For each fixture (date, session), assemble the exact inputs the TypeScript
selector engine consumed when producing the golden output. This makes the
fixtures fully replayable: the Python port reads input.json + positions.json
+ master_research subset, runs evaluate(), and must produce output that
byte-matches output.golden.json.

What we capture per fixture:
  - run.json: the daily_runs row (full record)
  - recent_regimes.json: last 5 PM regimes prior to this fixture's date
  - positions.json: known_positions.json snapshot (note: we use the CURRENT
    file, which is the same one the live engine used for these fixtures'
    capture moment of 2026-05-11. This is a known limitation: historical
    positions are not replayed because the system never versioned them.)
  - master_research_subset.json: cohort rows + forward returns relevant
    to this fixture's cohort
  - vintage_anchors.json: derived from option_chain_snapshot + lookback days

Outputs colocated in this directory; one subdirectory per fixture.

Usage:
    python capture_inputs.py
"""
from __future__ import annotations
import json
import os
import sqlite3
import sys
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
DB_PATH = "/home/user/workspace/asst-gamma-dashboard/data.db"
POSITIONS_PATH = "/home/user/workspace/asst-gamma-dashboard/selector_state/known_positions.json"
MASTER_RESEARCH_PATH = "/home/user/workspace/master_research_export/asst_research_master_v1.5_full.csv"

FIXTURES = [
    ("2026-05-11", "PM"),
    ("2026-05-08", "PM"),
    ("2026-05-07", "PM"),
    ("2026-04-14", "PM"),
    ("2026-04-09", "PM"),
]


def fetch_run(con: sqlite3.Connection, date: str, session: str) -> dict:
    """Return the daily_runs row for (date, session) as a dict."""
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM daily_runs WHERE date = ? AND session = ?",
        (date, session),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No row for {date} {session}")
    return dict(row)


def fetch_recent_regimes(con: sqlite3.Connection, date: str) -> list[str]:
    """Last 5 PM regimes prior to `date`, oldest-first."""
    rows = con.execute(
        "SELECT regime FROM daily_runs WHERE session='PM' AND date<? "
        "ORDER BY date DESC LIMIT 5",
        (date,),
    ).fetchall()
    return list(reversed([r[0] for r in rows]))


def load_positions() -> dict:
    return json.loads(Path(POSITIONS_PATH).read_text())


def fetch_cohort_subset(date: str, cohort_id: str) -> dict:
    """Return master_research rows relevant to the fixture's cohort.

    The master research export is partitioned by cohort_id. We capture
    the rows for this cohort whose observation date < fixture date,
    so the Python engine sees the same historical evidence the TS
    engine had at evaluation time.
    """
    if not Path(MASTER_RESEARCH_PATH).exists():
        return {"available": False, "reason": "master_research_export not on disk"}
    import csv
    matches = []
    with open(MASTER_RESEARCH_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("cohort_id") != cohort_id:
                continue
            # Only past observations
            if row.get("date", "") >= date:
                continue
            matches.append(row)
    return {
        "available": True,
        "cohort_id": cohort_id,
        "n_rows": len(matches),
        "rows": matches,
    }


def capture_one(con: sqlite3.Connection, date: str, session: str) -> dict:
    fix_dir = HERE / f"{date}_{session}"
    fix_dir.mkdir(exist_ok=True)

    run = fetch_run(con, date, session)
    recent = fetch_recent_regimes(con, date)
    positions = load_positions()

    # Load the previously-captured golden output to extract cohort_id
    golden_path = HERE / f"{date}_{session}.golden.json"
    if not golden_path.exists():
        raise RuntimeError(f"golden output not captured yet for {date} {session}")
    golden = json.loads(golden_path.read_text())
    cohort_id = golden.get("cohort_id", "?")

    cohort_subset = fetch_cohort_subset(date, cohort_id)

    # Persist inputs into the fixture's directory
    (fix_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (fix_dir / "recent_regimes.json").write_text(json.dumps(recent, indent=2))
    (fix_dir / "positions.json").write_text(json.dumps(positions, indent=2))
    (fix_dir / "cohort_subset.json").write_text(json.dumps(cohort_subset, indent=2))
    # Move/copy the golden output into the fixture dir too for self-containment
    (fix_dir / "output.golden.json").write_text(golden_path.read_text())

    return {
        "fixture": f"{date}_{session}",
        "dir": str(fix_dir),
        "files_written": ["run.json", "recent_regimes.json", "positions.json",
                          "cohort_subset.json", "output.golden.json"],
        "summary": {
            "cohort_id": cohort_id,
            "posture": golden.get("overall_posture"),
            "regime": run.get("regime"),
            "iv_band": run.get("iv_band"),
            "risk_zone": run.get("risk_zone"),
            "has_chain": run.get("option_chain_snapshot_json") is not None,
            "recent_regimes": recent,
            "cohort_history_rows": cohort_subset.get("n_rows", 0),
        },
    }


def main() -> int:
    con = sqlite3.connect(DB_PATH)
    manifest = {"fixtures": []}
    for date, session in FIXTURES:
        try:
            result = capture_one(con, date, session)
            manifest["fixtures"].append(result)
            print(f"\u2713 captured {result['fixture']}: cohort={result['summary']['cohort_id']}, "
                  f"posture={result['summary']['posture']!r}, "
                  f"cohort_history={result['summary']['cohort_history_rows']} rows")
        except Exception as e:
            print(f"\u2717 FAILED {date}_{session}: {e}")
            return 1
    manifest_path = HERE / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
P1.8 parity test \u2014 byte-identical comparison of Python selector vs
golden TS outputs across every fixture in engine/tests/fixtures/selector_golden/.

Per docs/v2_planning/08_BUILD_PLAN.md \u00a73.3, the Python port is BLOCKED
from promotion until every fixture passes. Approved exceptions (e.g.,
timestamp_utc differs by inherent design) are documented here.

Run: pytest engine/tests/test_selector_parity.py -v
Or:  python engine/tests/test_selector_parity.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.selector import evaluate, build_edges_from_subset  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "selector_golden"
FIXTURES = [
    "2026-05-11_PM",
    "2026-05-08_PM",
    "2026-05-07_PM",
    "2026-04-14_PM",
    "2026-04-09_PM",
]

# Fields known to differ between TS and Python outputs by design.
# Removed from both sides before comparison.
INHERENT_DIFFS = ["timestamp_utc"]


def _strip_inherent(d):
    """Recursively remove keys in INHERENT_DIFFS from a dict structure."""
    if isinstance(d, dict):
        return {k: _strip_inherent(v) for k, v in d.items() if k not in INHERENT_DIFFS}
    if isinstance(d, list):
        return [_strip_inherent(x) for x in d]
    return d


def _diff_paths(a, b, path=""):
    """Yield path strings where a and b differ. Order-sensitive on lists."""
    if type(a) != type(b):
        yield f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__}) "\
              f"-- got={a!r} vs golden={b!r}"
        return
    if isinstance(a, dict):
        akeys = set(a.keys())
        bkeys = set(b.keys())
        for k in akeys - bkeys:
            yield f"{path}.{k}: only in Python ({a[k]!r})"
        for k in bkeys - akeys:
            yield f"{path}.{k}: only in golden ({b[k]!r})"
        for k in akeys & bkeys:
            yield from _diff_paths(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}: length mismatch ({len(a)} vs {len(b)})"
        for i, (ai, bi) in enumerate(zip(a, b)):
            yield from _diff_paths(ai, bi, f"{path}[{i}]")
    else:
        if a != b:
            yield f"{path}: {a!r} != {b!r}"


def _run_fixture(name: str):
    d = FIX_DIR / name
    run = json.loads((d / "run.json").read_text())
    # parse JSON-encoded fields back into objects so the Python engine can
    # access them as Python types (matches what Drizzle on the TS side
    # produces). The fields that need re-parsing are option_chain_snapshot_json
    # (kept as string in v1 storage), pos_magnets/neg_magnets, csp_candidates_json,
    # stochastic_output_json, csp/leap/pmcc_suggestion_json.
    # However, the engine reads option_chain_snapshot_json as a string and
    # parses it itself via parse_chain_snapshot. Other JSON-typed fields are
    # not consumed by the selector. So we pass `run` as-is.
    recent = json.loads((d / "recent_regimes.json").read_text())
    positions = json.loads((d / "positions.json").read_text())
    cohort_subset = json.loads((d / "cohort_subset.json").read_text())
    golden = json.loads((d / "output.golden.json").read_text())

    edges = build_edges_from_subset(cohort_subset)

    output = evaluate({
        "run": run,
        "recentRegimes": recent,
        "edges": edges,
        "positions": positions,
        "vintage_anchors": [],   # P1.8 baseline: no vintage anchors path yet
    })

    g = _strip_inherent(golden)
    p = _strip_inherent(output)
    diffs = list(_diff_paths(p, g, f"{name}"))
    return diffs, output, golden


import pytest  # noqa: E402


@pytest.mark.parametrize("fixture", FIXTURES)
def test_selector_byte_identical(fixture: str):
    """Each golden fixture must replay byte-identically against the TS engine."""
    diffs, _output, _golden = _run_fixture(fixture)
    if diffs:
        msg = f"{fixture}: {len(diffs)} differences\n" + "\n".join(
            f"  {d}" for d in diffs[:30]
        )
        if len(diffs) > 30:
            msg += f"\n  ... and {len(diffs) - 30} more"
        pytest.fail(msg)


def main() -> int:
    total = 0
    passed = 0
    for fx in FIXTURES:
        total += 1
        diffs, output, golden = _run_fixture(fx)
        if not diffs:
            print(f"\u2713 {fx}: byte-identical")
            passed += 1
        else:
            print(f"\u2717 {fx}: {len(diffs)} differences")
            for d in diffs[:30]:
                print(f"    {d}")
            if len(diffs) > 30:
                print(f"    ... and {len(diffs) - 30} more")
    print(f"\n{passed}/{total} fixtures pass")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

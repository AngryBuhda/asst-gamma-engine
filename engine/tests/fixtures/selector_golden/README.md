# Selector engine golden fixtures (P1.8)

## Purpose

These fixtures define the byte-identical contract that the Python port of
the TypeScript `selectorEngine.ts` must satisfy before P1.8 promotes to
acceptance. Each fixture is a complete replay package:

- The exact inputs the TypeScript engine consumed
- The exact output the TypeScript engine produced

A Python port of the selector engine is considered passing only if, for
each fixture, calling `evaluate(inputs)` produces JSON that is
byte-identical to `output.golden.json`.

## Fixture directory layout

```
selector_golden/
├── README.md              (this file)
├── MANIFEST.json          (machine-readable index)
├── capture_inputs.py      (script that built these fixtures from v1)
└── {date}_{session}/
    ├── run.json           (daily_runs row \u2014 the primary input)
    ├── recent_regimes.json (last 5 PM regimes prior to this run)
    ├── positions.json     (known_positions.json snapshot)
    ├── cohort_subset.json (master research rows for this fixture's cohort)
    └── output.golden.json (TS engine output \u2014 the byte-identical target)
```

## Fixtures captured

| Fixture | Cohort | Posture | Notable |
|---|---|---|---|
| 2026-05-11_PM | EXPANSION\|RICH_VOL\|NEUTRAL | HARVEST: HOLD | Brand-new cohort (0 history rows); has chain |
| 2026-05-08_PM | RECOVERY\|EXTREME_CHEAP\|LONG | BUILD: LEAP ONLY | 11 cohort history rows; has chain |
| 2026-05-07_PM | EXPANSION\|EXTREME_RICH\|LONG | HARVEST: PMCC ON | 1 cohort history row; has chain |
| 2026-04-14_PM | RECOVERY\|CHEAP_VOL\|LONG | BUILD: LEAP ONLY | 2 cohort history rows; no chain (legacy era) |
| 2026-04-09_PM | RECOVERY\|EXTREME_CHEAP\|SHORT | BUILD: CSP + LEAP | 3 cohort history rows; rare SHORT_GAMMA family; no chain |

## Coverage gaps (documented for transparency)

- **iv_band = 2 (NORMAL_VOL):** zero rows in the entire v1 dataset. The
  Python port's NORMAL_VOL behavior must be validated by code review,
  not by fixture replay. Filed under R1 risk register as a known
  fixture-coverage gap.
- **short_gamma_strong regime:** also unseen in PM rows. Same disposition
  as NORMAL_VOL.
- **Empty positions:** all fixtures share the same `positions.json` because
  the v1 system never versioned known_positions over time. The Python
  port's behavior under variable positions must be tested separately
  via unit tests with synthetic position payloads.

## Discipline rule

The selector port (P1.8 in 08_BUILD_PLAN.md) is blocked from promotion
until **every** fixture in this directory replays byte-identically.
Approved exceptions (e.g., timestamp_utc differs by inherent design)
are documented in the parity-validation test code, not silently
accepted.

## Regenerating fixtures

If the TS engine semantics change before P1.8 ships:

```
# 1. Restart Express server with the latest dist/index.cjs
# 2. Re-capture goldens by re-hitting /api/selector/evaluate per fixture
# 3. Re-run capture_inputs.py to refresh the input bundles
cd engine/tests/fixtures/selector_golden && python3 capture_inputs.py
```

Note: after the v2 engine is in production, this directory becomes the
historical reference for what v1 produced and is no longer regenerated.
The Python port becomes the source of truth.

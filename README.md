# asst-gamma-engine

Open-source quantitative engine for the ASST Bitcoin treasury Gamma
Flywheel research project.

## What this is

A Python package that:

1. Fetches daily market data (gamma exposure, options chains, BTC
   on-chain metrics, ASST fundamentals) from external feeds
2. Computes derived signals (banding, stochastic state machine,
   iv_band, vintage anchors, BCI evidence)
3. Runs the selector engine to produce structured `SelectorOutput`
   recommendations
4. Persists everything to a versioned SQLite database
5. Emits research-grade exports (CSV, parquet, JSONL) suitable for
   downstream analysis

The engine is the **methodology**: open and reproducible. It is the
source of truth for what the system computes. The proprietary
terminal layer that visualizes the output lives in a separate private
repo.

## Status

🚧 In active rebuild from v1 (v1.5 of the data schema). This repo is
the **engine half** of the v2 two-repo split documented in the
project's planning artifacts. The companion read surface is being
built separately.

Production engine release: **not yet — v2.0 alpha pending**.

## Architecture (v2)

```
engine/
  feeds/         # external data sources (Tiingo, BGeometrics, Glassnode, chain parsing)
  compute/       # derived signals (banding, stochastics, iv_band, legacy suggestions)
  selector/      # SelectorOutput pipeline (ports the TS selector engine)
  orchestration/ # pipeline_state, gap checks, integrity sweep, regime alerts
  exports/       # master research, selector quant, BCI evidence digest
  migrations/    # forward-only SQL migrations
  tests/         # unit tests with golden-fixture data
```

## Schema

The engine writes a versioned SQLite database. The current schema
version is **v1.5**. Migrations are forward-only and idempotent;
every schema bump ships as a numbered SQL file in `migrations/`.

Consumers (including the proprietary terminal) **pin to an exact
engine release** so that schema evolution cannot silently break
downstream renderings.

## License

Apache License 2.0. See [LICENSE](LICENSE).

The engine is intentionally open: methodology, banding logic, and
selector behavior are public so that research consumers (and future
testers) can independently verify what every recommendation means.
The terminal repo on top of this engine is proprietary.

## Contributing

This is a personal research project; external contributions are not
solicited at this time but the code is available under Apache 2.0 for
inspection, learning, and forks.

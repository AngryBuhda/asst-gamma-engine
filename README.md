# data branch

This branch is the **data carrier** for asst-gamma-engine. The
operator-local fetch cron pushes the latest `data.db` here after each
session (AM/MID/PM/EOD). GitHub Actions workflows in the **main** branch
clone this branch to read the freshest snapshot during the build step.

## Contents

  data.db        SQLite database with daily_runs + stochastic_log + pipeline_runs
  meta.json      build-time freshness manifest (timestamp, row counts)

## Update cadence

Push-after-fetch hook (P1.18) commits + pushes here whenever the
operator-local fetch path successfully writes a new row.

## Why a separate branch?

Per docs/v2_planning/07_ENGINEERING_PIPELINE.md §3:

> Q-A data.db transport \u2014 git push from operator-local fetch cron to the
> engine repo's `data` branch. Simplest path; no third-party dependency
> on the build's critical path. Drive remains pure backup.

## Reading from CI

Workflows that need the latest data.db should:

```yaml
- name: Pull latest data.db from data branch
  run: |
    git fetch origin data:data
    git checkout data -- data.db meta.json
```

## Do NOT merge this branch into main

The data carries no source code. It exists as a parallel branch only so
the file is accessible without bloating main's git history.

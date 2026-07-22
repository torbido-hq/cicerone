# Architecture

This document describes how the code under `src/cicerone/` fits together.
For configuration and usage, see the main [README](../README.md).

## Module overview

```
config.py            load & resolve config/cicerone.toml (structural config + ${ENV_VAR} secrets)
feature_config.py     load config/features.toml (event weights, feature columns)
io/
  base.py             InputSource / OutputSink protocols
  factory.py          picks a concrete backend by IOSettings.kind ("dataset" | "db")
  dataset_store.py     backend: parquet files (S3-compatible or local disk)
  db_store.py          backend: SQLAlchemy-backed database tables/queries
  options.py           shared "require_option" validation helper
dataset.py            raw events/users/items -> weighted rectools Dataset (BuiltDataset)
model.py            BuiltDataset -> STRATEGIES registry (collaborative/item_based/
                     popular/latest) -> top-K recommendations, combined per user
job.py                orchestrates one end-to-end run (source -> dataset -> model -> sink)
scheduler.py           in-process cron loop that calls job.run() on config/cicerone.toml's cron_schedule
```

## Data flow

```mermaid
flowchart LR
    subgraph Input
        S3["dataset (S3/local parquet)"]
        DB1["db (SQLAlchemy)"]
    end
    S3 -->|InputSource| J[job.run]
    DB1 -->|InputSource| J
    J --> D[dataset.build_dataset]
    D --> M[model.train_and_recommend]
    M --> J
    subgraph Output
        S3O["dataset (S3/local parquet)"]
        DB2["db (SQLAlchemy)"]
    end
    J -->|OutputSink| S3O
    J -->|OutputSink| DB2
```

1. `job.run()` loads `Settings` (`config.load_settings`) and `FeatureConfig`
   (`feature_config.load_feature_config`), builds the configured
   `InputSource`/`OutputSink` via `io.factory`, and reads `events`
   (required) plus `users`/`items` (optional).
2. `dataset.build_dataset()` turns raw events into weighted interactions
   (event-type weights, quantity scaling, per-pair caps, exponential time
   decay — all driven by `FeatureConfig`) and explodes user/item feature
   columns into rectools' long format, then constructs a
   `rectools.dataset.Dataset`.
3. `model.train_and_recommend()` fits every strategy listed in
   `Settings.models` (`STRATEGIES` registry in `model.py`; defaults to
   `["collaborative", "popular"]`) and produces top-K recommendations:
   personalized strategies (`collaborative`, `item_based`) only run for "warm"
   users (any user present in the dataset, with or without interactions — see
   the cold-start note below); non-personalized strategies (`popular`,
   `latest`) run for every target user and backfill any warm user who didn't
   get enough personalized results after the availability filter. Strategies
   are combined either by priority order (default — earlier ones win ties)
   or, if `Settings.model_weights` is set, by weighted reciprocal rank fusion
   (`_combine_by_weighted_fusion`) — see `model.RRF_K` and the module
   docstring for the exact formula.
4. `job.run()` writes the combined recommendations and a small run manifest
   (counts, timestamp) back out via the configured `OutputSink`.
5. `scheduler.main()` is the container's actual entrypoint: it computes the
   next run time from `cron_schedule` with `croniter`, sleeps, calls
   `job.run()`, and loops forever — a failed run is logged but never kills
   the loop.

## Extensibility: adding a new I/O backend

Input and output are each just a `kind` (string) + a free-form `options`
dict (`config.IOSettings`) — the config loader never needs to know what
keys a given backend requires. To add a new backend (e.g. a message queue):

1. Add a module under `src/cicerone/io/` implementing the `InputSource`
   and/or `OutputSink` protocol (`io/base.py`) — read `options` yourself,
   validating required keys with `io.options.require_option`.
2. Register the new `kind` string in `io/factory.py`'s
   `build_input_source`/`build_output_sink`.
3. Document the new `kind` and its `options` in `config/cicerone.toml`.

Nothing in `config.py`, `job.py`, `dataset.py`, or `model.py` needs to
change — they only ever see the `InputSource`/`OutputSink` protocol and the
generic `IOSettings`.

## Cold-start behavior

A user only counts as truly "cold" (popularity-only) if they're absent from
the dataset entirely — no interactions **and** no features. A user with
only features (no interactions) is still "warm" to LightFM via hybrid
cold-start and can get personalized recommendations. See
`model._recommendable_item_ids` and `model.train_and_recommend` for exactly
how warm/cold users and the availability filter interact.

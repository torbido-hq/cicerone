# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Incremental events between full retrains: internal `EventSource` surface,
  webhook `POST /events`, micro-batch buffer/worker, and write-through
  updater for popular/latest slices (`[events]` config). Design:
  `docs/incremental-events.md`. Webhook `occurred_at` requires an explicit
  timezone (`Z` / offset) or Unix epoch seconds (UTC).

### Changed

- Bump GitHub Actions Pages deploy helpers: `actions/upload-pages-artifact`
  v3 → v5 (Dependabot #79), `actions/deploy-pages` v4 → v5 (#80).
- Bump `fastapi` 0.140.13 → 0.141.1 (Dependabot #69).
- Dependabot: ignore `numpy` major bumps (Python 3.11 CI) and `boto3>=1.43.57`
  (aiobotocore botocore pin).
- Project docs site (not part of the runtime product): Starlight under
  `website/`, synced from `docs/`, published at [cicerone.dev](https://cicerone.dev).

## [0.5.1] - 2026-08-12

### Added

- Serve OpenAPI / ReDoc ``x-codeSamples`` (Ruby, Python, JavaScript, Shell)
  for `/health` and `/recommendations/{user_id}`.

### Fixed

- Redis lock `release()` joins the refresher (≤250ms) and ignores in-flight
  refresh failures after an intentional stop, avoiding a `_mark_lost` race.
- Retrain Prometheus labels use the real trigger source (`cron`, `s3-poll`,
  `webhook`, …) instead of collapsing non-webhook to `poll`.
- Serve fails closed when `features.toml` cannot be loaded (no silent disable
  of availability filters).
- Input poller treats local `stat` errors like S3 failures (log and continue).

### Changed

- Blending warm path uses per-user / shared latest rankings without building
  Cartesian U×K latest frames; `latest_by_user` keys are normalized to `str`.
- Serve caches `generated_at` with the refresh loop; cold-start prefers the
  per-user index / refresh-time fallback; category filters use a refresh-time
  `category → item_id` map.
- Content-fallback fit avoids `iterrows`; recommend uses vectorized top-K.
- Event caps apply in one sort across all capped event types.
- Shared helpers: `cicerone.values` (`is_missing` / `as_list`),
  `io.options.read_parquet` / `validate_storage_options`, and
  `MISSING_TABLE_ERRORS`.
- `cicerone.serve` package `__all__` exports only the public API.

## [0.5.0] - 2026-08-11

### Added

- Serve-mode Prometheus metrics at `GET /metrics` (`prometheus-client`):
  request volume/latency, cache hit/miss/age/refresh, recommendation
  source tiers, retrain-trigger counts, and `cicerone_up`. Configured via
  `[serve].metrics_enabled` (default `true`) and optional
  `[serve].metrics_token` (`X-Metrics-Token` header; empty = open endpoint).
  Per-replica registry; no multiprocess mode.
- Optional `[model.<strategy>]` TOML tables for RecTools `model_from_config`
  (`collaborative`, `item_based`, `popular`, `latest`).
- Optional scheduler lock backends (`[job.trigger].lock_backend`:
  `in_process` default, `postgres`, `redis`) for multi-replica mutual
  exclusion; manifest records `lock_backend`. Redis is an optional
  install (`requirements-redis.txt`). Optional `lock_key` /
  `lock_ttl_seconds` that namespace shared Redis/Postgres lock stores.
  Redis lock TTL is refreshed while held so long runs stay exclusive.

### Changed

- Strategies built via RecTools `model_from_config` / `get_config`.
- Artifacts schema **v3**: RecTools `model.save` / `load_model` in a zip
  envelope; `content_fallback` still pickle. Schema v2 no longer loadable.
  `created_at` is stored as ISO-8601 in `meta.json` and loaded as `datetime`.
- Split `model` / `config` into packages; import paths unchanged. Cross-module
  helpers are public (`_` = module-local only). Tests follow the same split
  (`test_model_*.py`, `test_config_*.py`).
- Docs: architecture test/module map, tutorial `[model.*]` knobs, README /
  CONTRIBUTING test layout notes; serve `/metrics` in README, architecture,
  tutorial, and `examples/serve/`.
- `cicerone.serve` is a package (`serve/app.py`, `serve/metrics.py`);
  `python -m cicerone.serve` and `from cicerone.serve import create_app`
  unchanged.
- Shared defaults: item-based `K` from `DEFAULT_ITEM_BASED_K_NEIGHBORS`;
  latest window from `model_config.LATEST_WINDOW_DAYS` (re-exported by
  `model.constants`).

### Renamed config keys (legacy still accepted)

- `job.item_based.k_neighbors` → `model.item_based.model.K`. Conflicting
  values raise `ConfigError`.

## [0.4.1] - 2026-08-07

### Added

- Nested settings surfaces on `Settings`: `serve`, `trigger`, `dashboard`, and
  `automl` dataclasses (flat `settings.serve_host`-style accessors remain as
  compatibility properties). Invalid config knobs raise `ConfigError`
  (`ValueError` subclass); missing files / unset `${ENV}` still raise
  `RuntimeError`.
- Manifest field `partial_outputs` when a sink write fails after some outputs
  were already persisted (success is set only after all writes succeed).
- `[[boosts]]` accepted as an alias for `[[boost]]` in `features.toml`
  (defining both is an error).
- I/O factory kind registry; `configure_item_filters` on the
  `RecommendationReader` protocol; frozen `ModelArtifact`.

### Changed

- Event caps keep the **most recent** N events per `(user, item, event_type)`
  (sorted by `occurred_at` descending before `cumcount`).
- Aggregated `(user, item)` pairs with non-positive weight are **dropped**
  instead of being floored to `1e-3` (negative review sums no longer become
  weak positive LightFM signals).
- Serve cold-start heuristic is deterministic across dataset and DB backends:
  prefer `popular_fallback`, then `latest`, then lexicographic `user_id`;
  DB path picks one user then fetches that user's top-K (no `LIMIT k*20`
  under-fill).
- `recommend_with_models` split into cohort → recommend → combine → boost
  phases; blending uses per-user indexes and optional shared latest rankings;
  serve dataset reader indexes recommendations by `user_id` at refresh time.
- ProcessPool strategy now initializes LightFM with `num_threads=1` inside
  workers to avoid CPU oversubscription.

### Fixed

- `item_true` eligibility no longer treats non-empty strings such as
  `"false"` / `"0"` as true (`astype(bool)`); only explicit truthy tokens
  / non-zero numerics / bools pass.
- DB `_clear_table_for_replace` catches the same missing-table errors as the
  rest of the DB store (including SQLite `OperationalError` on `TRUNCATE`).

## [0.4.0] - 2026-08-06

### Added

- **Content cold-item fallback** (`content_fallback` strategy): recommends
  zero-interaction items by one-hot cosine similarity over configured
  `item_features` against each warm user's history. Gated by
  `[job.content_fallback].enabled` (default off); when enabled, auto-inserted
  before the first non-personalized strategy. Independent of `item_based`.
  Source label: `content_fallback`. Requires `scikit-learn`.
- **Weighted multi-source blending** (`[blending]` in `config/features.toml`):
  replaces the binary personalized-vs-`popular_fallback` choice with a
  gradual per-user mix of `personalized`, `popular`, and date-based
  `latest` (publication/`occurred_at`-style columns on `items`). A
  configurable sigmoid or linear curve maps interaction count →
  personalized weight; the remainder is split by `popular_share`. When no
  usable date column exists, `latest` is disabled and its weight moves to
  `popular`. Combined rows use `source = "blended"`. Availability /
  eligibility still filter every source before the blend. A shared
  `__cold_start__` row set is written for serve-mode fallback.
- **Serve read contract** aligned with a Gorse-style lookup (without
  adopting Gorse infra): `GET /recommendations/{user_id}` accepts
  `limit` / `k`, `category`, and `exclude_unavailable`, returns
  `{generated_at, user_id, fallback, items:[{item_id,rank,score,source}]}`
  plus an `X-Generated-At` header from the last run manifest, and falls
  back to the precomputed cold-start list for unknown users (not a bare
  404). Items are snapshotted to the output store
  (`items_snapshot.parquet` / `recommendation_items`) so filters stay on
  the configured output without loading ML deps. Documented in the README
  Serve section alongside the existing Dashboard style.
- **Serve OpenAPI contract**: response models for `/health` and
  `/recommendations/{user_id}` so FastAPI's `/openapi.json`, `/docs`, and
  `/redoc` document the real JSON shape (including `X-Generated-At`). A
  checked-in schema at `docs/openapi/serve.openapi.json` can be regenerated
  with `python -m cicerone.export_serve_openapi`.
- **Thin serve clients**: `cicerone.serve_client.ServeClient` (stdlib
  `urllib`, typed via `serve_schemas`) plus copy-paste examples under
  `examples/serve/` (Python, Node `fetch`, curl).

### Changed

- **Default model chain** is now `["collaborative", "item_based", "popular"]`
  (was `["collaborative", "popular"]`), so sparse warm users get item-KNN
  backfill before raw popularity.
- **`[job.item_based].k_neighbors`** configures `TFIDFRecommender(K=…)`
  (default `20`).
- Serve JSON response is an object (with `generated_at` / `items`) rather
  than a bare list; `k` remains accepted as an alias for `limit`.
- Docs: architecture data-flow covers the three combine paths (priority /
  RRF / `[blending]`) and the items snapshot write; tutorial gains a
  blending walkthrough and an updated serve-API section (response shape,
  `limit` / `category` / cold-start fallback, OpenAPI `/docs`,
  `examples/serve/` clients).
- Blending correctness: date-based `latest` is ranked per eligibility
  cohort (no cross-cohort allowlist union); `__cold_start__` uses the
  global item-scoped allowlist; multi-personalized strategies collapse to
  best rank before RRF; sigmoid maps `n=0 → 0`; strategy `latest` is
  skipped while blending is on; serve heuristic fallback never reuses
  warm `blended` rows.
- `cicerone.serve.main` imports I/O factory helpers lazily so OpenAPI export
  and `create_app` do not require dataset/DB backend imports at module load.
- Serve OpenAPI `info.version` follows `cicerone.__version__` (single source
  of truth with the changelog / release tag).
- `config.make_settings(**overrides)` is the shared Settings factory for
  tests and OpenAPI schema export (replacing duplicated default blocks).

## [0.3.2] - 2026-08-04

### Added

- Optional per-epoch LightFM training metrics (`[job].log_epoch_metrics`,
  default `false`; interval via `[job].epoch_metrics_every`, default `5`):
  when enabled, the collaborative strategy fits via `fit_partial` one epoch
  at a time and logs in-sample Precision@K/Recall@K over a seeded random
  user sample. Tunables (`epoch_metrics_max_users`, regression/plateau
  thresholds) live on `EpochMetricsSettings`. Significant regression or late
  plateau across logged epochs emits a WARN. Off by default so scheduled
  batch runs stay unchanged.
- Top-K ID-mapping regression test: synthetic catalog with sparse external
  IDs that must not collide with rectools' dense internal indices, asserting
  no duplicate items per user and no seen items in personalized rows.

### Changed

- Documented that per-strategy top-K extraction is rectools-native
  (`ModelBase.recommend()` + Dataset id maps) and that AutoML eval metrics
  already use `rectools.metrics.calc_metrics` (MAP/NDCG/Recall) — no custom
  metric implementations.

## [0.3.1] - 2026-08-04

### Changed

- README gains a top-level **Features** list (batch, strategies, AutoML,
  policies, serve, trigger, dashboard, artifacts).

- Priority combine fills top-K from earlier strategies first (stable
  priority + rank sort) instead of interleaving by per-strategy rank.
- `[job].max_workers` drives ProcessPool parallelism for AutoML folds and
  strategy fitting (default `1` / sequential; set `>1` to opt in).
- AutoML `primary_metric` matches an exact name or a single `NAME@k`;
  ambiguous `NAME@k` sets are rejected (no order-dependent first match).
- Shared I/O helpers: `object_key`, `sql_identifier`, S3 not-found checks.
- Model-artifact DB writes use portable SQLAlchemy `LargeBinary` /
  timezone-aware `DateTime` and clear via `DELETE` (dialect-agnostic).
- Config validates `top_k`, `half_life_days`, feature column types, and
  `model_weights` keys against `job.models`.
- Batch job builds artifacts in memory and writes outputs only after
  successful compute, reducing partial failed-run publishes.

### Fixed

- Optional dataset `users`/`items` reads only treat missing files / S3
  not-found as absent; other errors propagate.

## [0.3.0] - 2026-08-03

### Added

- **Business policy layer** (`config/features.toml` → `[[eligibility]]` /
  `[[boost]]`): declarative hard filters (region/nationality, market match,
  category allowlists) and soft ranking boosts (paying producers, plan
  tiers, numeric lifts). Applied at batch recommend time via
  `cicerone.policy`; serve mode stays a lookup of already-policy-aware rows.
  `item_availability_filters` remains sugar for global `item_true` rules.
  User-scoped eligibility groups users into cohorts whose allowed-item set
  is computed once and reused across strategies; missing `item_column`
  warnings are deduplicated per `(kind, rule, column)`. When boosts are
  configured, candidates are over-fetched (`boost_overfetch_factor` ×
  `top_k`, default 3 — tunable in `features.toml`) before score multipliers
  so a commercially boosted item just outside the raw top-K can still enter
  the final list. Cohorts whose eligibility excludes every item get an empty
  allowlist (no silent catalog fallback) and are skipped.
- Optional **model artifact** (`[job].save_model_artifact`): the batch job
  can write a versioned, portable fitted-model bundle
  (`model.artifact` / `model_artifacts` table) alongside recommendations.
  Load and recommend without re-fitting via `cicerone.artifact`. Serve mode
  still reads precomputed rows only — this does not add live inference.
  Artifacts persist the `users` frame (schema version **2**) so
  `recommend_from_artifact` can re-apply user-scoped eligibility offline.
- Optional `postgres` service in `docker-compose.yml` (Postgres 16, compose
  profile `db`) for local `kind = "db"` input/output. First boot creates
  `cicerone` (app/tutorial) and `cicerone_test` (pytest) databases.
  `INPUT_DATABASE_URL` / `OUTPUT_DATABASE_URL` stay unset unless provided
  via `.env`.
- System-style end-to-end test (`tests/test_system_db.py`) against a real
  Postgres: seeds shared conftest fixtures → `job.run` (db in/out + model
  artifact) → verify recommendations, manifest, artifact blob, and the
  serve/dashboard DB readers. Schema reset is module-scoped, reflects via
  SQLAlchemy metadata, and is gated by a test DB name check plus
  `ALLOW_SCHEMA_RESET_FOR_TESTS=1`.
- User-facing documentation for model artifacts and business policies:
  README sections, tutorial §8–9, architecture notes, annotated recipes in
  `config/features.toml`, and `model_artifact_table` in the commented db
  config example.

### Changed

- `model.train_and_recommend` is split into `fit_strategies` +
  `recommend_with_models` so fitted weights can be reused (AutoML cache,
  model artifacts) without a second fit.
- Tutorial database section uses
  `docker compose --profile db up -d postgres` instead of an ad-hoc
  `docker run` Postgres container.
- Local pytest guidance points `TEST_DATABASE_URL` at `cicerone_test` (with
  `ALLOW_SCHEMA_RESET_FOR_TESTS=1`) so tests do not wipe app data.

### Fixed

- Model-artifact DB writes use `CREATE TABLE IF NOT EXISTS` + `TRUNCATE` +
  `INSERT` (no `DROP`/`CREATE` race under concurrent jobs).
- `model_artifact_table` is validated as a simple SQL identifier before
  interpolation.
- `cicerone.artifact` documents that pickle loads are trusted-internal-only
  (never user-controlled payloads; not on the serve path).

## [0.2.1] - 2026-07-29

### Added

- AutoML candidate backtesting can now evaluate time-based folds in
  parallel: `evaluate_candidates(..., max_workers=1)` runs folds through a
  `ProcessPoolExecutor` when `max_workers > 1`, instead of sequentially
  (opt-in; default behavior/performance is unchanged).
- Strategy fitting in `train_and_recommend(..., max_workers=1)` can
  likewise fit independent, not-yet-cached strategies via a
  `ProcessPoolExecutor` when `max_workers > 1` (opt-in; default is
  unchanged).

### Changed

- The job's input reads (events, users, items) now run concurrently via a
  `ThreadPoolExecutor` instead of sequentially, speeding up runs against
  network-backed (S3) input sources.

## [0.2.0] - 2026-07-29

### Added

- **Serve mode**: an optional lightweight, read-only HTTP API
  (`[job].mode = "serve"`) that exposes precomputed recommendations for a
  given user without any model loaded in the request path. Protected by a
  bearer token (`[serve].auth_token` / `SERVE_AUTH_TOKEN`).
- **Event-driven retrain trigger**: an opt-in webhook
  (`[job.trigger].enabled = true`) that lets an external system request an
  immediate retrain instead of waiting for the next scheduled run, with an
  optional S3-poll mode and a debounce guard against duplicate/overlapping
  triggers. Protected by a bearer token (`[job.trigger].auth_token` /
  `TRIGGER_AUTH_TOKEN`).
- **Dashboard**: a small, standalone read-only status page
  (`[dashboard].enabled = true`) showing recent job run history and errors,
  independent of `[job].mode`. Protected by HTTP Basic Auth against a set of
  named users managed via the new `manage_dashboard_users` CLI. Built with
  htmx + Stimulus + Tailwind, polling a `/partials/status` endpoint.
- `docs/tutorial.md`: new walkthroughs for serve mode, the retrain trigger,
  and the dashboard.

### Changed

- README's Security section corrected: previously claimed the container
  "accepts no inbound connections" unconditionally; now documents that the
  serve API, trigger webhook, and dashboard each expose one port only when
  explicitly enabled, and describes how each is protected.
- `.env.example` now documents `SERVE_AUTH_TOKEN` and `TRIGGER_AUTH_TOKEN`.
- `config/cicerone.toml` now includes a commented `[dashboard]` example
  alongside the existing `[serve]` and `[job.trigger]` examples.

## [0.1.0] - 2026-07-22

### Added

- Initial batch recommender job: reads interaction events, trains a hybrid
  [rectools](https://github.com/MobileTeleSystems/RecTools) + LightFM model,
  and writes top-K recommendations per user.
- Pluggable input/output backends: parquet files (`dataset`, S3-compatible
  or local disk) and SQLAlchemy-backed database tables (`db`).
- Pluggable multi-model strategy registry (`collaborative`, `item_based`,
  `popular`, `latest`) with weighted reciprocal rank fusion across models.
- AutoML mode to automatically pick a model strategy.
- Generic, TOML-based configuration (`config/cicerone.toml`,
  `config/features.toml`) with `${ENV_VAR}` secret interpolation.
- Scheduler for running the job on a cron schedule.
- Getting-started tutorial (`docs/tutorial.md`) and architecture overview
  (`docs/architecture.md`).
- CI: ruff lint/format, mypy, pip-audit, CodeQL, Dependabot.

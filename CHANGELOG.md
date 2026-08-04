# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.2] - 2026-08-04

### Added

- Optional per-epoch LightFM training metrics (`[job].log_epoch_metrics`,
  default `false`; interval via `[job].epoch_metrics_every`, default `5`):
  when enabled, the collaborative strategy fits via `fit_partial` one epoch
  at a time and logs in-sample Precision@K/Recall@K. Significant regression
  or late plateau across logged epochs emits a WARN. Off by default so
  scheduled batch runs stay unchanged.
- Top-K ID-mapping regression test: synthetic catalog with sparse external
  IDs that must not collide with rectools' dense internal indices; asserts
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

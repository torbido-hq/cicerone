# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Optional `postgres` service in `docker-compose.yml` (Postgres 16, compose
  profile `db`) for local `kind = "db"` input/output. First boot creates
  `cicerone` (app/tutorial) and `cicerone_test` (pytest) databases.
- System-style end-to-end test (`tests/test_system_db.py`) against a real
  Postgres: seed catalog → `job.run` (db in/out + model artifact) →
  verify recommendations, manifest, artifact blob, and the serve/dashboard
  DB readers (session-scoped engine; schema reset via SQLAlchemy metadata
  reflection).

### Changed

- Tutorial database section uses
  `docker compose --profile db up -d postgres` instead of an ad-hoc
  `docker run` Postgres container.
- README / tutorial / architecture document the model-artifact feature and
  the compose Postgres workflow. Local pytest guidance points
  `TEST_DATABASE_URL` at `cicerone_test` so tests do not wipe app data.

## [0.2.1] - 2026-07-29

### Added

- Optional **model artifact** (`[job].save_model_artifact`): the batch job
  can write a versioned, portable fitted-model bundle
  (`model.artifact` / `model_artifacts` table) alongside recommendations.
  Load and recommend without re-fitting via `cicerone.artifact`. Serve mode
  still reads precomputed rows only — this does not add live inference.
- AutoML candidate backtesting can now evaluate time-based folds in
  parallel: `evaluate_candidates(..., max_workers=1)` runs folds through a
  `ProcessPoolExecutor` when `max_workers > 1`, instead of sequentially
  (opt-in; default behavior/performance is unchanged).
- Strategy fitting in `train_and_recommend(..., max_workers=1)` can
  likewise fit independent, not-yet-cached strategies in parallel via a
  `ProcessPoolExecutor` when `max_workers > 1` (opt-in; default is
  unchanged).

### Changed

- `model.train_and_recommend` is split into `fit_strategies` +
  `recommend_with_models` so fitted weights can be reused (AutoML cache,
  model artifacts) without a second fit.
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

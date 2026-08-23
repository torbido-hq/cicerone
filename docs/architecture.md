<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Architecture

This document describes how the code under `src/cicerone/` fits together.
For configuration and usage, see the main [README](../README.md). For the
pipeline and how strategies differ, see [how-it-works.md](how-it-works.md).
For `[events]` ingest (webhook, backends, HA), see
[incremental-events.md](incremental-events.md).

## Module overview

| Path | Role |
| --- | --- |
| `config/` | Load & resolve `config/cicerone.toml` (structural config + `${ENV_VAR}` secrets); package: constants / settings / validation / load / `events` / lock_url; nested Serve/Trigger/Dashboard/AutoML/Events settings (+ flat property aliases); `ConfigError` for invalid knobs; `make_settings(**overrides)` for tests / OpenAPI export |
| `feature_config.py` | Load `config/features.toml` (event weights, feature columns, eligibility/boost policy rules; `[[boost]]` / `[[boosts]]`) |
| `policy/` | Declarative eligibility masks (fail-open/fail-closed matrix), cohort grouping (`eligibility.py`), score boosts (`boosts.py`) |
| `blending.py` | Per-user weighted mix of personalized/popular/latest (optional) |
| `io/base.py` | `InputSource` / `OutputSink` / `RecommendationReader` protocols (including `configure_item_filters` and `replace_recommendations_for_users`); `BaseRecommendationReader` with empty defaults for custom readers |
| `io/recommendation_schema.py` | Shared recommendation column constants + SQL identifier helper for read/write paths |
| `io/db_errors.py` | Shared SQLAlchemy missing-table/column classifiers for recommendation I/O |
| `io/factory.py` | kind→backend registry (`"dataset"` or `"db"`) |
| `io/dataset_store.py` | Backend: parquet files (S3-compatible or local disk) |
| `io/db_store.py` | Backend: SQLAlchemy-backed database tables/queries |
| `io/recommendation_reader.py` | Read-only lookup of precomputed recs for serve mode (no rectools/lightfm import); dataset path indexes by `user_id` at refresh; shared cold-start selection rule |
| `io/manifest_reader.py` | Read-only lookup of job-run manifests for the dashboard (no rectools/lightfm import) |
| `io/options.py` | Shared `require_option` / `build_s3_client` helpers |
| `dataset.py` | Raw events/users/items → weighted rectools Dataset (`BuiltDataset`; keeps users+items frames for policy evaluation; caps keep most recent N) |
| `model/` | `BuiltDataset` → `STRATEGIES` registry → fit / recommend / combine |
| `model/strategies.py` | `RecommenderModel` protocol, `Strategy`, `STRATEGIES`, `build_strategy_model` |
| `model/fit.py` | `fit_strategies`, `plan_model_run`, ProcessPool workers |
| `model/recommend.py` | `recommend_with_models`, cohort plan, `train_and_recommend` |
| `model/combine.py` | Priority + weighted RRF combiners |
| `model/epoch_metrics.py` | Optional LightFM per-epoch Precision/Recall logging |
| `model/constants.py` | `RRF_K`, `DEFAULT_MODELS`, source column names |
| `model_config.py` | Default + TOML `[model.*]` RecTools `model_from_config` configs; sequential `architecture` → `cls`; legacy `job.item_based.k_neighbors` → `model.K` (no ML imports — safe for serve) |
| `content_fallback.py` | Optional content-based cold-item strategy (one-hot item features + cosine vs user history) |
| `artifact.py` | Optional versioned fitted-model bundle (schema **v3**: RecTools `save`/`load_model` for library models + pickle envelope; `content_fallback` still pickle) |
| `automl.py` | Optional: backtests candidate models/weights/`rrf_k` configs over time-based folds of event history and picks the best one |
| `cli.py` | `cicerone` console script (`start` (alias `run`) / `job` / `serve` / `dashboard` / `scheduler` / `users` / `export-openapi`; `--config`, `--log-level`, `--log-format`) |
| `packaging.py` | Wheel checks for the Docker `package` stage (`python -m cicerone.packaging`) |
| `job.py` | Orchestrates one end-to-end run (source → dataset → model → sink) |
| `scheduler.py` | In-process cron loop that calls `job.run()`; when `[job.trigger]` is enabled, also hosts the retrain-trigger HTTP server (`trigger.py`) |
| `serve/` | Serve mode package: FastAPI read API over precomputed recommendations |
| `serve/app.py` | Routes, middleware, refresh loop (`cicerone serve`) |
| `serve/item_filters.py` | Category / availability snapshot cache for serve requests |
| `serve/events_routes.py` | Optional `POST /events` webhook mount when `[events]` webhook is enabled |
| `serve/bootstrap_events.py` | Start/stop the serve-process event worker (micro-batch → write-through) |
| `serve/metrics.py` | Prometheus metric objects + helpers (default in-process registry) |
| `serve_schemas.py` | Pydantic models that drive the serve OpenAPI schema |
| `serve_client.py` | Thin stdlib HTTP client for the serve read API |
| `export_serve_openapi.py` | `cicerone export-openapi` — dump FastAPI OpenAPI JSON (`docs/openapi/…`) |
| `events/` | EventSource registry, micro-batch write-through — [incremental-events.md](incremental-events.md) |
| `trigger.py` | Event-driven retrain trigger: webhook + optional input-bucket poll, debounce guard (`RunGuard`) shared with the cron loop; increments `cicerone_retrain_trigger_total` (per replica) |
| `locks/` | Optional lock backends (`postgres.py` / `redis.py`) for `RunGuard` and the events apply lease; Redis `owned()` checks the fencing token, Postgres `owned()` checks `pg_locks` for this backend pid |
| `config/lock_url.py` | Postgres lock URL resolution for config load + lock builder |
| `http_auth.py` | Shared bearer-token (serve/trigger) and HTTP Basic Auth (dashboard) dependencies |
| `dashboard.py` | Standalone FastAPI dashboard: job status/history plus user-id lookup (`cicerone dashboard`) |
| `dashboard_lookup.py` | Output-store user lookup for the dashboard (fallback, category join, display formatting) |
| `dashboard_users.py` | Load/save the dashboard's Basic Auth users file (TOML, username → bcrypt hash) |
| `manage_dashboard_users.py` | CLI to add/remove/list dashboard users |
| `templates/`, `static/` | Jinja2 templates + vendored htmx/Stimulus/Tailwind assets for the dashboard |

Local Docker Compose (`docker-compose.yml`) also offers an optional
`postgres` service under profile `db`
(`docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres`)
so `kind = "db"` input/output can be exercised without an external database.
Credentials and database names live in `docker/postgres/defaults.env` (see
CONTRIBUTING.md). CI uses a separate throwaway instance via
`docker-compose.ci.yml`. The system-style check in `tests/test_system_db.py`
exercises the full job → recommendations/manifest/artifact →
serve/dashboard reader path against that real Postgres (resetting only
`cicerone.io.db_store.DEFAULT_DB_TABLES`).

Public imports stay stable after the package splits:
`from cicerone.model import …` and `from cicerone.config import …`.
Cross-module helpers are public; `_` is for true module-locals only.

## Tests

Test modules mirror the packages (same pattern as `tests/test_io_*.py`):

| Path | Covers |
| --- | --- |
| `tests/test_config_load.py` | TOML load / Settings / env placeholders |
| `tests/test_config_validation.py` | Weights, epoch metrics, max_workers helpers |
| `tests/test_model_strategies.py` | `STRATEGIES` registry / `RecommenderModel` checks |
| `tests/test_model_fit.py` | Fit cache, parallel fit, `resolve_run_models` |
| `tests/test_model_recommend.py` | `train_and_recommend` / boosts / `content_fallback` |
| `tests/test_model_combine.py` | Priority combiner unit tests |
| `tests/test_model_epoch_metrics.py` | LightFM per-epoch metric helpers |
| `tests/test_model_config.py` | RecTools `[model.*]` + save/load round trips |
| `tests/test_model_sequential.py` | SASRec/BERT4Rec TOML mapping, AutoML skip, serve no-torch |
| `tests/support/model_events.py` | Shared synthetic events helper |
| `tests/support/toml_config.py` | Shared `write_toml` helper |
| `tests/support/events.py` | Shared event payload helper for `test_events_*` |
| `tests/test_events_*.py` | EventSource registry / normalize / webhook / db / db_postgres / s3 / redis_streams / buffer / store / updater / worker / ha |
| `tests/test_config_events.py` | `[events]` coerce + TOML load |
| `tests/test_serve_events_routes.py` / `test_serve_bootstrap_events.py` | Serve webhook mount + worker bootstrap |

## Data flow

```mermaid
flowchart LR
    subgraph Input
        S3["dataset (S3/local parquet)"]
        DB1["db (SQLAlchemy)"]
    end
    S3 -->|InputSource| J[job.run]
    DB1 -->|InputSource| J
    J -->|if Settings.automl_enabled| A[automl: evaluate_candidates + select_best_candidate]
    A --> J
    J --> D[dataset.build_dataset]
    D --> M[model.train_and_recommend]
    M --> J
    subgraph Output
        S3O["dataset (S3/local parquet)"]
        DB2["db (SQLAlchemy)"]
    end
    J -->|OutputSink| S3O
    J -->|OutputSink| DB2
    Ev["optional EventSource"] -->|"write-through popular/latest"| S3O
    Ev -->|"write-through popular/latest"| DB2
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
   `Settings.models` (`STRATEGIES` in `model/strategies.py`; defaults to
   `["collaborative", "item_based", "popular"]`) via RecTools
   `model_from_config` using `Settings.model_configs` (from
   `model_config.resolve_model_configs` + optional `[model.*]` TOML) and
   produces top-K recommendations. When `[job.content_fallback].enabled` is true,
   `content_fallback` is inserted before the first non-personalized
   strategy if not already listed. Personalized strategies
   (`collaborative`, `item_based`, `sequential`, `content_fallback`) only run for "warm"
   users (any user present in the dataset, with or without interactions — see
   the cold-start note below); non-personalized strategies (`popular`,
   `latest`) run for every target user and backfill any warm user who didn't
   get enough personalized results after eligibility filtering. Before
   `recommend()`, `policy.resolve_eligibility()` merges
   `item_availability_filters` sugar with explicit `[[eligibility]]` rules.
   User-scoped rules group target users into cohorts that share the same
   allowed item set (rectools accepts one `items_to_recommend` list per
   call); each cohort's allowlist is computed once and reused across
   every strategy. Strategies are then combined in one of three ways:
   - **Priority order** (default) — earlier strategies fill top-K first;
     later ones only backfill.
   - **Weighted RRF** — if `Settings.model_weights` is set (even an empty
     table), `combine_by_weighted_fusion` sums `weight / (rrf_k + rank)`
     across strategies; `rrf_k` defaults to `model.RRF_K` and is
     overridable via `Settings.rrf_k`/`[job].rrf_k`. Combined `source`
     labels join contributing strategy labels in `enabled_models` order.
   - **Per-user blending** — if `FeatureConfig.blending.enabled`
     (`[blending]` in `features.toml`), `cicerone.blending` replaces the
     binary personalized-vs-popular choice with a gradual mix of
     `personalized`, `popular`, and (when `items` has a usable datetime
     column from `latest_date_columns`) date-based `latest`. A sigmoid or
     linear curve maps each user's distinct (user, item) count after
     aggregation → personalized weight; the remainder is split by
     `popular_share`. Availability / eligibility still filter every source
     *before* the blend; date-based `latest` is ranked **per cohort
     allowlist** (not a cross-cohort union). An item gets
     `source = "blended"` only when more than one source contributed it. A
     shared `__cold_start__` user is ranked against the global item-scoped
     allowlist for serve-mode fallback. When no date column is usable,
     `latest` is disabled and its weight moves to `popular`. Strategy
     `latest` (trending PopularModel) is skipped while blending is on.
   `Settings.max_workers` (`[job].max_workers`, default `1`) parallelizes
   AutoML fold evaluation and strategy fitting via `ProcessPoolExecutor`
   when set `>1`. Per-strategy top-K is rectools-native
   (`ModelBase.recommend()` maps external↔internal IDs via the Dataset's
   id maps — Cicerone does not hand-roll that conversion). When
   `[job].log_epoch_metrics = true`, the collaborative LightFM strategy is
   fitted epoch-by-epoch via `fit_partial` and logs in-sample
   Precision@K/Recall@K every `[job].epoch_metrics_every` epochs over a
   seeded random user sample (default off). Tunables are grouped in
   `EpochMetricsSettings`.
   If `[[boost]]` rules are configured, candidates are over-fetched by
   `FeatureConfig.boost_overfetch_factor` (default 3× `top_k`), scores are
   multiplied by the product of boost factors, and ranks are rewritten
   before truncating to `top_k`. Cohorts with an empty allowlist (eligibility
   filtered out every item) are skipped.
   An optional `strategy_cache` parameter (keyed by strategy name, caching
   the *fitted model*) lets a caller who is evaluating multiple configs
   against the same `BuiltDataset` — namely `automl.evaluate_candidates()` —
   skip re-fitting a strategy shared by more than one candidate. A separate
   `recommend_cache` memoizes per-strategy `recommend()` frames, keyed by
   strategy, cohort, `recommend_k`, allowlist and dataset fingerprint;
   AutoML passes one per fold, so candidates that differ only in how they
   combine strategies reuse the scored frames and recompute just the
   combination. The batch job also passes a `strategy_cache` when
   `[job].save_model_artifact = true` so fitted weights can be serialized
   without a second fit.
4. If `Settings.automl_enabled` (`[job.automl].enabled`), before step 3
   `automl.evaluate_candidates()` backtests a list of candidate
   `models`/`weights`/`rrf_k` configs (defaults to `automl.DEFAULT_CANDIDATES`,
   overridable via `[[job.automl.candidates]]`) over `Settings.automl_n_splits`
   time-based folds of the raw event history — each fold trains a fresh
   `BuiltDataset` on everything before a `Settings.automl_test_days`-day
   held-out window and scores its recommendations against that window with
   `rectools.metrics` (MAP@k/NDCG@k/Recall@k). Within a fold,
   `evaluate_candidates()` passes a `strategy_cache` dict (reset per fold,
   shared across every candidate scored against that fold) to
   `train_and_recommend()` so candidates sharing a strategy reuse its fitted
   model instead of retraining it per candidate. `sequential` is dropped from
   the pool (INFO log) when `rectools[torch]` is missing or median distinct
   items/user is below `Settings.sequential_min_median_interactions`.
   `select_best_candidate()` then picks the highest scorer by
   `Settings.automl_primary_metric`
   (matched by prefix, e.g. `"MAP"` matches `"MAP@10"`), and its
   `models`/`weights`/`rrf_k` replace the static config for that run's call
   to `model.train_and_recommend()`.
5. `job.run()` writes the combined recommendations and a small run manifest
   (counts, timestamp, effective `models`/`model_weights`/`rrf_k`,
   `artifact_written` / `artifact_schema_version` when a model artifact was
   saved, and `automl_metrics` when AutoML ran) back out via the configured
   `OutputSink`. When items were loaded, it also writes an items snapshot
   (`items_snapshot.parquet` / `recommendation_items`) so serve mode can
   apply `?category=` and `exclude_unavailable` without reading the input
   store. When `Settings.save_model_artifact` is true, it also writes
   a versioned fitted-model artifact (`model.artifact` for the dataset
   backend, `model_artifacts` table for db) via
   `OutputSink.write_model_artifact`. Serve mode never loads this artifact.
6. For batch, `cicerone start` runs `scheduler.main()`: it
   computes the next run time from `cron_schedule` with `croniter`, sleeps,
   calls `job.run(triggered_by="cron")` (with `fence_check=backend.owned` when a
   distributed lock is held), and loops forever — a failed run is
   logged but never kills the loop. When `Settings.trigger_enabled`
   (`[job.trigger].enabled`), `scheduler.main()` instead delegates to
   `_run_with_trigger()`, which runs the same cron loop on a background
   thread and additionally serves `trigger.create_app()` (see below) in the
   main thread — both funnel through one `trigger.RunGuard` so at most one
   run happens at a time regardless of what triggered it.

## Serve mode and the retrain trigger

Selected via `[job].mode = "serve"`, `cicerone.serve` is a separate entrypoint
(`cicerone serve`) from the batch scheduler — a serve-only
deployment never imports `cicerone.model`/`dataset`/`automl` (no
rectools/lightfm/implicit/torch needed in that process or its request path):

- `io.factory.build_recommendation_reader(settings.output)` builds a
  `RecommendationReader` (`io/recommendation_reader.py`) matching the
  configured output `kind` — `DatasetRecommendationReader` caches the whole
  parquet file (and optional `items_snapshot.parquet`) in memory and refreshes
  it on a background timer (`serve.app`'s `_start_refresh_loop`);
  `DbRecommendationReader` queries the recommendations table directly per
  request and caches the `recommendation_items` snapshot for filters. Both
  readers record cache hit/miss and refresh success/failure/duration metrics
  (`cicerone.serve.metrics`).
- `serve.create_app()` exposes `GET /health` and
  `GET /recommendations/{user_id}` (`limit`/`k`, `category`,
  `exclude_unavailable`) behind `http_auth.require_bearer_token`. Unknown
  users fall back to the precomputed `__cold_start__` list rather than 404.
  Responses include `generated_at` from the run manifest. Pydantic models in
  `serve_schemas.py` populate `/openapi.json` (and `/docs` / `/redoc`);
  `export_serve_openapi` writes the checked-in copy under `docs/openapi/`.
  Integrators can call the same contract via `serve_client.ServeClient`, the
  snippets in `examples/serve/`, or the `x-codeSamples` embedded in OpenAPI /
  ReDoc (Ruby, Python, JavaScript, Shell).
- When `[serve].metrics_enabled` (default `true`), `GET /metrics` exposes
  Prometheus text-format process metrics (request volume/latency, cache
  health, recommendation source tiers, `cicerone_up`). It does **not** use
  the recommendation bearer token; optional `[serve].metrics_token` gates it
  via the `X-Metrics-Token` header. When the token is empty, bind serve to a
  trusted network or reverse proxy. Metrics are per replica (default
  `prometheus-client` registry; no multiprocess mode) — cross-replica
  aggregation is Prometheus's job via scrape targets.

`cicerone.trigger` implements the event-driven retrain trigger, additive to
(not a replacement for) `scheduler.py`'s cron loop:

- `RunGuard` (thread-safe) debounces concurrent trigger sources: a run
  already in flight, or one that finished within `debounce_seconds`, causes
  a new trigger to be skipped rather than queued or run in parallel.
- `create_app()` exposes `POST /trigger/retrain` (webhook, behind
  `http_auth.require_bearer_token`) which calls `guard.trigger("webhook")`.
- `poll_input_forever()` (only started when
  `Settings.trigger_poll_input_bucket`) periodically fingerprints the input
  source (`_current_marker`: local file mtime, or S3 `head_object`'s
  `LastModified`) and calls `guard.trigger("s3-poll")` when it changes. This
  is a deliberate substitute for real S3 event notifications, which require
  SNS/SQS/Lambda wiring and aren't portable across S3-compatible backends
  (R2, MinIO) — polling avoids adding that infra while still being
  event-driven from the operator's point of view.
- Every successful run's manifest (written by `job.run()`) records
  `triggered_by` (`"cron"`, `"webhook"`, or `"s3-poll"`) and
  `lock_backend`.
- Debounce exclusion defaults to an in-process `threading.Lock` with no
  distributed backend (`lock_backend = "in_process"`). Optional `postgres` /
  `redis` backends (see `cicerone.locks`) coordinate across scheduler
  replicas; clients are imported only when selected. Prefer `postgres`
  when a DB URL is available; use `redis` (`cicerone-recommender[redis]` /
  `requirements-redis.txt`) for dataset/S3-only HA. Redis `owned()` checks the
  fencing token before commit so an expired TTL cannot split-brain a long
  apply/retrain; Postgres advisory locks are session-scoped (disconnect
  releases; `owned()` checks `pg_locks` for this backend pid). Cron and
  `RunGuard` pass `owned()` into `job.run` as `fence_check` so a lost lock
  skips artifact and recommendation writes.

Serve replicas scale on the read path. Incremental `[events]` apply is
single-writer unless `events.ha = true` with `lock_backend` postgres/redis
(leader-only lease, separate key from the retrain lock).

## Dashboard

`cicerone.dashboard` is a standalone entrypoint (`cicerone dashboard`;
compose maps port `8090`) for checking whether
the last job run succeeded and inspecting a user's current top-K — it is
**not** gated by `[job].mode` like serve/batch, so it's available even in
plain batch-only deployments with no other HTTP surface. Like serve mode, it
never imports `cicerone.model`/`dataset`/`automl`.

- `io.factory.build_manifest_reader(settings.output)` builds a
  `ManifestReader` (`io/manifest_reader.py`) matching the configured output
  `kind`: `DatasetManifestReader` only ever has the latest run (a `dataset`
  output's `manifest.json` is overwritten every run, not appended — no
  history for that backend), while `DbManifestReader` queries the
  `recommendation_runs` table for real history (`read_recent(limit)`).
- `io.factory.build_recommendation_reader(settings.output)` builds a
  `RecommendationReader` for the user-id inspector (`dashboard_lookup.py`).
  Lookup reads the output store directly (no serve hop). `k` is
  `min(job.top_k, dashboard.lookup_k)` (default 20). Missing users fall
  back to `__cold_start__` / popular-latest rows with a badge; `category`
  is joined from the items snapshot when that column exists.
- `job.run()` writes exactly one manifest per run via a `try`/`finally`,
  with a consistent key set (`status: "success"|"failed"`, `error`) on both
  the success and failure paths, so a failed run is no longer silently
  invisible to the dashboard.
- Auth is HTTP Basic (`http_auth.require_basic_auth`), not the bearer-token
  pattern used by serve/trigger — a browser-navigable page needs a login
  prompt, since a human can't attach a custom `Authorization` header to a
  plain top-level page load. Users live in a small TOML file
  (`dashboard_users.py`: username -> bcrypt hash) managed via
  `cicerone users add <username>` (optional `--users-path`, or `--config`
  pointing at the dashboard TOML).
- `dashboard.create_app()` exposes `GET /health` (no auth), `GET
  /partials/status` (Basic Auth, an htmx-polled fragment — see
  `templates/_status.html`), `GET /partials/recommendations` (Basic Auth,
  user-id lookup fragment — see `templates/_recommendations.html`), and
  `GET /dashboard` (Basic Auth, the full page). The page polls
  `/partials/status` via `hx-trigger="load, every Ns"`
  (`Settings.dashboard_refresh_interval_seconds`) instead of a websocket or
  client-side JS framework. The lookup form is outside that poll target so
  a status refresh does not wipe results.
- Frontend stack: server-rendered Jinja2 templates + htmx (polling) +
  Stimulus (a small `time-ago` controller for relative timestamps) +
  Tailwind CSS, all vendored under `src/cicerone/static/` — no CDN at
  runtime and no Node/npm dependency at runtime or for end users. Tailwind
  is compiled ahead of time in a dedicated `frontend` Docker build stage
  (`node:26.7.0-slim`, pinned via `package.json`/`package-lock.json`) whose only
  output (`static/tailwind.css`) is copied into the runtime image and the
  PyPI wheel (`package` Docker stage).

## Business policies

`config/features.toml` can declare hard eligibility filters and soft ranking
boosts (`[[eligibility]]` / `[[boost]]`). These are evaluated in
`cicerone.policy` during `recommend_with_models` — not at serve time — so
precomputed recommendation rows already respect region/nationality gates,
paying-producer boosts, plan tiers, and similar ecommerce rules. See the
annotated recipes in `config/features.toml` and the README data-contract
section.

Implementation details:

- **Cohorts:** user-scoped eligibility groups target users by the
  fingerprint of the user attributes those rules read. Each cohort's
  allowed-item list is computed once and shared by every strategy's
  `recommend()` call.
- **Boost over-fetch:** when any `[[boost]]` is configured,
  strategies request `boost_overfetch_factor * top_k` candidates (default
  factor 3, set in `features.toml`) so a commercially boosted item that
  would otherwise sit just outside the raw top-K can still be promoted after
  score multiplication.
- **Empty cohorts:** if eligibility excludes every catalog item for a
  cohort, the allowlist is empty (logged once per such cohort) and that
  cohort is skipped — there is no silent fallback to the full catalog.
- **Missing columns:** a configured `item_column` absent from the items
  frame fails open (rule skipped). The warning is emitted once per
  `(kind, rule name, column)` for the process lifetime to avoid log spam
  when eligibility runs per cohort.
- **Artifacts:** schema version **3** persists RecTools models with
  `model.save` / `load_model` (zip envelope); `users`/`items` frames and
  `content_fallback` stay pickle-backed so `recommend_from_artifact` can
  re-apply user-scoped eligibility offline. Schema v2 bare pickles are not
  loadable.

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

Nothing in `config/`, `job.py`, `dataset.py`, or `model/` needs to
change — they only ever see the `InputSource`/`OutputSink` protocol and the
generic `IOSettings`.

## Incremental events

Serve-process ingest lives in `events/` plus `serve/events_routes.py` and
`serve/bootstrap_events.py`. Operator guide:
[incremental-events.md](incremental-events.md).

## Cold-start behavior

A user only counts as truly "cold" (popularity-only) if they're absent from
the dataset entirely — no interactions **and** no features. A user with
only features (no interactions) is still "warm" to LightFM via hybrid
cold-start and can get personalized recommendations. See
`cicerone.policy.resolve_eligibility` / `allowed_items_for_cohort` and
`model.train_and_recommend` for exactly how warm/cold users and eligibility
interact.

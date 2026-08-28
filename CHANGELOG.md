# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.8.0] - 2026-08-28

### Added

- Dashboard Configuration page (`GET /dashboard/config`) shows the loaded
  Settings and `features.toml` with tokens, URLs, and keys redacted.
  Section chips, on/off badges, and nested panels.
- Config keys and section titles can open a one-line hint; some include a
  Docs link to cicerone.dev.
- Optional Kafka and RabbitMQ extras: `[events].kind = "kafka"` /
  `"rabbitmq"` ingest (consumer group / queue) and `[publish]` to emit
  per-user recommendation JSON after the `[output]` store write. Serve
  still looks up from dataset/db. `pip install 'cicerone-recommender[kafka]'`
  or `[rabbitmq]`. Prefer Redis Streams when you already run Redis for the
  lock.

### Fixed

- Dashboard `Cache-Control: private, no-store` skips only `/static` assets,
  not other paths that happen to start with that prefix.

### Security

- Dashboard pages send `X-Robots-Tag` and HTML `noindex` so crawlers skip
  them if the process is accidentally public; `GET /robots.txt` disallows
  `/`, and OpenAPI `/docs` is off.
- Config page redacts `*_url` option keys and any value with embedded URL
  credentials.

## [0.7.1] - 2026-08-31

### Changed

- ProcessPool fit and AutoML pickle the shared dataset/config once per worker,
  not once per strategy or fold.
- Blending expands identical per-user latest item/rank order with the same
  vectorized path as `shared_latest`.
- Content-fallback fit stores only the last 50 history items per user,
  ordered by event datetime when present (the window recommend already scored).
- Content-fallback recommend starts an inner user thread pool only on the
  process main thread (not from a worker thread).
- Online collaborative refresh uses `[job].max_workers` for strategy recommend
  threads.
- Importing `cicerone.job` no longer calls `logging.basicConfig` (`cicerone job`
  and `python -m cicerone.job` still configure logging).
- Ranking sorts use pandas `mergesort` so score ties follow item id (same as
  weighted RRF).
- Boost boolean/value_map factors use `item_true_mask` / vectorized map instead
  of per-cell lambdas. `PRIMARY_METRIC_WEIGHTED` and the log format string live
  in one constants module.

### Fixed

- Online rewrite skips sequential when torch is missing instead of dropping
  sequential / RRF / blend rows that share a part.
- Online artifact replace on S3 is refused; `[events.online]` requires db or
  a local dataset path.
- AutoML drops `content_fallback` from candidates when that strategy is off.
- Experiment metrics use first exposure, events after exposure and before
  promote, and ITT that ignores the promoted arm.
- Three-plus variants promote the unique best mean (Bonferroni-adjusted
  alpha).
- Incremental boost keeps existing reasons when the event item is already
  in the list.
- Empty `_source_contribs` falls back to `source`; reasons are validated at
  write.
- Serve promote-state reads reuse the last successful value on failure.
- Named variant filters ignore missing / NaN `variant` values.
- Incremental merge without a `variant` column keeps the unlabelled prior
  on control only.
- Legacy exposures tables missing `experiment_id` are ignored.
- Experiment time windows drop untimed events; invalid `promoted_at` blocks
  promote. DB promote-state errors reuse the cached winner.

## [0.7.0] - 2026-08-28

### Added

- Optional `[events.online]`: the serve events worker continues LightFM
  (`fit_partial`) on IDs already in the last model artifact and rewrites
  personalized / item-KNN / content-fallback rows for affected users.
  `GET /recommendations` stays a lookup. New catalog IDs and sequential
  models still wait for `job.run()`.

- Optional sequential architecture `hstu` (`HSTUModel`). Sequences are still
  last-touch aggregated `(user, item)` pairs, so HSTU relative-time bias is
  weak on Cicerone data.
- Opt-in AutoML `[job.automl].debias` (RecTools `DebiasConfig` on MAP/NDCG/Recall;
  default off).
- Sequential per-epoch Precision/Recall logs when `[job].log_epoch_metrics`
  is on (same knobs as collaborative).
- Serve recommendations include optional `reasons` (contributing sources,
  boost hits, similar history items / matched attributes), persisted at
  batch time when `[job.explain]` is enabled (default on). Existing DB
  tables need `ALTER TABLE … ADD COLUMN reasons TEXT`.
- Dashboard user lookup shows recent `[input]` events next to current top-K
  (overlap highlighting, source mix). `dashboard.lookup_events` defaults to 20.
  User attributes render only when `dashboard.lookup_user_attrs` is set.
- Dashboard Pause updates / Resume updates and Refresh for the status poll
  (starts paused when `prefers-reduced-motion` is set).
- `[experiment]` sticky A/B tests of whole ranking recipes (models + combiner +
  blending knobs). The job fits the union once, writes a `variant` column, and
  serve hashes `user_id` onto one list. Dashboard Experiments page: always-valid
  CIs, catalog guardrails, optional exposure log, AutoML challenger, promote
  winner to 100% traffic.

### Changed

- Sequential SASRec defaults to the eSASRec recipe (`sampled_softmax`,
  `n_negatives = 256`, LiGR layers).
- CI compose runs sequential extra tests in a separate `test-sequential`
  image with `rectools[torch]` (main test and runtime images stay torch-free).
- Bump `rectools` 0.13.0 → 0.19.0, `scipy` 1.12.0 → 1.17.0, and switch the
  implicit pin to `pm-implicit` 0.7.3 (RecTools 0.18+ on Python 3.11).
- Bump `uvicorn` 0.52.3 → 0.52.4, `ruff` 0.16.3 → 0.16.4, `mypy` 2.3.0 → 2.3.1,
  `wheel` 0.44.0 → 0.48.0.
- Bump `scipy` 1.17.0 → 1.17.1 and `moto` 5.2.2 → 5.2.3.
- Bump `actions/upload-artifact` v4 → v7 and `actions/download-artifact` v4 → v8.
- Dashboard latest-run card shows stale on the card, visible Latest run /
  Recent runs headings, relative times in history, and plainer operator copy.

### Fixed

- Dashboard no longer paints unknown or overdue latest runs as success.
- Dashboard history no longer styles empty errors as failures.
- Dashboard lookup keeps recommendations when a user attribute is a list,
  Series, or array.
- Serve collapses leftover `variant` rows to `control` (else the
  lexicographically first remaining name)
  when `[experiment]` is off, instead of mixing control and treatment ranks.
- Experiment promote state is a single-row replace (no `DROP TABLE`);
  reads order by `promoted_at`, and fall back if that column is missing.
- Online LightFM persists the artifact only after a successful apply and
  event `ack`; a failed refresh nacks the batch.
- Online collaborative rewrite is skipped while `[experiment]` is enabled.
- Dashboard unknown staleness (`is_stale` unset) is amber, not success green.
- Exposure-conditional experiment metrics ignore rows from other
  `experiment_id`s.
- Incremental write-through collapses leftover `variant` rows when
  `[experiment]` is off (same control-else-first rule as serve).
- Experiment promote state is read live (no per-process TTL cache) so
  replicas agree after promote.
- `log_exposures` with `events.ha` requires db output.
- Online persist after ack retries, then drops the pending fit if it still
  fails or the apply lease is lost.
- SQS and Redis Streams heartbeat in-flight messages for the duration of
  incremental apply.
- Leftover `variant` collapse ignores missing/NaN names instead of treating
  them as `"nan"`.
- Dashboard lookup refresh TTL tracks one reader, not an unbounded id map.
- Experiment coverage guardrail uses the items-snapshot catalog size, not
  distinct recommended item ids (so concentrated lists cannot relax the floor).
- Online persist after ack is skipped when a full retrain holds the lock or
  replaced the model artifact.
- Online LightFM caps extra interactions on top of the last job artifact
  (`events.online.max_extra_interactions`, default 50_000).
- Dashboard Experiments can resume the split after promote.
- Writing `reasons` or `variant` to an existing DB table missing those
  columns raises a clear `ALTER TABLE` error.
- Unknown dashboard staleness is announced as unknown, not success.
- Sequential `log_epoch_metrics` calls transformer `fit_partial` with min
  and max epoch (LightFM still gets epochs only).
- AutoML scores candidates with `job.content_fallback.enabled`, matching
  the recipe the job ships.
- AutoML-challenger incremental apply uses `control`/`treatment` when
  `[[experiment.variants]]` is empty.
- Challenger control recipes prefer the prior run's `experiment_variants`
  control arm over the union `models` list.
- Experiment promote is blocked when recommendations or `variant` are
  missing (catalog guardrails cannot run).
- A failed first in-flight heartbeat nacks the batch instead of applying.
- SQS receive uses a 5-minute visibility timeout so micro-batch plus apply
  can outlast the queue default.
- `[events.online]` with `[experiment]` logs a warning when the online
  rewrite is skipped.
- HSTU keeps an explicit `[model.sequential].loss = "sampled_softmax"`
  instead of rewriting it to `softmax`.
- Sequential epoch-metric `fit_partial` detects RecTools transformers by
  `min_epochs`/`max_epochs`, so LightFM `num_threads` stays at its default.
- Online artifact persist is a compare-and-swap against the baseline
  fingerprint (local file lock / DB `DELETE … WHERE written_at`), after
  serializing the blob, so a finishing retrain is not overwritten.
- Dashboard experiment promote/resume redirects stay on `/dashboard/experiments`
  even when the variant or error text is hostile.

## [0.6.2] - 2026-08-24

### Changed

- After fit, strategy×cohort `recommend()` calls run in a thread pool when
  `job.max_workers > 1` (default remains 1).
- Blend RRF scores with per-user source weights in pandas instead of a Python loop
  over users (same top-K).
- Incremental popular ranking and training interactions share vectorized
  per-row event weights.
- Dashboard lookup formats rank/score with Python coercion instead of
  pandas scalars.
- Dashboard lookup uses shared `is_missing` (empty source/category still "—").
- Weighted fusion keeps source-label join as an O(groups) Python agg; a scale
  test records the baseline.

### Fixed

- Incremental popular/latest/boost ignores unknown, zero-weight, and negative events, and does not rewrite popular-only users in a mixed batch.
- DB watermarks compare synthetic `id` / `ctid` numerically so same-timestamp `id:9` does not skip `id:10`.
- DB poll/lag skip lexical `event_id >` when the watermark is a synthetic numeric identity (`id:9` vs `id:10` in the `event_id` column).
- DB same-timestamp numeric-identity pages use a padded SQL sort key and LIMIT.
- SQLite event watermarks compare fractional seconds; SQL identity keys keep the `id:` / `rowid:` / `ctid:` prefix for non-numeric suffixes.
- Items filter cache retries when the items version moves during rebuild.
- Recommend cache keys include the user set; run manifest user counts omit `__cold_start__`.
- Dashboard lookup treats `pd.NA` source/category as missing instead of failing the lookup.

## [0.6.1] - 2026-08-20

### Changed

- AutoML reuses per-strategy recommend frames across candidates in a fold.
- Content-fallback scoring parallelizes across users; incremental merge groups events
  by user once; latest ranking expansion uses NumPy repeat/tile.
- Split `locks`, `policy`, and serve item-filter helpers into smaller modules
  (public imports unchanged).

### Fixed

- Serve availability filters treat `"false"` / `"0"` as unavailable (same coercion as training).
- Incremental apply no longer deletes popular-only users on unknown/zero-weight events, and
  keeps the best preserved ranks when boost slots truncate top-K.
- Recommendation replace raises on a missing `user_id` column instead of ACKing events
  after a no-op write (DB) or dropping other users' rows (dataset).
- S3 list-mode event source retries unreadable objects before skipping them; webhook
  `nack` no longer duplicates already-pending ids.
- Full retrain skips artifact/recommendation writes if the distributed lock is lost
  before write (cron and trigger paths pass `owned()` as a fence).
- DB event source distinguishes same-payload rows at one timestamp via table identity
  (`id` / SQLite `rowid` / Postgres `ctid`) so the watermark cannot skip a twin.

## [0.6.0] - 2026-08-20

### Added

- Docs site homepage: latest dated CHANGELOG release and a GitHub **what changed**
  link to that section.
- Dashboard user lookup: inspect a `user_id`'s current precomputed top-K
  (rank, item, score, source, optional category) from the job output store,
  with cold-start fallback, on the Basic-Auth status page.
  `GET /dashboard?user_id=` fills the lookup on load.
- PyPI distribution `cicerone-recommender` (`import cicerone`; the name
  `cicerone` is taken). Wheel includes compiled dashboard CSS. A GitHub
  Release publishes via trusted publishing (`.github/workflows/publish.yml`).
- `cicerone` CLI (`start`/`job`/`serve`/`dashboard`/`scheduler`/`users` /
  `export-openapi`) with `--config` for a TOML path, plus `--log-level` /
  `--log-format` (or `CICERONE_LOG_LEVEL` / `CICERONE_LOG_FORMAT`). Runtime
  image pip-installs the wheel; entrypoint is `cicerone start`.

- Optional project-site articles at `/articles/` (static Markdown under
  `website/src/content/docs/articles/`). No nav, RSS, or index until a
  published post exists. Article pages use IBM Plex Serif and a ~65ch
  measure. Brand accents invert for dark theme. Listing keeps an h1;
  posts use `description` for meta. Website-only PRs skip Docker lint/test
  jobs; the `ci` job still succeeds.

- Optional **sequential** strategy (`SASRecModel` / `BERT4RecModel`) via
  `[model.sequential]` (`architecture = "sasrec"` or `"bert4rec"`). Requires
  `cicerone-recommender[sequential]` / `requirements-sequential.txt`
  (`rectools[torch]`); serve mode never imports torch. AutoML drops it from
  the candidate pool when the extra is missing or median distinct items/user
  is below `[job.sequential].min_median_interactions` (default 5), and logs
  the skip.

- Incremental events horizontal HA: leader-only apply lease
  (`{lock_key}:events:apply`) when `events.ha = true` with
  `job.trigger.lock_backend` postgres/redis. Fan-out sources acquire the
  lease only when a micro-batch is ready. Metrics:
  `cicerone_events_lock_total`, `cicerone_events_leader`,
  `cicerone_events_apply_busy_total`.

- Redis Streams EventSource (`events.kind = "redis_streams"`): consumer-group
  poll via `XREADGROUP` / `XACK`, idle PEL recovery with `XAUTOCLAIM`, and
  stream entry id fallback when `event_id` is omitted. Requires
  `cicerone-recommender[redis]` / `requirements-redis.txt` (same optional
  `redis` pin as the lock backend).
- User-scoped incremental write-through: load/replace only affected users
  (plus `__cold_start__`) via `OutputSink.replace_recommendations_for_users`
  (returns post-write distinct user count) instead of full-frame overwrite.
  Updater keeps an LRU-bounded per-user cache (default 2048); dataset
  `count_recommendation_users` projects only `user_id` from parquet, and
  `load_recommendations_for_users` uses parquet `filters` for `user_id` when
  the engine supports predicate pushdown.
- Incremental events Prometheus metrics on serve `/metrics` (source lag /
  connected, flush counters, last success timestamp, tick errors) and an
  incremental-events panel on the Basic-Auth dashboard (from manifests).
- Incremental events between full retrains: internal `EventSource` surface,
  webhook `POST /events`, micro-batch buffer/worker, and write-through
  updater for popular/latest slices (`[events]` config). Operator guide:
  `docs/incremental-events.md`. Webhook `occurred_at` requires an explicit
  timezone (`Z` / offset) or Unix epoch seconds (UTC).
- DB event source (`events.kind = "db"`): watermark poll over
  `events_table` / `events_query`, durable optional `watermark_path`,
  watermark advances only on successful flush ack.
- S3-compatible event source (`events.kind = "s3"`), R2-first: list/marker
  poll via the same `build_s3_client` / `endpoint_url` options as dataset
  I/O; optional AWS-only SQS mode (rejected with `endpoint_url`). JSON
  object/array payloads; ack advances marker or deletes the SQS message.

### Changed

- Docs: `docs/how-it-works.md` (pipeline, strategies, papers); incremental
  events operator guide; tutorial webhook step; site sidebar/homepage
  cards; OpenAPI `x-codeSamples` for `POST /events`; PyPI extras
  (`cicerone-recommender[sequential]` / `[redis]`) next to `requirements-*.txt`;
  example TOML / OpenAPI regenerate command for pip hosts; missing-package
  errors name the PyPI extras; operator ingest recap lives in
  `docs/incremental-events.md` (other pages link it).
- Parse article `draft` from YAML frontmatter; article layout CSS keys off
  `data-cicerone-articles` rather than starlight-blog class names.
- Share the articles URL prefix between the Starlight plugin and layout
  classifier; `robots.txt` allows the site and disallows `/pagefind/`.
- Article plugin gating passes `{ production }` explicitly; layout kind
  matches starlight-blog listing routes and ignores a missing route id.
- Drop the Starlight “Edit page” footer; site content is edited in git.
- Docker `package` stage validates the wheel via `python -m cicerone.packaging`
  (selects `cicerone_recommender-<version>` including PEP 440 local versions
  and numeric wheel build tags).

- README and docs-site dashboard screenshot include the user recommendation lookup.
- Docs site copies `docs/images/` into `website/public/images/docs/` at build time.
- Dashboard lookup form is labeled, results are announced, and job-run
  tables expose captions / column headers; helper text contrast is higher.
- Dashboard inspector k is `min(job.top_k, dashboard.lookup_k)` (default 20).

- Bump `pyarrow` 25.0.0 → 25.0.1 (#85), `SQLAlchemy` 2.0.51 → 2.0.52 (#86),
  `uvicorn` 0.52.1 → 0.52.3 (#87), `ruff` 0.16.2 → 0.16.3 (#88).
- Bump GitHub Actions Pages deploy helpers: `actions/upload-pages-artifact`
  v3 → v5 (Dependabot #79), `actions/deploy-pages` v4 → v5 (#80).
- Bump `fastapi` 0.140.13 → 0.141.1 (Dependabot #69).
- Dependabot: ignore `numpy` major bumps (Python 3.11 CI) and `boto3>=1.43.57`
  (aiobotocore botocore pin).
- Project docs site (not part of the runtime product): Starlight under
  `website/`, synced from `docs/`, published at [cicerone.dev](https://cicerone.dev).

### Fixed

- OpenAPI `POST /events` curl sample JSON-encodes `USER_ID` with `python3`
  (falls back to `python`, then `jq`) and errors if none is on PATH.
  `examples/serve/curl_examples.sh` sources the same
  `src/cicerone/serve/python_detect.sh` snippet the OpenAPI samples embed.
- Docker test image includes `examples/serve/` so CI can read `curl_examples.sh`.
- Tutorial webhook step starts serve with `cicerone --config … serve` (same as
  the HTTP API step).
- `cicerone users` with a config path requires enabled `dashboard.users_path`
  (or an explicit `--users-path`); the error names the loaded config and the
  dashboard settings that were resolved.

- Dashboard still starts if the recommendation store cannot be opened (lookup disabled).
- Dashboard lookup errors show a generic message; details stay in the logs.
- Dashboard lookup URL updates keep the hash fragment.
- Dashboard lookup disables the Look up button during the htmx request.
- Postgres `is_locked()` logs and re-raises probe failures instead of
  treating a dead database as “lock free”; `owned()` logs before fail-closed.
- S3 EventSource `nack` returns events to the local pending queue (and
  extends SQS visibility) instead of dropping the batch. SQS HA lock-busy
  nacks can retry immediately; list-mode array payloads no longer lose
  sibling events when one id is nacked.
- Event worker ack/nack bookkeeping: buffer duplicates are acked (not left
  in-flight), capacity overflow is nacked for redelivery, and stop drains
  the buffer once before closing the source.
- DB event source poll uses `(occurred_at, event_id)` cursor/order when an
  `event_id` column exists so same-timestamp pages cannot skip rows.

## [0.5.1] - 2026-08-12

### Added

- Serve OpenAPI / ReDoc ``x-codeSamples`` (Ruby, Python, JavaScript, Shell)
  for `/health` and `/recommendations/{user_id}`.

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

### Fixed

- Redis lock `release()` joins the refresher (≤250ms) and ignores in-flight
  refresh failures after an intentional stop, avoiding a `_mark_lost` race.
- Retrain Prometheus labels use the real trigger source (`cron`, `s3-poll`,
  `webhook`, …) instead of collapsing non-webhook to `poll`.
- Serve fails closed when `features.toml` cannot be loaded (no silent disable
  of availability filters).
- Input poller treats local `stat` errors like S3 failures (log and continue).

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

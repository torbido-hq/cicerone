<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Configuration

Cicerone reads one TOML file (`cicerone --config PATH` or
`CICERONE_CONFIG_PATH`; default `/app/config/cicerone.toml`) plus
`features.toml` at `[job].feature_config_path` (default
`/app/config/features.toml`). Annotated examples live in those shipped
files. This page is the user-facing reference for the current 0.8 line.

`${ENV_VAR}` in **parsed string values** is required at load; an unset
variable fails the process. Comments are not resolved. Write a literal
`${…}` as `$${NAME}`.

`[input]` and `[output]` are required and each needs `kind`. Every other
table below is optional unless a later rule says otherwise.

## CLI

| Command | What it does |
| --- | --- |
| `cicerone job` | One batch `job.run()`, then exit. |
| `cicerone start` (alias `run`) | `job.mode = "batch"` (default): one job immediately, then the scheduler loop. `job.mode = "serve"`: start the read API only (no job, no cron). |
| `cicerone scheduler` | Cron loop (and `[job.trigger]` if enabled). Does **not** run a job first; it sleeps until the next `cron_schedule` tick. |
| `cicerone serve` | Read API. Requires `job.mode = "serve"`. |
| `cicerone dashboard` | Basic-Auth status UI. Independent of `job.mode`. |
| `cicerone users` | `add` / `remove` / `list` dashboard users. `--users-path` or, with `--config`, `[dashboard].users_path`. |
| `cicerone export-openapi` | Write the serve OpenAPI document (`-o PATH`). |

Global flags: `-c` / `--config`, `--log-level` (default `INFO`),
`--log-format`, `-V`. `--config` may sit before or after the subcommand.

## Input and output

Required. `kind` is `"dataset"` or `"db"`. Input and output are independent
(read Postgres, write local parquet, or the reverse).

**`kind = "dataset"`** — parquet on local disk or S3-compatible storage.

| Option | Role |
| --- | --- |
| `storage_backend` | `"local"` or `"s3"` |
| `path` | Local directory (`storage_backend = "local"`) |
| `endpoint_url`, `access_key_id`, `secret_access_key`, `bucket`, `prefix` | S3-compatible (`storage_backend = "s3"`) |

**`kind = "db"`** — SQLAlchemy. Typical option: `database_url`. Optional
`events_query` / `users_query` / `items_query` (trusted deploy-time
`SELECT` only) read an existing schema instead of `events` / `users` /
`items` tables. Default write tables: `recommendations`,
`recommendation_runs`, `model_artifacts`, `recommendation_items`.

`[output].artifact_hmac_key` (16+ bytes) is required when
`[events.online]` is on. Serve never reads `[input]`; the loader still
requires the table (use a placeholder path).

## `[job]`

| Key | Default | Notes |
| --- | --- | --- |
| `mode` | `"batch"` | `"batch"` or `"serve"` |
| `top_k` | `10` | Rows written per user |
| `half_life_days` | `90` | Recency decay on `occurred_at` |
| `cron_schedule` | `"0 3 * * *"` | 5-field cron, UTC |
| `feature_config_path` | `/app/config/features.toml` | Host installs must point this at a real file |
| `models` | `["collaborative", "item_based", "popular"]` | Empty list is an error. Opt-in names: `sequential`, `ease`, `als`, `content_fallback`, `popular_in_category`, `latest`, `random` |
| `model_weights` | unset | Set (even `{}`) to enable weighted RRF |
| `rrf_k` | `60` when fusion is on | Must be positive; place it **above** `[job.model_weights]` in TOML |
| `save_model_artifact` | `false` | Writes `model.artifact` (dataset) or `model_artifacts` (db) |
| `max_workers` | `1` | Process pool for fit / AutoML |
| `log_epoch_metrics` | `false` | Per-epoch Precision/Recall for collaborative / sequential |
| `content_fallback.enabled` | `false` | Cold catalog items by feature cosine |
| `sequential.min_median_interactions` | `5` | AutoML drops sequential below this |
| `explain.enabled` | `true` | Persist `reasons` JSON on each output row |

`[job.automl]`: `enabled` default `false`; `n_splits = 2`,
`test_days = 14`, `primary_metric = "MAP"`, `debias = false`. Override the
search space with `[[job.automl.candidates]]`.

Algorithms: [how-it-works.md](how-it-works.md).

## Job trigger

Optional retrain webhook **in addition to** cron. Runs inside
`cicerone start` / `cicerone scheduler`, not `cicerone serve`.

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | — |
| `auth_token` | unset | **Required** when enabled |
| `host` | `127.0.0.1` | Compose / `docker run -p` needs `0.0.0.0` |
| `port` | `8080` | `POST /trigger/retrain` |
| `debounce_seconds` | `60` | In-flight or recent runs are skipped, not queued |
| `poll_input_bucket` | `false` | Dataset input mtime / S3 `LastModified` |
| `poll_interval_seconds` | `300` | — |
| `lock_backend` | `"in_process"` | `"postgres"` or `"redis"` for multi-replica |

## `[job.eval]`

Optional production replay of **previously written** lists against later
`[input]` events (HitRate / MAP / NDCG / …). This is not CTR.
`enabled` default `false`.

## `[serve]`

Required `auth_token` when `job.mode = "serve"`.

| Key | Default | Notes |
| --- | --- | --- |
| `host` | `127.0.0.1` | Published Docker ports need `0.0.0.0` |
| `port` | `8000` | — |
| `default_k` | `10` (max 100) | Used when the client omits `limit` / `k` |
| `refresh_interval_seconds` | `60` | Dataset output: reload parquet cache. DB output: reload the items snapshot only |
| `category_column` | `"category"` | `?category=` filter |
| `metrics_enabled` | `false` | `GET /metrics`; needs `metrics_token` |
| `log_impressions` | `false` | Requires `[track]`. Records **returned** GET items, not browser renders. The HTTP response can return before the row is written |

`GET /recommendations/{user_id}` is a lookup of materialized rows. It does
not fit a model. **200** returns the object; **401** missing/invalid
bearer; **404** no rows and no fallback. Dataset vs DB visibility:
[how-it-works.md](how-it-works.md#what-get-recommendations-does).

## `[events]`

Serve-process incremental ingest. Off until `enabled = true`.
`GET /recommendations` never drives this worker.

| Key | Default | Notes |
| --- | --- | --- |
| `kind` | `"webhook"` | `webhook`, `db`, `s3`, `redis_streams`, `kafka`, `rabbitmq` |
| `ha` | `false` | Requires `job.trigger.lock_backend` `postgres` or `redis` |
| `incremental.batch_size` | `100` | Flush when the buffer hits this **or** the window |
| `incremental.batch_window_seconds` | `60` | — |
| `incremental.poll_interval_seconds` | `1` | Worker tick |
| `options.max_pending` | `10000` (minimum 100) | Webhook HTTP 429 when full |
| `options.max_body_bytes` | `1048576` (1 MiB) | Shared with `POST /track` |
| `options.auth_token` | `serve.auth_token` | Webhook Bearer |

`POST /events` **202** means the webhook queue accepted the event. It does
not mean recommendations changed. Webhook pending state is **in-memory**.
Operator guide: [incremental-events.md](incremental-events.md).

### `[events.online]`

Optional LightFM `fit_partial` in the **events worker** (not on GET).
Skipped while `[experiment]` is on. Requires `[events]` enabled, a signed
model artifact (`save_model_artifact = true` plus
`output.artifact_hmac_key`), and `collaborative` as LightFM.

| Key | Default |
| --- | --- |
| `enabled` | `false` |
| `fit_partial_epochs` | `1` (`0` = frozen weights + history refresh) |
| `fit_min_events` | `100` |
| `max_extra_interactions` | `50000` |

## `[track]`

Quality contract. Rows never enter `[event_weights]` or LightFM.

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Mounts `POST /track`. Requires db output or a **local** dataset path (S3 JSONL append is refused) |
| `attribution_window_hours` | `24` | Click ↔ impression match window |
| `conversion_event_types` | empty → `"purchase"` when scoring CTR/conversion | `[input]` event types treated as conversions |
| `min_impressions` | `100` | Gates Promote / Thompson volume. Does **not** change the CTR formula |

`kind` is `impression` or `click` only. Details:
[evaluation.md](evaluation.md).

## `[experiment]`

Sticky user → variant assignment over **materialized per-variant lists**.
Not a per-request coin flip and not a browser cookie.

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Needs `id` and at least two `[[experiment.variants]]` (unless `automl_challenger`) |
| `primary_metric` | `"weighted"` | Or `"ctr"`, `"conversion"`, or an `event_type` |
| `attribution` | `"user"` | `user` (0.7.0 ITT) / `click` / `impression` / `recommended` |
| `allocation` | `"fixed"` | `"thompson"` needs `[track]` and `cicerone-recommender[bandits]` |
| `alpha` | `0.05` | Sequential CI level |
| `log_exposures` | `false` | Local JSONL or db table; S3 append refused |
| `explore_traffic` | `0.5` | Thompson only |
| `rotate_min_prob` | `0.9` | Thompson only |

Variant keys: `name`, `traffic` (remainder of a sum `< 1` goes to the last
variant), optional `models` / `weights` / `rrf_k` / `combiner` /
`blending` / `boosts` / `eligibility`. Changing `traffic` remaps users.
[experiments.md](experiments.md).

## `[dashboard]`

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Required for `cicerone users` to read `users_path` from this file |
| `host` | `127.0.0.1` | Compose needs `0.0.0.0` |
| `port` | `8090` | — |
| `users_path` | `/app/config/dashboard_users.toml` | HTTP Basic |
| `refresh_interval_seconds` | `30` | Status poll |
| `history_limit` | `20` | Meaningful for db output only (dataset `manifest.json` is overwritten) |
| `lookup_k` | `20` | Inspector recs, capped by `job.top_k` |
| `lookup_events` | `20` | Inspector `[input]` events |
| `lookup_user_attrs` | empty | Allowlisted user columns |

## `[publish]`

Optional sidecar after `[output]` writes. `enabled` default `false`;
`kind` `"kafka"` or `"rabbitmq"`. Serve still reads `[output]`.
Independent of `[events]`.

## `[model.*]`

RecTools `model_from_config` tables (`cls` plus hyperparameters).
`content_fallback` stays under `[job.content_fallback]`. Examples:
`config/cicerone.toml`. Changing `[model.collaborative].cls` away from
`LightFMWrapperModel` breaks `[events.online]`.

## Feature config

Separate file. Root keys must precede table headers.

| Key / table | Role |
| --- | --- |
| `[event_weights]` | Base weight per `event_type`. Types present in events but missing here are dropped |
| `quantity_scaled_events` | Those types also scale by `log1p(quantity)` (shipped default includes `purchase`) |
| `[event_caps]` | Keep the most recent N events per `(user, item, event_type)` |
| `[[user_features]]` / `[[item_features]]` | `"categorical"` or `"list"` columns |
| `item_availability_filters` | Boolean item columns that must all be true (shipped: `published`, `in_stock`) |
| `[[eligibility]]` / `[[boost]]` | Hard allowlists and soft re-rank. `boost_overfetch_factor` default `3` |
| `[blending]` | Per-user mix of personalized / popular / date-based latest. Wins over RRF if both are set |

## Where state lives

| Layer | What | Survives process restart? |
| --- | --- | --- |
| Webhook `_pending` / `_in_flight` | Accepted `POST /events` not yet flushed | **No** (in-memory) |
| Serve dataset cache | `recommendations.parquet` + items snapshot | Reloads from output; a hard kill mid-refresh keeps the previous cache |
| Serve DB reader | Recs queried per GET; items snapshot cached | Recs yes (SQL); snapshot reloads on the refresh timer |
| `[output]` | Recommendation rows, manifest, optional artifact / items snapshot | Yes |
| Track store | `track.jsonl` or `recommendation_track` | Yes, after `POST /track` returns 202. `log_impressions` may still be in-flight |
| Event backends `db` / `s3` / Redis / Kafka / RabbitMQ | Source cursor or broker backlog | Yes, per [incremental-events.md](incremental-events.md#delivery-semantics) |
| Experiment promote | `experiment_state.json` or `experiment_state` table | Yes |

Architecture package map: [architecture.md](architecture.md).

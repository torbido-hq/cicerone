<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Configuration

Cicerone reads one TOML file (`cicerone --config PATH` or
`CICERONE_CONFIG_PATH`; default `/app/config/cicerone.toml`) plus
`features.toml` at `[job].feature_config_path` (default
`/app/config/features.toml`). Annotated examples live in those shipped
files. This page is the user-facing reference for the current 0.8 line.

`${ENV_VAR}` is resolved at load in `[input]` / `[output]` `options`,
`output.artifact_hmac_key`, `serve.auth_token` / `metrics_token`,
`job.trigger.auth_token` / `postgres_url` / `redis_url`, `events.options`
and `publish.options`; an unset variable fails the process. Anywhere else
`${…}` stays literal. Comments are not resolved. Write a literal `${…}` as
`$${NAME}`.

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
| `cicerone users` | `add` / `remove` / `list` dashboard users. `--users-path` or, with `--config`, `[dashboard].users_path`. Passwords are at most 72 bytes (UTF-8). |
| `cicerone export-openapi` | Write the serve OpenAPI document (`-o PATH`). |

Global flags: `-c` / `--config`, `--log-level` (default `INFO`, or
`CICERONE_LOG_LEVEL`), `--log-format` (or `CICERONE_LOG_FORMAT`), `-V`.
`--config` may sit before or after the subcommand.

## Input and output

Required. `kind` is `"dataset"` or `"db"`. Input and output are independent
(read Postgres, write local parquet, or the reverse).

**`kind = "dataset"`** — parquet on local disk or S3-compatible storage.

| Option | Role |
| --- | --- |
| `storage_backend` | `"local"` (default) or `"s3"` |
| `path` | Local directory (`storage_backend = "local"`) |
| `endpoint_url`, `access_key_id`, `secret_access_key`, `bucket`, `prefix` | S3-compatible (`storage_backend = "s3"`) |
| `region` | S3 region, default `"auto"` |

**`kind = "db"`** — SQLAlchemy. Typical option: `database_url`. Optional
`events_query` / `users_query` / `items_query` (trusted deploy-time
`SELECT` only) read an existing schema instead of `events` / `users` /
`items` tables (`events_table` / `users_table` / `items_table`). Default
write tables, each renamed by the `[output.options]` key in brackets:
`recommendations` (`recommendations_table`), `recommendation_runs`
(`manifest_table`), `model_artifacts` (`model_artifact_table`),
`recommendation_items` (`recommendation_items_table`),
`incremental_applied_events` (`applied_events_table`, see
[incremental events](incremental-events.md)), `recommendation_exposures`
(`exposures_table`), `experiment_state` (`experiment_state_table`),
`recommendation_track` (`track_table`), `recommendation_eval`
(`eval_table`) and `recommendation_history` (`history_table`).

The db events source supports Postgres and SQLite, not MySQL. The track, eval,
history, experiment-state and exposures tables also support MySQL: install
`pip install 'cicerone-recommender[mysql]'` (or `requirements-mysql.txt`) and
use `database_url = "mysql+pymysql://user:password@host:3306/db?ssl_ca=/path/to/ca.pem"`.
`ssl_ca` turns on TLS and verifies the server's certificate and host name;
PyMySQL uses plaintext without it, which is only suitable on a trusted local
network. The Docker image doesn't ship the driver, so extend it to
use MySQL. On MySQL, `POST /track` answers 400 for an `event_id` over 255
characters or another text value over 65,535 UTF-8 bytes, and nothing from
that batch is written. Strict mode is required: on a non-strict server an
out-of-range `rank` can be clamped, and an oversized eval report or history
value is truncated silently, so a truncated report replaces the good one and
then reads as missing. On a strict server an eval report over 16 MiB (or over
`max_allowed_packet`, if that is lower) or a history value over 65,535 UTF-8
bytes fails that write: the job logs it, carries on and keeps the rows already
stored. Config load refuses an `experiment.id` over 255 characters on MySQL.
The eval, history and exposures tables have no primary key, so
`sql_require_primary_key` and Group Replication refuse them; the job logs the
error on every run, and serve on every exposure. Cicerone alters no existing
table, except that experiment state adds a missing `payload` column to a
hand-made table, as it does on Postgres and SQLite. The job skips the eval
report and history on MariaDB with a logged warning; experiment state isn't
supported there and its writes fail with a server error, including during
Thompson jobs. Requires MySQL 8.0 or later; CI tests 8.4. Track ingest, the
eval report and experiment state need the `CREATE` privilege on every write;
history and exposures need it only to create their tables.
New history tables on any backend store `rank` as `BIGINT`, so a fractional
rank rounds on Postgres and MySQL.
Known limit: the history table has no index and no retention, so each read
scans it and it grows every run.
A track table dropped while the store runs isn't recreated until
restart. With `[track]` enabled, serve refuses to start on a database other
than Postgres, SQLite or MySQL.
On MySQL/MariaDB the `recommendations` and `recommendation_items` tables
must use a transactional engine (InnoDB): a full write empties them with
`DELETE` and rolls back on failure.
New `model_artifacts` tables use `LONGBLOB` and `DATETIME(6)` there. A table
created before 0.8.6 stays a 64 KB `BLOB`, and its one-second `written_at` lets
a same-second write slip past the online update's check, until you run
`ALTER TABLE model_artifacts MODIFY payload LONGBLOB NOT NULL, MODIFY written_at DATETIME(6) NOT NULL`
(use your `model_artifact_table` name). The server's `max_allowed_packet` (64 MiB by
default on MySQL) still caps an artifact; raise it for larger ones.

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
| `epoch_metrics_every` / `_max_users` / `_regression_drop` / `_plateau_eps` / `_plateau_window` | `5` / `500` / `0.25` / `0.01` / `3` | Only with `log_epoch_metrics` |
| `item_based.k_neighbors` | `20` | Legacy; prefer `[model.item_based.model].K` |
| `content_fallback.enabled` | `false` | Cold catalog items by feature cosine |
| `content_fallback.max_neighbors` | `50` | — |
| `sequential.min_median_interactions` | `5` | AutoML drops sequential below this |
| `explain.enabled` | `true` | Persist `reasons` JSON on each output row |
| `explain.max_similar_items` / `explain.max_attributes` | `3` / `5` | Caps inside `reasons` |

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
| `port` | `8080` (1–65535; 0 deprecated) | `POST /trigger/retrain` (`202` started, `429` busy or debounced); `GET /health` |
| `debounce_seconds` | `60` | From the last guarded run's start. In-flight or recent runs are skipped, not queued |
| `poll_input_bucket` | `false` | Dataset input only: `events.parquet` mtime / S3 `LastModified` |
| `poll_input_single_replica` | `false` | **Required** with `poll_input_bucket` and a `postgres` / `redis` lock; enable polling on one replica only |
| `poll_interval_seconds` | `300` | — |
| `lock_backend` | `"in_process"` | `"postgres"` or `"redis"` for multi-replica |
| `postgres_url` | unset | Postgres lock; falls back to `[output.options].database_url` when output is `db`, else load fails |
| `redis_url` | unset | **Required** with `lock_backend = "redis"` |
| `lock_key` | `"cicerone:scheduler:run_guard"` | Non-empty; namespaces the lock |
| `lock_ttl_seconds` | `86400` | Redis TTL, must be positive |

## `[job.eval]`

Optional production replay of **previously written** lists against later
`[input]` events (HitRate / MAP / NDCG / …). This is not CTR.
`enabled` default `false`. Optional `event_types` (default: the
`track.conversion_event_types` rule below) and `ks` (default `5`, `10` and
`top_k`, capped at `top_k`).

## `[serve]`

Required `auth_token` when `job.mode = "serve"`.

| Key | Default | Notes |
| --- | --- | --- |
| `host` | `127.0.0.1` | Published Docker ports need `0.0.0.0` |
| `port` | `8000` (1–65535; 0 deprecated) | — |
| `default_k` | `10` (max 100) | Used when the client omits `limit` / `k` |
| `refresh_interval_seconds` | `60` | Dataset output: reload parquet cache. DB output: reload the items snapshot only |
| `category_column` | `"category"` | `?category=` filter |
| `metrics_enabled` | `false` | `GET /metrics`; needs `metrics_token` |
| `metrics_token` | unset | **Required** when `metrics_enabled`; scrapes send it as `X-Metrics-Token` |
| `log_impressions` | `false` | Requires `[track]`. Records **returned** GET items, not browser renders. The HTTP response can return before the row is written; with 1000 appends pending the rows are dropped and counted in `cicerone_serve_appends_dropped_total` |

`GET /recommendations/{user_id}` is a lookup of materialized rows. It does
not fit a model. **200** returns the object; **400** `limit` and `k`
disagree; **401** missing/invalid bearer; **404** no rows and no fallback;
**422** `limit` / `k` outside 1–100. User ids containing `/` are
not supported (an encoded `%2F` is decoded before routing and still
404s). `exclude_unavailable` (default `true`) re-applies
`item_availability_filters`. Dataset vs DB visibility:
[how-it-works.md](how-it-works.md#what-get-recommendations-does).

## `[events]`

Serve-process incremental ingest. Off until `enabled = true`.
`GET /recommendations` never drives this worker.

| Key | Default | Notes |
| --- | --- | --- |
| `kind` | `"webhook"` | `webhook`, `db`, `s3`, `redis_streams`, `kafka`, `rabbitmq` (checked only when enabled) |
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
| `enabled` | `false` | Mounts `POST /track`. Requires db output or a **local** dataset path (S3 JSONL append is refused), unless `redis_url` is set |
| `redis_url` | unset | Impression/click log in Redis (`cicerone-recommender[redis]`). History and eval stay on `[output]`. Allows `events.ha` without db output. New rows go to Redis; rows already on `[output]` are still read, deduped by `event_id`. Turning it off again stops reading the Redis rows, and they are not copied back. Set it only once every replica runs a version that supports it: older versions ignore the key (with an `Unknown config key` warning) and never read the Redis log |
| `redis_key_prefix` | derived from `[output]` | Prefix of the Redis log key (`<prefix>:track`). By default it is derived from the `[output]` location (credentials excluded), so deployments with different outputs never share a log; set it to share or rename one. Serve logs `Track log: redis key <key> (<n> rows)` at startup, so a changed prefix shows as a new key or an empty log. Client calls time out after 5 s |
| `attribution_window_hours` | `24` | Click ↔ impression match window |
| `conversion_event_types` | empty → `"purchase"`, or the experiment's event-type `primary_metric` | `[input]` event types treated as conversions |
| `min_impressions` | `100` | Gates Promote / Thompson volume; the sequential test needs this many users per arm. Does **not** change the CTR formula |

`kind` is `impression` or `click` only. Details:
[evaluation.md](evaluation.md).

## `[experiment]`

Sticky user → variant assignment over **materialized per-variant lists**.
Not a per-request coin flip and not a browser cookie.

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Needs `id` and at least two `[[experiment.variants]]` (unless `automl_challenger`) |
| `primary_metric` | `"weighted"` | Or `"ctr"`, `"conversion"`, or an `event_type` |
| `attribution` | `"user"` | `user` (0.7.0 ITT) / `click` / `impression` / `recommended`. `click` / `impression` go with `ctr` / `conversion`, both ways |
| `allocation` | `"fixed"` | `"thompson"` needs `[track]`, `cicerone-recommender[bandits]`, `primary_metric = "conversion"` and `click` / `impression` attribution |
| `automl_challenger` | `false` | Last successful manifest vs this run's AutoML pick; variants optional |
| `alpha` | `0.05` | Sequential CI level |
| `log_exposures` | `false` | Local JSONL or db table; S3 append refused |
| `explore_traffic` | `0.5` | Thompson only, in `[0, 1]` |
| `rotate_min_prob` | `0.9` | Thompson only, in `(0, 1)` |

Variant keys: `name`, `traffic` (remainder of a sum `< 1` goes to the last
variant), optional `models` / `weights` (alias `model_weights`) / `rrf_k` /
`combiner` / `blending` / `boosts` (or `[[experiment.variants.boost]]`
tables) / `eligibility`. Changing `traffic` remaps users.
[experiments.md](experiments.md).

## `[dashboard]`

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Required to start `cicerone dashboard` (which also needs at least one user) and for `cicerone users` to read `users_path` from this file |
| `host` | `127.0.0.1` | Compose needs `0.0.0.0` |
| `port` | `8090` (1–65535; 0 deprecated) | — |
| `users_path` | `/app/config/dashboard_users.toml` | HTTP Basic |
| `refresh_interval_seconds` | `30` | Status poll |
| `history_limit` | `20` | Meaningful for db output only (dataset `manifest.json` is overwritten) |
| `lookup_k` | `20` | Deprecated and ignored (removed in 0.9.0); warns when the dashboard is enabled; Inspect shows `serve.default_k` rows |
| `lookup_events` | `20` | Inspector `[input]` events |
| `lookup_user_attrs` | empty | Allowlisted user columns |
| `trigger_url` | derived | Retrain webhook base URL as the dashboard reaches it (e.g. `http://recommender:8080`). Unset: `http://127.0.0.1:<[job.trigger].port>` (`[::1]` for an IPv6 wildcard), only when `[job.trigger].host` is loopback or a wildcard. Set it to show the Retrain button when the dashboard config has no `[job.trigger]`, and whenever the dashboard runs in a different container or host from the trigger: the derived `127.0.0.1` is the dashboard's own. No query or fragment. `http` to a non-loopback host is refused unless `trigger_allow_http` is set |
| `trigger_auth_token` | `[job.trigger].auth_token` of this config | Bearer token for the Retrain request; `${ENV}` placeholders are resolved. The fallback only works when this same config file has `[job.trigger].auth_token`; with separate dashboard and job configs, set it here |
| `trigger_allow_http` | `false` | Allow a plain-`http` `trigger_url` to a non-loopback host. The token is then sent in clear, so only on a private network (e.g. a compose network) |

The dashboard loads `[job].feature_config_path` at startup to drop unavailable items from Inspect user, as serve does.
Inspect takes `[serve]`, `[job].feature_config_path` and `[experiment]` from the dashboard's own config file, so they must match serve's.
After a promote, Inspect can show the new variant up to one serve `refresh_interval_seconds` before serve does.

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
| `[[eligibility]]` / `[[boost]]` | Hard allowlists (`on_missing_user` `"exclude"` default or `"allow"`) and soft re-rank (`[[boosts]]` alias). `boost_overfetch_factor` default `3` |
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

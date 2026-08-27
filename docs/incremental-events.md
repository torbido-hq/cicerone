<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Incremental events

Cicerone can fold new interaction events into recommendations **between**
full batch retrains. Enable `[events]` on the **serve** process
(`cicerone serve` or the serve container): sources normalize to the same
event contract, micro-batch, then write-through to the same `[output]`
store serve already reads.

This is not live ranking on `GET /recommendations`. LightFM / item-KNN /
content-fallback rows wait for the next `job.run()` **unless**
`[events.online]` is enabled: the serve events worker then continues LightFM
(`fit_partial`) on IDs already in the last model artifact and rewrites
personalized / item-KNN / content-fallback rows for affected users.
Sequential never runs `fit_partial`. The default runtime image is
torch-free. New catalog IDs still wait for a full retrain.

The incremental path always refreshes **popular / latest slices** (and recency
boosts) for affected users plus `__cold_start__`. When `[experiment]` is
on, that popular/latest refresh runs **in every variant**. Online LightFM
rewrite is skipped while `[experiment]` is on so arms stay isolated.
[how-it-works.md](how-it-works.md) explains the split. Experiments:
[experiments.md](experiments.md).

Preserved personalized / `item_based` / `sequential` / `content_fallback`
/ `blended` rows keep their batch `reasons` unless `[events.online]` replaced
them. New popular, latest, and
`incremental` rows get a minimal `{sources:[{label}]}` payload — no
history-overlap `similar_items` on this path.

Batch I/O and serve packages: [architecture.md](architecture.md).

## What it does not do

- Request-path LightFM / SASRec inference (`GET` stays a lookup)
- Growing LightFM embeddings for brand-new user/item IDs (those wait for `job.run()`)
- Sequential `fit_partial` (SASRec/BERT4Rec/HSTU stay batch)
- A public plugin API (`EventSource` is internal, same spirit as `io/` kinds)

## Event contract

Same columns as batch input:

| Column | Required | Notes |
| --- | --- | --- |
| `user_id` | yes | string |
| `item_id` | yes | string |
| `event_type` | yes | must appear in `features.toml` `[event_weights]` |
| `quantity` | no | default `1` |
| `occurred_at` | yes | timezone-aware: ISO-8601 with `Z` or offset, or Unix epoch seconds (UTC) |

Optional transport fields:

- `event_id` — idempotency / ack key (generated if omitted)
- `idempotency_key` — alias accepted by the webhook JSON body

## Enable the webhook

On the serve config (`config/cicerone.serve.toml` or your local copy):

```toml
[events]
enabled = true
ha = false
kind = "webhook"

[events.options]
# auth_token = "${EVENTS_AUTH_TOKEN}"  # optional; defaults to serve.auth_token
# max_pending = 10000                  # HTTP 429 when full; minimum 100

[events.incremental]
batch_size = 100
batch_window_seconds = 60
poll_interval_seconds = 1
```

### Online collaborative refresh

Optional `[events.online]` continues LightFM on the last model artifact and
rewrites personalized / item-KNN / content-fallback rows for **affected
users only**. While `[experiment]` is on, that rewrite is skipped so arms
stay isolated (popular/latest still refresh every variant).
`GET /recommendations` is still a lookup.

```toml
[events.online]
enabled = true
fit_partial_epochs = 1          # 0 = frozen weights + history refresh only
fit_min_events = 100            # skip SGD until this many known-ID events
```

Startup fails if the `[output]` store has no artifact — the batch job must
set `[job].save_model_artifact = true`. An event is trained only when both
its `user_id` and `item_id` already exist in that artifact (including a new
interaction between two known IDs). Unknown IDs still get popular / latest /
`incremental` boosts and wait for the next `job.run()`. After
`fit_partial`, item factors move globally but only the flush's users are
rewritten (same class of staleness as Gorse's cache between worker
passes). Sequential never runs `fit_partial`. If the sequential extra is
missing, existing sequential rows are left as-is; if it is installed,
affected users are re-scored from the batch-fitted sequential model.

`POST /events` uses Bearer auth (`events.options.auth_token` or
`serve.auth_token`). Body: one event object, a JSON array, or
`{"events":[...]}`. Accepted events return **202** with `accepted` and
`event_ids`. Invalid JSON / contract → **400**. Full backlog
(`max_pending`) → **429**.

```sh
curl -sS -X POST \
  -H "Authorization: Bearer $SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","item_id":"ipa-001","event_type":"purchase","quantity":1,"occurred_at":"2026-08-19T12:00:00Z"}' \
  http://localhost:8000/events
```

Flushes run when the buffer hits `batch_size` **or**
`batch_window_seconds` elapses. Then serve's refresh loop (dataset output)
or the next DB read picks up the new rows. OpenAPI documents the route when
webhook events are enabled (`/docs`, `/redoc`, checked-in
`docs/openapi/serve.openapi.json`).

Full retrain remains `[job]` cron + `[job.trigger]`. Incremental updates
are a separate cheap path between those runs.

## Other sources

`kind` is one of the **shipped** backends: `webhook`, `db`, `s3`,
`redis_streams`. Annotated examples: `config/cicerone.toml`.

### `db`

Watermark poll over `events_table` / `events_query`. No `POST /events`.
Required: `events.options.database_url`. Optional durable `watermark_path`
and `initial_watermark`. The watermark advances only after a successful
flush `ack`. When an `event_id` column exists, poll orders by
`(occurred_at, event_id)` so same-timestamp pages cannot skip rows.
Without `event_id`, table sources use `id`, SQLite `rowid`, or Postgres
`ctid` so identical payloads at the same timestamp stay distinct.
`events_query`, when set, must be a single read-only `SELECT` from
**trusted deploy-time config** (same rule as `input.options.events_query`).
It has no table identity: project `event_id` or `id`; otherwise same-payload twins
share a cursor.

### `s3`

S3-compatible (R2-first): same `endpoint_url` + credentials shape as
dataset I/O. Default `mode = "list"` (list/marker under `prefix`; durable
`marker_path`). Objects are JSON event objects or arrays; missing
`event_id` uses `bucket/key|etag|index`. List mode is a **single consumer**
per bucket/prefix.

AWS-only `mode = "sqs"` (`queue_url`) is rejected when `endpoint_url` is
set. `nack` requeues locally and extends SQS visibility so a lock-busy
replica can retry without waiting out the visibility timeout.

### `redis_streams`

Consumer group (`XREADGROUP` / `XACK` / `XAUTOCLAIM` for idle PEL
recovery). Required: `redis_url`, `stream`, `consumer_group`. Optional
`consumer_name` (default hostname), `group_start_id` (`0-0`), `block_ms`,
`claim_idle_ms`. Flat hash fields match the event contract; missing
`event_id` uses the stream entry id. Requires
`pip install 'cicerone-recommender[redis]'` or
`pip install -r requirements-redis.txt` (same extra as the Redis lock).

## High availability

Default is **one writer process**. Dataset output is whole-object
read-modify-write — multi-replica serve is only safe with a leader.

Set `events.ha = true` **and** `job.trigger.lock_backend` to `postgres` or
`redis`. Config fails fast otherwise. Serve takes a **separate** apply
lease (`{lock_key}:events:apply`) around write-through (not the 24h
retrain TTL; Redis apply lease is 60s, refreshed while held). Fan-out
sources poll without the lease and acquire it when a micro-batch is ready
to flush; db / s3-list still take the lease to poll (single consumer). On
lock busy the flush nacks so lag stays honest. Redis `owned()` fences
writes if the lease expires mid-apply; Postgres `owned()` checks `pg_locks`
for this session. A dead Postgres lock probe **fails closed** (logged and
re-raised), not “lock free”. The same `owned()` callback is passed into
full `job.run()` (cron and `RunGuard` trigger) so a lost retrain lock skips
artifact and recommendation writes.

Fan-out sources **heartbeat** in-flight messages for the duration of apply:
S3 SQS extends visibility (5 minutes) so `fit_partial` cannot outrun the
receive window; Redis Streams `XCLAIM`s to the same consumer with idle 0 so
`XAUTOCLAIM` does not steal the PEL. Online persist after `ack` retries,
then drops the pending fit if it still fails or the lease is lost — serving
rows from that flush stay written.

| Source | Multi-replica |
| --- | --- |
| webhook | Single-ingress or sticky session; do not claim multi-replica ingest |
| db | Single poller (non-leader skips poll when the apply lease is configured) |
| s3 list | Single consumer; non-leader skips poll |
| s3 sqs | Delivery fan-out OK; apply still under the lease |
| redis_streams | Consumer groups + unique `consumer_name`; apply is leader-only |

The retrain interlock only engages when something supplies a busy check.
`start_events_runtime` builds the retrain probe solely when
`[events].ha = true`, and its optional `busy_check` argument is left unset by
the only caller in shipped code (`serve.app`) — no batch path starts the
events runtime at all. So with `ha = false` a flush is never deferred while a
full retrain writes, and `cicerone_events_apply_busy_total{reason="retrain"}`
stays at zero unless you embed the worker yourself and pass a check.

## Ops

Loads and writes are **user-scoped** (affected users + `__cold_start__`):
`OutputSink.replace_recommendations_for_users` updates those users without
a full-table overwrite. The updater keeps an LRU cache of those rows
(default 2048 users) between micro-batches.

Serve `/metrics` (when events are enabled):

| Metric | Meaning |
| --- | --- |
| `cicerone_events_source_lag` | Source backlog (`-1` if unknown / events off). Webhook/S3-list: pending + in-flight. S3-SQS: that plus approximate visible queue depth. Redis Streams: `XPENDING` + group `lag`. DB: rows after watermark. |
| `cicerone_events_source_connected` | `1` when the source reports connected |
| `cicerone_events_flush_total{status=}` | `success` / `busy` / `error` |
| `cicerone_events_flush_events_total` | Events applied on successful flushes |
| `cicerone_events_last_success_timestamp_seconds` | Last successful flush (Unix seconds) |
| `cicerone_events_tick_errors_total` | Unexpected exceptions outside handled flush paths |
| `cicerone_events_lock_total{status=}` | Apply-lease `acquired` / `skip` |
| `cicerone_events_leader` | `1` while this replica owns the apply lock |
| `cicerone_events_apply_busy_total{reason=}` | Skip due to `lock` or `retrain` |

Lag/connected refresh each poll cycle (not on `/metrics` scrape). The
dashboard (when `[events]` is enabled) shows the latest incremental
**success** from recent manifests. With a dataset output, a later full
retrain overwrites `manifest.json`, so the panel may go empty until the
next flush (prefer a DB output for history).

## Delivery semantics

| Source | Typical delivery | Idempotency |
| --- | --- | --- |
| Webhook | At-least-once (client retries) | `event_id` / `idempotency_key`; short dedupe window |
| DB watermark | Near exactly-once | Advance watermark only after successful flush |
| S3 list (R2) / SQS | At-least-once | Object key + ETag dedupe |
| Redis Streams | At-least-once | `XACK` after successful flush; stream entry id fallback |

Duplicate delivery can inflate weights for `quantity_scaled_events` on the
popular/latest path. Online LightFM persists the model artifact only after
the event source `ack`, so a nack/redelivery does not append the same
interactions twice. If that persist still fails after retries, the pending
fit is dropped (rows already written).

## Internals

`EventSource` (`connect` / `poll` / `ack` / `health`) is an internal
interface. Built-in backends register by `kind` at import time; unknown
kinds raise `ValueError`. Package layout:
[architecture.md](architecture.md) (`events/`, `serve/events_routes.py`,
`serve/bootstrap_events.py`).

## Roadmap

Shipped: webhook, db watermark, S3 list / AWS SQS, Redis Streams.
Possible later backends (not configured today): RabbitMQ, Kafka. Prefer
Redis Streams when you already run Redis for the lock; do not add Kafka
only for this feature.

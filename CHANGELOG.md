# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.9.0] - 2026-09-11

### Added

- Job writes catalog `item_scores` (`popular_score`, `latest_score`, `n_users`) next to recommendations. Dataset: `item_scores.parquet`. DB: `item_scores` table (`item_scores_table` option). First 0.9.0 job creates the table.
- Serve `GET /item-scores` (bearer, cursor pagination, optional `item_id`) for search-index pull. See [docs/search-weights.md](docs/search-weights.md).
- Serve-time hide of consumed items (`[serve].exclude_consumed`, default on)
  using `[input]` history plus a process-local incremental overlay capped at
  `consumed_lookback` items per user. Short lists fill from popular/latest
  (`[serve].fallback_fill`) when hide or availability filters drop rows.
- Catalog writes on `[input]` (`PUT/GET/DELETE /users/{id}`, `/items/{id}`,
  `/catalog/events`) so a host can upsert users, items, and events without
  a separate dump.
- Named surfaces on the serve API: `GET /popular`, `GET /latest`,
  `GET /similar/{item_id}`, and `POST /session/recommendations`. The job
  writes popular / latest / item-neighbor snapshots next to recommendations.
- Job writes `__cold_start__` under priority and weighted RRF (popular-only)
  as well as blending, so serve fallback does not depend on the combiner.

### Changed

- Dashboard user lookup applies the same item availability filters as serve.

### Fixed

- Popular, latest, and neighbor snapshots drop missing or blank item/user ids instead of casting them to `"nan"`.
- DB popular/latest/similar reads re-raise SQL connectivity errors instead of treating every `OperationalError` as a missing table.
- Serve refreshes surface snapshots in the same loop as `generated_at` so the stamp and surfaces stay on one job.
- `/popular`, `/latest`, `/similar`, and session recommendations increment source-tier serve metrics.
- Dataset surface refresh treats a complete unstamped trio as the cached baseline, so later unstamped file-by-file writes cannot mix snapshots.
- Job popular/latest snapshots keep at least `serve.default_k * 5` rows so `/popular`, `/latest`, and session fallback can over-fetch.
- Session neighbor lookups read every session item from one surface snapshot.
- Job neighbor snapshots keep at least `serve.default_k * 5` rows per item so session over-fetch still has candidates.
- Dataset surface refresh applies a new popular/latest/neighbors snapshot only when all three files match the job stamp; a missing stamp or unreadable parquet keeps the last consistent trio.
- Dataset `write_surfaces` rechecks the writer lease before each surface file.
- Job builds popular/latest/neighbor frames only when the sink can write surfaces.
- `ServeClient` popular/latest/similar send `exclude_unavailable` and `user_id`.
- Popular snapshots break distinct-user ties by event count, then `item_id`.
- Job marks `partial_outputs` before the multi-file surface write so a mid-write failure is not reported as clean.
- `POST /session/recommendations` accepts only an `items` list of item ids.
- Serve keeps the last `item_scores` catalog when a refresh cannot read scores (I/O error, missing file or table after a load, or invalid values). Before the first successful load, missing data still serves empty.
- Serve rejects `item_scores` catalogs with blank or duplicate `item_id`s or non-integral `n_users` and keeps the last valid cache.
- `GET /item-scores` treats an empty-string `cursor` as a seek point instead of restarting the first page.
- Job replaces a legacy `item_scores` table that is missing score columns instead of appending into the old schema.
- Job writes `item_scores` before recommendations so a score-write failure does not leave a new recs file without scores.
- DB manifest append adds missing columns (including `n_item_scores`) on pre-0.9.0 `recommendation_runs`.
- Dataset `item_scores` writes use the same validation as the DB sink.
- DB manifest ALTER uses a fixed type map so a failed first post-upgrade run cannot create `n_item_scores` as TEXT.
- Job uses one weighting timestamp for training interactions and `latest_score`.
- Item-score build strips item IDs before grouping so padded IDs do not crash the catalog write.
- Job skips `write_item_scores` when the sink does not implement it and records `n_item_scores` as null.
- New DB `recommendation_runs` tables get typed columns on first create, not TEXT from a null first write.
- Job builds `item_scores` before taking the recommendations writer lock.
- README says incremental write-through updates recommendations, not catalog `item_scores`.
- `OutputSink` no longer requires `write_item_scores`; that method lives on optional `ItemScoresWriter`.
- Job builds catalog `item_scores` only when the sink can write them.
- Serve reuses a cached sorted `item_id` index for `GET /item-scores` pagination.
- Serve snapshots the `item_scores` frame and id index under one lock so a refresh cannot pair a new index with an old page.
- DB `item_scores` schema inspect and legacy-table replace run under the writer lock.
- Search-weights docs say catalog writes apply to sinks that implement `ItemScoresWriter`.
- Job sets `partial_outputs` only after a snapshot, score, or recommendation write succeeds.
- Serve sorts and validates `item_scores` from readers that do not snapshot a cached id index.
- In-memory SQLite `[input]` history is readable from serve worker threads, so default consumed hide still applies.
- Catalog writes share the in-memory SQLite `[input]` engine, keep a writable `event_id`, and return 400 for blank path IDs. Query-backed db `[input]` stays 501.
- Dataset `DELETE /users/{id}` ignores a nonempty `users.parquet` that has no `user_id` column instead of raising.
- Dataset catalog PUTs merge into the existing user/item row instead of dropping omitted columns.
- Catalog event upserts reconcile the consumed overlay from remaining catalog rows in one write.
- `POST /catalog/events` uses the shared byte-limited JSON reader and documents 401/413/422.
- Incremental catalog persist uses `replace_events` so a reused event id updates the consumed overlay.
- Catalog GET rows map pandas missing timestamps (`NaT`/`NA`) to JSON `null`.
- Dataset `DELETE /users/{id}` removes the user row and their events under one lock.
- Catalog and webhook ingest return 400 for out-of-range `occurred_at` epochs.
- Dataset `GET /catalog/events/{user_id}` uses parquet `user_id` predicate pushdown, with a full-file fallback.
- Dataset catalog event writes keep columns from an empty `events.parquet` schema.
- Catalog user and item upserts reject a missing id instead of storing `"None"`.
- Catalog writes and incremental persist update the consumed overlay under the same lock as the store mutation.
- `DELETE /items/{id}` also removes that item's events so leftover interactions cannot resurrect it.
- DB catalog PUT→GET keeps extra user/item fields and returns structured values as objects, not JSON strings.

## [0.8.3] - 2026-09-18

### Fixed

- Recipe `eligibility = false` / subset / replace skips
  `item_availability_filters` sugar.
- Job optional-eval and sidecar catches log exception type and message,
  and only swallow store/eval I/O errors. Sidecar failures use
  `PublishError`, including Kafka produce/flush/close and a failed
  RabbitMQ retry or handle close. Kafka `Producer()` construction is a
  connect failure (`ConfigError`), same as `list_topics`. The sidecar
  generation check only swallows store I/O errors. Lock-loss and
  unexpected `RuntimeError`s fail the run and rewrite that generation to
  failed when no newer manifest exists. The failed rewrite keeps the
  original success generation as the skip cutoff and writes a later
  `generated_at` so a DB reader does not tie with the success row.
  Catalog size and experiment overlay swallow expected store I/O,
  including S3 and SQL errors (a transient `OperationalError` included);
  catalog size also treats a parquet `ArrowInvalid` as missing;
  an unexpected `RuntimeError` fails Thompson and eval. Experiment DB
  state only treats a missing table or column as absent, not a generic
  `ProgrammingError`. An unexpected
  `RuntimeError` on Kafka or RabbitMQ close fails the run after both
  RabbitMQ handles have been closed. A sidecar failure after a complete
  recs write does not set `partial_outputs`. Incremental publish only
  swallows expected sidecar I/O. An unexpected `RuntimeError` during
  RabbitMQ recover is not wrapped as `PublishError`.
- DB serve keeps `AND variant=` when table inspect fails, and prefers the leftover
  fallback arm in the same `LIMIT` query when no arm is assigned.
- Thompson serve hashes the sticky pair or names on disk, not the full config
  set. Snapshot names are read only when the pair is missing. Unreadable
  `experiment_state.json` is a read error.
- Incremental apply keeps parked variant rows when the hashed arm is updated.
- Experiments evaluate uses the same champion/challenger overlay as serve.
- Promoting a 3+ arm Thompson winner keeps the previous champion as challenger
  and resets pair volume.
- Serve caches the assignment overlay on the recommendations refresh loop.
- Dataset serve filters cold-start fallback outside the cache lock.
- Publish sidecar failures do not un-succeed a recs write or livelock ingest,
  including a broker that is down at connect. Connect and publish run after
  the writer lock is released, and the apply/retrain fence is re-checked
  before and after connect. A newer manifest generation skips publish.
- RabbitMQ publish confirms deliveries, stamps a stable `message_id`, recovers
  the channel after a broker error, and retries only unconfirmed users.
- Kafka publish fails when a delivery callback reports an error.
- Fingerprint ingest acks apply only to generated event ids. Switching a live
  buffer to generated-only rebuilds the fingerprint set.
- Writer-lock busy nacks restore events the same way as other apply failures.
- Event worker stop closes the source after a tick that never started a loop
  thread.
- A dispatched RabbitMQ job is not started after its I/O timeout.
- Local dataset writers take `msvcrt.locking` when `fcntl` is missing.
- `POST /track` acquires the writer lock off the serve event loop.
- Concurrent database `/track` writes in one process accept a given
  `event_id` once.
- CTR is capped at one click per impression, matching CVR. Duplicate
  clicks on the same impression count once. Blank impression IDs are
  repaired before matching.
- Track source annotation does not invent a variant from a later snapshot.
  Exact snapshot matches can fill variant; the latest-snapshot fallback
  stays source-only and does not overwrite an existing source.
  Recs without `generated_at` still merge source and variant.
- Track `since` drops untimed rows; `experiment_id=` excludes blank ids.
  DB reads apply a conservative date bound. Invalid `since` is an empty query.
  Dataset `track.jsonl` streams local lines and drops out-of-window rows.
  A failed DB track read raises unless the table is missing.
- Quality and Experiments use an attribution lookback on track and events.
  Experiments keep only exposures for users in that window, and a failed
  track read does not drop healthy exposures.
- Metric event `since` is pushed into SQL and parquet. Missing `occurred_at`
  or an unsupported bound is empty. `quantity` is optional and kept when
  the source has it. Missing S3 events are empty. Custom `events_query`
  is wrapped unless it already has a top-level `LIMIT`/`OFFSET`.
  SQL comments do not count as pagination. A failed DB engine on a
  bounded read is empty.
- Thompson skips window trials when a pair exists but `window_started_at` is empty.
- The job loads track and recommendations once for eval and Thompson.
  A failed track or recs read is retried independently, not treated as empty.
  Thompson windows the shared rows in memory so eval keeps full history.
  Shared preload keeps an empty recommendations frame instead of reloading,
  and runs in parallel on local/S3.
  Eval ignores a preloaded track when track is disabled.

### Security

- Serve, trigger, and dashboard bind `127.0.0.1` by default. `0.0.0.0` is a
  TOML opt-in.
- `[events.online]` requires `[output].artifact_hmac_key` and verifies it
  before unpickle. Artifact and storage reads reject payloads over 512 MiB.
- Thompson promote accepts only the champion/challenger pair.
- Custom `input.options.*_query` and `events.options.events_query` reject
  more Postgres file/admin functions (`pg_ls_dir`, `dblink`, large objects).

## [0.8.2] - 2026-09-15

### Changed

- Incremental apply takes the postgres/redis lease whenever
  `lock_backend` is distributed, even if `events.ha` is false. Retrain
  probe follows the same rule.

### Fixed

- Kafka and RabbitMQ ingest and publish time out broker calls after 10s (`timeout_seconds`).
- Broker `timeout_seconds` values that overflow client millisecond limits raise `ConfigError`.
- Kafka `timeout_seconds` below 10 ms raises `ConfigError` (librdkafka `socket.timeout.ms`).
- A timed-out RabbitMQ I/O call queued during the idle pump is not executed.
- Event worker stop closes the source even when the worker thread misses its join deadline.
- Event worker stop interrupts a hung source without taking the reconnect lock.
- Event worker stop does not close the source during an in-flight apply/ack.
- Event worker stop does not close the source during an in-flight tick.
- A failed RabbitMQ I/O worker replies abandoned to a job already dequeued.
- RabbitMQ ack does not drop a new connection's delivery tag after reconnect.
- The event worker reconnects when the source reports disconnected after a broker timeout.
- Event worker stop closes a reconnect that finishes after shutdown.
- Event worker stop does not wait on a reconnect that is still opening a broker connection.
- Abandoned RabbitMQ I/O does not return a late poll result onto a new connection.
- RabbitMQ poll drops a delivery whose I/O handle was replaced before the tag is recorded.
- A RabbitMQ heartbeat timeout fails closed so apply nacks before writing.
- Event worker start aborts if stop wins during the initial connect.
- Event worker stop does not wait on the initial connect lock.
- RabbitMQ I/O rejects new broker calls after close reserves the thread.
- A failed RabbitMQ idle pump wakes queued I/O callers instead of leaving them until timeout.
- RabbitMQ poll does not return a delivery whose tag was cleared by reconnect or close.
- Event worker reconnect closes the source after a failed in-flight connect when stop is set.
- RabbitMQ close reserves the I/O thread for shutdown so a new idle pump cannot queue ahead of it.
- A RabbitMQ idle-pump socket failure marks the source disconnected so the worker reconnects.
- RabbitMQ reconnect keeps the previous I/O thread on its own connection.
- Manual `popular_in_category` fails at job fit when items lack the category column.
- Overall track CVR uses the same impression-slice attribution as rank, source, and variant.
- Event worker drain runs before the source close on shutdown.
- Event worker stop does not close during an in-flight reconnect.
- RabbitMQ health reports disconnected when the I/O worker is closing or replaced.
- Event worker stop closes the source once.
- RabbitMQ heartbeat fails closed when the I/O worker is already failed or closing.
- Event worker tick does not poll after stop has closed the source.
- Event worker start abort drains before close.
- RabbitMQ poll drops a stale delivery when the same event id is reborn after reconnect.
- RabbitMQ close still closes channel handles when the failed I/O thread has already exited.
- RabbitMQ poll drops a claimed event after reconnect even if the new channel reuses the same tag.
- Event worker skips poll after a failed reconnect until connect succeeds.
- Event worker start abort drains when the initial connect raises.
- RabbitMQ poll stops getting once reconnect replaces the I/O handle.
- RabbitMQ ack does not resolve tags from a connection that was replaced.
- Event worker stop returns false when the initial connect still holds the source lock.
- Event worker health probes run under the tick lock so stop cannot close during health.
- RabbitMQ nack ignores an event whose I/O handle was replaced after poll.
- Event worker start does not clear a stop that already finished before connect.
- RabbitMQ ack drops ownership stamps so the poll map does not grow without bound.
- Event worker stop does not block on the source lock for a leftover dead thread.
- Event worker start refreshes source health before the first tick.
- Event worker reconnect requeues buffered events after the new connection is up and restores only what nack did not keep.
- Event worker restores a batch when the source cannot keep a nack after a broker timeout.
- Event worker reconnect takes the tick lock so it cannot interleave with poll/apply.
- A timed-out event-source ack after a successful apply persists and does not nack.
- Event worker start health runs under the tick lock.
- Event worker start abort does not close a worker started after a later stop.
- Event worker skips the first poll when startup health reports disconnected.
- A timed-out RabbitMQ I/O job is not started after submit fails.
- RabbitMQ I/O timeout abandons an unclaimed job so it cannot run after submit fails.
- A RabbitMQ I/O timeout abandons a claimed job before the broker call starts.
- A RabbitMQ I/O timeout cannot start a job after the waiter has already failed.
- Event worker stop does not close the source during a never-started in-flight tick.
- A RabbitMQ heartbeat broker error marks the I/O worker failed and fails closed.
- Event worker overflow nacks restore events the source did not keep.
- Event worker start abort waits for the tick lock before closing a launched worker.
- A RabbitMQ I/O timeout cannot start a job after it was marked running.
- RabbitMQ nack does not keep a batch if I/O fails during requeue.
- Event worker stop does not close a worker started after the previous thread joined.
- Event worker does not apply a reconnect-restored event again when the broker redelivers it.
- Event worker start abort after launch leaves drain to the worker thread.
- Event worker does not ack a redelivery while the same unapplied event is still buffered.
- Event worker persists after a successful apply if ack fails, instead of nacking written events.
- Kafka ack keeps local offsets when a partition commit fails.
- Event worker remembers numeric event ids so a timed-out ack cannot apply them twice.
- A RabbitMQ I/O timeout cannot start a job after it was marked started.
- RabbitMQ messages without an event id get a generated id mapped to the delivery tag.
- Event worker remembers applied fingerprints only for sources whose event ids change on reconnect.
- Event worker does not ack a fingerprint-matching redelivery while the original is still unapplied.
- Event worker acks a deferred fingerprint duplicate after the original is applied.
- Event worker retries a failed post-apply ack instead of leaving the source offset stuck.
- Event fingerprints length-prefix fields so values containing `|` cannot collide.
- A RabbitMQ `basic_get` failure marks the I/O worker failed so health reconnects.
- RabbitMQ reconnect keeps nacked local pending events across the new connection.
- Event worker stop waits for an in-flight tick before closing a joined worker.
- Event worker retries a failed unbuffered or deferred ack on the next tick.
- A RabbitMQ I/O timeout abandons a job that has not been dispatched to the broker.
- Event worker stop does not close a worker started after a leftover dead thread.
- A RabbitMQ I/O timeout cannot run a job after invoke starts but before the broker call.
- A RabbitMQ I/O timeout detaches the channel so a late get or ack cannot use the abandoned connection.
- Event worker start takes the tick lock before the source lock.
- Kafka ack keeps local offsets when a partition watermark cannot advance.
- Event worker fingerprint dedupe applies only to generated event ids.
- Event worker stop returns pending acks when a drain ack fails.
- RabbitMQ ack clears ownership for events carried across reconnect.
- Event worker reconnect does not nack post-apply ack retries.
- Event source ack returns only ids bound to a live delivery.
- A RabbitMQ I/O timeout cannot start a dispatched job.
- RabbitMQ get, ack, declare, and heartbeat raise after the I/O channel is detached instead of succeeding.
- RabbitMQ abandoned cleanup closes the timed-out connection and any leftover live handle.
- Dataset recommendation writes and user-replace share one host lock;
  JSONL track/exposure appends do the same. A distributed lock serializes
  those writers when `lock_backend` is postgres or redis.
- Dashboard lookup keeps recommendation and history reads on the request
  thread so in-memory SQLite does not miss rows.
- `cicerone_events_leader` reports apply-lease ownership, not HA-only.
- Local dataset file locks serialize in-process writers when `fcntl` is missing.
- Incremental apply reloads cached users under the lease so another
  replica's write is not overwritten.
- Thompson and served-eval store reads stay on the job thread so
  in-memory SQLite is not empty.
- Dataset writes re-check the caller fence after waiting for the writer
  lock. Writer-lease loss is logged as itself, not as apply-lease loss.
- Incremental apply reloads and remakes affected users under the dataset
  writer lock so a finished retrain is not overwritten.
- Dataset writer-lock re-entry is per-thread, so a second writer cannot
  skip the host/distributed lease.
- The job writes a successful manifest under the same dataset write lock
  as recommendations.
- Nested dataset writes re-check writer ownership before replacing
  parquet. A failed job does not write a manifest after fence loss.
- Writer-lock contention is a busy nack, not an apply error. Serve
  impression/exposure appends retry that busy path.
- Dataset `write_manifest` takes the same writer lock and fence as
  recommendation writes.
- The job publishes artifact, items, recommendations, and the manifest
  under one write lock. A failed job does not overwrite a newer
  incremental manifest.
- Nested recommendation writes re-check ownership after parquet serialize.
- Artifact, items snapshot, Thompson state, and publish re-check the writer
  fence immediately before each replacement. Incremental publish does the
  same. Online persist after ack takes the dataset write lock again.
- Failure manifests compare freshness under the writer lock. A stale Redis
  `held_writer_lock` cannot release a later acquire on the same object.
- Incremental apply rechecks the apply fence immediately before replace.
  Track and exposure appends recheck writer ownership before the bytes
  write. Database failure manifests skip when a newer row already exists.
- Redis acquire starts a new TTL refresher after release so a joining
  previous thread cannot leave the new holder without refresh.
- The job rechecks the retrain fence immediately before writing a
  success manifest. Track appends fence ownership after JSON encode.
  A late Redis refresh failure cannot `_mark_lost` a newer acquire.
- Database recommendation, manifest, and artifact writes honor the
  caller fence. Online persist after ack fences even without a dataset
  writer lock.
- Job Thompson state and dashboard promote share the dataset writer
  lock. A late Redis `_mark_lost` cannot stop a newer refresher.
  Legacy `write_manifest` fallback only applies when the sink lacks
  `skip_if_newer_than`.
- Writer-lock ownership is generation-aware after a same-object
  reacquire. Job track eval/history writes take that lease. Local
  dataset file locks time out instead of blocking forever.
- A direct job takes the retrain lease when `lock_backend` is
  distributed. Experiment state writes re-enter only on the owning
  thread and honor the retrain fence on DB output. Online persist
  rechecks retrain-busy after the writer wait. Redis release does not
  stop a newer refresher.
- Dashboard promote reads and writes experiment state under one writer
  lock. The job rematches `promoted_variant` before writing Thompson
  state. Track eval and history writes honor the retrain fence.
- Writer-lock acquire binds the Redis generation in the same step as
  SET NX, so a later reacquire cannot fence stale work.
- Experiment DB state fallback and exposure appends honor the caller
  fence. DB manifests recheck it before append. Track sidecar lock
  errors stay best-effort. DB recommendation writes take the same
  output writer lease as dataset.
- DB sink writes check writer ownership, not only the caller fence.
  Failure manifests take `recommendations_write` again. Exposure
  appends recheck after the local file lock.
- DB mutating sink methods and dataset artifact writes take the writer
  lease themselves. A stale Redis `try_acquire` does not replace a
  newer refresher.
- Dashboard promote maps writer-lock busy and loss to flash errors.
  DB recommendation, artifact, and item writes recheck the fence after
  each destructive step. Redis `SET NX` drops a token rotated while
  the call was in flight. Track JSONL appends honor the caller fence.
- Dataset writes re-enter a lock already held on this thread. DB track
  append, eval, and history honor the writer lease and recheck the
  fence before INSERT. Experiment state and exposure DB writes recheck
  after DELETE. `POST /track` returns 503 on writer busy or lock loss.
- DB sink, track, and experiment writes recheck the fence after the
  last statement so a lost lease rolls back before commit. Local
  `flock` retries only `EAGAIN`/`EACCES`.
- DB exposure and history appends run `to_sql` on a transaction
  connection so a failed final fence rolls the insert back.
- Experiment state CREATE/ALTER stay in the fenced write transaction
  so a lost lease does not leave an empty or migrated table.

## [0.8.1] - 2026-09-11

### Fixed

- Production replay scores the served experiment variant, not the union of all lists.
- Click-through CVR ignores unmatched clicks.
- JSONL track idempotency records ids only after a successful append.
- Thompson writes the champion/challenger pair after the recommendation write.
- Kafka and RabbitMQ reconnect clear in-flight ack maps.
- History snapshot lookup compares instants; unmatched `generated_at` does not take a later job.
- Quality live CTR prefers history snapshots and clears the load-error banner when live metrics succeed.
- Dashboard config redacts webhook URLs that have no userinfo.
- Experiments manifest recipe parse treats malformed items as missing recipes.
- Inheriting all experiment boost/eligibility rules rejects duplicate names, same as a named subset.
- The recommendation publish sidecar rejects NaN scores instead of emitting non-JSON.
- Job eval and history writes on `kind = "db"` run one at a time so SQLite does not drop either.

## [0.8.0] - 2026-09-08

### Added

- `[track]` ingest: `POST /track` records recommendation impressions and clicks
  off the training event path (db table or local JSONL). Hosts report what was
  actually rendered; `GET /recommendations` is not an impression unless
  `[serve].log_impressions` is on.
- CTR and attributed conversion (view-through and click-through) with an
  attribution window, sliced by rank, source, and variant. The job writes
  `track_eval`; the dashboard Quality page shows it.
- `[experiment].primary_metric` `ctr` / `conversion` and
  `attribution = click | impression | user | recommended`, with a volume floor
  (`track.min_impressions`) before promote.
- Optional `[job.eval]` production replay (HitRate / NDCG / Recall / MRR /
  Precision / catalog coverage / novelty of the previous lists against later
  events) and append-only recommendation history for windows that span jobs.
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
- Opt-in RecTools strategies `ease` (`EASEModel`), `als`
  (`ImplicitALSWrapperModel` with item/user features), `popular_in_category`,
  and `random`. `[model.collaborative]` stays LightFM; `[events.online]`
  rejects a swapped collaborative `cls`.
- Item-KNN can use RecTools `CosineRecommender` or `BM25Recommender` via
  `[model.item_based.model].cls`. LightFM `loss` (`warp` / `bpr` / `logistic`
  / `warp-kos`) is the same `[model.collaborative.model]` passthrough.
- Serve Prometheus `cicerone_track_ingest_total` for `POST /track`.
- Experiment variants can inherit, drop, subset, or replace `[[boost]]` /
  `[[eligibility]]` rules (`boosts = ["featured"]` or
  `[[experiment.variants.boost]]` tables). Duplicate subset names are rejected.
- Job-time Thompson sampling (`allocation = "thompson"`) writes a sticky
  champion/challenger pair from live CVR via Fidelity MABWiser. Requires
  `cicerone-recommender[bandits]`. Read errors or empty track fail closed
  to fixed (every named recipe). Serve still hashes; Ship is Promote.

### Changed

- Dashboard Config hints are in-flow disclosures; Explain keys reveals per-key
  help. Quality shows CTR as percents with click and conversion counts and an
  as-of time. Experiments promote confirms before sending 100% traffic and
  uses a one-shot flash cookie.
- Dashboard header wraps on narrow viewports, underlines the current page,
  marks Quality and Experiments as off when unused, and offers Sign out.
- Dashboard Configuration includes `[publish]`, `[track]`, and `[job.eval]`.
  Experiments CI is labelled a mixture interval.
- Dashboard Experiments loads events, recommendations, track, exposures, and
  catalog size in parallel, and pushes event-type / experiment_id filters into
  the store. Job eval and Quality live read only the history snapshots
  referenced by track `generated_at`. Explain reasons scan interactions for
  recommended users only.
- Bump `boto3` 1.43.83 → 1.43.88, `psycopg` 3.3.4 → 3.3.5, and `ruff`
  0.16.5 → 0.16.6.

### Fixed

- Quality as-of uses the scored previous run's timestamp, not the job that
  just finished.
- Quality labels live track metrics when a stored eval has no usable
  track_eval.
- Quality omits as-of when CTR/CVR is computed live from the track store.
- Track ingest is idempotent without a host `event_id` (stable JSON hashes,
  assigned ids, JSONL file lock, `accepted` from ids not already stored).
  `since` compares instants. Slices join the matching recommendation snapshot
  and count by impression `event_id`. Prometheus counts match accepted rows.
- `POST /track` `event_ids` lists newly written rows only, matching
  `accepted`.
- Recommendation history writes one parquet file per job snapshot. Filtered
  and `since` reads skip legacy unstamped files and parse slugged part names.
- Incremental `[publish]` runs after the output write. Connect, config, or
  publish failures fail the job or incremental tick so the batch is nacked.
- Kafka and RabbitMQ ingest commit or ack after flush (contiguous watermark;
  poison non-UTF-8 skipped; AMQP on one I/O thread). Publish uses the
  configured routing key and reports setup failure.
- Incremental popular/latest write-through is the assigned (or promoted)
  variant only.
- Experiment `ctr` / `conversion` use the track store only for `click` or
  `impression` attribution, require `track.enabled`, and take the impression
  `variant` when present. Empty track stays event ITT.
- Served-eval `CatalogCoverage` uses the item catalog. Experiments surfaces
  variant policy `ConfigError` instead of an empty-variant message. Job
  `[eval]` still reads recommendation history when `[track]` is off.
- In-flight event apply nacks the batch if a later heartbeat fails. Dashboard
  `Cache-Control` skips only `/static` assets.

### Security

- Dashboard pages send `X-Robots-Tag` and HTML `noindex` so crawlers skip
  them if the process is accidentally public; `GET /robots.txt` disallows
  `/`, and OpenAPI `/docs` is off.
- Config page redacts `*_url` option keys, `access_key_id` /
  `aws_access_key_id`, and any value with embedded URL credentials.
- `POST /track` rejects bodies larger than 1 MiB (or
  `events.options.max_body_bytes`).
- Dashboard flash cookies are an allowlist; form variant names never enter
  `Set-Cookie`.

## [0.7.3] - 2026-09-03

### Changed

- Serve `GET /metrics` is off unless `[serve].metrics_enabled = true` and a
  non-empty `metrics_token` is set. Scrapes must send `X-Metrics-Token`.
- Compose publishes serve `:8000`, trigger `:8080`, and dashboard `:8090` on
  `127.0.0.1` only (same as Postgres).

### Fixed

- Dashboard promote / resume-split POSTs require a CSRF cookie + form token.
- Dashboard and trigger no longer expose `/docs`, `/redoc`, or `/openapi.json`.
- Serve `limit` / `k` and `default_k` cap at 100. `POST /events` rejects bodies
  larger than 1 MiB (or `events.options.max_body_bytes`).
- Input `events_query` / `users_query` / `items_query` use the same read-only
  SELECT gate as `events.options.events_query`. Custom SQL is not logged.
- Model artifacts refuse legacy bare pickle and unexpected zip members.
- Malformed dashboard bcrypt hashes fail closed as 401, not 500.

### Security

- Serve, dashboard, and trigger send `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, and `CSP frame-ancestors 'none'` (HSTS on HTTPS).

## [0.7.2] - 2026-09-02

### Changed

- I/O factories and the event-source registry import SQLAlchemy, boto3, and
  Redis backends only when that kind is selected. Local parquet helpers no
  longer import boto3 at module load.

### Fixed

- Experiment promote-state reads order null `promoted_at` last (Postgres
  `DESC` otherwise prefers the legacy null row).
- Online LightFM skips `fit_partial` and artifact persist when sequential is
  in the last artifact and torch is missing.
- Online refresh rewrites every user in the pending extra window when
  `fit_min_events` fires, not only the current batch.
- Incremental write-through applies ranking boosts only to the assigned
  experiment variant.
- Experiment first-exposure uses a later timestamped row when the first log
  line has no `exposed_at`; events before that start are dropped.
- Empty or failed exposure logs stay exposure-conditional instead of
  silently switching to ITT.
- Dashboard promote reads reuse the last successful state when the store
  fails (same fail-closed cache as serve).
- Job target users stringify IDs so int/str duplicates do not split.
- Job target users skip missing IDs so NaN does not become a literal `"nan"`
  user.
- Incremental write-through uses the promoted experiment winner, same as serve.
- Explain similar-item overlap counts each history item once.
- Content-fallback top-K ties follow item id.

### Removed

- Unused `s3fs` pin. S3 reads and writes use boto3 and pyarrow.

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

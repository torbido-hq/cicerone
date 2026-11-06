---
title: The transformer still sleeps until cron
description: SASRec trains in the nightly job and writes a top-K table. Serve GET stays a lookup. Torch stays off the default image and off that request.
date: 2026-11-06
excerpt: Put sequential in models and the job fits a transformer. The homepage still reads last night's rows. A GET does not load the artifact.
authors:
  - nicholas
---

You want next-item ranking, so you add a transformer. You put SASRec behind the homepage and expect each click to move the weights. You did not get a live sequence model. You got another nightly top-K. Rows that model won are `source = sequential`. The lookup reads the table.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) post already writes ranks with LightFM and bestsellers. This walkthrough swaps the personalized half for Cicerone's opt-in `sequential` strategy. RecTools SASRec is the default. BERT4Rec and HSTU are the other two classes. The storefront still reads the recommendation table, or it calls serve. A GET does not load the artifact and does not import torch.

```text
[job].models includes "sequential"
        │
        ▼
   job process with the sequential extra
   fits SASRec / BERT4Rec / HSTU
        │
        ▼
   recommendations table
   (source can be sequential)
        │
        ▼
   SELECT … ORDER BY rank
   or GET /recommendations/{user_id}
        │
        ▼
   stored rows
```

Four nouns.

| Noun | What it is |
| --- | --- |
| **Sequence** | One row per `(user, item)` after weighting. Order is that row's datetime. |
| **Architecture** | `sasrec`, `bert4rec`, or `hstu` under `[model.sequential]`. Default class is `SASRecModel`. |
| **Fit** | `job.run()` calls `model.fit` on the aggregated dataset and writes rows. |
| **Lookup** | `GET /recommendations/{user_id}` or a `SELECT`. Reads rows. Does not load the artifact. |

A GET does not load the artifact. The serve process opens it only when `[events.online]` is on. How that load works is in the re-score section.

## The job image needs the extra

`pip install 'cicerone-recommender[sequential]'`, or `pip install -r requirements-sequential.txt` from a checkout. `pyproject.toml` declares that extra as `rectools[torch]>=0.19.0,<0.20`. `requirements-sequential.txt` pins `rectools[torch]==0.19.0`, the same RecTools version as `requirements.txt`. The extra counts as installed when `torch` and `pytorch_lightning` are importable.

`docker/Dockerfile` does not end on the image you want for serve. Three names:

| Name | What it is |
| --- | --- |
| `runtime` | Production serve stage. Does not install `requirements-sequential.txt`. Torch is absent. |
| `test-sequential` | Last stage of that Dockerfile. A test image, not the production serve image. It installs the sequential extra so the sequential tests can import torch. |
| `cicerone-sequential` | Local tag you build yourself, for the batch fit. It is not a stage of `docker/Dockerfile`. |

Run the builds from the repository root. A bare `docker build -f docker/Dockerfile .` builds `test-sequential`. For the torch-free serve image, name the `runtime` stage:

```sh
docker build -t cicerone -f docker/Dockerfile --target runtime .
```

TOML load still succeeds. The job raises `ConfigError` when it builds the strategy. The message is:

```text
strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'
```

`Dockerfile.sequential` starts `FROM cicerone`. That tag has to be the `runtime` image from the command above. The runtime stage ends as `USER cicerone` with `ENTRYPOINT ["cicerone"]` and `CMD ["start"]`. This file does not replace either, so a run with no arguments is `cicerone start`. With `job.mode = "batch"` that runs one job, then the scheduler. If `cicerone` is the last stage of `docker/Dockerfile`, this file extends `test-sequential`. That stage inherits `ENTRYPOINT []` from the test stage, replaces the command with pytest, and already has torch. The result is neither the batch job image nor a torch-free serve image.

Save the file in the repo root so the build context can see `requirements-sequential.txt`. The runtime stage's last user is `cicerone`, so the snippet switches to root for `pip` and back. Tag the result `cicerone-sequential`.

```dockerfile
FROM cicerone
USER root
COPY requirements-sequential.txt /tmp/requirements-sequential.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sequential.txt \
    && rm /tmp/requirements-sequential.txt
USER cicerone
```

```sh
docker build -t cicerone-sequential -f Dockerfile.sequential .
```

Run the batch job from `cicerone-sequential`. Keep the serve process on the `cicerone` image and leave `[events.online]` off. The BFF does not need the extra either: `pip install cicerone-recommender` on Python 3.11 is enough for `ServeClient`, and `fastapi` is already a dependency. That install does not pull in torch.

## Turn sequential on

These blocks are edits to the nightly files. They are not a full config.

```toml
# features.toml — the nightly file sets enabled = true
[blending]
enabled = false
```

```toml
# cicerone.toml — same [input] / [output] as the nightly post
[job]
mode = "batch"
top_k = 20
models = ["sequential", "popular"]
cron_schedule = "0 3 * * *"
feature_config_path = "/app/config/features.toml"

[model.sequential]
architecture = "sasrec"
# epochs = 3
# session_max_len = 50
# train_min_user_interactions = 2

# [job.sequential]
# min_median_interactions = 5
```

`architecture` selects the RecTools class: `sasrec` → `SASRecModel`, `bert4rec` → `BERT4RecModel`, `hstu` → `HSTUModel`. Setting `architecture` and a conflicting `cls` is a `ConfigError`.

Cicerone starts from its SASRec config unless you override a key: `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`, `deterministic = true`, `verbose = 0`. It expands `LiGRLayers` to `rectools.models.nn.transformers.ligr.LiGRLayers` before RecTools sees it.

That list is not the RecTools constructor. RecTools SASRec differs on `n_factors = 256`, `session_max_len = 100`, `loss = softmax`, `n_negatives = 1`, and `deterministic = false`. `verbose = 0` is the same. `bert4rec` keeps Cicerone's config, including `deterministic = true`. `hstu` drops LiGR and rewrites an inherited `sampled_softmax`. It keeps `deterministic` and `verbose`.

| Architecture | Class | What Cicerone actually passes |
| --- | --- | --- |
| `sasrec` (default) | `SASRecModel` | Shifted next-item. RecTools leaves causal attention on. Layers stay LiGR. Loss stays `sampled_softmax`. |
| `bert4rec` | `BERT4RecModel` | Masked-item training. RecTools `mask_prob` stays `0.15`. A drawn position keeps the original item id as the loss target. Of those inputs, 80% become MASK, 10% become a random catalog item, and 10% stay the original item. A position that is not drawn keeps its input and has its loss target set to `0`. The loss skips target `0`. Causal attention stays off. Cicerone still passes LiGR and `sampled_softmax`. RecTools' own BERT4Rec defaults are PreLN layers and `softmax`. |
| `hstu` | `HSTUModel` | Same shifted preparator as SASRec, on HSTU layers. Cicerone drops LiGR. An inherited `sampled_softmax` becomes `softmax`, and `n_negatives` is dropped, unless you set `loss`. `relative_time_attention` defaults to true. |

`architecture = "hstu"` with that default trains, then fails in this job's recommend call. The flag sets `HSTUModel.require_recommend_context`. RecTools raises `ValueError` when that property is set and `context` is missing. It continues when the frame is passed. This call does not pass it, so the exception lands before `_write_job_outputs` and that run publishes no table.

The message, including the missing space before `Check`, is:

```text
This model requires `context` to be provided for recommendations generation (model.require_recommend_context is True).Check docs and examples for details.
```

Set `relative_time_attention = false` under `[model.sequential]` and the same call recommends. `add_unix_ts` stays off, so those timestamps are not in the sequence.

Users with no interactions skip sequential. `popular` still scores them, and those rows are `popular_fallback`. Personalized recommend passes `filter_viewed`, so a sequential row is an item that is not already in that user's aggregated history.

With no `model_weights`, combine is priority. `models = ["sequential", "popular"]` lets sequential win a duplicate item, and that row keeps `source = sequential`. A `popular` row that still fits in the top-K is `popular_fallback`. Set `model_weights` and the combiner is weighted RRF: a row that both strategies contributed is `sequential+popular_fallback`, in `models` order.

Blending is a different combiner. It relabels every personalized frame to `personalized` before the mix. A row drawn from more than one of `personalized`, `popular_fallback`, and `latest` is `blended`. Blending is also what writes `'__cold_start__'`. This config does not. The nightly Rails helper that falls back to that sentinel gets no rows for a guest.

## A sequence is not a click log

Before the model sees the data, the job turns events into one row per `(user, item)`.

Where `[event_caps]` names an event type, it keeps that many most recent events per `(user, item, event_type)`. Half-life decay multiplies what remains. The weight is the sum. The datetime is `max(occurred_at)`. A sum that is not positive is dropped. RecTools then sorts by that datetime, stably, and builds one sequence per user.

Two purchases of the same SKU are one position, at the later timestamp. Buy A, then B, then A again, and the sequence is B then A when that last A is later than B and both aggregated weights stay positive. A non-positive sum drops that item. Intermediate timestamps are not in the sequence.

`session_max_len` (default 50) is the recommend window for SASRec and HSTU: the tail of that unique history. Training keeps one more item (`session_max_len + 1`) because the shifted target needs it. BERT4Rec training keeps `session_max_len` items. At recommend time BERT4Rec appends a MASK token inside that same window, so the model sees `session_max_len - 1` real items.

`train_min_user_interactions` (default 2) drops shorter users from training only. A user left out of training can still be scored when they hold at least one item the fitted model already knows. RecTools leaves out a target user who has none of those items.

HSTU is the architecture that reads a clock, and only the clock aggregation left behind. With `relative_time_attention` on, RecTools sets `add_unix_ts` and turns each aggregated datetime into seconds. The bias is the gap between last touches of distinct items. It is not the gap between raw repeats. SASRec and BERT4Rec do not get `unix_ts` on these defaults.

## The table is the morning read

A known user's `SELECT … ORDER BY rank` returns that user's stored rows. The nightly `JOIN` does that.

Serve is the other reader. This walkthrough uses the stdlib client in the package.

```python
import os
from fastapi import FastAPI, HTTPException
from cicerone.serve_client import ServeClient, ServeClientError

app = FastAPI()
client = ServeClient(
    os.environ.get("CICERONE_SERVE_URL", "http://localhost:8000"),
    token=os.environ["CICERONE_SERVE_TOKEN"],
)

@app.get("/for-you/{user_id}")
def for_you(user_id: str, limit: int = 8):
    try:
        body = client.recommendations(user_id, limit=limit)
    except ServeClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return {
        "user_id": body.user_id,
        "fallback": body.fallback,
        "items": [
            {"item_id": row.item_id, "rank": row.rank, "source": row.source}
            for row in body.items
        ],
    }
```

`source` on a sequential win is the string `sequential`. LightFM's wins stay `personalized`. `popular` in config is `popular_fallback` in the table.

`cicerone serve` is a second process. It refuses to start unless `job.mode = "serve"`, and that mode refuses to load unless `[serve].auth_token` is set. The batch file above stays `mode = "batch"` and does not set a token. A `SELECT` does not send one.

`CICERONE_SERVE_URL` and `CICERONE_SERVE_TOKEN` are environment variables this BFF reads. They are not Cicerone settings. Set `CICERONE_SERVE_TOKEN` to the same string as `[serve].auth_token`. `ServeClient` omits the header when `token` is left out. `GET /recommendations` on this process rejects that call because the token is configured.

`limit` and `k` are the same cap. Pass one. Send both with different values and serve returns 400, detail `limit and k disagree; pass only one (or the same value)`. `exclude_unavailable` defaults to true. A category or availability filter that removes every row is 200 with an empty `items` list.

## Re-score is not a fit

Optional `[events.online]` runs in the serve process, after the [checkout](/articles/this-afternoons-checkout-can-move-the-row/) flush. It is not transformer training.

| Operation | Changes model weights? | Changes recommendation sequence? |
| --- | --- | --- |
| Batch fit | Yes | Yes |
| Online re-score | No | Yes |
| LightFM `fit_partial` | Yes, for LightFM only | Not applicable to the transformer |

Startup calls `ensure_loaded` before the process listens. That unpickles the artifact and installs it before the process accepts a request. A GET does not.

A later `refresh` with no events returns before it reads the store. Otherwise `_reload` runs first. A refresh that is still pending does not read. A matching fingerprint skips the read, and the model already in memory stays. A different byte digest is unpickled and installed then, before this call can recommend.

Reload, merge, and re-score stay three steps. The re-score is `recommend` on the fitted sequential model, and only after a kept event has been merged. It does not replace the sequential weights.

A re-score keeps an event only when both its `user_id` and its `item_id` are already in the loaded artifact's interactions. Anything else is counted as dropped. Kept events go through the same weighting as the job. A non-positive aggregate is not merged. The merge sums weights, keeps the later datetime, and stays one row per `(user, item)`.

`recommend` then runs on the fitted sequential model with that dataset. The sequence can change. The sequential weights do not. `filter_viewed` drops items now in that user's history.

`fit_partial` is the third row. Cicerone calls it only on the collaborative LightFM object, after `fit_min_events` known events (default 100) and only when `fit_partial_epochs` is positive (default 1). This walkthrough's `models` list has no `collaborative`, so that call does not run. The sequential model is not passed to it.

With `[events.online]` on, config load requires `[events]` on, `[output].artifact_hmac_key` set, and output kind `db` or a local dataset. A key that is present is measured with `len(key.encode("utf-8"))` and rejected below 16. The `ConfigError` says `output.artifact_hmac_key must be at least 16 bytes`. Config load does not read `[job].save_model_artifact`.

The batch job writes the artifact only when that flag is true. Online serve reads the store. A missing artifact is not a config error. `ensure_loaded` raises `OnlineArtifactError` before the process listens. The message is `events.online.enabled requires a model artifact in [output]; the batch job must set save_model_artifact = true`. `[events.online]` still requires `[model.collaborative].cls` to be `LightFMWrapperModel`. That is the default class. It does not mean collaborative is in `models`.

Leave `[experiment]` off. With it on, the online trainer is not constructed. The log is `Online collaborative refresh skipped while [experiment] is enabled`. Popular / latest inject can still run.

On a torch-free serve, that startup unpickle is where torch is imported. `loads_artifact` calls RecTools `load_model`. `load_model` unpickles the sequential class, and that module imports torch. The import fails inside `ensure_loaded`. The process never listens, so `refresh` never runs.

The INFO line `Online refresh skipped (sequential in artifact, torch extra missing)` is logged only inside `refresh`, after `loads_artifact` has returned. Config load does not stop for the missing extra. It warns: `[events.online] is enabled while job.models includes sequential and the torch extra is not installed; online recommend will be skipped`. With online off, this process never opens the artifact, and GET returns last night's rows.

## AutoML and an unknown user

AutoML is off in this walkthrough, so `[job].models` is what the job fits. Turn it on and the winning candidate replaces that list for the run.

When a candidate contains `sequential`, AutoML logs `Excluding sequential from AutoML candidate pool: …` if torch is missing, or if the median count of distinct `item_id` values per user in the event frame is below `[job.sequential].min_median_interactions` (default 5). The median is computed on events, before aggregation. Sequential is removed from every candidate, and a candidate that becomes empty is dropped. The log line and the exception are different strings. A search that contained only sequential raises `AutoML has no candidates left after excluding sequential; ` plus that same skip reason. With AutoML off, the median does not stop a hand-listed `models` entry from fitting.

An unknown `user_id` is still a lookup. Serve does not score a new sequence. With blending off there is no `__cold_start__` row.

Serve picks one user who still has a `popular_fallback` or `latest` row. `popular_fallback` beats `latest`. Among users who have that winning source, both readers take the lexicographically smallest `user_id`. The database pick is one row: source order, then `user_id ASC`, `LIMIT 1`. The read that follows loads that user's rows.

The response is that user's stored rows, every source in the slice, ordered by `rank` and cut at the request limit. `fallback` is true. A `sequential` row in that slice is returned with the others.

A user is eligible when those stored rows include `popular_fallback` or `latest`. A warm user whose top-K is entirely `sequential` has neither, so that user is not eligible. If the table has neither fallback source, the status is 404 and the detail is `No recommendations for user_id='…'`.

| You change | What happens |
| --- | --- |
| `models` includes `sequential`, job image has no torch | Config loads. The fit raises `ConfigError` with `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'`. |
| `architecture = "bert4rec"` | Next fit is `BERT4RecModel`. LiGR and `sampled_softmax` stay. `mask_prob` stays `0.15`. Causal attention stays off. `source` stays `sequential`. |
| `architecture = "hstu"` | Recommend raises `ValueError` because this job omits `context`. That run publishes no table. Passing `context` gets past that check. |
| `architecture = "hstu"` and `relative_time_attention = false` | Recommend runs. No `unix_ts`. LiGR stays dropped. Inherited `sampled_softmax` still becomes `softmax` unless `loss` is set. |
| Turn blending on | Personalized frames are relabeled `personalized` before the mix. A multi-source row is `blended`. `__cold_start__` is written. |
| Set `model_weights` | RRF. A shared item's `source` joins the contributors with `+`. |
| AutoML on, median distinct items below the floor | INFO `Excluding sequential from AutoML candidate pool:` plus `median distinct items per user is N.N, below job.sequential.min_median_interactions=5`. A candidate that still has another model can win. A pool that is empty afterward raises. |
| `[events.online]` on, torch-free serve, artifact contains `sequential` | Startup fails while loading the artifact. Turn online off on that image. |
| `[events.online]` on, sequential extra installed, both ids known | `recommend` on the frozen model with the merged interactions. No `fit_partial` on the transformer. A new id is dropped. |
| `[experiment]` on while online is on | Online trainer is not started. Log: `Online collaborative refresh skipped while [experiment] is enabled`. |
| GET for an unknown user, blending off | `fallback` true when some user has `popular_fallback` or `latest`. That user's stored rows, cut at the limit. Else 404 `No recommendations for user_id='…'`. |

## When this is the wrong tool

- You need the next item from the click that just happened, inside this request. That is request-path inference. Cicerone's GET is a lookup.
- Median history is a handful of distinct SKUs. Leave sequential off. AutoML drops it below `min_median_interactions`. `popular` still runs.
- You wanted HSTU relative time on a raw click stream. Aggregation already collapsed the repeats, and the default flag fails during recommend, above.
- You only needed bestsellers. Do not pull torch for that.

## In the morning

Last night's job fitted SASRec and wrote the table. The BFF called serve and printed `source = sequential` for the ranks that model won. A purchase at noon did not run SGD on those weights. Cron still owns the fit.

The transformer still sleeps until cron. The list it wrote is already in the table. A click does not wake the model.

<details>
<summary>Reference</summary>

Exact strings. The product page is [how it works](/how-it-works/).

**Extra.** `cicerone-recommender[sequential]` or `requirements-sequential.txt` (`rectools[torch]==0.19.0`). Importable means `torch` and `pytorch_lightning`. Serve image: `docker build -t cicerone -f docker/Dockerfile --target runtime .`. Job image, from the repo root, after that tag exists: `docker build -t cicerone-sequential -f Dockerfile.sequential .`.

**SASRec config Cicerone starts from.** `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`, `deterministic = true`, `verbose = 0`. RecTools' SASRec constructor default for `deterministic` is false. `bert4rec` keeps this config. `[job.sequential].min_median_interactions = 5`, AutoML only.

**HSTU overrides.** Drop `transformer_layers_type`. Inherited `sampled_softmax` becomes `softmax` and `n_negatives` is removed. An explicit `loss` is left as written, and `n_negatives` stays. `deterministic` and `verbose` stay. `relative_time_attention = true` unless you set it false.

**Online log when the artifact did load and torch is missing.** `Online refresh skipped (sequential in artifact, torch extra missing)`.

| What you did | Exact failure |
| --- | --- |
| Fitting `sequential` with no torch | `ConfigError`. Message: `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'` |
| `architecture` and `cls` disagree | `ConfigError`. Message starts `Conflicting sequential settings:` and names both values. |
| HSTU recommend, relative time still on | ValueError: This model requires `context` to be provided for recommendations generation (model.require_recommend_context is True).Check docs and examples for details. |
| AutoML drops sequential | `Excluding sequential from AutoML candidate pool: ` plus either `optional sequential extra is not installed (install with: pip install 'cicerone-recommender[sequential]')` or `median distinct items per user is N.N, below job.sequential.min_median_interactions=5` |
| AutoML pool empty after that drop | `AutoML has no candidates left after excluding sequential; ` plus the skip reason |
| `[experiment]` on, online on | `Online collaborative refresh skipped while [experiment] is enabled` |
| Unknown GET, no `popular_fallback` or `latest` row | `No recommendations for user_id='…'` |

</details>

---
title: The transformer still sleeps until cron
description: SASRec trains in the nightly job and writes a top-K table. Serve GET stays a lookup. Torch stays off the default image and off that request.
date: 2026-11-06
excerpt: Put sequential in models and the job fits a transformer. The homepage still reads last night's rows. A GET does not load the artifact.
authors:
  - nicholas
---

You want next-item ranking, so you add a transformer. You put SASRec behind the homepage and expect each click to move the weights. You did not get a live sequence model. You got another nightly top-K, stamped `source = sequential`, and a lookup that reads those rows.

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

Four nouns, then stop mixing them.

| Noun | What it is |
| --- | --- |
| **Sequence** | One row per `(user, item)` after weighting. Order is that row's datetime. |
| **Architecture** | `sasrec`, `bert4rec`, or `hstu` under `[model.sequential]`. Default class is `SASRecModel`. |
| **Fit** | `job.run()` calls `model.fit` on the aggregated dataset and writes rows. |
| **Lookup** | `GET /recommendations/{user_id}` or a `SELECT`. Reads rows. Does not load the artifact. |

A serve process loads the artifact only when `[events.online]` is on, and it does that at startup. That load is not a GET.

## The job image needs the extra

`pip install 'cicerone-recommender[sequential]'`, or `pip install -r requirements-sequential.txt` from a checkout. `pyproject.toml` declares that extra as `rectools[torch]>=0.19.0,<0.20`. `requirements-sequential.txt` pins `rectools[torch]==0.19.0`, the same RecTools version as `requirements.txt`. The extra counts as installed when `torch` and `pytorch_lightning` are importable.

`docker/Dockerfile` does not end on the image you want for serve. Three names:

| Name | What it is |
| --- | --- |
| `runtime` | Production stage. Does not install `requirements-sequential.txt`. Torch is absent. |
| `test-sequential` | Last stage of that Dockerfile. A test image. It installs the sequential extra so the sequential tests can import torch. |
| `cicerone-sequential` | Local tag you build yourself, for the batch fit. It is not a stage of `docker/Dockerfile`. |

A bare `docker build -f docker/Dockerfile .` builds the last stage, `test-sequential`. Tag the torch-free stage yourself:

```sh
docker build -t cicerone -f docker/Dockerfile --target runtime .
```

TOML load does not check the extra. The job raises `ConfigError` when it builds the strategy. The message is:

```text
strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'
```

`Dockerfile.sequential` starts `FROM cicerone`. That tag has to be the `runtime` image from the command above. If `cicerone` is the default last stage, this file extends `test-sequential` instead, and serve is no longer torch-free. Save it in the repo root so the build context can see `requirements-sequential.txt`. The runtime stage's last user is `cicerone`, so the snippet switches to root for `pip` and back. Tag the result `cicerone-sequential`.

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

Run the batch job from `cicerone-sequential`. Keep the serve process on the `cicerone` image and leave `[events.online]` off. The BFF does not need the extra either: `pip install cicerone-recommender` on Python 3.11 is enough for `ServeClient`, and `fastapi` is already a dependency. That install does not import torch.

## Turn sequential on

These tables are edits to the nightly files. They are not a full config.

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

`architecture` selects the RecTools class: `sasrec` → `SASRecModel`, `bert4rec` → `BERT4RecModel`, `hstu` → `HSTUModel`. Setting `architecture` and a conflicting `cls` is a `ConfigError`. The other keys stay on the SASRec defaults unless you override them: `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`. Cicerone expands `LiGRLayers` to `rectools.models.nn.transformers.ligr.LiGRLayers` before RecTools sees it.

| Architecture | Class | What Cicerone actually passes |
| --- | --- | --- |
| `sasrec` (default) | `SASRecModel` | Shifted next-item. RecTools leaves causal attention on. Layers stay LiGR. Loss stays `sampled_softmax`. |
| `bert4rec` | `BERT4RecModel` | Masked-item training. RecTools `mask_prob` stays `0.15`. Of the positions it picks, 80% become MASK, 10% become a random item, and 10% keep the input item while the target is set to `0`. Causal attention stays off. Cicerone still passes LiGR and `sampled_softmax`. RecTools' own BERT4Rec defaults are PreLN layers and `softmax`. |
| `hstu` | `HSTUModel` | Same shifted preparator as SASRec, on HSTU layers. Cicerone drops LiGR. An inherited `sampled_softmax` becomes `softmax`, and `n_negatives` is dropped, unless you set `loss`. `relative_time_attention` is turned on, which sets `HSTUModel.require_recommend_context`. The job's `recommend` call does not pass `context`. The fit can finish. Recommend raises `ValueError`. `_write_job_outputs` does not run, so that run publishes no recommendation table. `relative_time_attention = false` lets recommend run and turns the timestamps off. |

`architecture = "hstu"` with the defaults is not a working end-to-end batch job. RecTools raises `ValueError` from `recommend` because `context` is missing. The message, including the missing space before `Check`, is:

```text
This model requires `context` to be provided for recommendations generation (model.require_recommend_context is True).Check docs and examples for details.
```

The job never supplies that argument. The fit trains the HSTU weights. The exception is raised before `_write_job_outputs`, so the recommendation table from that run is not published. Set `relative_time_attention = false` under `[model.sequential]` and recommend runs. That setting also leaves `add_unix_ts` off.

Users with no interactions skip sequential. `popular` still scores them, and those rows are `popular_fallback`. Personalized recommend passes `filter_viewed`, so a sequential row is an item that user has not already aggregated.

With no `model_weights`, combine is priority. `models = ["sequential", "popular"]` lets sequential win a duplicate item, and that row keeps `source = sequential`. A `popular` row that still fits in the top-K is `popular_fallback`. Set `model_weights` and the combiner is weighted RRF: a row that both strategies contributed is `sequential+popular_fallback`, in `models` order.

Blending is a different combiner. It relabels every personalized frame to `personalized` before the mix. A row drawn from more than one of `personalized`, `popular_fallback`, and `latest` is `blended`. Blending is also what writes `'__cold_start__'`. This config does not. The nightly Rails helper that falls back to that sentinel gets no rows for a guest.

## A sequence is not a click log

Before the model sees the data, the job weights events and aggregates to one row per `(user, item)`. Where `[event_caps]` names an event type, the job keeps that many most recent events per `(user, item, event_type)`. Half-life decay multiplies what remains. The aggregated weight is the sum. The datetime is `max(occurred_at)`. Pairs whose sum is not positive are dropped. RecTools sorts the remaining rows by that datetime, stably, and builds one sequence per user.

Two purchases of the same SKU are one position, at the later timestamp. Buy A, then B, then A again, and the sequence is B then A. Intermediate timestamps are not in the sequence.

`session_max_len` (default 50) is the recommend window for SASRec and HSTU: the tail of that unique history. Training keeps one more item (`session_max_len + 1`) because the shifted target needs it. BERT4Rec training keeps `session_max_len` items. At recommend time BERT4Rec appends a MASK token inside that same window, so the model sees `session_max_len - 1` real items.

`train_min_user_interactions` (default 2) drops shorter users from training only. A user left out of training can still be scored when they hold at least one item the fitted model already knows. RecTools leaves out a target user who has none of those items.

HSTU is the architecture that reads a clock, and only the clock aggregation left behind. With `relative_time_attention` on, RecTools sets `add_unix_ts` and turns each aggregated datetime into seconds. The bias is the gap between last touches of distinct items. It is not the gap between raw repeats. Do not quote a session-stream paper this fit did not train. SASRec and BERT4Rec do not get `unix_ts` on these defaults. Turning the flag off, so recommend can run, removes those timestamps.

## The table is the morning read

A known user's `SELECT … ORDER BY rank` returns that user's stored rows. The nightly `JOIN` does that. It does not grow a `__cold_start__` list.

Serve is the other reader. This walkthrough uses the stdlib client in the package.

```python
import os
from fastapi import FastAPI, HTTPException
from cicerone.serve_client import ServeClient, ServeClientError

app = FastAPI()
client = ServeClient(
    os.environ["CICERONE_SERVE_URL"],
    token=os.environ.get("CICERONE_SERVE_TOKEN"),
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

Bearer only if `[serve].auth_token` is set. `limit` and `k` are the same cap; pass one. Send both with different values and serve returns 400, detail `limit and k disagree; pass only one (or the same value)`. `exclude_unavailable` defaults to true. A category or availability filter that removes every row is 200 with an empty `items` list.

## Re-score is not a fit

Optional `[events.online]` runs in the serve process, after the [checkout](/articles/this-afternoons-checkout-can-move-the-row/) flush. It is not transformer training.

| Operation | Changes model weights? | Changes recommendation sequence? |
| --- | --- | --- |
| Batch fit | Yes | Yes |
| Online re-score | No | Yes |
| LightFM `fit_partial` | Yes, for LightFM only | Not applicable to the transformer |

The worker loads the artifact once, at startup, and only when `[events.online]` is on. A GET does not load it. For a known `user_id` and a known `item_id` — both already in the artifact's interaction table — it weights the new events the same way the job did, then merges them onto the loaded interactions: weights sum, datetime stays the max, still one row per `(user, item)`. It calls `recommend` on the fitted sequential model with that updated dataset. The sequence the model reads can change. The model weights do not. `filter_viewed` then drops items now in that user's history. A new user or a new SKU is counted as dropped and is not in this refresh.

`fit_partial` is the third row, and it is not the re-score. Cicerone calls it only on the collaborative LightFM object, after `fit_min_events` known events (default 100) and only when `fit_partial_epochs` is positive. This walkthrough's `models` list has no `collaborative`, so that call does not run. The sequential model is not passed to it.

Online also needs `[events]` on, `[job].save_model_artifact = true`, `[output].artifact_hmac_key`, and output kind `db` or a local dataset. `[events.online]` still requires `[model.collaborative].cls` to be `LightFMWrapperModel`. That is the default class. It does not mean collaborative is in `models`.

Leave `[experiment]` off. With it on, the online trainer is not constructed. The log is `Online collaborative refresh skipped while [experiment] is enabled`. Popular / latest inject can still run.

Put the sequential extra on the job process (`cicerone-sequential`). Put it on the serve process only when that process must unpickle the artifact. A torch-free serve (`runtime`, tagged `cicerone`) with `[events.online]` on calls `ensure_loaded` at startup, before it accepts a GET. RecTools `load_model` has to import `SASRecModel`, `BERT4RecModel`, or `HSTUModel`, and those classes import torch, so the process dies during that unpickle. The INFO line `Online refresh skipped (sequential in artifact, torch extra missing)` is logged later, inside `refresh`, and only after `loads_artifact` has already returned. A startup that died in `ensure_loaded` never reaches it. Config load may also warn that online recommend will be skipped. That warning is not the unpickle. Last night's rows are what that torch-free serve returns when online is off, because that process never opens the artifact.

## AutoML and an unknown user

AutoML is off in this walkthrough, so `[job].models` is what the job fits. Turn it on and the winning candidate replaces that list for the run.

When a candidate contains `sequential`, AutoML logs `Excluding sequential from AutoML candidate pool: …` if torch is missing, or if the median count of distinct `item_id` values per user in the event frame is below `[job.sequential].min_median_interactions` (default 5). The median is computed on events, before aggregation. Sequential is removed from every candidate, and a candidate that becomes empty is dropped. A search that contained only sequential then raises `AutoML has no candidates left after excluding sequential`. With AutoML off, the median does not stop a hand-listed `models` entry from fitting.

An unknown `user_id` is still a lookup. Serve does not score a new sequence. With blending off there is no `__cold_start__` row. Serve picks one user who still has a `popular_fallback` or `latest` row in the table: `popular_fallback` beats `latest`. Ties break on string order of `user_id` in the dataset reader, and on `ORDER BY user_id ASC` in the database reader. The response is that user's stored rows, ordered by `rank`, cut at the request limit, every source in that slice. A warm user whose top-K is entirely `sequential` has no fallback row and is not eligible. When sequential fills the top-K, the eligible users are the ones sequential skipped, and their rows are `popular_fallback`. If the chosen user's slice also contains `sequential`, those rows are in the response. `fallback` is true. If the table has neither fallback source, the status is 404 and the detail is `No recommendations for user_id='…'`.

| You change | What happens |
| --- | --- |
| `models` includes `sequential`, job image has no torch | Config loads. The fit raises `ConfigError` with `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'`. |
| `architecture = "bert4rec"` | Next fit is `BERT4RecModel`. LiGR and `sampled_softmax` stay. `mask_prob` stays `0.15`. Causal attention stays off. `source` stays `sequential`. |
| `architecture = "hstu"` | The fit builds `HSTUModel` with `relative_time_attention` on. Recommend raises `ValueError` because `context` is missing. The job does not publish that run. |
| `architecture = "hstu"` and `relative_time_attention = false` | Recommend runs. No `unix_ts`. LiGR stays dropped. Inherited `sampled_softmax` still becomes `softmax` unless `loss` is set. |
| Turn blending on | Personalized frames are relabeled `personalized` before the mix. A multi-source row is `blended`. `__cold_start__` is written. |
| Set `model_weights` | RRF. A shared item's `source` joins the contributors with `+`. |
| AutoML on, median distinct items below the floor | INFO `Excluding sequential from AutoML candidate pool:` plus the median and `job.sequential.min_median_interactions`. The winner is what fits, and sequential is not in it. |
| `[events.online]` on, torch-free serve, artifact contains `sequential` | Startup fails while loading the artifact. Turn online off on that image. |
| `[events.online]` on, sequential extra installed, both ids known | `recommend` on the frozen model with the merged interactions. No `fit_partial` on the transformer. A new id is dropped. |
| `[experiment]` on while online is on | Online trainer is not started. Log: `Online collaborative refresh skipped while [experiment] is enabled`. |
| GET for an unknown user, blending off | `fallback` true when some user has `popular_fallback` or `latest`. That user's stored rows, cut at the limit. Else 404 `No recommendations for user_id='…'`. |

## When this is the wrong tool

- You need the next item from the click that just happened, inside this request. That is request-path inference. Cicerone's GET is a lookup.
- Median history is a handful of distinct SKUs. Leave sequential off. AutoML drops it below `min_median_interactions`. `popular` still runs.
- You wanted HSTU relative time on a raw repeat-rich stream. Aggregation already collapsed the repeats, and the default flag makes recommend demand a `context` frame the job does not pass.
- You only needed bestsellers. Do not pull torch for that.

## In the morning

Last night's job fitted SASRec and wrote the table. The BFF called serve and printed `source = sequential` for the ranks that model won. A purchase at noon did not run SGD on those weights. Cron still owns the fit.

The transformer still sleeps until cron. The list it wrote is already in the table. A click does not wake the model.

<details>
<summary>Reference</summary>

Exact strings. The product page is [how it works](/how-it-works/).

**Extra.** `cicerone-recommender[sequential]` or `requirements-sequential.txt` (`rectools[torch]==0.19.0`). Importable means `torch` and `pytorch_lightning`. `docker build -t cicerone -f docker/Dockerfile --target runtime .`, then `Dockerfile.sequential` from the repo root, tagged `cicerone-sequential`. A bare build of `docker/Dockerfile` is `test-sequential`, not `runtime`.

**Defaults that stay unless overridden.** `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`. `[job.sequential].min_median_interactions = 5`, AutoML only.

**HSTU overrides.** Drop `transformer_layers_type`. Inherited `sampled_softmax` becomes `softmax` and `n_negatives` is removed. An explicit `loss` is left as written, and `n_negatives` stays. `relative_time_attention = true` unless you set it false.

**Online log when the artifact did load and torch is missing.** `Online refresh skipped (sequential in artifact, torch extra missing)`.

| What you did | Exact failure |
| --- | --- |
| Fitting `sequential` with no torch | `ConfigError`. Message: `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'` |
| `architecture` and `cls` disagree | `ConfigError`. Message starts `Conflicting sequential settings:` and names both values. |
| HSTU recommend, relative time still on | ValueError: This model requires `context` to be provided for recommendations generation (model.require_recommend_context is True).Check docs and examples for details. |
| AutoML drops sequential | `Excluding sequential from AutoML candidate pool: ` plus either `optional sequential extra is not installed (install with: pip install 'cicerone-recommender[sequential]')` or `median distinct items per user is N.N, below job.sequential.min_median_interactions=5` |
| AutoML pool empty after that drop | `AutoML has no candidates left after excluding sequential` |
| `[experiment]` on, online on | `Online collaborative refresh skipped while [experiment] is enabled` |
| Unknown GET, no `popular_fallback` or `latest` row | `No recommendations for user_id='…'` |

</details>

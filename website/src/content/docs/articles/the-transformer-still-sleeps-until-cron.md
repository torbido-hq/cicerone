---
title: The transformer still sleeps until cron
description: SASRec trains in the nightly job and writes a top-K table. Serve GET stays a lookup. Torch stays off the default image and off that request.
date: 2026-11-06
excerpt: Put sequential in models and the job fits a transformer. The homepage still reads last night's rows. A GET does not load the artifact.
authors:
  - nicholas
---

You want next-item ranking, so you add a transformer. You put SASRec behind the homepage and expect each click to move the weights. You did not get a live sequence model. You got another nightly top-K, stamped `source = sequential`, and a lookup that reads those rows.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) post already writes ranks with LightFM and bestsellers. This walkthrough swaps the personalized half for Cicerone's opt-in `sequential` strategy — RecTools SASRec by default, or BERT4Rec / HSTU. The storefront still reads a table, or it calls serve. The GET does not load the fitted model.

```text
[job].models includes "sequential"
        │
        ▼
   job image with the sequential extra
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
   stored rows, no model load
```

Four nouns, then stop mixing them.

| Noun | What it is |
| --- | --- |
| **Sequence** | One row per `(user, item)` after weighting. Order is that row's datetime. |
| **Architecture** | `sasrec`, `bert4rec`, or `hstu` under `[model.sequential]`. Default class is `SASRecModel`. |
| **Fit** | `job.run()` calls `model.fit` on the aggregated dataset and writes rows. |
| **Lookup** | `GET /recommendations/{user_id}` or a `SELECT`. Reads rows. Does not load the artifact. |

## The default image does not fit this

`pip install 'cicerone-recommender[sequential]'`, or `pip install -r requirements-sequential.txt` from a checkout. The requirements file pins `rectools[torch]==0.19.0`, the same RecTools version as the runtime image. The extra counts as installed when `torch` and `pytorch_lightning` are importable. The default Docker runtime image does **not** install it. A GET on that image does not import torch: the handler never loads a model.

TOML load does not check the extra. The job raises when it builds the strategy:

```text
ConfigError: strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'
```

Build the default image the way the [nightly](/articles/a-nightly-table-next-to-your-orders/) post does (`docker build -t cicerone -f docker/Dockerfile .`). Save the next file as `Dockerfile.sequential` in the repo root, on top of that local tag. The runtime stage's last user is `cicerone`, so the snippet switches to root for `pip` and back.

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

Run the batch job from `cicerone-sequential`. Keep serve on `cicerone` and leave `[events.online]` off. A serve process loads the artifact only for online refresh, and that unpickle imports torch.

## Turn sequential on

These tables are edits to the nightly files. They are not a full config.

Priority combine keeps the `sequential` label. Blending relabels every personalized frame to `personalized` before it mixes, so turn blending off in the nightly `features.toml`. Leave `[event_weights]`, caps, and item features as they are.

```toml
# features.toml — the nightly file already has this table
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

`architecture` selects the RecTools class: `sasrec` → `SASRecModel`, `bert4rec` → `BERT4RecModel`, `hstu` → `HSTUModel`. Setting `architecture` and a conflicting `cls` is a `ConfigError`. The other keys stay on the SASRec defaults unless you override them: `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`. HSTU is the exception: Cicerone drops `transformer_layers_type`, and an inherited `sampled_softmax` is rewritten to `softmax` unless you set `loss`.

| Architecture | Class | What that class trains |
| --- | --- | --- |
| `sasrec` (default) | `SASRecModel` | Shifted next-item. Causal attention stays on. Layers stay LiGR. |
| `bert4rec` | `BERT4RecModel` | Masked items (MLM, mask probability 0.15). Causal attention stays off. LiGR stays unless you change `transformer_layers_type`. |
| `hstu` | `HSTUModel` | Shifted next-item on HSTU layers. `relative_time_attention` is turned on. Loss becomes `softmax` unless you set it. |

Users need interactions. Feature-only users skip sequential the way they skip `item_based`. `popular` still scores them; those rows are `popular_fallback`. Personalized recommend passes `filter_viewed`, so a sequential row is an item that user has not already aggregated.

Blending is what wrote `'__cold_start__'` in the nightly post. This config does not. The nightly Rails helper that falls back to that sentinel gets no rows for a guest.

## A sequence is not a click log

Before the model sees the data, Cicerone weights events and aggregates to **one row per `(user, item)`**. The weight is the sum of the decayed row weights. The datetime is `max(occurred_at)`. RecTools then sorts those rows by datetime and builds the sequence from that order.

Two purchases of the same SKU are one position, at the later timestamp. A string of distinct clicks stays a string of distinct items, ordered by each item's last time. Intermediate timestamps are not in the sequence.

`session_max_len` (default 50) keeps the tail of that unique history. SASRec's training preparator keeps one extra item on top of that cap. Recommend still uses `session_max_len`. `train_min_user_interactions` (default 2) drops shorter users from **training**. A user left out of training can still be scored when they hold at least one item the fitted model already knows.

HSTU is the architecture that reads the clock. With `relative_time_attention` on, RecTools adds `unix_ts` from that aggregated datetime. The bias is the gap between last touches of distinct items. It is not the gap between raw repeats, because the repeats are already gone. Do not quote a session-stream paper this fit did not train.

SASRec and BERT4Rec, on these defaults, get the same order and the same weights. They do not get `unix_ts`.

AutoML is a separate gate. When a candidate contains `sequential`, it logs `Excluding sequential from AutoML candidate pool: …` if torch is missing, or if the median distinct items per user in the event frame is below `[job.sequential].min_median_interactions` (default 5). That edits the search, not a job with AutoML off: `models` still fits. With AutoML on, the job fits the winning candidate. Sequential is gone from that candidate once it was excluded. A search that contained only sequential then raises `AutoML has no candidates left after excluding sequential`.

## Read it without torch

A known user's `SELECT … ORDER BY rank` returns that user's rows, `sequential` and `popular_fallback` together. The nightly `JOIN` does that. It does not grow a `__cold_start__` list.

Serve is the other reader. This walkthrough uses the stdlib client in the package. The BFF is Python 3.11. `pip install cicerone-recommender` is enough for the import (`fastapi` is already a dependency). That install does not include the sequential extra, so this process does not import torch.

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

The serve container that answers `GET /recommendations` stays on the default image. `source` on a sequential win is the string `sequential`. LightFM's wins stay `personalized`. `popular` in config is `popular_fallback` in the table.

Bearer only if `[serve].auth_token` is set. `limit` and `k` are the same cap; pass one. `category` filters after the list is chosen.

An unknown `user_id` does not get someone else's sequential list. With no `__cold_start__` rows, serve returns one user's rows whose `source` is `popular_fallback` or `latest`. A user who has `popular_fallback` wins over a user who only has `latest`. Ties break on the smallest `user_id`. `fallback` is true. If the table has neither source, the status is 404 and the detail is `No recommendations for user_id='…'`.

## Fit, re-score, fit_partial

Three operations. Only the first one trains the transformer.

**Fit.** `job.run()` aggregates, calls `model.fit`, and writes the table. Weights move here. This is the cron.

**Re-score.** Optional `[events.online]` on the serve process, after the [checkout](/articles/this-afternoons-checkout-can-move-the-row/) flush has a known `user_id` and a known `item_id`. The worker loads the artifact, appends those interactions (weights sum, datetime stays the max), and calls `recommend` on the fitted sequential model. The sequence the model reads can change. The model weights do not. `filter_viewed` then drops items now in that user's history. Both ids have to be in the last artifact. A new user or a new SKU is not in this refresh.

**`fit_partial`.** Cicerone calls it only on the collaborative LightFM object, and only after `fit_min_events` known events. This walkthrough's `models` list has no `collaborative`, so that call does not run. The sequential model is not passed to it.

Online also needs what the checkout post already sets: `[events]` on, `[job].save_model_artifact = true`, `[output].artifact_hmac_key`, and `db` or a local dataset. `[events.online]` still requires `[model.collaborative].cls` to be `LightFMWrapperModel`. That is the default class. It does not mean collaborative is in `models`.

Leave `[experiment]` off. With it on, the online trainer is not constructed (`Online collaborative refresh skipped while [experiment] is enabled`). Popular / latest inject can still run.

Put the sequential extra on the job. Put it on serve only when that process must unpickle the artifact. A torch-free serve with `[events.online]` on does not reach a polite skip: `ensure_loaded` unpickles `SASRecModel` / `BERT4RecModel` / `HSTUModel`, and that import needs torch. The process fails at startup. Last night's rows are still what a torch-free serve returns when online is off.

| You change | What happens |
| --- | --- |
| `models` includes `sequential`, job image has no torch | Config loads. Fit raises `ConfigError`: `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'`. |
| `architecture = "bert4rec"` | Next fit is `BERT4RecModel` (masked items). LiGR and `sampled_softmax` stay unless you override them. `source` stays `sequential`. |
| `architecture = "hstu"` | Next fit is `HSTUModel`. LiGR is dropped. Inherited `sampled_softmax` becomes `softmax` unless `loss` is set. `relative_time_attention` is on. `source` stays `sequential`. |
| Turn blending on | Personalized frames are relabeled `personalized` before the mix. A multi-source row is `blended`. `sequential` is not a `source`. |
| AutoML, median distinct items below the floor | INFO `Excluding sequential from AutoML candidate pool:` plus the median and `job.sequential.min_median_interactions`. AutoML off: `models` still fits. AutoML on: the winning candidate fits, without sequential. |
| `[events.online]` on, torch-free serve, artifact contains `sequential` | Startup fails while unpickling the sequential model. Turn online off on that image. |
| `[events.online]` on, sequential extra installed, both ids known | Frozen `recommend` for users in that flush. No `fit_partial` on the transformer. A new id is dropped from this refresh. |
| `[experiment]` on while online is on | Online trainer is not started. Log: `Online collaborative refresh skipped while [experiment] is enabled`. |
| GET for an unknown user, blending off, `popular` rows exist | `fallback` true. One other user's `popular_fallback` or `latest` rows. A `popular_fallback` user wins. Not a sequential list. |
| GET for an unknown user, no `popular_fallback` or `latest` rows | 404 `No recommendations for user_id='…'`. |

## When this is the wrong tool

- You need the next item from the click that just happened, inside this request. That is request-path inference. Cicerone's GET is a lookup.
- Median history is a handful of distinct SKUs. Leave sequential off. AutoML drops it below `min_median_interactions`. `popular` still runs.
- You wanted HSTU relative time on a raw repeat-rich stream. Aggregation already collapsed the repeats.
- You only needed bestsellers. Do not pull torch for that.

## In the morning

Last night's job fitted a transformer and wrote the table. The BFF called serve and printed `source = sequential` for the ranks that model won. A purchase at noon did not run SGD on SASRec. Cron still owns the weights.

The transformer still sleeps until cron. The list it wrote is already in the table. A click does not wake the model.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [how it works](/how-it-works/).

**Extra.** `pip install 'cicerone-recommender[sequential]'` or `requirements-sequential.txt` (`rectools[torch]==0.19.0`). `sequential_extra_available` is true when `torch` and `pytorch_lightning` are importable. Default runtime image is torch-free. GET does not load the artifact.

**Config.** Opt-in name in `[job].models`. `[model.sequential].architecture`: `sasrec` (default) / `bert4rec` / `hstu`. Conflicting `architecture` + `cls` is `ConfigError`. SASRec defaults that stay unless overridden: `n_factors = 64`, `n_blocks = 2`, `n_heads = 4`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`. `[job.sequential].min_median_interactions` default `5` (AutoML only). The snippets above are edits to the nightly files.

**Sequence.** Aggregate: sum of decayed weights, datetime = `max(occurred_at)`, one row per `(user, item)`. RecTools sorts by that datetime. `session_max_len` keeps the tail. SASRec training keeps one extra item. `train_min_user_interactions` filters training, not recommend. HSTU with `relative_time_attention` (Cicerone sets it) adds `unix_ts` from that datetime.

**Source.** Strategy label is `sequential`. Blending on: those frames are relabeled `personalized` before the blend. Priority or RRF, blending off, keeps `sequential`. `filter_viewed` is on for personalized recommend.

**Serve fallback, blending off.** No `__cold_start__` rows. Unknown user: one user's `popular_fallback` or `latest` rows. A user with `popular_fallback` wins; else `latest`. Ties break on the smallest `user_id`. 404 detail `No recommendations for user_id='…'` when neither source exists.

**Online.** `fit_partial` is called on collaborative LightFM only. Sequential `recommend` uses the loaded model weights and the updated interactions. Both ids must already be in the artifact. Missing extra while the artifact contains `sequential`: `ensure_loaded` unpickles that model and the process needs torch. `[experiment]` on: trainer not started; log `Online collaborative refresh skipped while [experiment] is enabled`. AutoML on: `job.run()` fits the winning candidate, not the `[job].models` list.

**Failures.**

| What you did | What happens |
| --- | --- |
| `models` includes `sequential`, no torch on the job | Fit: `ConfigError` with the install hint. Config load succeeds. |
| Online on, torch-free serve, sequential in the artifact | Startup fails while `ensure_loaded` unpickles the sequential model. |
| Online on, extra installed, new user or new item | That event is not in the online refresh. Popular / latest inject can still move. |
| Blending enabled | `source` is `personalized` or `blended`, not `sequential`. |
| AutoML, median below the floor, or no torch | INFO `Excluding sequential from AutoML candidate pool:`. AutoML on replaces `models` with the winner. |
| Feature-only user | Sequential skipped for that user. `popular_fallback` can still fill. |
| Unknown GET, no popular or latest rows | 404 `No recommendations for user_id='…'`. |

</details>

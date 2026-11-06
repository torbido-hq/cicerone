---
title: The transformer still sleeps until cron
description: SASRec trains in the nightly job and writes a top-K table. Serve GET stays a lookup. Torch stays off the default image and off the request path.
date: 2026-11-06
excerpt: Put sequential in models and the job fits a transformer. The homepage still reads last night's rows. GET does not load torch.
authors:
  - nicholas
---

You want next-item ranking, so you add a transformer. You put SASRec behind the homepage and expect each click to move the list. You did not get a live sequence model. You got another nightly top-K, stamped `source = sequential`, and a lookup that never imports torch.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) post already writes ranks with LightFM and bestsellers. This walkthrough swaps the personalized half for Cicerone's opt-in `sequential` strategy — RecTools SASRec by default, or BERT4Rec / HSTU. The storefront still reads a table, or it calls serve. There is still no model in the request.

```text
[job].models includes "sequential"
        │
        ▼
   job image with torch extra
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
   no torch on that path
```

Four nouns, then stop mixing them.

| Noun | What it is |
| --- | --- |
| **Sequence** | Distinct items for one user, ordered by last-touch time after aggregation. Not a raw click stream with repeats. |
| **Architecture** | `sasrec`, `bert4rec`, or `hstu` under `[model.sequential]`. Default is SASRec. |
| **Fit** | Tonight's `job.run()` that trains the transformer and writes rows. |
| **Lookup** | `GET /recommendations/{user_id}` or a `SELECT`. Reads rows. Does not load the artifact. |

## The default image does not fit this

`pip install 'cicerone-recommender[sequential]'` (or `requirements-sequential.txt` from a checkout). That pulls RecTools' torch extra. The default Docker runtime image does **not** install it. Serve's GET path never imports it either.

List `"sequential"` in `[job].models` on a torch-free job and fit raises `ConfigError`: `strategy 'sequential' requires torch; install with: pip install 'cicerone-recommender[sequential]'`. Build a job image that has the extra. Leave the serve image torch-free unless you also want online re-score of sequential rows (below).

Build the default image the way the [nightly](/articles/a-nightly-table-next-to-your-orders/) post does (`docker build -t cicerone -f docker/Dockerfile .`), then add the extra:

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

Run the batch job from `cicerone-sequential`. Keep serve on `cicerone` unless you mean frozen online re-score.

## Turn sequential on

Priority combine keeps the `sequential` label visible. Blending collapses every personalized strategy into `source = personalized` / `blended`, so turn it off in `features.toml` for this walkthrough. Keep the rest of that file from the nightly post (`[event_weights]`, caps, item features).

```toml
# features.toml
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

`architecture` is the Cicerone knob. It maps to RecTools `SASRecModel`, `BERT4RecModel`, or `HSTUModel`. Defaults follow RecTools eSASRec (`sampled_softmax` + LiGR layers). Do not set both `architecture` and a conflicting `cls`.

| Architecture | Attention | What the fit predicts |
| --- | --- | --- |
| `sasrec` (default) | Causal | Next item from the left context |
| `bert4rec` | Bidirectional | Masked items (cloze) |
| `hstu` | Pointwise; optional relative time | Next item; time bias wants raw event order |

Users need interaction history. Feature-only warm users skip sequential the way they skip `item_based`. `popular` backfills holes. Blending wrote the `'__cold_start__'` sentinel in the nightly post. With blending off there is no sentinel: serve substitutes one `popular_fallback` / `latest` user's top-K, and returns 404 if the table has neither.

## A sequence is not a session log

Before any model sees the data, Cicerone weights events and aggregates to **one row per `(user, item)`**. The sequential strategy then orders those distinct items by last-touch time. Two purchases of the same SKU collapse. A bounce-heavy click stream becomes a short unique-item list.

That is why HSTU's relative-time bias is weak here. The paper wants ordered raw events. Cicerone hands it last-touch uniques. Do not quote a session-stream result this binary does not train.

`session_max_len` (default 50) caps how much of that unique history the model sees. `train_min_user_interactions` (default 2) is RecTools' floor for who enters training. AutoML is a different gate: it **drops** `sequential` from the candidate pool (INFO log) when torch is missing or the median distinct items per user is below `[job.sequential].min_median_interactions` (default 5). A hand-listed `models = ["sequential", …]` still fits when the extra is present; AutoML just will not pick it on a thin catalog.

## Read it without torch

The [nightly](/articles/a-nightly-table-next-to-your-orders/) `JOIN` still works. So does serve. This walkthrough uses the thin stdlib client that ships with the package — no FastAPI dependency inside Cicerone, no torch on the BFF.

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

`pip install cicerone-recommender fastapi` on the BFF is enough for that file. The serve container that answers `GET /recommendations` stays on the default torch-free image. `source` on a sequential win is the string `sequential`, not `personalized`. LightFM's wins stay `personalized`. Model name and `source` label match for this strategy; they do not for `popular` / `popular_fallback`.

Bearer only if `[serve].auth_token` is set. `limit` / `k` and `category` still apply after the list is chosen.

## What this afternoon's checkout does

The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) path can still refresh popular / latest after a flush. Sequential **never** runs `fit_partial`. Weights stay whatever tonight's job wrote.

| Serve image | `[events.online]` | What happens to sequential rows |
| --- | --- | --- |
| Torch-free (default) | off | Last night's rows until the next `job.run()`. Popular / latest can still move. |
| Torch-free | on, artifact includes `sequential` | Online rewrite is **skipped entirely** (`Online refresh skipped (sequential in artifact, torch extra missing)`). Collaborative re-score does not run either. Popular / latest inject still can. |
| Has sequential extra | on | Known users can be **re-scored** from the frozen sequential model. Still no SGD on the transformer. New user / item ids wait for `job.run()`. |

Put torch on the job. Put it on serve only if you mean that frozen re-score. GET still does not load the model.

| You change | What happens |
| --- | --- |
| Add `"sequential"` without the extra | Fit: `ConfigError` requiring torch. |
| `architecture = "bert4rec"` | Next job fits BERT4Rec. Same `source = sequential` label. |
| Turn blending on | Personalized slots relabel to `personalized` / `blended`. The `sequential` string disappears from `source`. |
| AutoML on a sparse catalog | Sequential dropped from candidates; INFO log names the reason. |
| Afternoon purchase, online on, torch-free serve | No online personalized rewrite while `sequential` is in the artifact. |
| Afternoon purchase, online on, torch on serve | Frozen sequential re-score for known ids in the flush. No `fit_partial`. |

## When this is the wrong tool

- You need the next item inside this request from the live session. That is request-path inference. Cicerone's GET is a lookup.
- Median history is a handful of distinct SKUs. Leave sequential off; keep `popular` / LightFM. AutoML will drop it anyway below `min_median_interactions`.
- You wanted HSTU relative-time on a raw repeat-rich stream. Aggregation already removed the repeats.
- You only needed bestsellers. Do not pull torch for that.

## In the morning

Last night's job fitted a transformer and wrote twenty rows. The BFF called serve and printed `source = sequential` for the ranks that model won. A purchase at noon did not run SGD on SASRec. Cron still owns the weights.

The transformer still sleeps until cron. The list it wrote is already there. A click does not wake the model.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [how it works](/how-it-works/).

**Extra.** `pip install 'cicerone-recommender[sequential]'` or `pip install -r requirements-sequential.txt`. Needs importable `torch` and `pytorch_lightning`. Default runtime image is torch-free. GET never imports torch.

**Config.** Opt-in name in `[job].models`. `[model.sequential].architecture`: `sasrec` (default) / `bert4rec` / `hstu`. Conflicting `architecture` + `cls` is `ConfigError`. Defaults: `n_factors = 64`, `epochs = 3`, `loss = sampled_softmax`, `n_negatives = 256`, `transformer_layers_type = LiGRLayers`, `session_max_len = 50`, `train_min_user_interactions = 2`, `batch_size = 128`, `lr = 0.001`. `[job.sequential].min_median_interactions` default `5` (AutoML skip only).

**Source.** Strategy label is `sequential`. With blending on, personalized frames are relabeled `personalized` before the blend, so you will not see `sequential` in the table. Priority or RRF keeps it.

**Online.** No sequential `fit_partial`. Missing extra + sequential in the artifact → whole online recommend rewrite skipped. Extra present → frozen re-score for affected known users. New ids wait for `job.run()`. `[experiment]` on skips online LightFM rewrite; popular / latest still follow the assigned variant.

**Failures.**

| What you did | What happens |
| --- | --- |
| `models` includes `sequential`, no torch on job | `ConfigError`: requires torch + install hint. |
| Torch-free serve, online on, sequential in artifact | Online refresh skipped; no personalized rewrite that flush. |
| Blending enabled | `source` shows `personalized` / `blended`, not `sequential`. |
| AutoML, median distinct items below floor | Sequential excluded from candidates; INFO log. |
| AutoML, no torch | Same exclusion; hint names the extra. |
| Feature-only user, no interactions | Sequential skipped for that user; popular can still fill. |

</details>

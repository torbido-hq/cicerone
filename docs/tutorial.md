<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Tutorial: from zero to a full local run

This is a hands-on, step-by-step walkthrough of Cicerone's main features
using a handful of made-up users/items/events on local disk — no S3 bucket
or database required until the optional "database backend" section.
Commands below use Docker so the host stays clean. To run the same TOML
with pip instead, see [README Installation](../README.md#installation)
(`pip install cicerone-recommender`, then `cicerone --config …`; point
`feature_config_path` and local dataset paths at host files).
`docker-compose.yml` is for local developer convenience only — do not run
it as a production deployment. For the full configuration reference, see
the [README](../README.md); for algorithms and how strategies differ, see
[how-it-works.md](how-it-works.md); for `[events]` ingest between retrains,
see [incremental-events.md](incremental-events.md); for how the code is
structured, see [architecture.md](architecture.md).

1. [Create a sample dataset](#1-create-a-sample-dataset)
2. [Point cicerone.toml at it](#2-point-ciceronetoml-at-it)
3. [Run the job once](#3-run-the-job-once)
4. [Inspect the recommendations](#4-inspect-the-recommendations)
5. [Try different model strategies](#5-try-different-model-strategies)
6. [Weighted reciprocal rank fusion](#6-weighted-reciprocal-rank-fusion)
7. [Per-user weighted blending](#7-per-user-weighted-blending)
8. [Let AutoML pick a strategy for you](#8-let-automl-pick-a-strategy-for-you)
9. [Tune interaction weights & features](#9-tune-interaction-weights--features)
10. [Save a fitted model artifact](#10-save-a-fitted-model-artifact)
11. [Try the database backend](#11-try-the-database-backend-optional)
12. [Serve recommendations over an HTTP API](#12-serve-recommendations-over-an-http-api)
13. [Ingest incremental events](#13-ingest-incremental-events-optional)
14. [Trigger a retrain on demand](#14-trigger-a-retrain-on-demand)
15. [Check job status with the dashboard](#15-check-job-status-with-the-dashboard)
16. [Run continuously, on a schedule](#16-run-continuously-on-a-schedule)
17. [Next steps](#17-next-steps)

## 1. Create a sample dataset

Cicerone reads `events.parquet` (required) and `users.parquet`/
`items.parquet` (optional) from a directory when `storage_backend = "local"`.
Create one under `data/input/`:

```sh
mkdir -p data/input data/output
```

Build the test image once (reused for every command below) and generate the
three parquet files with a throwaway container:

```sh
docker build --target test -t cicerone-test -f docker/Dockerfile .
docker run --rm -v "$PWD/data":/data cicerone-test python -c "
import pandas as pd
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
events = pd.DataFrame([
    {'user_id': 'alice', 'item_id': 'ipa-001', 'event_type': 'purchase', 'quantity': 2, 'occurred_at': now - timedelta(days=1)},
    {'user_id': 'alice', 'item_id': 'stout-002', 'event_type': 'view', 'quantity': 1, 'occurred_at': now - timedelta(days=3)},
    {'user_id': 'bob', 'item_id': 'ipa-001', 'event_type': 'purchase', 'quantity': 1, 'occurred_at': now - timedelta(days=2)},
    {'user_id': 'bob', 'item_id': 'lager-003', 'event_type': 'purchase', 'quantity': 3, 'occurred_at': now - timedelta(days=5)},
    {'user_id': 'carol', 'item_id': 'stout-002', 'event_type': 'purchase', 'quantity': 1, 'occurred_at': now - timedelta(days=1)},
])
users = pd.DataFrame([
    {'user_id': 'alice', 'favorite_styles': ['IPA'], 'region_slug': 'north'},
    {'user_id': 'bob', 'favorite_styles': ['Lager'], 'region_slug': 'south'},
    {'user_id': 'carol', 'favorite_styles': ['Stout'], 'region_slug': 'north'},
])
items = pd.DataFrame([
    {'item_id': 'ipa-001', 'category': 'beer', 'primary_style': 'IPA', 'producer_id': 'p1', 'region_slug': 'north', 'abv_bucket': 'medium', 'published': True, 'in_stock': True, 'published_at': now - timedelta(days=30)},
    {'item_id': 'stout-002', 'category': 'beer', 'primary_style': 'Stout', 'producer_id': 'p1', 'region_slug': 'north', 'abv_bucket': 'high', 'published': True, 'in_stock': True, 'published_at': now - timedelta(days=3)},
    {'item_id': 'lager-003', 'category': 'beer', 'primary_style': 'Lager', 'producer_id': 'p2', 'region_slug': 'south', 'abv_bucket': 'low', 'published': True, 'in_stock': True, 'published_at': now - timedelta(days=1)},
])
events.to_parquet('/data/input/events.parquet')
users.to_parquet('/data/input/users.parquet')
items.to_parquet('/data/input/items.parquet')
print('sample dataset written to data/input/')
"
```

`event_type`/user/item columns here match the defaults in
`config/features.toml` — see [step 9](#9-tune-interaction-weights--features)
for how to adapt them to your own catalog.

## 2. Point `cicerone.toml` at it

Copy the shipped config so you can experiment freely without touching the
version-controlled original, and switch both `[input]`/`[output]` to the
local backend:

```sh
cp config/cicerone.toml config/cicerone.local.toml
```

Edit `config/cicerone.local.toml`'s `[input.options]`/`[output.options]` to:

```toml
[input.options]
storage_backend = "local"
path = "/data/input"

[output.options]
storage_backend = "local"
path = "/data/output"
```

You'll keep editing this same file's `[job]` section through the rest of
this tutorial.

## 3. Run the job once

```sh
docker run --rm \
  -v "$PWD/config/cicerone.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config/features.toml":/app/config/features.toml:ro \
  -v "$PWD/data":/data \
  cicerone-test cicerone --config /app/config/cicerone.toml job
```

You'll re-run this exact command after every config change below.

## 4. Inspect the recommendations

```sh
docker run --rm -v "$PWD/data":/data cicerone-test python -c "
import pandas as pd
print(pd.read_parquet('/data/output/recommendations.parquet'))
import json
print(json.dumps(json.load(open('/data/output/manifest.json')), indent=2))
"
```

You should see up to `top_k` ranked `item_id`s per user, each tagged with
the `source` strategy that produced it (`personalized` for the default
`collaborative` model, `popular_fallback` for users without enough
personalized results). With `[job.explain]` on (default), each row also
has a `reasons` JSON string: contributing sources, boost hits, and similar
history items. Serve returns that as `item.reasons`. Disable with
`[job.explain].enabled = false`. See
[how-it-works.md](how-it-works.md#why-this-item). A `manifest.json` has
run metadata (event/user/item counts, the models/weights actually used,
timestamps).

## 5. Try different model strategies

`[job].models` picks which strategies to fit and combine, in priority order
(earlier entries win ties for the same user/item pair). What each algorithm
actually is, and how they differ: [how-it-works.md](how-it-works.md). Add
this to `config/cicerone.local.toml`'s `[job]` section:

```toml
[job]
top_k = 10
models = ["collaborative", "item_based", "popular", "latest"]
```

Re-run the command from [step 3](#3-run-the-job-once) and check the
manifest's `models` field — it now lists all four. With a catalog this
small (3 items), `collaborative` alone already covers every unseen item per
user, so `item_based`/`latest` won't visibly win any ties yet; you'll see
them contribute once you turn on weighted fusion in the next step.
Available strategies:

- `collaborative`: `LightFMWrapperModel` — hybrid CF using user/item
  features for cold-start. Personalized, warm users only.
- `item_based`: `ImplicitItemKNNWrapperModel` — item-item similarity.
  Neighbor count is RecTools `model.item_based.model.K` (default 20);
  legacy `[job.item_based].k_neighbors` still works. Personalized; users
  with interactions only (feature-only warm users stay on
  collaborative/popular).
- `sequential`: RecTools `SASRecModel` (default eSASRec), `BERT4RecModel`,
  or `HSTUModel` (`architecture = "hstu"`). Opt-in;
  not in the default Docker image (`pip install 'cicerone-recommender[sequential]'`
  or `pip install -r requirements-sequential.txt`). Sequences are unique
  items by last-touch time. AutoML skips it on sparse data or a missing
  extra. Skip it in this tutorial.
- `content_fallback`: zero-interaction items via categorical feature
  similarity (opt-in: `[job.content_fallback].enabled = true`).
- `popular`: `PopularModel` — global popularity. Non-personalized, backfills
  every warm user without enough personalized results. Optional
  `[model.popular]`.
- `latest`: `PopularModel` restricted to the last two weeks of interactions —
  trending items. Non-personalized, same backfill role as `popular`.
  Optional `[model.latest]` (`period = { days = 14 }`).

Optional RecTools hyperparameters (omit to keep built-in defaults):

```toml
[model.item_based]
cls = "ImplicitItemKNNWrapperModel"
[model.item_based.model]
cls = "TFIDFRecommender"
K = 20
```

## 6. Weighted reciprocal rank fusion

Instead of the default priority order (earlier `models` entries always win
ties), you can blend every enabled strategy's ranks by weight. Replace the
`[job]` section with:

```toml
[job]
top_k = 10
models = ["collaborative", "item_based", "popular", "latest"]

[job.model_weights]
collaborative = 1.0
item_based = 0.6
popular = 0.3
latest = 0.4
```

Re-run and re-inspect: fused (user, item) pairs now get a combined `source`
label (e.g. `"popular_fallback+latest"`) joined in `models`' configured
order, and `score` is the summed `weight / (rrf_k + rank)` across every
strategy that recommended that pair. Tune the fusion constant with
`rrf_k = 60` (the default; must be a positive number, placed *above*
`[job.model_weights]` — see the TOML gotcha note in `config/cicerone.toml`)
if you want top ranks to matter more or less relative to lower ones.

## 7. Per-user weighted blending

Section 6's RRF uses **fixed** per-strategy weights for every user. Optional
`[blending]` in `config/features.toml` instead grows the personalized weight
with each user's interaction count (sigmoid or linear), and splits the
remainder between `popular` and date-based `latest`:

```toml
[blending]
enabled = true
curve = "linear"
saturate_at = 5.0
popular_share = 0.7
latest_date_columns = ["published_at", "created_at", "occurred_at"]
```

Copy `config/features.toml` next to your local job config (or edit the
mounted one), enable the block above, keep `models` including
`collaborative` (blending auto-adds `popular` if missing), and re-run
[step 3](#3-run-the-job-once). Inspect `data/output/recommendations.parquet`:

- cold / low-history users lean on `popular_fallback` / `latest`
- richer users lean on `personalized`
- an item is `source = "blended"` only when more than one source contributed it
- a sentinel `__cold_start__` user is written (global availability allowlist)
  for serve-mode fallback

The blend curve uses each user's **distinct (user, item) count** after
dataset aggregation (not raw event rows). If `items` has no usable date
column among `latest_date_columns`, `latest` is skipped and its weight
moves to `popular` (check the job log). Independent of
`[job.model_weights]` RRF — enable one or the other as the primary
combiner (blending wins and logs a warning if both are set). See the
annotated `[blending]` block in `config/features.toml` and the README's
Interaction weights section.

## 8. Let AutoML pick a strategy for you

Instead of hand-picking `models`/`model_weights`, AutoML backtests a set of
candidate configs over time-based folds of your own event history and picks
the best one automatically, every run. Our sample dataset only spans a few
days, so use small `n_splits`/`test_days` values to get at least one valid
fold (production configs typically use the defaults, `n_splits = 2` /
`test_days = 14`, over months of real history). By default AutoML tries
every strategy alone, the default priority combo, and one all-strategy
weighted-fusion blend. On a catalog this tiny that mostly wastes folds:
`item_based` is left with a couple of training interactions and no usable
neighbours, so its candidates score zero on every metric while the
non-personalized ones actually rank something. It does not fail — test users
with no interactions in that fold are dropped before `recommend()` rather
than raising — it just has nothing to say. Override the search space with
`[[job.automl.candidates]]` to a couple of `item_based`-free options instead:

```toml
[job]
top_k = 10

[job.automl]
enabled = true
n_splits = 1
test_days = 2
primary_metric = "MAP"

[[job.automl.candidates]]
models = ["collaborative", "popular"]
[[job.automl.candidates]]
models = ["collaborative", "popular"]
[job.automl.candidates.weights]
collaborative = 1.0
popular = 0.3
```

Re-run the job — the log output (`docker logs`, or just watch stdout) will
show a line per candidate like `AutoML candidate 'collaborative+popular'
scored {...} over 1 fold(s)`, followed by `AutoML selected '...' (metrics=...,
over 1 fold(s))`. In the manifest, `automl_enabled` is only the config flag
and `automl_metrics` is how the winner scored; what was actually picked lands
in `models` / `model_weights` / `rrf_k`. See `config/cicerone.toml` for the
full annotated example of the default search space (safe to use as-is once
your dataset has enough history for every strategy to see every backtested
user).

To watch collaborative or sequential training (separate from AutoML's fold
scores), set `log_epoch_metrics = true` under `[job]` — you'll see per-epoch
Precision@K/Recall@K lines, plus a WARN if metrics regress or plateau. Off
by default; details in the
[README model strategies](../README.md#model-strategies) section.

## 9. Tune interaction weights & features

`config/features.toml` (mounted read-only, already used by every run above)
controls signal weighting and cold-start features without touching code:

- `[event_weights]`: base weight per `event_type` before time-decay (e.g.
  `purchase = 4.0`, `view = 0.3`); an `event_type` present in `events.parquet`
  but missing here is dropped, with a warning.
- `quantity_scaled_events`: event types whose weight also scales by
  `log1p(quantity)` (default: `["purchase"]` — buying 3 matters more than
  buying 1).
- `[event_caps]`: caps how many times a given `event_type` counts per (user,
  item) pair before decay, so noisy high-frequency signals (e.g. `view`)
  can't drown out rarer ones.
- `[[user_features]]` / `[[item_features]]`: which `users.parquet`/
  `items.parquet` columns feed the model, and whether each is
  `"categorical"` (single value) or `"list"` (multi-valued, e.g. our sample
  data's `favorite_styles`).
- `item_availability_filters`: boolean `items.parquet` columns that must all
  be `true` for an item to ever be recommended (default:
  `["published", "in_stock"]`).
- `[[eligibility]]` / `[[boost]]`: optional hard per-user item filters
  (e.g. nationality ∈ `available_countries`) and soft commercial re-rank
  boosts (e.g. paying producers). Eligibility groups users into cohorts that
  share an allowed-item set; boosts over-fetch candidates
  (`boost_overfetch_factor` × `top_k`, default 3) before re-ranking so a
  boosted item just outside the raw top-K can still make the cut. See the
  annotated recipes in `config/features.toml`.
- `[blending]`: optional per-user mix of personalized / popular / latest
  (see [step 7](#7-per-user-weighted-blending)).

Try lowering `view`'s weight to `0.1` or raising `purchase` to `6.0` in your
own copy of `config/features.toml`, re-run, and compare the output —
`half_life_days` in `[job]` (default 90) additionally decays all of this by
recency.

## 10. Save a fitted model artifact

By default the batch job discards fitted strategy weights after writing
recommendations. With `[job].save_model_artifact = true` it also writes a
versioned portable artifact you can reload later without re-training:

```toml
[job]
save_model_artifact = true
# ... keep the rest of your local job settings ...
```

Re-run the job from [step 3](#3-run-the-job-once). Alongside
`recommendations.parquet` / `manifest.json` you should now see
`model.artifact` in `data/output/`. The manifest gains
`artifact_written = true` and `artifact_schema_version = 3`.

Load it and recommend without calling `job.run` again:

```sh
docker run --rm -v "$PWD/data":/data cicerone-test python -c "
from cicerone.artifact import load_artifact, recommend_from_artifact
artifact = load_artifact('/data/output/model.artifact')
print('models:', artifact.models, 'schema:', artifact.schema_version)
print(recommend_from_artifact(artifact, ['alice', 'bob'], top_k=3))
"
```

Serve mode never loads this file on `GET /recommendations` — it still
only reads precomputed recommendation rows. With `[events.online]` the
events worker loads it between retrains. Artifacts are for offline reload /
that write-through path; only load ones your own batch job wrote (pickle is
not safe on untrusted bytes). See the README's
[Model artifacts](../README.md#model-artifacts) section.

## 11. Try the database backend (optional)

Input/output don't have to be static files — `kind = "db"` reads/writes a
relational database via SQLAlchemy instead (independently for input and
output). `docker-compose.yml` already includes a Postgres 16 service —
start just that:

```sh
docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres
```

(Credentials, DB names, and host port live in
[`docker/postgres/defaults.env`](../docker/postgres/defaults.env) — see
[CONTRIBUTING.md](../CONTRIBUTING.md#local-postgres-defaults). From the host
use `localhost`; from another compose container use `postgres`. For
pytest, use the pytest DB from that file and
[CONTRIBUTING.md](../CONTRIBUTING.md) for `TEST_DATABASE_URL`. Opt-in via
`--profile db` so a plain `docker compose up` does not require Postgres.)

Load the sample dataset into `events`/`users`/`items` tables:

```sh
docker run --rm --network host -v "$PWD/data":/data cicerone-test python -c "
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone')
for name in ('events', 'users', 'items'):
    df = pd.read_parquet(f'/data/input/{name}.parquet')
    for col in df.columns:
        # list-typed columns (e.g. favorite_styles) round-trip through parquet
        # as numpy arrays, which psycopg can't adapt directly — plain lists work.
        if df[col].apply(lambda v: hasattr(v, 'tolist')).any():
            df[col] = df[col].apply(lambda v: v.tolist() if hasattr(v, 'tolist') else v)
    df.to_sql(name, engine, if_exists='replace', index=False)
print('sample dataset loaded into Postgres')
"
```

Switch `config/cicerone.local.toml`'s `[input]`/`[output]` to:

```toml
[input]
kind = "db"

[input.options]
database_url = "postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone"

[output]
kind = "db"

[output.options]
database_url = "postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone"
# optional when [job].save_model_artifact = true:
# model_artifact_table = "model_artifacts"
```

Re-run the command from [step 3](#3-run-the-job-once) with `--network host`
added, then check the results straight from Postgres:

```sh
docker run --rm --network host cicerone-test python -c "
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone')
print(pd.read_sql('SELECT * FROM recommendations', engine))
print(pd.read_sql('SELECT status, models, artifact_written FROM recommendation_runs', engine))
"
```

A fresh `recommendations` table includes `reasons`. An existing table from
an older Cicerone needs `ALTER TABLE recommendations ADD COLUMN reasons TEXT`
before the extra column will persist (`pandas` `to_sql(append)` will not
add it).

Input and output can be mixed (e.g. read from Postgres, write to S3, or
vice versa), and raw SQL overrides (`events_query`/`users_query`/
`items_query`) let you read straight from an existing application schema
instead of requiring materialized tables — see the README's
[Configuration section](../README.md#configuration-configciceronetoml).

Clean up when you're done:

```sh
docker compose --profile db down   # or: docker compose --profile db stop postgres
```

## 12. Serve recommendations over an HTTP API

Everything so far has run the batch job directly. `[job].mode = "serve"`
switches to a separate, lightweight **read** API over whatever the batch job
last wrote to `[output]` — `GET` never imports lightfm/implicit/torch, never
trains, and never loads a model artifact. With `[events.online]` the events
worker does load the artifact for write-through. (It does import `rectools`
itself, for `Columns`, so that one stays in a serve-only image.) Reuse the
local `data/output/` from
[step 3](#3-run-the-job-once):

```sh
cp config/cicerone.serve.toml config/cicerone.serve.local.toml
```

In `config/cicerone.serve.local.toml`, **replace** the `[output.options]` and
`[serve]` tables the copy already ships with — do not append these, because a
second `[output.options]` or `[serve]` header is a duplicate table and TOML
parsing fails. Replacing `[output.options]` also drops the `${OUTPUT_S3_*}`
placeholders, which would otherwise resolve against environment variables you
have not set. Everything omitted from `[serve]` keeps its default (`host`
`0.0.0.0`, `port` 8000, `default_k` 10, `refresh_interval_seconds` 60):

```toml
[output.options]
storage_backend = "local"
path = "/data/output"

[serve]
auth_token = "tutorial-token"
category_column = "category"
```

Start it in the background and query a user's recommendations. Keep the
token in a shell variable rather than typing it directly into the `curl`
command (and out of your shell history):

```sh
docker run --rm -d --name cicerone-tutorial-serve -p 8000:8000 \
  -v "$PWD/config/cicerone.serve.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config/features.toml":/app/config/features.toml:ro \
  -v "$PWD/data":/data \
  cicerone-test cicerone --config /app/config/cicerone.toml serve

read -s -p "Serve auth token: " SERVE_TOKEN && echo
curl -s -H "Authorization: Bearer $SERVE_TOKEN" \
  "http://localhost:8000/recommendations/alice?limit=5" | python -m json.tool
```

The response is an object (not a bare list):

```json
{
  "generated_at": "2026-08-04T03:00:00+00:00",
  "user_id": "alice",
  "fallback": false,
  "items": [
    {
      "item_id": "lager-003",
      "rank": 1,
      "score": 0.91,
      "source": "blended",
      "reasons": {
        "sources": [
          {"label": "personalized", "rank": 4, "weight": 0.72, "contribution": 0.0113}
        ],
        "boosts": [],
        "similar_items": [{"item_id": "lager-001", "score": 0.67}],
        "matched_attributes": [{"column": "style", "value": "lager"}]
      }
    }
  ]
}
```

Try a few filters (same auth header):

```sh
# Category filter (column from [serve].category_column, default "category")
curl -s -H "Authorization: Bearer $SERVE_TOKEN" \
  "http://localhost:8000/recommendations/alice?limit=5&category=beer"

# Unknown user → cold-start fallback (popular_fallback/latest/blended);
# 404 only when the table has no sentinel and no popular/latest rows
curl -s -H "Authorization: Bearer $SERVE_TOKEN" \
  "http://localhost:8000/recommendations/nobody?limit=5"

# Prometheus metrics (no bearer token; optional X-Metrics-Token if configured)
curl -s "http://localhost:8000/metrics" | head
```

`GET /metrics` is enabled by default (`[serve].metrics_enabled = true`). It
does not require the recommendation bearer token. Set `[serve].metrics_token`
and send `X-Metrics-Token: …` if you want a separate scrape secret; when
empty, treat the endpoint as network-boundary protected.

Optional `POST /events` (webhook incremental ingest) is off until you set
`[events]` — the next step walks through that. OpenAPI (Swagger UI) is at
`http://localhost:8000/docs`; ReDoc (with language `x-codeSamples`,
including Ruby) is at `http://localhost:8000/redoc`. The JSON schema is
`/openapi.json` (`/metrics` is omitted from the schema). For a runnable
client, use the thin package helper or the snippets under
[`examples/serve/`](../examples/serve/):

```sh
docker run --rm --network host -e PYTHONPATH=/app/src \
  -e CICERONE_SERVE_URL=http://localhost:8000 \
  -e CICERONE_SERVE_TOKEN="$SERVE_TOKEN" \
  -v "$PWD":/app -w /app cicerone-test \
  python examples/serve/python_client.py
```

From a PyPI install, drop `PYTHONPATH` and run
`python examples/serve/python_client.py` on the host with the same env vars.

For a `dataset` output (as here), the recommendations file and optional
`items_snapshot.parquet` are cached in memory and reloaded every
`[serve].refresh_interval_seconds` (default 60s) — re-run
[step 3](#3-run-the-job-once) and query again after that interval to see
updated results without restarting the container. For a `db` output, every
request queries the table directly instead. Clean up when you're done:

```sh
docker stop cicerone-tutorial-serve
```

## 13. Ingest incremental events (optional)

Enable webhook `[events]` on serve and POST one interaction. Other backends,
HA, and optional `[events.online]` LightFM `fit_partial`:
[incremental-events.md](incremental-events.md).

Add this to `config/cicerone.serve.local.toml` (keep the `[output]` /
`[serve]` edits from [step 12](#12-serve-recommendations-over-an-http-api)):

```toml
[events]
enabled = true
kind = "webhook"

[events.incremental]
batch_size = 1
batch_window_seconds = 2
```

Start serve again, POST one event, wait for the micro-batch window, then
read Alice's rows (and `cicerone_events_*` on `/metrics`):

```sh
docker run --rm -d --name cicerone-tutorial-serve -p 8000:8000 \
  -v "$PWD/config/cicerone.serve.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config/features.toml":/app/config/features.toml:ro \
  -v "$PWD/data":/data \
  cicerone-test cicerone --config /app/config/cicerone.toml serve

curl -sS -X POST -H "Authorization: Bearer $SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","item_id":"ipa-001","event_type":"purchase","quantity":1,"occurred_at":"2026-08-19T12:00:00Z"}' \
  http://localhost:8000/events

sleep 3
curl -s -H "Authorization: Bearer $SERVE_TOKEN" \
  "http://localhost:8000/recommendations/alice?limit=5" | python -m json.tool
```

`occurred_at` must include a timezone (`Z` / offset) or be Unix epoch
seconds. A full backlog returns HTTP 429. Stop serve when you are done:

```sh
docker stop cicerone-tutorial-serve
```

## 14. Trigger a retrain on demand

`[job.trigger]` adds an event-driven retrain trigger *in addition to* the
cron schedule, running inside the same `cicerone.scheduler` process (not a
separate container). Add this to `config/cicerone.local.toml`:

```toml
[job.trigger]
enabled = true
auth_token = "tutorial-token"
port = 8080
```

Start the scheduler in the background (this also runs the batch job
immediately, then re-enters the cron loop) and fire the webhook. As above,
keep the token in a shell variable instead of inlining it:

```sh
docker run --rm -d --name cicerone-tutorial-scheduler -p 8080:8080 \
  -v "$PWD/config/cicerone.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config/features.toml":/app/config/features.toml:ro \
  -v "$PWD/data":/data \
  cicerone-test cicerone --config /app/config/cicerone.toml start

read -s -p "Trigger auth token: " TRIGGER_TOKEN && echo
curl -X POST -H "Authorization: Bearer $TRIGGER_TOKEN" http://localhost:8080/trigger/retrain
```

A trigger fired while a run is already in flight, or within
`[job.trigger].debounce_seconds` (default 60) of the last one, is skipped
rather than queued — check `docker logs cicerone-tutorial-scheduler` to see
it happen if you call the webhook twice in a row. Single-instance locking is
the default; see the README Event-driven retrain trigger section for optional
`postgres` / `redis` backends when running multiple scheduler replicas. The
written manifest also records `triggered_by` (`"cron"`, `"webhook"`, or
`"s3-poll"` if `poll_input_bucket = true`) and `lock_backend`. Clean up when
you're done:

```sh
docker stop cicerone-tutorial-scheduler
```

## 15. Check job status with the dashboard

`cicerone.dashboard` is a small, standalone status page over job run
history — `cicerone dashboard`, independent of `[job].mode`.
Reuse the same local `data/output/`:

```sh
cp config/cicerone.dashboard.toml config/cicerone.dashboard.local.toml
```

Edit `config/cicerone.dashboard.local.toml`'s `[output.options]` to point at
the same local directory, and its `[job].cron_schedule` to match
`config/cicerone.local.toml`'s:

```toml
[output.options]
storage_backend = "local"
path = "/data/output"

[job]
cron_schedule = "0 3 * * *"
```

Add a login user (prompts for a password interactively, note `-it`):

```sh
mkdir -p config
docker run --rm -it \
  -v "$PWD/config":/app/config \
  cicerone-test cicerone users --users-path /app/config/dashboard_users.toml add tutorial-user
```

Then start the dashboard and open `http://localhost:8090/dashboard` in a
browser (log in with the user just created), or
`http://localhost:8090/dashboard?user_id=alice` to fill the inspector on
load. HTTP Basic credentials stay in the browser until it forgets them
for this origin — closing the tab is not a sign-out; use a private window
or clear saved passwords. The **Inspect user** form shows that `user_id`'s
recent `[input]` events beside current top-K (including cold-start
fallback). Shared item ids are highlighted. When `reasons` is present, a
Why column summarizes source labels and the top similar history item.
Recs row count is `min(job.top_k, dashboard.lookup_k)` (`lookup_k`
defaults to 20, so 10 with the default `top_k`); events use
`dashboard.lookup_events` (default 20). User attributes show only when
`dashboard.lookup_user_attrs` lists columns. The
incremental-events panel is gated on the `[events]` block of the config the
**dashboard** was started with, not the one from
[step 13](#13-ingest-incremental-events-optional): copy `enabled = true` into
`config/cicerone.dashboard.local.toml` too, or the panel never renders. Once
it does render, it shows the last flush from the latest manifest, or "no
incremental flushes recorded" until one lands (dataset `manifest.json` is
overwritten by the next full job run).

```sh
docker run --rm -d --name cicerone-tutorial-dashboard -p 8090:8090 \
  -v "$PWD/config/cicerone.dashboard.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config":/app/config \
  -v "$PWD/data":/data \
  cicerone-test cicerone --config /app/config/cicerone.toml dashboard
```

With our `dataset` output, only the latest run is ever shown (its
`manifest.json` is overwritten every run, not appended) — switch to a `db`
output (see [step 11](#11-try-the-database-backend-optional)) to see a real
run history table instead. Clean up when you're done:

```sh
docker stop cicerone-tutorial-dashboard
```

## 16. Run continuously, on a schedule

Everything above ran the job once via `docker run`. In practice, Cicerone
runs continuously as a long-lived container: `docker-compose.yml` runs the
job immediately on boot, then again on `[job].cron_schedule` (a 5-field cron
expression evaluated in UTC; default: every night at 03:00). Use that compose
file to exercise recommender + serve + dashboard locally — it is for
developer convenience, not production. Point it at your real input/output
backend (S3-compatible storage or a database — see
`.env.example`/`config/cicerone.toml`) and run:

```sh
cp .env.example .env   # fill in the secrets your cicerone.toml references
docker compose up --build
```

## 17. Next steps

- Swap in your own data, following the [data contract](../README.md#data-contract).
- Install from PyPI (`pip install cicerone-recommender`) and run
  `cicerone --config ./config/cicerone.toml start` — set
  `feature_config_path` (and dashboard `users_path`) to host files; see
  [README Installation](../README.md#installation).
- Read [how-it-works.md](how-it-works.md) for algorithms,
  the [model strategies](../README.md#model-strategies),
  [AutoML](../README.md#automl), and
  [model artifacts](../README.md#model-artifacts) reference for every
  tunable knob covered above.
- Point input/output at S3-compatible object storage (R2, AWS S3, MinIO) —
  see the README's [Configuration](../README.md#configuration-configciceronetoml)
  section.
- Sticky A/B tests of ranking recipes: [experiments.md](experiments.md).
- Production incremental ingest (db / s3 / Redis Streams, HA, online
  LightFM): [incremental-events.md](incremental-events.md).
- Run the test suite (`docker compose -f docker-compose.ci.yml up --build
  --abort-on-container-exit --exit-code-from test test`) if you're contributing
  code — see [CONTRIBUTING.md](../CONTRIBUTING.md). That suite includes a
  system-style Postgres end-to-end check (`tests/test_system_db.py`).

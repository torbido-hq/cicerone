# Tutorial: from zero to a full local run

This is a hands-on, step-by-step walkthrough of Cicerone's main features
using a handful of made-up users/items/events on local disk — no S3 bucket
or database required until the optional "database backend" section.
Everything runs in Docker (nothing gets installed on the host). For the
full configuration reference, see the [README](../README.md); for how the
code is structured, see [architecture.md](architecture.md).

1. [Create a sample dataset](#1-create-a-sample-dataset)
2. [Point cicerone.toml at it](#2-point-ciceronetoml-at-it)
3. [Run the job once](#3-run-the-job-once)
4. [Inspect the recommendations](#4-inspect-the-recommendations)
5. [Try different model strategies](#5-try-different-model-strategies)
6. [Weighted reciprocal rank fusion](#6-weighted-reciprocal-rank-fusion)
7. [Let AutoML pick a strategy for you](#7-let-automl-pick-a-strategy-for-you)
8. [Tune interaction weights & features](#8-tune-interaction-weights--features)
9. [Try the database backend](#9-try-the-database-backend-optional)
10. [Run continuously, on a schedule](#10-run-continuously-on-a-schedule)
11. [Next steps](#11-next-steps)

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
    {'item_id': 'ipa-001', 'category': 'beer', 'primary_style': 'IPA', 'producer_id': 'p1', 'region_slug': 'north', 'abv_bucket': 'medium', 'published': True, 'in_stock': True},
    {'item_id': 'stout-002', 'category': 'beer', 'primary_style': 'Stout', 'producer_id': 'p1', 'region_slug': 'north', 'abv_bucket': 'high', 'published': True, 'in_stock': True},
    {'item_id': 'lager-003', 'category': 'beer', 'primary_style': 'Lager', 'producer_id': 'p2', 'region_slug': 'south', 'abv_bucket': 'low', 'published': True, 'in_stock': True},
])
events.to_parquet('/data/input/events.parquet')
users.to_parquet('/data/input/users.parquet')
items.to_parquet('/data/input/items.parquet')
print('sample dataset written to data/input/')
"
```

`event_type`/user/item columns here match the defaults in
`config/features.toml` — see [step 8](#8-tune-interaction-weights--features)
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
  -e CICERONE_CONFIG_PATH=/app/config/cicerone.toml \
  cicerone-test python -m cicerone.job
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
personalized results), plus a `manifest.json` with run metadata (event/user/
item counts, the models/weights actually used, timestamps).

## 5. Try different model strategies

`[job].models` picks which of the four built-in strategies to fit and
combine, in priority order (earlier entries win ties for the same
user/item pair). Add this to `config/cicerone.local.toml`'s `[job]` section:

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
  Personalized, warm users only.
- `popular`: `PopularModel` — global popularity. Non-personalized, backfills
  every warm user without enough personalized results.
- `latest`: `PopularModel` restricted to the last two weeks of interactions —
  trending items. Non-personalized, same backfill role as `popular`.

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
`[job.model_weights]` — see the TOML gotcha note in the README) if you want
top ranks to matter more or less relative to lower ones.

## 7. Let AutoML pick a strategy for you

Instead of hand-picking `models`/`model_weights`, AutoML backtests a set of
candidate configs over time-based folds of your own event history and picks
the best one automatically, every run. Our sample dataset only spans a few
days, so use small `n_splits`/`test_days` values to get at least one valid
fold (production configs typically use the defaults, `n_splits = 2` /
`test_days = 14`, over months of real history). By default AutoML tries
every strategy alone, the default priority combo, and one all-strategy
weighted-fusion blend — with a catalog this tiny, `item_based`'s candidates
can end up being asked to recommend for a user it never saw interact in
that fold, which `ImplicitItemKNNWrapperModel` doesn't support, so override
the search space with `[[job.automl.candidates]]` to a couple of
`item_based`-free options instead:

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
over 1 fold(s))`, and the manifest's `automl_enabled`/`automl_metrics` fields
record what was picked and how it scored. See `config/cicerone.toml` for the
full annotated example of the default search space (safe to use as-is once
your dataset has enough history for every strategy to see every backtested
user).

## 8. Tune interaction weights & features

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

Try lowering `view`'s weight to `0.1` or raising `purchase` to `6.0` in your
own copy of `config/features.toml`, re-run, and compare the output —
`half_life_days` in `[job]` (default 90) additionally decays all of this by
recency.

## 9. Try the database backend (optional)

Input/output don't have to be static files — `kind = "db"` reads/writes a
relational database via SQLAlchemy instead (independently for input and
output). Start a throwaway local Postgres:

```sh
docker run --rm -d --name cicerone-tutorial-db -p 5432:5432 \
  -e POSTGRES_USER=cicerone -e POSTGRES_PASSWORD=cicerone -e POSTGRES_DB=cicerone \
  postgres:16-alpine
```

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
```

Re-run the command from [step 3](#3-run-the-job-once) with `--network host`
added, then check the results straight from Postgres:

```sh
docker run --rm --network host cicerone-test python -c "
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone')
print(pd.read_sql('SELECT * FROM recommendations', engine))
"
```

Input and output can be mixed (e.g. read from Postgres, write to S3, or
vice versa), and raw SQL overrides (`events_query`/`users_query`/
`items_query`) let you read straight from an existing application schema
instead of requiring materialized tables — see the README's
[Configuration section](../README.md#configuration-config-cicerone-toml).

Clean up the throwaway database when you're done:

```sh
docker stop cicerone-tutorial-db
```

## 10. Run continuously, on a schedule

Everything above ran the job once via `docker run`. In practice, Cicerone
runs continuously as a long-lived container: `docker-compose.yml` runs the
job immediately on boot, then again on `[job].cron_schedule` (a 5-field cron
expression evaluated in UTC; default: every night at 03:00). Point
`docker-compose.yml` at your real input/output backend (S3-compatible
storage or a database — see `.env.example`/`config/cicerone.toml`) and run:

```sh
cp .env.example .env   # fill in the secrets your cicerone.toml references
docker compose up --build
```

## 11. Next steps

- Swap in your own data, following the [data contract](../README.md#data-contract).
- Read the full [model strategies](../README.md#model-strategies) and
  [AutoML](../README.md#automl) reference for every tunable knob covered
  above.
- Point input/output at S3-compatible object storage (R2, AWS S3, MinIO) —
  see the README's [Configuration](../README.md#configuration-config-cicerone-toml)
  section.
- Run the test suite (`docker compose -f docker-compose.ci.yml up --build
  --abort-on-container-exit --exit-code-from test`) if you're contributing
  code — see [CONTRIBUTING.md](../CONTRIBUTING.md).

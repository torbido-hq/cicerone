# Tutorial: your first run with local sample data

This walks through running Cicerone end-to-end on your own machine, with a
handful of made-up users/items/events on local disk — no S3 bucket or
database required. For the full configuration reference, see the
[README](../README.md); for how the code is structured, see
[architecture.md](architecture.md).

## 1. Create a sample dataset

Cicerone reads `events.parquet` (required) and `users.parquet`/
`items.parquet` (optional) from a directory when `storage_backend = "local"`.
Create one under `data/input/`:

```sh
mkdir -p data/input data/output
```

Generate the three parquet files with a throwaway container (everything
runs in Docker — nothing gets installed on the host):

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

## 2. Point `cicerone.toml` at it

Copy the shipped config and switch both `[input]`/`[output]` to the local
backend:

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

## 3. Run the job once

```sh
docker run --rm \
  -v "$PWD/config/cicerone.local.toml":/app/config/cicerone.toml:ro \
  -v "$PWD/config/features.toml":/app/config/features.toml:ro \
  -v "$PWD/data":/data \
  -e CICERONE_CONFIG_PATH=/app/config/cicerone.toml \
  cicerone-test python -m cicerone.job
```

## 4. Inspect the recommendations

```sh
docker run --rm -v "$PWD/data":/data cicerone-test python -c "
import pandas as pd
print(pd.read_parquet('/data/output/recommendations.parquet'))
print(pd.read_parquet('/data/output/manifest.parquet'))
"
```

You should see up to `top_k` ranked `item_id`s per user, each tagged with
the `source` strategy that produced it (`personalized` for the default
`collaborative` model, `popular_fallback` for users without enough
personalized results).

## Next steps

- Swap in your own data, following the [data contract](../README.md#data-contract).
- Try enabling more strategies or weighted fusion (`[job.model_weights]` in
  `cicerone.toml`) — see [README.md#model-strategies](../README.md#model-strategies).
- Let AutoML pick the best combination for you instead — see
  [README.md#automl](../README.md#automl).
- Point `docker-compose.yml` at a real S3-compatible bucket or database and
  run it continuously (`docker compose up --build`) instead of one-off
  `docker run` calls.

# cicerone

Low-volume batch job that suggests drinks for **torbido**'s users. No API:
it reads the input, trains a hybrid [rectools](https://github.com/MobileTeleSystems/RecTools)
+ LightFM model, and writes out the recommendations. Everything runs in
Docker (Python 3.11 only lives inside the image, nothing to install on the
host).

## Flow

```
input source (R2/S3/local dataset, or a Postgres DB)
                                        |
                                        v
                              cicerone (container "recommender")
                                 1. reads events/users/items
                                 2. weighs interactions (see below,
                                    config/features.yml)
                                 3. trains LightFMWrapperModel (rectools)
                                 4. generates top-K per user + popularity fallback
                                        |
                                        v
                     output destination (R2/S3/local dataset, or a Postgres DB)
```

Scheduling is handled in-process (`croniter`, no system cron): it runs once
at boot, then again on `CRON_SCHEDULE` (default: every night at 03:00 UTC).

## I/O configuration (`INPUT_*` / `OUTPUT_*`)

Input and output are configured **independently** of each other. Each side
can be:

- **`KIND=dataset`**: static parquet files, on S3-compatible object storage
  (R2, AWS S3, MinIO — `STORAGE_BACKEND=s3`) or on a mounted local disk
  (`STORAGE_BACKEND=local`, handy for tests or manual import/export).
- **`KIND=db`**: a Postgres table/query via SQLAlchemy (`DATABASE_URL`),
  with the option to override the read queries (`*_QUERY`) to read directly
  from torbido's own schema instead of requiring materialized
  `events`/`users`/`items` tables.

The two sides can be freely mixed, e.g. read from a Postgres replica and
write recommendations to R2, or vice versa. See [.env.example](.env.example)
for the full list of variables.

## Data contract (`INPUT_*`)

`events` (required):

| column      | type      | notes                                                         |
|-------------|-----------|----------------------------------------------------------------|
| user_id     | str       | torbido user UUID                                               |
| item_id     | str       | torbido product UUID                                            |
| event_type  | str       | see `config/features.yml` → `event_weights`                     |
| quantity    | int       | optional, used for the types listed in `quantity_scaled_events` |
| occurred_at | datetime  | UTC                                                              |

Recommended mapping from torbido's models to `event_type` (customizable in
`config/features.yml`):
- `order_items` (via `orders` with a completed status) → `purchase` (quantity = order_items.quantity)
- `reviews` with `feeling` in `ecstatic/delighted/satisfied` → `review_positive`
- `reviews` with `feeling` in `disappointed/awful` → `review_negative`
- `user_saved_products` → `saved`
- `analytics_events` with `event_type: cart_add` → `cart_add`
- `analytics_events` with `event_type: product_view` → `view`

`users` (optional, enables user features for cold-start): columns are
configurable in `config/features.yml` → `user_features` (default:
`favorite_styles` as a list, `region_slug` as categorical).

`items` (optional, enables item features + the availability filter): columns
are configurable in `config/features.yml` → `item_features` (default:
`category`, `primary_style`, `producer_id`, `region_slug`, `abv_bucket`). The
availability filter (`item_availability_filters`, default `published` +
`in_stock`) always excludes unavailable products from the recommendations.

## Output (`OUTPUT_*`)

`recommendations`: `user_id, item_id, rank, score, source`
(`source` = `personalized` or `popular_fallback`).

`manifest`: metadata about the latest run (counts, timestamps) for monitoring.

## Interaction weights & cold-start

All weighting logic is configurable without rebuilding the image via
`config/features.yml` (mounted as a volume, see `docker-compose.yml`):
`event_weights`, `quantity_scaled_events`, `event_caps`, `user_features`,
`item_features`, `item_availability_filters`. Exponential decay with a
configurable half-life (`INTERACTION_HALF_LIFE_DAYS`, default 90 days) gives
more weight to recent activity. Users without enough interactions still get
a fallback list from `PopularModel` (rectools), still honoring the
availability filter.

## Usage

```sh
cp .env.example .env   # pick INPUT_KIND/OUTPUT_KIND and set credentials
docker compose up --build
```

## Tests & CI

```sh
docker compose -f docker-compose.ci.yml up --build --abort-on-container-exit --exit-code-from test
```

Runs the whole pytest suite (with an ephemeral Postgres for the `db` backend
tests) inside Docker — nothing to install on the host. The minimum required
coverage is 95% (`pyproject.toml`, `[tool.coverage.report].fail_under`) and
is enforced on every PR by `.github/workflows/ci.yml`.

## Security

- Credentials (S3/DB) should be scoped to the bare minimum (read on the
  input side, write on the output side, no delete/admin permissions).
- No personal data other than `user_id` (a UUID) is ever read or written.
- No ports exposed: the container accepts no inbound connections.
- Credentials only ever live in environment variables (`.env`, not committed).

## License

[Beerware](LICENSE) — if we meet someday and you find this useful, buy me a
beer (or one on Torbido).



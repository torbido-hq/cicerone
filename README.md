# Cicerone

[![CI](https://github.com/torbido-hq/cicerone/actions/workflows/ci.yml/badge.svg)](https://github.com/torbido-hq/cicerone/actions/workflows/ci.yml)
[![CodeQL](https://github.com/torbido-hq/cicerone/actions/workflows/codeql.yml/badge.svg)](https://github.com/torbido-hq/cicerone/actions/workflows/codeql.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Beerware](https://img.shields.io/badge/license-Beerware%20🍺-f28e1c.svg)](LICENSE)

A generic, self-hosted batch recommender system. No API, no cache: it reads
your interaction data, trains a hybrid [rectools](https://github.com/MobileTeleSystems/RecTools)
+ LightFM model, and writes out top-K recommendations per user. Everything
runs in Docker (Python 3.11 only lives inside the image, nothing to install
on the host).

Cicerone isn't tied to any particular product, shop, or domain — it works
for any catalog of "users" and "items" with interaction events (purchases,
views, reviews, ...): drinks, books, courses, tracks, you name it. Input and
output are pluggable and configured through a single TOML file, so wiring it
up to your own data doesn't require touching any code.

> **Why "Cicerone"?** In the world of beer, a [Cicerone](https://www.cicerone.org)
> is a certified expert on beer's history, styles, ingredients, brewing, and
> — most importantly — what to pair or recommend for a given taste. Think of
> it as the beer world's equivalent of a wine sommelier. It felt like a
> fitting name for a project whose whole job is recommending the right drink
> to the right person, even though the underlying engine works just as well
> for any other kind of product catalog.

## Flow

```
input source (S3-compatible/local dataset, or a database)
                                        |
                                        v
                              cicerone (container "recommender")
                                 1. reads events/users/items
                                 2. weighs interactions (see below,
                                    config/features.toml)
                                 3. trains LightFMWrapperModel (rectools)
                                 4. generates top-K per user + popularity fallback
                                        |
                                        v
                     output destination (S3-compatible/local dataset, or a database)
```

Scheduling is handled in-process (`croniter`, no system cron): it runs once
at boot, then again on `[job].cron_schedule` in `config/cicerone.toml`
(default: every night at 03:00 UTC).

## Configuration (`config/cicerone.toml`)

All structural configuration — which backend to use for input/output,
bucket/table names, scheduling, tuning — lives in one version-controlled
TOML file, `config/cicerone.toml` (mounted read-only, see
`docker-compose.yml`; override the path with `CICERONE_CONFIG_PATH`).
Secrets are never written into it directly: reference them with
`${ENV_VAR_NAME}` placeholders, resolved from the environment at load time
(see [.env.example](.env.example)).

Input and output are configured **independently** of each other, each with
a `kind` and a backend-specific `options` table:

- **`kind = "dataset"`**: static parquet files, on S3-compatible object
  storage (R2, AWS S3, MinIO — `storage_backend = "s3"`) or on a mounted
  local disk (`storage_backend = "local"`, handy for tests or manual
  import/export).
- **`kind = "db"`**: a database table/query via SQLAlchemy
  (`database_url`), with the option to override the read queries
  (`events_query` / `users_query` / `items_query`) to read directly from
  your own schema instead of requiring materialized `events`/`users`/`items`
  tables.

The two sides can be freely mixed, e.g. read from a Postgres replica and
write recommendations to S3, or vice versa. New backends can be added under
`src/cicerone/io/` without changing the configuration format — see
`config/cicerone.toml` for the full annotated example (including the `db`
variant, commented out).

## Data contract

`events` (required):

| column      | type      | notes                                                            |
|-------------|-----------|-------------------------------------------------------------------|
| user_id     | str       | any stable user identifier                                        |
| item_id     | str       | any stable item/product identifier                                |
| event_type  | str       | see `config/features.toml` → `event_weights`                       |
| quantity    | int       | optional, used for the types listed in `quantity_scaled_events`   |
| occurred_at | datetime  | UTC                                                                |

`event_type` is entirely up to you — map your own events to whatever names
you list in `config/features.toml` → `event_weights`. A typical e-commerce
mapping looks like:
- a completed order line → `purchase` (quantity = line quantity)
- a positive review/rating → `review_positive`
- a negative review/rating → `review_negative`
- a wishlist/save action → `saved`
- an "add to cart" analytics event → `cart_add`
- a "product viewed" analytics event → `view`

`users` (optional, enables user features for cold-start): columns are
configurable in `config/features.toml` → `user_features` (default:
`favorite_styles` as a list, `region_slug` as categorical — rename/replace
these for your own domain).

`items` (optional, enables item features + the availability filter): columns
are configurable in `config/features.toml` → `item_features` (default:
`category`, `primary_style`, `producer_id`, `region_slug`, `abv_bucket` —
again, adapt these to your catalog). The availability filter
(`item_availability_filters`, default `published` + `in_stock`) always
excludes unavailable items from the recommendations.

## Output

`recommendations`: `user_id, item_id, rank, score, source`
(`source` = `personalized` or `popular_fallback`).

`manifest`: metadata about the latest run (counts, timestamps) for monitoring.

## Interaction weights & cold-start

All weighting logic is configurable without rebuilding the image via
`config/features.toml` (mounted as a volume, see `docker-compose.yml`):
`event_weights`, `quantity_scaled_events`, `event_caps`, `user_features`,
`item_features`, `item_availability_filters`. Exponential decay with a
configurable half-life (`[job].half_life_days` in `config/cicerone.toml`,
default 90 days) gives more weight to recent activity. Users without enough
interactions still get a fallback list from `PopularModel` (rectools), still
honoring the availability filter.

## Usage

```sh
cp .env.example .env   # set the secrets referenced by config/cicerone.toml
# edit config/cicerone.toml: pick input/output kind & backend for your setup
docker compose up --build
```

## Tests & CI

```sh
docker compose -f docker-compose.ci.yml up --build --abort-on-container-exit --exit-code-from test
```

Runs the whole pytest suite (with an ephemeral Postgres for the `db` backend
tests) inside Docker — nothing to install on the host. The minimum required
coverage is 95% (`pyproject.toml`, `[tool.coverage.report].fail_under`) and
is enforced on every PR by `.github/workflows/ci.yml`, which also runs
[Ruff](https://docs.astral.sh/ruff/) (lint + format check) in the same test
image. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run tests/lint
locally and [docs/architecture.md](docs/architecture.md) for how the code is
structured.

## Security

- Credentials (S3/DB) should be scoped to the bare minimum (read on the
  input side, write on the output side, no delete/admin permissions).
- No personal data other than `user_id` (an opaque identifier) is ever read
  or written.
- No ports exposed: the container accepts no inbound connections.
- Credentials only ever live in environment variables (`.env`, not
  committed), referenced from `config/cicerone.toml` via `${...}`
  placeholders — never written into the config file itself.
- CI also runs `pip-audit` (dependency CVE scan) and
  [CodeQL](.github/workflows/codeql.yml) (static analysis) on every PR;
  Dependabot (`.github/dependabot.yml`) opens PRs for outdated pip/Docker/
  Actions pins.

## License

[Beerware](LICENSE) — if we meet someday and you find this useful, buy me a
beer (or, even better, one straight from [Torbido](https://torbido.it)).



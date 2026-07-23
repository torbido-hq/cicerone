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
                                 3. trains the configured model strategies
                                    (collaborative/item-based/popular/latest)
                                 4. combines them into top-K recs per user
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

## Model strategies

`[job].models` in `config/cicerone.toml` picks which strategies to fit and
combine, in priority order (earlier entries win ties for the same
user/item pair). Defaults to `["collaborative", "popular"]` if omitted:

- `collaborative`: `LightFMWrapperModel` (rectools) — hybrid CF, uses user/item
  features for cold-start. Personalized, warm users only.
- `item_based`: `ImplicitItemKNNWrapperModel` (rectools) — item-item
  similarity. Personalized, warm users only.
- `popular`: `PopularModel` (rectools) — global popularity. Non-personalized,
  runs for every target user and backfills any warm user without enough
  personalized results.
- `latest`: `PopularModel` restricted to the last two weeks of interactions —
  trending/recently active items. Non-personalized, same backfill role as
  `popular`.

By default, strategies are combined in priority order: earlier ones win ties
for the same user/item pair, non-personalized ones only backfill users who
didn't get enough personalized results. Optionally, `[job.model_weights]`
switches to a weighted reciprocal rank fusion instead — every enabled
strategy's rank contributes `weight / (rrf_k + rank)` to each item's fused
score, summed across strategies, so results from heterogeneous strategies
blend without needing to normalize their raw scores. `rrf_k` (`[job].rrf_k`,
default `60`) is tunable and only applies when `model_weights` is set — it
must be positive. An explicitly empty `[job.model_weights]` table still
enables fusion mode, with every enabled strategy defaulting to weight `1.0`.
Weight values must be non-negative. When a fusion result's (user, item) pair
was produced by more than one strategy, its `source` label joins each
contributing strategy's label in `models`' configured order (e.g.
`"popular_fallback+latest"` when `models = ["popular", "latest"]`), not
alphabetically — so the label reflects your configured priority regardless
of how the underlying strategy labels happen to sort.

## AutoML

Instead of a fixed `models`/`model_weights` config, `[job.automl]` can pick
the best combination automatically for every run:

```toml
[job.automl]
enabled = true
n_splits = 2       # time-based folds to backtest each candidate over
test_days = 14     # size of each fold's held-out window, in days
primary_metric = "MAP" # matched by prefix, e.g. "MAP@10"
```

Each run, `cicerone.automl.evaluate_candidates()` splits your event history
into `n_splits` non-overlapping, most-recent-first `test_days`-day windows;
for each candidate strategy/weight combination, it trains on everything
before the window and scores the recommendations against what actually
happened during it (`MAP@k`, `NDCG@k`, `Recall@k`, via `rectools.metrics`).
`select_best_candidate()` then picks the highest-scoring candidate by
`primary_metric`, and that candidate's `models`/`weights`/`rrf_k` are used
for the run in place of the static config, ties broken by candidate order.

The default candidate search space tries every strategy alone, the default
priority combo, and one weighted-fusion blend across all four strategies —
override it with `[[job.automl.candidates]]` (same shape as
`models`/`model_weights`/`rrf_k` above, one array-of-tables entry per
candidate) if you want to try a different set. Unlike top-level
`[job.model_weights]`, a candidate's `weights` table (if present) must give
an explicit weight for every one of its `models` — there's no implicit
default for an omitted model, to avoid silently backtesting a weighting you
didn't intend. AutoML raises if there isn't enough event history for at
least one fold — reduce `n_splits`/`test_days` or provide more historical
events.

Within each backtested fold, candidates that enable the same strategy (e.g.
two fusion candidates that both include `popular`) reuse that strategy's
already-fitted model instead of re-fitting it per candidate — fitting still
happens once per fold per distinct strategy, and `recommend()` still runs
fresh for every candidate, so this is purely a training-cost optimization
and doesn't change scoring.

## Output

`recommendations`: `user_id, item_id, rank, score, source` (`source` is the
label of whichever strategy produced that row: `personalized`, `item_based`,
`popular_fallback`, or `latest`).

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



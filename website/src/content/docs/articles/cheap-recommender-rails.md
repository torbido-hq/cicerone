---
title: A cheap recommender for a small Rails shop
description: Batch recommendations on the Postgres you already have — compared with rolling your own on sparse data, not with a recs SaaS.
date: 2026-08-20
excerpt: Point Cicerone at your orders table, let it write top-K rows overnight, and JOIN them in Rails. Honest about what sparse data will and will not do.
authors:
  - nicholas
---

Canonical URL: [https://cicerone.dev/articles/cheap-recommender-rails/](https://cicerone.dev/articles/cheap-recommender-rails/)

Most small shops that want “recommended for you” do not start by buying a recs platform. They start in-house, on not much data: a bestsellers query, “customers who bought this also bought”, maybe a weekend with [LightFM](https://making.lyst.com/lightfm/docs/home.html). That is the right comparison for [Cicerone](https://cicerone.dev) — a self-hosted **batch** recommender that reads your interactions, writes a top-K table, and otherwise stays out of the request path.

I wrote it for a catalog that is not Netflix. The engine does not care that the original catalog was drinks. Users, items, events.

This walkthrough wires it to a typical Rails + Postgres shop **without** the HTTP serve API. One job container, the database you already have, ActiveRecord on the way out.

## What you are actually getting

Cicerone trains **offline**. On a cron (default 03:00 UTC) it:

1. Runs SQL you give it (`events`, optional `users` / `items`)
2. Weighs those events, fits a couple of strategies (here: LightFM + popularity)
3. Writes `cicerone_recommendations` (`user_id`, `item_id`, `rank`, `score`, `source`)

Your app `SELECT`s that table and joins `products`. Guests and brand-new accounts use a sentinel row set, `user_id = '__cold_start__'`, which is why we turn on per-user blending below.

There is no model in the Rails process. A purchase this afternoon does not change tonight’s ranks. Personalized LightFM only moves on a full retrain.

## Compare this with the in-house thing you would write anyway

On a thin event log, **collaborative filtering is mostly popularity with extra steps**. Overlap is the scarce resource: two customers have to have bought some of the same things before matrix factorization has anything to say. Until then, a decent `GROUP BY product_id ORDER BY COUNT(*) DESC` homepage is not naïve — it is often the honest global model.

| Approach | What it costs you | What sparse data does to it |
| --- | --- | --- |
| Bestsellers / “also bought” SQL | An afternoon | Works immediately; no personalization |
| Your own LightFM + cron + fallbacks | Days of glue, then you own it | Same math as Cicerone; you still write weights, cold-start, and a job that cannot overlap itself |
| Cicerone on this Postgres | A container, two TOML files, a `JOIN` | Same math, less glue; **still** falls back to popular when history is thin |

Cicerone is worth it when you want that second row without becoming a recs team: time-decayed event weights, hybrid CF with optional item features, a popularity backfill, and a table Rails already knows how to read. It is not worth it if you wanted live inference, session-level next-click, or a model that “just knows” from fifty checkouts a week.

**Keep the bestsellers query.** After the first job, look at `source`. If almost every signed-in user is `popular_fallback` (or `blended` that is still mostly popular), the fancy part is not earning its keep yet. That is a data problem, not a config problem.

Other downsides, said plainly:

- Each successful job **truncates then rewrites** the recommendations table. Point `recommendations_table` at anything you care about and you will empty it. Prefix the name.
- Default input tables are `users` / `events` / `items`. Rails already has `users`. If you omit `users_query`, Cicerone will `SELECT * FROM users` and then look for a `user_id` column that is actually `id`. Always set the queries.
- IDs are strings. `42` and `"42"` are the same only if you `::text` in SQL and `id.to_s` in Ruby.
- Output identifiers cannot be schema-qualified (`cicerone.recommendations` is rejected). Tables live in `public` with a prefix, or you use the same database role as the app.
- The job needs `CREATE` (first run) and `TRUNCATE` on its output tables. A dedicated role that is `SELECT`-only on `orders` is the right idea; granting `CREATE` on `public` is the awkward part. Many small shops just use the app role and are careful with the table name. I would still not use the default name `recommendations`.
- `docker-compose.yml` in the Cicerone repo is developer convenience, not a production deploy.
- [Beerware](https://github.com/torbido-hq/cicerone/blob/main/LICENSE). You operate the box.

## Map a Rails schema to the event contract

Cicerone wants **events** (required):

| column | notes |
| --- | --- |
| `user_id` | stable string |
| `item_id` | stable string |
| `event_type` | must appear in `[event_weights]` or the row is dropped |
| `quantity` | optional; purchases can scale with `log1p(quantity)` |
| `occurred_at` | UTC |

Paid order lines are enough to start. Product views help a bit; they will not rescue a catalog nobody has bought.

```sql
SELECT
  o.user_id::text AS user_id,
  oi.product_id::text AS item_id,
  'purchase'::text AS event_type,
  oi.quantity,
  o.created_at AS occurred_at
FROM order_items oi
INNER JOIN orders o ON o.id = oi.order_id
WHERE o.status IN ('paid', 'complete', 'completed')
  AND o.user_id IS NOT NULL
```

If you also store authenticated product views, `UNION ALL` them as `event_type = 'view'` with `quantity = 1`. Skip `user_id IS NULL` (guest checkouts, anonymous pageviews) — there is no one to personalize.

Items (optional, but you want them so we can hide unpublished / out-of-stock and use `category`):

```sql
SELECT
  p.id::text AS item_id,
  p.category,
  p.created_at,
  COALESCE(p.published, TRUE) AS published,
  COALESCE(p.in_stock, TRUE) AS in_stock
FROM products p
```

Rename `published` / `in_stock` to whatever booleans you actually have (`active`, `inventory_count > 0`, …). If those columns do not exist, either add expressions or set `item_availability_filters = []` so Cicerone does not look for them. A missing filter column is skipped (fail-open), which is easy to miss in the logs.

Users can be as small as:

```sql
SELECT id::text AS user_id FROM users
```

You do not need user features for this setup. The query is there so Cicerone does not try to eat the Devise `users` table raw.

## Two TOML files

Put these next to the Rails app (or in a small `cicerone/` directory you mount into the container). Secrets stay in the environment; the TOML only references `${INPUT_DATABASE_URL}` and `${OUTPUT_DATABASE_URL}`. Both should be the **same** database, with the SQLAlchemy driver prefix:

```text
postgresql+psycopg://USER:PASS@HOST:5432/DBNAME
```

That is not Rails’ `postgres://` URL. From Compose, `HOST` is the Postgres **service name** (`postgres`, `db`, …), not `localhost`.

`cicerone.toml`:

```toml
[job]
mode = "batch"
top_k = 10
half_life_days = 90
cron_schedule = "0 3 * * *"
feature_config_path = "/app/config/features.toml"
models = ["collaborative", "popular"]

[input]
kind = "db"

[input.options]
database_url = "${INPUT_DATABASE_URL}"
events_query = """
SELECT
  o.user_id::text AS user_id,
  oi.product_id::text AS item_id,
  'purchase'::text AS event_type,
  oi.quantity,
  o.created_at AS occurred_at
FROM order_items oi
INNER JOIN orders o ON o.id = oi.order_id
WHERE o.status IN ('paid', 'complete', 'completed')
  AND o.user_id IS NOT NULL
"""
users_query = """
SELECT id::text AS user_id FROM users
"""
items_query = """
SELECT
  p.id::text AS item_id,
  p.category,
  p.created_at,
  COALESCE(p.published, TRUE) AS published,
  COALESCE(p.in_stock, TRUE) AS in_stock
FROM products p
"""

[output]
kind = "db"

[output.options]
database_url = "${OUTPUT_DATABASE_URL}"
recommendations_table = "cicerone_recommendations"
manifest_table = "cicerone_recommendation_runs"
```

`features.toml` — slim on purpose. Do not copy the beer-oriented `favorite_styles` / `abv_bucket` columns from the repo defaults. Root keys have to sit **above** the first `[table]` header (TOML assigns keys to whichever table they follow):

```toml
quantity_scaled_events = ["purchase"]
item_availability_filters = ["published", "in_stock"]

[event_weights]
purchase = 4.0
view = 0.3

[event_caps]
view = 5

[[item_features]]
column = "category"
type = "categorical"

[blending]
enabled = true
curve = "linear"
saturate_at = 5.0
popular_share = 0.7
latest_date_columns = ["created_at"]
```

Blending is what writes `__cold_start__`. Linear `saturate_at = 5` means a customer with a handful of distinct products already leans personalized; people with one order stay mostly on popular / latest. On a small shop that is the behaviour you want. If `products` has no usable date column among `latest_date_columns`, latest is skipped and its weight moves to popular (you will see that in the job log).

A `view` weight with no view rows is harmless. An `event_type` in SQL that you forget to list under `[event_weights]` is dropped, with a warning.

## Own the output table in Rails

Cicerone will `CREATE TABLE` on first write if the table is missing (pandas `to_sql`). I still want a migration, so `schema.rb` / `structure.sql` knows the name and you do not discover types by accident:

```ruby
class CreateCiceroneRecommendations < ActiveRecord::Migration[7.1]
  def change
    create_table :cicerone_recommendations, id: false do |t|
      t.string :user_id, null: false
      t.string :item_id, null: false
      t.integer :rank, null: false
      t.float :score
      t.string :source
    end

    add_index :cicerone_recommendations, [:user_id, :rank]
  end
end
```

No `id`. The natural key is `(user_id, item_id)`. After each job the table is emptied and filled again; do not put foreign keys on it that would block `TRUNCATE`. Leave `cicerone_recommendation_runs` to Cicerone — the manifest columns are job metadata, not something ActiveRecord should own.

## Run the job once

Build the image from a checkout (there is no published image tag; the [Dockerfile](https://github.com/torbido-hq/cicerone/blob/main/docker/Dockerfile) is the supported path). Python 3.11 and LightFM live inside it.

```sh
git clone --depth 1 https://github.com/torbido-hq/cicerone.git
cd cicerone
docker build -t cicerone -f docker/Dockerfile .
```

Then, from the Rails app, with the two TOML files in `./cicerone/` and the app’s Compose network name substituted:

```sh
export INPUT_DATABASE_URL='postgresql+psycopg://USER:PASS@postgres:5432/myapp_production'
export OUTPUT_DATABASE_URL="$INPUT_DATABASE_URL"

docker run --rm \
  --network myapp_default \
  -e INPUT_DATABASE_URL \
  -e OUTPUT_DATABASE_URL \
  -v "$PWD/cicerone/cicerone.toml:/app/config/cicerone.toml:ro" \
  -v "$PWD/cicerone/features.toml:/app/config/features.toml:ro" \
  cicerone job --config /app/config/cicerone.toml
```

The image `ENTRYPOINT` is already `cicerone`, so the command is `job --config …`, not `cicerone job …`.

If the process cannot see Postgres, it is almost always the hostname (container DNS vs `localhost`) or the URL scheme (`postgres://` vs `postgresql+psycopg://`).

Sanity check:

```sql
SELECT source, COUNT(*) FROM cicerone_recommendations GROUP BY 1;
SELECT * FROM cicerone_recommendations WHERE user_id = '__cold_start__' ORDER BY rank;
SELECT * FROM cicerone_recommendations WHERE user_id = '42' ORDER BY rank;
```

You want a non-empty sentinel. For user `42`, mixed `personalized` / `blended` / `popular_fallback` is success; **only** `popular_fallback` means that user did not give the collaborative model enough to work with.

## Read it from Rails

The table has no ActiveRecord primary key. Treat it as a query, not as a `belongs_to :product` (string `item_id` vs integer `products.id`):

```ruby
class CiceroneRecommendation < ApplicationRecord
  self.table_name = "cicerone_recommendations"
  self.primary_key = false

  def readonly?
    true
  end

  def self.for_user(user)
    where(user_id: user.id.to_s).order(:rank)
  end

  def self.cold_start
    where(user_id: "__cold_start__").order(:rank)
  end

  def self.product_ids_for(user, limit: 8)
    scope = user ? for_user(user) : cold_start
    ids = scope.limit(limit).pluck(:item_id)
    ids = cold_start.limit(limit).pluck(:item_id) if user && ids.empty?
    ids.map { |id| Integer(id) }
  end
end
```

`self.primary_key = false` is Rails 7.1. On 7.0 you can still `pluck` without it; skip `find`.

```ruby
def recommended_products(user, limit: 8)
  ids = CiceroneRecommendation.product_ids_for(user, limit: limit)
  Product.where(id: ids).sort_by { |p| ids.index(p.id) }
end
```

A template is then ordinary: iterate `@products`. If the job has never succeeded, `ids` is empty — show nothing, or your old bestsellers partial. Empty is better than inventing a failure UI.

## Keep it running

Paste a service into **your** Compose file, same network as Postgres. `start` runs one job immediately, then the cron expression (UTC):

```yaml
cicerone:
  image: cicerone
  command: ["start", "--config", "/app/config/cicerone.toml"]
  environment:
    INPUT_DATABASE_URL: postgresql+psycopg://USER:PASS@postgres:5432/myapp_production
    OUTPUT_DATABASE_URL: postgresql+psycopg://USER:PASS@postgres:5432/myapp_production
  volumes:
    - ./cicerone/cicerone.toml:/app/config/cicerone.toml:ro
    - ./cicerone/features.toml:/app/config/features.toml:ro
  restart: unless-stopped
  depends_on:
    - postgres
```

No ports. Batch mode does not listen. Disk is a couple of TOML files; CPU is a nightly LightFM fit. For a shop-sized catalog that is a small VM, not a GPU story.

## When this is the wrong tool

- You need the list to change inside the same request as “add to cart”. That is a different product (and Cicerone’s optional [serve API](https://cicerone.dev/) plus incremental events still do not retrain LightFM on the request path).
- You have almost no overlapping buyers. Ship bestsellers, collect events, come back.
- You already enjoy operating a Python training stack and want SASRec, AutoML, eligibility rules, a dashboard. Those exist — they are not this article. See the [tutorial](https://cicerone.dev/tutorial/) and [how it works](https://cicerone.dev/how-it-works/).

If you do try this and `source` stays popular for everyone who matters, write me off until the catalog is denser. The cheap part is not pretending sparse data is dense. It is not spending a quarter building the glue around a model that, on your data, should mostly agree with `ORDER BY COUNT(*) DESC` anyway.

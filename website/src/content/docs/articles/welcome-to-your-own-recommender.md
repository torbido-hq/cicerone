---
title: Welcome to your own recommender
description: Cicerone stays out of your stack — any language, your database, a nightly top-K table. Rails is the walkthrough, not a dependency.
date: 2026-08-20
excerpt: Point it at your orders table, JOIN the ranks in whatever you already run. Honest about what sparse data will and will not do.
authors:
  - nicholas
---

<img src="https://cicerone.dev/images/docs/cicerone-logo.svg" alt="Cicerone" width="200">

Canonical URL: [https://cicerone.dev/articles/welcome-to-your-own-recommender/](https://cicerone.dev/articles/welcome-to-your-own-recommender/)

If you have been meaning to put “recommended for you” on the site, you already have the ingredients. Most shops start in-house, on not much data: a bestsellers query, “customers who bought this also bought”, maybe a weekend with [LightFM](https://making.lyst.com/lightfm/docs/home.html). That is the right comparison for [Cicerone](https://cicerone.dev) — a self-hosted **batch** recommender that reads your interactions, writes a top-K table, and otherwise stays out of the request path.

The name is a [beer sommelier](https://www.cicerone.org): someone who knows styles and what to pour next. I wrote the first version for a drinks catalog — not Netflix, not a recs team, just “what should this person try”. The engine never needed the SKUs to be bottles. Users, items, events.

It also does not care what you wrote the shop in. There is no Rails gem, no Django package, no Node SDK to keep in lockstep. Cicerone is a container that speaks SQL (or parquet / S3) and writes rows. Your request path stays yours. Rails is this article’s example because it is a common small-shop stack; the same two TOML files sit next to Laravel, Django, Phoenix, or a Go service. The HTTP serve API is optional and we will not use it here.

## What you are actually getting

Cicerone trains **offline**. On a cron (default 03:00 UTC) it:

1. Runs SQL you give it (`events`, optional `users` / `items`)
2. Weighs those events, fits a couple of strategies (here: LightFM + popularity)
3. Writes `cicerone_recommendations` (`user_id`, `item_id`, `rank`, `score`, `source`)

Your app `SELECT`s that table and joins `products`. Guests and brand-new accounts use a sentinel row set, `user_id = '__cold_start__'`, which is why we turn on per-user blending below.

Nothing Cicerone-shaped loads in the web process. A purchase this afternoon does not change tonight’s ranks. Personalized LightFM only moves on a full retrain.

That is the trick worth getting excited about: **Netflix-shaped math on a shop-sized catalog**, without putting a Python model on the hot path. [LightFM](https://arxiv.org/abs/1507.08439) (WARP) embeds users, items, and side features — here, `category` — in one latent space, so a newly listed IPA can sit near other IPAs before it has a sales history. Popularity covers people with one order. Per-user blending mixes those ranks so you do not fall off a cliff from “personalized” to “everyone sees the same ten SKUs”. You get that as a table. The storefront never imports `rectools`.

## Compare this with the in-house thing you would write anyway

On a thin event log, **collaborative filtering is mostly popularity with extra steps**. Overlap is the scarce resource: two customers have to have bought some of the same things before matrix factorization has anything to say. Until then, a decent `GROUP BY product_id ORDER BY COUNT(*) DESC` homepage is not naïve — it is often the honest global model.

| Approach | What it costs you | What sparse data does to it |
| --- | --- | --- |
| Bestsellers / “also bought” SQL | An afternoon | Works immediately; no personalization |
| Your own LightFM + cron + fallbacks | Days of glue, then you own it | Same math as Cicerone; you still write weights, cold-start, and a job that cannot overlap itself |
| Cicerone beside the database | A container, two TOML files, a `JOIN` in any language | Same math, less glue; **still** falls back to popular when history is thin |

Cicerone is worth it when you want that second row without becoming a recs team — and without marrying the shop to a recs framework: time-decayed event weights, hybrid CF with optional item features, a popularity backfill, and a table any app can `SELECT`. It is not worth it if you wanted live inference, session-level next-click, or a model that “just knows” from fifty checkouts a week.

**Keep the bestsellers query.** After the first job, look at `source`. If almost every signed-in user is `popular_fallback` (or `blended` that is still mostly popular), the fancy part is not earning its keep yet. That is a data problem, not a config problem.

Other downsides, said plainly:

- Each successful job **truncates then rewrites** the recommendations table. Point `recommendations_table` at anything you care about and you will empty it. Prefix the name.
- Default input tables are `users` / `events` / `items`. Rails already has `users`, often with `email`, `encrypted_password`, and reset tokens. If you omit `users_query`, Cicerone will `SELECT * FROM users` and then look for a `user_id` column that is actually `id`. Always set a query that aliases `id` and selects nothing else:

```sql
SELECT id::text AS user_id FROM users
```
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

If you also store authenticated product views, fold them in. Same contract, a second `event_type` the weights already know:

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
UNION ALL
SELECT
  v.user_id::text,
  v.product_id::text,
  'view',
  1,
  v.created_at
FROM product_views v
WHERE v.user_id IS NOT NULL
```

Skip `user_id IS NULL` (guest checkouts, anonymous pageviews) — there is no one to personalize. A `view` weight with no view table yet is harmless; paste the `UNION ALL` when you have the rows.

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

Users can be as small as that same query — `id` only, no `email` / `encrypted_password` / tokens. You do not need user features for this setup. The query is there so Cicerone does not `SELECT *` the Devise `users` table.

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

      t.index [:user_id, :item_id], unique: true
      t.index [:user_id, :rank]
    end
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
  cicerone \
  job --config /app/config/cicerone.toml
```

The image is tagged `cicerone`; its `ENTRYPOINT` is already the CLI. Every `docker run` in this article is therefore `docker run … cicerone <subcommand> …` — `job`, `users`, `dashboard`. Do not pass a second `cicerone` (`cicerone cicerone job` will fail). Compose `command:` is only the subcommand, because Compose keeps that ENTRYPOINT.

If the process cannot see Postgres, it is almost always the hostname (container DNS vs `localhost`) or the URL scheme (`postgres://` vs `postgresql+psycopg://`).

## See what it did

The job always writes two things: a **manifest row** (did this run succeed, how much data, which models) and the **recommendations table** (what each user got). Dataset/parquet output overwrites the last manifest; **db output appends**, so you keep a short history. That is one reason to share Postgres.

Stdout of the job container is the first place to look (`Job finished: {…}` plus any `WARN` about dropped `event_type`s or missing feature columns). Then SQL:

```sql
SELECT generated_at, status, error,
       n_events, n_target_users, n_users_with_recommendations, n_items,
       models, triggered_by
FROM cicerone_recommendation_runs
ORDER BY generated_at DESC
LIMIT 5;
```

```sql
SELECT source, COUNT(*) AS n
FROM cicerone_recommendations
GROUP BY source
ORDER BY n DESC;
```

`status` should be `success`. `n_events` should match the order lines you think you have. `models` should list `collaborative,popular` (blending may still add popular if you omitted it). If almost every row is `popular_fallback`, the collaborative model did not have enough overlap yet — keep the bestsellers query.

Then look at a real person, with names:

```sql
SELECT r.rank, r.source, r.score, p.name, p.category
FROM cicerone_recommendations r
INNER JOIN products p ON p.id::text = r.item_id
WHERE r.user_id = '42'
ORDER BY r.rank;
```

```text
 rank | source           | score | name            | category
    1 | blended          |  0.81 | Hazy IPA 440ml  | beer
    2 | personalized     |  0.44 | Oat Stout       | beer
    3 | popular_fallback |  0.31 | House Lager     | beer
```

Illustrative rows — your catalog, your scores. `blended` means LightFM and popularity both voted; `personalized` is the latent space on its own. Swap `'42'` for `'__cold_start__'` to see guests.

If you would rather click than query, Cicerone ships a small Basic-Auth **dashboard** that reads the same two tables: latest run (and history, because this is a db output), plus a user-id lookup of current top-K. It never loads LightFM. Put this next to the other TOML as `cicerone.dashboard.toml`:

```toml
[job]
cron_schedule = "0 3 * * *"

[input]
kind = "dataset"

[input.options]
storage_backend = "local"
path = "/tmp/unused-in-dashboard-mode"

[output]
kind = "db"

[output.options]
database_url = "${OUTPUT_DATABASE_URL}"
recommendations_table = "cicerone_recommendations"
manifest_table = "cicerone_recommendation_runs"

[dashboard]
enabled = true
host = "0.0.0.0"
port = 8090
users_path = "/app/config/dashboard_users.toml"
lookup_k = 20
```

`cron_schedule` must match the batch job so the page can tell you the run looks overdue. `[input]` is required by the config loader and unused.

Add a login (prompts for a password; note `-it`), then start it. Same convention as the job: image name, then the subcommand.

```sh
docker run --rm -it \
  -v "$PWD/cicerone:/app/config" \
  cicerone \
  users --config /app/config/cicerone.dashboard.toml add you

docker run --rm --name cicerone-dashboard -p 127.0.0.1:8090:8090 \
  --network myapp_default \
  -e OUTPUT_DATABASE_URL \
  -v "$PWD/cicerone:/app/config" \
  cicerone \
  dashboard --config /app/config/cicerone.dashboard.toml
```

Open `http://127.0.0.1:8090/dashboard`, sign in, type a `user_id` (the string form, `'42'`). Bind only to localhost unless this sits behind your own auth.

![Cicerone dashboard: user lookup of current top-K, latest job status, and run history](https://cicerone.dev/images/docs/dashboard.png)

## Read it from the app

This is a `SELECT`, not an SDK. ActiveRecord is one way to do it; Eloquent, SQLAlchemy, Ecto, or `database/sql` would ask for the same columns. The table has no `id`. Set `primary_key` to `nil` so ActiveRecord does not assume one. Query it (`where` / `pluck`); do not `find`, `update`, or `belongs_to :product` (string `item_id` vs integer `products.id`):

```ruby
class CiceroneRecommendation < ApplicationRecord
  self.table_name = "cicerone_recommendations"
  self.primary_key = nil

  scope :for_user, ->(user) { where(user_id: user.id.to_s).order(:rank) }
  scope :cold_start, -> { where(user_id: "__cold_start__").order(:rank) }

  def readonly?
    true
  end

  class << self
    def product_ids_for(user, limit: 8)
      ids = ids_for(user.id.to_s, limit) if user
      ids = ids_for("__cold_start__", limit) if ids.blank?
      ids
    end

    private

    def ids_for(user_id, limit)
      where(user_id: user_id).order(:rank).limit(limit).pluck(:item_id).map(&:to_i)
    end
  end
end
```

```ruby
class HomeController < ApplicationController
  def index
    ids = CiceroneRecommendation.product_ids_for(current_user)
    @recommended = Product.in_order_of(:id, ids)
  end
end
```

```erb
<% if @recommended.any? %>
  <section aria-labelledby="recommended-heading">
    <h2 id="recommended-heading">Recommended for you</h2>
    <ul>
      <% @recommended.each do |product| %>
        <li><%= link_to product.name, product %></li>
      <% end %>
    </ul>
  </section>
<% end %>
```

If the job has never succeeded, `@recommended` is empty — hide the section, or fall back to your old bestsellers partial. Empty is better than inventing a failure UI.

Same table, no Rails. A Python worker or a Node renderer is the same `SELECT`:

```python
user_key = str(user_id) if user_id else "__cold_start__"
rows = conn.execute(
    """
    SELECT item_id FROM cicerone_recommendations
    WHERE user_id = %s ORDER BY rank LIMIT 8
    """,
    (user_key,),
).fetchall()
ids = [int(item_id) for (item_id,) in rows]
```

```js
const userKey = userId != null ? String(userId) : "__cold_start__";
const { rows } = await pool.query(
  `SELECT item_id FROM cicerone_recommendations
   WHERE user_id = $1 ORDER BY rank LIMIT 8`,
  [userKey],
);
const ids = rows.map((row) => Number(row.item_id));
```

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

No ports on the job. Batch mode does not listen. Disk is a couple of TOML files; CPU is a nightly LightFM fit. For a shop-sized catalog that is a small VM, not a GPU story.

Optional: keep the dashboard up too. The config directory is read-write so `users add` can persist the bcrypt file:

```yaml
cicerone-dashboard:
  image: cicerone
  command: ["dashboard", "--config", "/app/config/cicerone.dashboard.toml"]
  environment:
    OUTPUT_DATABASE_URL: postgresql+psycopg://USER:PASS@postgres:5432/myapp_production
  ports:
    - "127.0.0.1:8090:8090"
  volumes:
    - ./cicerone:/app/config
  restart: unless-stopped
  depends_on:
    - postgres
```

## When this is the wrong tool

- You need the list to change inside the same request as “add to cart”. That is a different product (and Cicerone’s optional [serve API](https://cicerone.dev/) plus incremental events still do not retrain LightFM on the request path).
- You have almost no overlapping buyers. Ship bestsellers, collect events, come back.
- You already enjoy operating a Python training stack and want SASRec, AutoML, eligibility rules. Those exist — they are not this article. See the [tutorial](https://cicerone.dev/tutorial/) and [how it works](https://cicerone.dev/how-it-works/).

If you try this and `source` stays popular for everyone who matters, that is fine: keep bestsellers on the homepage, let events accumulate, and come back. You are not behind. You are just early. The honest win is skipping a quarter of glue around a model that, on today’s data, should mostly agree with `ORDER BY COUNT(*) DESC` — and still having a real table to grow into.

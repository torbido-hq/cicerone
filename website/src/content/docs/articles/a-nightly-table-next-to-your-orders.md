---
title: A nightly table next to your orders
description: Nightly product recs as a Postgres table. Rails walkthrough; no Python in the request.
date: 2026-08-20
excerpt: Cron reads order_items, writes a ranked table, Rails JOINs it. Honest when the data is too thin to personalize.
authors:
  - nicholas
---

<img src="https://cicerone.dev/images/docs/cicerone-logo.svg" alt="Cicerone" width="200">

Canonical URL: [https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/](https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/)

You have an `order_items` table and you want a “Recommended for you” row on the homepage. What you do not want is Python running inside Rails, and you do not have a recommendations team to hand the problem to.

The usual answer is `ORDER BY sold_count DESC` under a friendlier heading. There is nothing dishonest about that query, but it shows the same row to everybody.

[Cicerone](https://cicerone.dev) is a Docker job that runs at night: it reads the orders you already store, writes a ranked table, and goes back to sleep. Your app only ever `SELECT`s from that table, so there is no gem to install and nothing Cicerone-shaped anywhere in the request path.

I built it for Torbido, a bottle shop that no longer trades; [torbido.co](https://torbido.co) is a landing page these days. The SKUs there were drinks, which is why the repo’s default `features.toml` still carries beer columns like `favorite_styles` and `abv_bucket`; ignore them unless you happen to have those fields. The job itself only cares about `user_id`, `item_id`, and `event_type`.

Rails is the example because a lot of small shops look like this, but the same two TOML files work just as well next to Laravel, Django, Phoenix, or a Go service. There is an optional HTTP API too, which this walkthrough does not use.

## What you get

On a cron (default 03:00 UTC) the job:

1. Runs SQL you give it (`events`, optional `users` / `items`)
2. Fits two lists — a personalized one ([LightFM](https://making.lyst.com/lightfm/docs/home.html)) and store-wide bestsellers — then mixes them
3. Writes `cicerone_recommendations` (`user_id`, `item_id`, `rank`, `score`, `source`)

Your app joins that table to `products`. Because the ranks are written once a night, a purchase this afternoon will not move them until the next run. ([Incremental events](https://cicerone.dev/incremental-events/) can refresh the bestsellers list between jobs, but LightFM still waits for the cron.)

| Name | Where | Meaning |
| --- | --- | --- |
| `popular` | `[job].models` | Store-wide bestsellers. |
| `personalized` | `source` | LightFM won this rank. |
| `popular_fallback` | `source` | The `popular` list filling in. Same bestsellers, different label. |
| `latest` | `source` | Newest by item date (`latest_date_columns`). Not the `"latest"` model. |
| `blended` | `source` | More than one of those lists voted on this rank. |

**Model names are not `source` labels.** `[job].models` lists what you train, which here is `collaborative` and `popular`. The `source` column records what actually won a given row, which is one of `personalized`, `popular_fallback`, `latest`, or `blended`. The bestsellers list therefore appears under two names: `popular` in the config, `popular_fallback` in the table.

### Guest list: `'__cold_start__'`

Guests and brand-new accounts have no purchase history, so the job writes one shared list under a sentinel `user_id`. The value stored in the column is `__cold_start__`, and the quotes you see in SQL are syntax rather than part of the string. It is text like any other `user_id`, not an integer, a `NULL`, or a bare identifier.

| Place | Literal |
| --- | --- |
| Database (`user_id` text) | `__cold_start__` |
| SQL | `'__cold_start__'` |
| Dashboard lookup | `__cold_start__` (no quotes) |

## What the job does

You do not have 1–5 star ratings to work with. You have purchases, which are a weak yes, and silence for everything else. Before any model sees them, those rows become training weights: recent purchases count for more, while caps and `log1p(quantity)` keep one bulk order from drowning out the rest. The full recipe is [interaction weighting](https://cicerone.dev/how-it-works/#interaction-weighting).

Each night:

1. LightFM builds a personalized list from people with similar purchases, and optional tags such as `category` give brand-new SKUs a way in. Alongside it, `popular` counts what the whole store buys.
2. The two lists are combined by **rank** rather than by raw score, because a LightFM number and a sales count do not measure the same thing. Customers with little history lean toward bestsellers; customers with enough overlap lean toward LightFM.
3. The result goes into the table, with `source` set to one of the labels above.

If no two customers have ever bought the same product, there is nothing for the personalized half to work with. That is thin data rather than a misconfigured job.

This walkthrough trains only `collaborative` and `popular`. The other models, the mixing formula, and the papers behind them are in [how it works](https://cicerone.dev/how-it-works/).

## When this is worth it

On a thin order log, collaborative filtering is mostly bestsellers with extra steps, because the personalized list has nothing to say until two customers have bought some of the same things. Until that happens, `GROUP BY product_id ORDER BY COUNT(*) DESC` is the honest global model.

| Approach | What it costs you | What thin data does to it |
| --- | --- | --- |
| Bestsellers / “also bought” SQL | An afternoon | Works immediately; no personalization |
| Your own LightFM + cron + fallbacks | Days of glue, then you own it | Same math as Cicerone; you still write weights, guest rows, and a job that cannot overlap itself |
| Cicerone beside the database | A container, two TOML files, a `JOIN` in any language | Same math, less glue; **still** falls back to `popular` when history is thin |

Cicerone earns its place if you want that second row without becoming a recommendations team and without running a Python model inside the shop process. It will not work miracles, though: fifty checkouts a week is not enough history for LightFM to “just know.”

**Keep the bestsellers query.** Once the first job has run, look at `source`. If almost every signed-in user comes back as `popular_fallback`, or as `blended` rows that are still mostly bestsellers, the personalized half is not earning its keep yet, and no amount of config will change that.

One trap is worth knowing before you run anything: every successful job truncates the recommendations table and rewrites it. Point `recommendations_table` at something you care about and the first run will empty it, so give the name a prefix you would never use elsewhere.

## Map a Rails schema to the event contract

The events query is the only required one:

| column | notes |
| --- | --- |
| `user_id` | stable string (`id::text` / `id.to_s`; `42` and `"42"` are not the same until you do) |
| `item_id` | stable string (same rule) |
| `event_type` | must appear in `[event_weights]` or the row is dropped |
| `quantity` | optional; purchases can scale with `log1p(quantity)` |
| `occurred_at` | UTC |

Paid order lines are enough to start with, using whichever timestamp means the money actually moved. The query below assumes `paid_at`, so fall back to `created_at` if you have no such column. Product views help a little, but they will not rescue a catalog nobody has bought from.

```sql
SELECT
  o.user_id::text AS user_id,
  oi.product_id::text AS item_id,
  'purchase'::text AS event_type,
  oi.quantity,
  o.paid_at AS occurred_at  -- no paid_at? use o.created_at
FROM order_items oi
INNER JOIN orders o ON o.id = oi.order_id
WHERE o.status IN ('paid', 'complete', 'completed')
  AND o.user_id IS NOT NULL
```

If you also store product views for signed-in users, append them with `UNION ALL`. It is the same five columns and a second `event_type` that the weights already know about:

```sql
SELECT
  v.user_id::text AS user_id,
  v.product_id::text AS item_id,
  'view'::text AS event_type,
  1,
  v.created_at AS occurred_at
FROM product_views v
WHERE v.user_id IS NOT NULL
```

Both queries drop rows with a `NULL` `user_id`, such as guest checkouts and anonymous pageviews, because there is nobody to personalize for. Leaving a `view` weight in the config with no view table behind it does no harm; paste the `UNION ALL` in once you have the rows.

The items query is optional, but you probably want it, because it lets you hide unpublished or out-of-stock products and feeds `category` into LightFM:

```sql
SELECT
  p.id::text AS item_id,
  p.category,
  p.created_at,
  COALESCE(p.published, TRUE) AS published,
  COALESCE(p.in_stock, TRUE) AS in_stock
FROM products p
```

Rename `published` and `in_stock` to whatever booleans you actually have, such as `active` or `inventory_count > 0`. If nothing matches, either write the expressions yourself or set `item_availability_filters = []` so Cicerone stops looking for them. A filter column that is missing gets skipped rather than raising, which is easy to overlook in the logs.

The users query can be a single column, since this setup needs no user features and has no business reading `email`, `encrypted_password`, or session tokens. Omit `users_query` entirely and Cicerone will `SELECT * FROM users` — your whole Devise table — and then look for a `user_id` column that is really called `id`. Alias it and select nothing else:

```sql
SELECT id::text AS user_id FROM users
```

## Two TOML files

Put these next to the Rails app, or in a small `cicerone/` directory you mount into the container. Secrets stay in the environment, since the TOML only refers to `${INPUT_DATABASE_URL}` and `${OUTPUT_DATABASE_URL}`, and both should point at the same database with the SQLAlchemy driver prefix:

```text
postgresql+psycopg://USER:PASS@HOST:5432/DBNAME
```

That is not the `postgres://` URL Rails uses. From Compose, `HOST` is the Postgres **service name** — `postgres`, `db`, and so on — rather than `localhost`.

Output identifiers cannot be schema-qualified, so `cicerone.recommendations` is rejected and the tables live in `public` behind a prefix. The job needs `CREATE` on its first run and `TRUNCATE` from then on. A dedicated role with `SELECT`-only access to `orders` is the right instinct, though granting it `CREATE` on `public` is the awkward part; plenty of small shops use the app role instead and stay careful about the table name. Either way, do not leave it as the default `recommendations`.

The queries in `cicerone.toml` are the purchase, users, and items SELECTs from the previous section, plus the view `UNION ALL` if you have that table:

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
  o.paid_at AS occurred_at  -- no paid_at? use o.created_at
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

`features.toml` is small on purpose. Root keys have to sit **above** the first `[table]` header, because TOML assigns every key to whichever table precedes it. The repo’s `favorite_styles` and `abv_bucket` defaults can stay where they are, since this walkthrough does not use them:

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

Blending is also what writes the `'__cold_start__'` rows. Setting `saturate_at = 5` means a customer with five distinct products is scored entirely by LightFM, while someone with a single order stays mostly on bestsellers and newest-by-date. On a small shop that is the behavior you want.

**Two things are called “latest.”** The `latest_date_columns` setting ranks products by an item timestamp, `created_at` in this config. The `"latest"` you can list in `[job].models` is something else: a recency-window sales list built from events, a third model rather than a flavor of `popular`. This walkthrough uses the date list only, and you should not add `"latest"` to `models` while blending is on, because the two rankings fight each other; [how it works](https://cicerone.dev/how-it-works/) has the details. If `products` has no usable date column among the names you gave, the date list is skipped and its weight moves to `popular`, which the job log will tell you.

A `view` weight with no view rows behind it is harmless, but an `event_type` that appears in your SQL and not in `[event_weights]` is dropped with a warning.

## Own the output table in Rails

Cicerone will `CREATE TABLE` on the first write if the table is missing, courtesy of pandas `to_sql`. I still write the migration myself, so that `schema.rb` or `structure.sql` knows the name and nobody discovers the column types by accident:

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

There is no `id` column, because the natural key is `(user_id, item_id)`. The table is emptied and refilled after every job, so avoid foreign keys that would block a `TRUNCATE`. Leave `cicerone_recommendation_runs` to Cicerone: those columns are job metadata, not something ActiveRecord should own.

## Run the job once

Build the image from a checkout. There is no published image tag, so the [Dockerfile](https://github.com/torbido-hq/cicerone/blob/main/docker/Dockerfile) is the supported path, and Python 3.11 and LightFM live inside it rather than on your host.

```sh
git clone --depth 1 https://github.com/torbido-hq/cicerone.git
cd cicerone
docker build -t cicerone -f docker/Dockerfile .
```

Then, from the Rails app, with the two TOML files in `./cicerone/` and your own Compose network name substituted:

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

The image is tagged `cicerone` and its `ENTRYPOINT` is already the CLI, so what follows the image name is only the subcommand: `job`, `users`, or `dashboard`, never a second `cicerone`. The same goes for `command:` in Compose.

If the process cannot see Postgres, the cause is almost always the hostname, where container DNS and `localhost` disagree, or the URL scheme, where `postgres://` should have been `postgresql+psycopg://`.

## See what it did

Every run writes two things: a **manifest row** recording whether the job succeeded, how much data it saw, and which models it trained, and the **recommendations table** itself. Dataset and parquet output overwrite the manifest each time, while a db output appends to it, which is one of the reasons to share Postgres.

The job container’s stdout is the first place to look, for a `Job finished: {…}` line plus any `WARN` about dropped `event_type`s or missing feature columns. After that, SQL:

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

`status` should read `success`, `n_events` should be close to the number of order lines you think you have, and `models` should read `collaborative,popular`. If almost every row comes back `popular_fallback`, the overlap between customers was too thin and the job is still serving bestsellers.

Then look at one real customer, with product names attached:

```sql
SELECT r.rank, r.source, r.score, p.name, p.category
FROM cicerone_recommendations r
INNER JOIN products p ON p.id::text = r.item_id
WHERE r.user_id = '42'   -- or '__cold_start__'
ORDER BY r.rank;
```

```text
 rank | source           | score | name            | category
    1 | blended          |  0.81 | Hazy IPA 440ml  | beer
    2 | personalized     |  0.44 | Oat Stout       | beer
    3 | popular_fallback |  0.31 | House Lager     | beer
```

Those rows are illustrative; your catalog and your scores will differ. Swap `'42'` for `'__cold_start__'` to read the guest list.

If you would rather click than query, Cicerone ships a small Basic-Auth **dashboard** that reads those same two tables: the latest run, the history a db output gives you, and a lookup of the current top-K for any user id. It never loads LightFM. Put its config next to the other TOML as `cicerone.dashboard.toml`:

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
lookup_k = 10
```

`cron_schedule` has to match the batch job so the page can tell when a run looks overdue. The `[input]` block is required by the config loader and otherwise unused.

Add a login, which prompts for a password and therefore needs `-it`, then start the dashboard. The convention is the same as the job: image name, then subcommand.

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

Open `http://127.0.0.1:8090/dashboard`, sign in, and type `42` or `__cold_start__` into the lookup. Keep it bound to localhost unless it sits behind your own auth.

![Cicerone dashboard: user lookup of current top-K, latest job status, and run history](https://cicerone.dev/images/docs/dashboard.png)

## Read it from the app

Reading the table is a `SELECT`, not an SDK call. ActiveRecord is one way to write it, and Eloquent, SQLAlchemy, Ecto, or `database/sql` would ask for the same columns. Because the table has no `id`, set `primary_key` to `nil` so ActiveRecord stops assuming one. `for_user` takes a string, so pass `current_user.id.to_s`, and `cold_start` covers the guest rows. Read with `pluck` rather than `find` or `update`, and resist `belongs_to :product`, since `item_id` is text while `products.id` is an integer:

```ruby
class CiceroneRecommendation < ApplicationRecord
  self.table_name = "cicerone_recommendations"
  self.primary_key = nil

  scope :for_user, ->(user_id) { where(user_id: user_id.to_s).order(:rank) }
  scope :cold_start, -> { where(user_id: '__cold_start__').order(:rank) }

  def readonly?
    true
  end

  class << self
    def product_ids_for(user, limit: 8)
      ids = for_user(user.id.to_s).limit(limit).pluck(:item_id).map(&:to_i) if user
      ids = cold_start.limit(limit).pluck(:item_id).map(&:to_i) if ids.blank?
      ids
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

If the job has never succeeded, `@recommended` comes back empty, so hide the section or fall back to your old bestsellers partial rather than inventing a failure state for it. The `.map(&:to_i)` assumes the usual integer `products.id`; drop it and pass the strings straight through if your primary keys are UUIDs.

None of this is specific to Rails. A Python worker or a Node renderer runs the same `SELECT`, falling back to `'__cold_start__'` when a signed-in user has no rows yet:

```python
sql = """
SELECT item_id FROM cicerone_recommendations
WHERE user_id = %s ORDER BY rank LIMIT 8
"""
user_key = str(user_id) if user_id else '__cold_start__'
rows = conn.execute(sql, (user_key,)).fetchall()
if user_id and not rows:
    rows = conn.execute(sql, ('__cold_start__',)).fetchall()
ids = [int(item_id) for (item_id,) in rows]
```

```js
const sql = `SELECT item_id FROM cicerone_recommendations
 WHERE user_id = $1 ORDER BY rank LIMIT 8`;
const userKey = userId != null ? String(userId) : '__cold_start__';
let { rows } = await pool.query(sql, [userKey]);
if (userId != null && rows.length === 0) {
  ({ rows } = await pool.query(sql, ['__cold_start__']));
}
const ids = rows.map((row) => Number(row.item_id));
```

## Keep it running

Paste a service into **your** Compose file, on the same network as Postgres. The `start` subcommand runs one job immediately and then follows the cron expression, in UTC:

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

The job needs no ports, because batch mode never listens. It costs you two TOML files on disk and one nightly LightFM fit of CPU, which a shop-sized catalog will manage on a small VM without a GPU. The `docker-compose.yml` in the Cicerone repo is developer convenience, not this deployment.

You can keep the dashboard up alongside it. Its config directory is mounted read-write so that `users add` can persist the bcrypt file:

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

The license is [Beerware](https://github.com/torbido-hq/cicerone/blob/main/LICENSE), and the box is yours to operate.

## When this is the wrong tool

- You need the row to change within the same request as “add to cart.” That is a different product, and even Cicerone’s optional [serve API](https://cicerone.dev/openapi/) and [incremental events](https://cicerone.dev/incremental-events/) do not retrain LightFM on the request path.
- You have almost no overlapping buyers, in which case you should ship bestsellers, collect events, and come back later.
- You already enjoy operating a Python training stack and want SASRec, AutoML, and eligibility rules. All of that exists, but it is not what this article is about; see the [tutorial](https://cicerone.dev/tutorial/), [how it works](https://cicerone.dev/how-it-works/), and [architecture](https://cicerone.dev/architecture/).

## In the morning

Last night’s cron has run, or you ran `job` yourself. Either way, look at `source`, because on day one it is the only metric that matters.

If it is mostly `popular_fallback`, the job is still serving bestsellers, so keep that query on the homepage until more customers have bought the same things. That is not a failure. It is what a young order log looks like.

If `personalized` and `blended` are showing up for people who actually buy, put the `SELECT` on the homepage and leave the cron running in Compose. That is the whole product this walkthrough set up: a table your shop already knows how to read.

The drink columns in the repo’s default config were never part of the contract. It only ever asked for `user_id`, `item_id`, and `event_type`. Everything after that is a cron job writing ranks and going back to sleep.

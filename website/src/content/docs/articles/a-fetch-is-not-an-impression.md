---
title: A fetch is not an impression
description: Cicerone records CTR from host-reported POST /track rows. GET /recommendations is a lookup. Quality matches a click to a prior (user, item) impression inside the window.
date: 2026-09-17
excerpt: log_impressions counts every GET. The Blade widget POSTs /track when it renders. A click sent to /events trains LightFM and never reaches Quality.
authors:
  - nicholas
---

You want click-through on the homepage widget. You turn on `serve.log_impressions` and have Laravel `Http::get` the serve API. Horizon warms the same URL for last week's buyers. A health check hits it too. You did not measure CTR. You measured fetches.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough already said `source` only tells you which list won, not whether anyone clicked. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) post already said impressions and clicks are `POST /track`, not `POST /events`. This is that contract.

Cicerone cannot see the Blade. A `SELECT` is not a view. A `GET /recommendations/{user_id}` is not a view. The host has to say what it rendered.

```text
Blade renders the widget
        │
        ├─ SELECT … ORDER BY rank
        │     or GET /recommendations/{user_id}
        ▼
POST /track   kind=impression, rank ≥ 1
        │
user taps a SKU
        ▼
POST /track   kind=click
        │
recommendation_track  /  track.jsonl
        │
        ▼
Quality  CTR / CVR
```

Four nouns, then stop mixing them.

| Noun | What it is |
| --- | --- |
| **Event** | A training or incremental row on `POST /events`. It enters `[event_weights]` and LightFM. |
| **Impression** | The host says this SKU was rendered. `POST /track` with `kind` `impression`. Rank is required and ≥ 1. |
| **Click** | The host says this SKU was tapped. `POST /track` with `kind` `click`. CTR counts it only after a prior impression of the same `(user_id, item_id)` inside the window. |
| **Conversion** | An `[input]` row (default `purchase`) attributed to a prior impression (view-through) or a matched click (click-through). |

A purchase this afternoon can still be an **event** on the Stripe path. That trains. It is not an impression.

## Turn track on

`[track]` is off by default. `serve.log_impressions` is off by default. The second flag is a `ConfigError` without the first: `serve.log_impressions requires track.enabled = true`.

```toml
[track]
enabled = true
# attribution_window_hours = 24
# conversion_event_types = ["purchase"]
# min_impressions = 100
```

Put that table on the **serve** process that accepts `POST /track`. Put it on the **job** if tonight's run should write `track_eval`. Put it on the **dashboard** config if you want Quality to compute live before any eval file exists.

Storage sits next to `[output]`: a local `track.jsonl`, or a `recommendation_track` db table. Object-store JSONL append is refused (`track.enabled requires output kind = "db" or a local dataset path; object-store JSONL append is not atomic`). `events.ha = true` requires db output (`track.enabled with events.ha requires output kind = "db"`). A dataset writer lock can refuse the append with 503 (`Writer lock is busy` / `Writer lock was lost`).

Leave `log_impressions` off. Serve would write one impression per **returned** item after `limit` / `k` and `category`, on a FastAPI background task, with a fresh `uuid4` every GET. Retries are new rows. Prefetch is a row. That is the flag you turned on in the first paragraph.

## Post when Blade renders

The [nightly](/articles/a-nightly-table-next-to-your-orders/) `SELECT` does not need to become a GET. SQL-join shops that never call serve still `POST /track` when the widget renders. Laravel is the host here because the earlier posts already used Rails and Node.

Bearer only if `[serve].auth_token` is set. Same token as `GET /recommendations`. One object, an array, or `{"events":[...]}`. Bodies larger than 1 MiB (or `events.options.max_body_bytes`) return 413. Impression without `rank` is 400: `impression requires rank >= 1`. `occurred_at` needs a timezone (`Z` or an offset) or Unix epoch seconds.

```php
use Illuminate\Support\Facades\Http;

function trackEvents(array $events): void
{
    $url = rtrim(env('CICERONE_SERVE_URL'), '/').'/track';
    $request = Http::acceptJson()->asJson();
    $token = env('CICERONE_SERVE_TOKEN');
    if (is_string($token) && $token !== '') {
        $request = $request->withToken($token);
    }
    $response = $request->post($url, ['events' => $events]);
    if ($response->status() === 401 || $response->serverError()) {
        report(new \RuntimeException('Cicerone /track '.$response->status()));
    }
}

function impressionEvents(string $userId, iterable $rows, ?string $generatedAt = null): array
{
    $events = [];
    $rank = 1;
    foreach ($rows as $row) {
        $events[] = [
            'kind' => 'impression',
            'user_id' => $userId,
            'item_id' => (string) $row->item_id,
            'rank' => $rank,
            'occurred_at' => now()->utc()->format('Y-m-d\TH:i:s\Z'),
            'event_id' => 'imp:'.$userId.':'.$row->item_id.':'.$rank.':'.($generatedAt ?? ''),
            'variant' => $row->variant ?? null,
            'generated_at' => $generatedAt,
        ];
        $rank++;
    }
    return $events;
}
```

Call `trackEvents(impressionEvents(...))` from the HTML homepage action after you load the rows you are about to put on the page. Do not call it from a warmup job that only fetched. Do not call it from a JSON API the widget never painted.

`event_id` (or `idempotency_key`) is the idempotency id. Omit it and Cicerone hashes `kind`, `user_id`, `item_id`, `occurred_at`, `rank`, `variant`, `experiment_id`, `generated_at`. A retry with a new `now()` is a new row. Send a stable id if Laravel retries the HTTP call. A duplicate `event_id` comes back **202** with `accepted` 0 and `event_ids` `[]`.

`[track]` off means the route is not mounted. That is 404, not a silent drop.

## Post the tap

Clicks use the same path. Rank is optional. CTR still needs a prior impression of that `(user_id, item_id)`.

```php
public function click(Request $request)
{
    $data = $request->validate([
        'item_id' => ['required', 'string'],
        'user_id' => ['required', 'string'],
    ]);
    trackEvents([[
        'kind' => 'click',
        'user_id' => $data['user_id'],
        'item_id' => $data['item_id'],
        'occurred_at' => now()->utc()->format('Y-m-d\TH:i:s\Z'),
        'event_id' => 'clk:'.$data['user_id'].':'.$data['item_id'].':'.$request->session()->token(),
    ]]);
    return response()->noContent();
}
```

Do not `POST /events` with `event_type = "click"` unless you mean to train on it. Track rows never enter `[event_weights]`. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) mapper stays on `/events`. This mapper stays on `/track`.

## How Quality counts

Dashboard Quality reads `track_eval` when the last successful job with `[track]` wrote it, and labels that block "As of" the eval `generated_at`. POSTs you send after that job wait for the next snapshot. Before any eval file exists, a dashboard config with `[track]` computes live from the store and says "Live from the track store." Empty copy is "No impressions yet."

CTR is matched clicks divided by impressions. A click matches the latest earlier impression of the same `(user_id, item_id)` whose `occurred_at` delta is ≥ 0 and ≤ `attribution_window_hours` (default 24). A click with no such impression is stored. The ratio ignores it. Conversions are capped so they cannot exceed impressions. Quality also splits **CVR (click)** and **CVR (view)**, and can break down by rank, `source`, and `variant` once those columns are on the impression.

`min_impressions` is 100 **impression rows**. It gates experiment Promote for `ctr` / `conversion`. It does not hide Quality. Assignment and Promote are [The same customer keeps the same list](/articles/the-same-customer-keeps-the-same-list/).

The interval on Experiments is a mixture bound. Quality CTR is a matched-click ratio inside a wall-clock window. It is not a causal lift. Replay (`[job.eval]`) scores last night's rows against later `[input]` events (HitRate, MAP, NDCG, and the rest). That is not CTR. Do not quote a paper this binary does not implement.

| You did | What happens |
| --- | --- |
| `POST /track` from Blade | 202. New `event_id`s are written. Duplicates are skipped. |
| `POST /events` with a tap | Queued as an **event**. LightFM can train. Quality does not see a **click**. |
| `GET /recommendations` | Lookup. No impression unless `log_impressions` is on. |
| `log_impressions = true` | One impression per returned item, background task, new `uuid4` every GET. |
| Impression, no `rank` | 400 `impression requires rank >= 1`. |
| Click, no prior impression | Row stored. CTR numerator stays 0 for that tap. |
| Click before the impression | Delta < 0. No match. |
| Click 25h later | Outside the default window. No match. |
| Job with `[track]` writes `track_eval` | Quality is "As of" that stamp. Later POSTs wait. |
| No eval file, dashboard `[track]` on | Quality computes live. |
| Object-store output | Config load fails. Not atomic. |
| `[track]` off | `POST /track` is 404. |

## When this is the wrong tool

- You wanted PostHog, GA, or a pixel on the storefront. Different product. Cicerone never sees the DOM.
- You wanted clicks to train the ranker. That is `POST /events` with an `event_type` in `[event_weights]`, not `/track`.
- You wanted request-path ranking. Serve GET stays a lookup.
- You wanted GrowthBook exposures. One `[experiment]` per file; host `/track` is not that layer.

## After a week

Horizon still prefetches. Those GETs are still fetches. Blade still POSTs `/track` when the widget is on the page. A tap still POSTs a **click**. Quality still matches `(user_id, item_id)` inside the window.

Turn on `log_impressions` only if you mean every returned GET item to become an impression, including the ones no shopper saw. Change `attribution_window_hours` if you mean a different match window. Change `event_id` if you mean Laravel retries to stay one row.

A fetch is not an impression. The widget is.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [evaluation](/evaluation/).

**Normalize.** Required: `kind`, `user_id`, `item_id`, `occurred_at`. `kind` is `impression` or `click` (case folded). Impression requires `rank` ≥ 1. Click rank is optional. `occurred_at` is ISO-8601 with timezone or Unix epoch seconds. `event_id` or `idempotency_key` is optional; else a uuid5 of the fields above plus `variant` / `experiment_id` / `generated_at`.

**HTTP.** 202 writes new rows (`accepted` is that count). Duplicate `event_id` → `accepted` 0. 400 invalid payload. 401 missing or bad Bearer. 413 body too large (1 MiB default). 503 dataset writer lock busy or lost. `[track]` off → 404.

**Quality.** CTR = matched clicks / impressions. Match is latest prior impression, same `(user_id, item_id)`, 0 ≤ Δt ≤ window. CVR click / CVR view cap conversions at impressions. `min_impressions` default 100, Promote only. `[job.eval]` is replay, not CTR.

**Failures.**

| What you did | What happens |
| --- | --- |
| `log_impressions` without `[track]` | `ConfigError`: `serve.log_impressions requires track.enabled = true`. |
| Track on S3/object store | `ConfigError`: append is not atomic. |
| `events.ha` + local dataset | `ConfigError`: `track.enabled with events.ha requires output kind = "db"`. |
| Naive `now()` and no `event_id` | Every retry is a new row. |
| `log_impressions` GET retry | New `uuid4`. New row. |
| Stored `track_eval` after a job | Live Quality does not run. Wait for the next job. |

</details>

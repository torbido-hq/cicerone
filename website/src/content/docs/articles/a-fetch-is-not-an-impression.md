---
title: A fetch is not an impression
description: Cicerone records CTR from host-reported POST /track rows. GET /recommendations is a lookup. Quality matches a click to a prior (user, item) impression inside the window.
date: 2026-09-17
excerpt: log_impressions counts every returned GET item. The Blade widget POSTs /track when it renders. A tap sent to /events is not a Quality click.
authors:
  - nicholas
---

You want click-through on the homepage widget. You turn on `serve.log_impressions` and have Laravel `Http::get` the serve API. Horizon warms the same URL for last week's buyers. A health check hits it too. You did not measure CTR. You measured fetches.

Cicerone cannot see Blade. `GET /recommendations/{user_id}` **returns** a list. `SELECT … ORDER BY rank` **returns** a list. An impression is a `/track` row that says a SKU was shown. The host has to send it.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough already said `source` only tells you which list won. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) post already said impressions and clicks are `POST /track`, not `POST /events`. This is that contract.

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
| **Event** | A training or incremental row on `POST /events` (route mounted only when `[events]` is on and `kind = "webhook"`). It is not a Quality click. Unknown `event_type`s never enter `[event_weights]`. |
| **Impression** | A `/track` row with `kind` `impression`. Rank is required and ≥ 1. Quality counts the row. Whether anyone saw the SKU is the host's claim. |
| **Click** | A `/track` row with `kind` `click`. CTR counts it only after a prior impression of the same `(user_id, item_id)` inside the window. |
| **Conversion** | An `[input]` row (default type `purchase`) that Quality attributes to a prior impression (view-through) or a matched click (click-through). It remains an input event. Quality's name for the attribution is conversion. |

A purchase on the Stripe path can still be an **event**. That is training or write-through. It is not an impression.

## Returned is not rendered

`[track]` is off by default. `serve.log_impressions` is off by default. The second flag is a `ConfigError` without the first: `serve.log_impressions requires track.enabled = true`.

```toml
[track]
enabled = true
# attribution_window_hours = 24
# conversion_event_types = ["purchase"]
# min_impressions = 100
```

The serve process that accepts `POST /track` needs that table. A SQL-join shop can skip GET. It cannot skip serve if it wants `/track`. Put `[track]` on the job if tonight's run should write `track_eval`. Put it on the dashboard config if you want Quality to compute live before any eval file exists.

Leave `log_impressions` off. Serve would write one impression row per **returned** item after `limit` / `k` and `category`, on a FastAPI background task, with a fresh `uuid4` every GET. The GET can already be 200. Prefetch is a row. A retry is a new row. Items the widget never painted are rows. That is the flag you turned on in the first paragraph.

Storage, HA, and lock errors are in Reference.

## Post when Blade renders

The [nightly](/articles/a-nightly-table-next-to-your-orders/) `SELECT` does not need to become a GET. Laravel is the host here because the earlier posts already used Rails and Node.

Bearer only if `[serve].auth_token` is set. Same token as GET. One object, an array, or `{"events":[...]}` — that HTTP field is named `events`; the rows are still track rows, not **events**. Impression without `rank` is 400: `impression requires rank >= 1`. `occurred_at` needs a timezone (`Z` or an offset) or Unix epoch seconds.

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
    if (!$response->successful()) {
        report(new \RuntimeException('Cicerone /track '.$response->status()));
    }
}

function impressionEvents(string $userId, iterable $rows, string $occurredAt): array
{
    $events = [];
    $rank = 1;
    foreach ($rows as $row) {
        $events[] = [
            'kind' => 'impression',
            'user_id' => $userId,
            'item_id' => (string) $row->item_id,
            'rank' => $rank,
            'occurred_at' => $occurredAt,
            'event_id' => 'imp:'.$userId.':'.$row->item_id.':'.$rank.':'.$occurredAt,
            'variant' => $row->variant ?? null,
        ];
        $rank++;
    }
    return $events;
}
```

Stamp `$occurredAt` once for the render (`now()->utc()->format('Y-m-d\TH:i:s\Z')`). Pass **only** the rows Blade is about to paint. A retry of that same array keeps the same `event_id`s. A later page view gets a new stamp and new rows. An `event_id` that omits the stamp collapses every repeat view of last night's list into one impression.

`event_id` (or `idempotency_key`) is the idempotency id. Omit it and Cicerone hashes `kind`, `user_id`, `item_id`, `occurred_at`, `rank`, `variant`, `experiment_id`, `generated_at`. A retry that rebuilds `now()` is a new row. A duplicate `event_id` comes back **202** with `accepted` 0 and `event_ids` `[]`. That is a successful retry, not a failed render. Report every non-2xx. The homepage still renders.

`[track]` off means the route is not mounted. That is 404, not a silent drop.

## Post the tap

Clicks use the same path. Rank is optional. CTR still needs a prior impression of that `(user_id, item_id)`.

```php
public function click(Request $request)
{
    $data = $request->validate([
        'item_id' => ['required', 'string'],
        'user_id' => ['required', 'string'],
        'event_id' => ['required', 'string'],
    ]);
    $occurredAt = now()->utc()->format('Y-m-d\TH:i:s\Z');
    trackEvents([[
        'kind' => 'click',
        'user_id' => $data['user_id'],
        'item_id' => $data['item_id'],
        'occurred_at' => $occurredAt,
        'event_id' => $data['event_id'],
    ]]);
    return response()->noContent();
}
```

Mint `event_id` once at tap time (`crypto.randomUUID()` is enough). Send that same id if the browser retries this Laravel route. A second tap mints a second id. Do not build the id in the controller from `now()` — that request is a new row every retry.

Do not `POST /events` with `event_type = "click"` unless you mean a training **event**. Track rows never enter `[event_weights]`. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) mapper stays on `/events`. This mapper stays on `/track`.

## How Quality counts

Dashboard Quality reads `track_eval` when the last successful job with `[track]` wrote it, and labels that block "As of" the eval `generated_at`. POSTs you send after that job wait for the next snapshot. Before any eval file exists, a dashboard config with `[track]` computes live from the store and says "Live from the track store." Empty copy is "No impressions yet."

CTR is **matched click rows** divided by **impression rows**. A click matches the latest earlier impression of the same `(user_id, item_id)` whose `occurred_at` delta is ≥ 0 and ≤ `attribution_window_hours` (default 24). Rank is not in the join. A click with no such impression is stored. The ratio ignores it. Two matched clicks on one impression count as two. CTR is not capped at 1. Conversions **are** capped so they cannot exceed impressions. Quality splits **CVR (click)** and **CVR (view)** — those labels are the two attribution paths, not a third noun.

Quality CTR is a matched-click ratio inside a wall-clock window. It is not causal lift. Replay (`[job.eval]`) is HitRate and friends on last night's lists against later `[input]` events. That is not CTR.

| You did | What happens |
| --- | --- |
| `POST /track` from Blade | 202. New `event_id`s are written. Duplicates are skipped. |
| `POST /events` with a tap | Only if the webhook is mounted: ingested as an **event**. Quality does not see a **click**. Otherwise 404. |
| `GET /recommendations` | Lookup. No impression row unless `log_impressions` is on. |
| `log_impressions = true` | One impression row per returned item, background task, new `uuid4` every GET. 200 does not mean the write landed. |
| Impression, no `rank` | 400 `impression requires rank >= 1`. |
| Click, no prior impression | Row stored. That tap is not in the CTR numerator. |
| Click before the impression | Delta < 0. No match. |
| Click 25h later | Outside the default window. No match. |
| Job with `[track]` writes `track_eval` | Quality is "As of" that stamp. Later POSTs wait. Live does not run. |
| No eval file, dashboard `[track]` on | Quality computes live. |
| `[track]` off | `POST /track` is 404. |

## When this is the wrong tool

- You wanted PostHog, GA, or a pixel on the storefront. Different product. Cicerone never sees the DOM.
- You wanted taps to train the ranker. That is `POST /events` with an `event_type` in `[event_weights]`, not `/track`.
- You wanted request-path ranking. Serve GET stays a lookup.

## After a week

Horizon still prefetches. Those GETs are still fetches. Blade still POSTs `/track` when the widget is on the page. A tap still POSTs a **click**. Quality still matches `(user_id, item_id)` inside the window.

Turn on `log_impressions` only if you mean every **returned** GET item to become an impression row, including the ones no shopper saw. Change `attribution_window_hours` if you mean a different match window. Change `event_id` if you mean Laravel retries to stay one row.

A fetch is not an impression. The POST from the widget is.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [evaluation](/evaluation/).

**Normalize.** Required: `kind`, `user_id`, `item_id`, `occurred_at`. `kind` is `impression` or `click` (case folded). Impression requires `rank` ≥ 1. Click rank is optional. `occurred_at` is ISO-8601 with timezone or Unix epoch seconds. `event_id` or `idempotency_key` is optional; else a uuid5 of those fields plus `variant` / `experiment_id` / `generated_at`.

**HTTP.** 202 writes new rows (`accepted` is that count). Duplicate `event_id` → `accepted` 0. 400 invalid payload. 401 missing or bad Bearer. 413 body too large (1 MiB default, or `events.options.max_body_bytes`). 503 dataset writer lock (`Writer lock is busy` / `Writer lock was lost`). `[track]` off → 404.

**Storage.** Next to `[output]`: local `track.jsonl`, or `recommendation_track` on db. Object-store JSONL append is refused (`track.enabled requires output kind = "db" or a local dataset path; object-store JSONL append is not atomic`). `events.ha = true` requires db (`track.enabled with events.ha requires output kind = "db"`).

**Quality.** CTR = matched click rows / impression rows (uncapped). Match is latest prior impression, same `(user_id, item_id)`, 0 ≤ Δt ≤ window. Rank is not in the join. `source` / `variant` breakdowns use columns on the impression, filled from the POST when present or joined from recommendation rows. CVR click / CVR view cap conversions at impressions. `min_impressions` default 100 gates experiment Promote for `ctr` / `conversion`; it does not hide Quality. `[job.eval]` is replay, not CTR. Experiments CIs are a mixture bound; that page is [The same customer keeps the same list](/articles/the-same-customer-keeps-the-same-list/).

**Failures.**

| What you did | What happens |
| --- | --- |
| `log_impressions` without `[track]` | `ConfigError`: `serve.log_impressions requires track.enabled = true`. |
| Track on S3/object store | `ConfigError`: append is not atomic. |
| `events.ha` + local dataset | `ConfigError`: `track.enabled with events.ha requires output kind = "db"`. |
| Naive `now()` and no `event_id` | Every retry is a new row. |
| `event_id` without a render stamp | Repeat views of the same list share one impression. |
| `log_impressions` GET retry | New `uuid4`. New row. |
| Stored `track_eval` after a job | Live Quality does not run. Wait for the next job. |

</details>

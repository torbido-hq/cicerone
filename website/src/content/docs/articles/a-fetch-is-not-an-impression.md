---
title: A fetch is not an impression
description: "GET /recommendations is a lookup. Quality CTR counts impression rows: host POST /track, or serve.log_impressions on returned GET items. A click matches a prior (user, item) inside the window."
date: 2026-09-17
excerpt: log_impressions counts every returned GET item. The Blade widget POSTs /track when it renders. A tap sent to /events is not a Quality click.
authors:
  - nicholas
---

You want click-through on the homepage widget. You turn on `serve.log_impressions` and have Laravel `Http::get` the serve API. Horizon warms the same URL for last week's buyers. A health check hits it too. You did not measure what shoppers saw. You turned fetches into impression rows.

`GET /recommendations/{user_id}` **returns** a list. `SELECT … ORDER BY rank` **returns** a list. Prefetch returns the same list. A health check returns it again. None of those prove a shopper saw a SKU.

Cicerone cannot see Blade. A **host-reported impression** is a `/track` row the host sends after it paints the widget. The host has to report it.

```text
recommendation lookup
        │
        ├─ SELECT … ORDER BY rank
        │     or GET /recommendations/{user_id}
        ▼
items the host will render
        │
        ▼
Blade renders the widget
        │
        ▼
POST /track   kind=impression
        │
user taps a SKU
        │
        ▼
POST /track   kind=click
        │
        ▼
Quality
```

## Returned is not rendered

Four facts, then stop collapsing them.

| What happened | What Cicerone has |
| --- | --- |
| Serve **returned** items on GET (after `limit` / `k` and `category`) | A lookup. Not a host-reported impression. |
| Blade **rendered** those items on the homepage | Still no host-reported impression. Cicerone did not see the paint. |
| Host `POST /track` with `kind` `impression` | An impression row. Quality counts it. The host claimed the SKU was shown. |
| `serve.log_impressions = true` | Serve writes an impression row per **returned** GET item, on a background task. New `uuid4` per item, every GET. Not render-aware. |

`log_impressions` turns returned GET items into impression rows. It is not render-aware. It does not wait for Blade. Prefetch, a GET retry, and items the widget never painted are still rows. The GET can already be **200** before the write runs. Leave the flag off unless you mean that.

That is not “the shopper looked at rank 3.” Cicerone never sees the DOM. A host-reported impression is the host saying it showed the SKU.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough already said `source` only tells you which list won. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) post already said impressions and clicks are `POST /track`, not `POST /events`.

## Four nouns

| Noun | Wire | What it is |
| --- | --- | --- |
| **Event** | `POST /events` | Training or incremental row. Not a Quality click. |
| **Impression** | `kind=impression` (host `/track` or `log_impressions`) | A stored `kind=impression` row (host POST or `log_impressions`). Rank ≥ 1. Quality counts it. |
| **Click** | `POST /track` `kind=click` | Host-reported tap. CTR needs a matching impression. |
| **Conversion** | `[input]` + Quality | An `[input]` row (default type `purchase`) attributed to a prior impression or a matched click. Attribution result. Still an input **event**. |

`POST /events` exists only when `[events]` is on and `kind = "webhook"`. `event_type`s missing from `[event_weights]` are not trained. A purchase on the Stripe path can still be an **event**. That is not an impression and not a Quality click.

The `/track` JSON field is named `events`. Those objects are track rows.

## Turn `/track` on

`[track]` is off by default. `serve.log_impressions` is off by default. The flag is a `ConfigError` without the table: `serve.log_impressions requires track.enabled = true`.

```toml
[track]
enabled = true
# attribution_window_hours = 24
# conversion_event_types = ["purchase"]
# min_impressions = 100
```

The serve process that accepts `POST /track` needs that table. A SQL-join shop can skip GET. It cannot skip serve if it wants `/track`. Put `[track]` on the job if tonight's run should write `track_eval`. Put it on the dashboard config to show Quality. Snapshot versus live is in Reference.

`[track]` off means the route is not mounted. That is 404, not a silent drop. HTTP, storage, `events.ha`, and lock errors are in Reference.

## Report the impression

Laravel is the host here because the earlier posts already used Rails and Node. The [nightly](/articles/a-nightly-table-next-to-your-orders/) `SELECT` does not need to become a GET.

Stamp **one** render time. Build the list from the rows Blade will paint. Retry that same POST. Do not stamp again.

```php
use Illuminate\Support\Facades\Http;

function postTrack(array $rows): void
{
    $url = rtrim(env('CICERONE_SERVE_URL'), '/').'/track';
    $request = Http::acceptJson()->asJson();
    $token = env('CICERONE_SERVE_TOKEN');
    if (is_string($token) && $token !== '') {
        $request = $request->withToken($token);
    }
    $response = $request->post($url, ['events' => $rows]);
    if (!$response->successful()) {
        report(new \RuntimeException('Cicerone /track '.$response->status()));
    }
}

function impressionRows(string $userId, iterable $painted, string $occurredAt): array
{
    $rows = [];
    $rank = 1;
    foreach ($painted as $row) {
        $rows[] = [
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
    return $rows;
}

$occurredAt = now()->utc()->format('Y-m-d\TH:i:s\Z');
postTrack(impressionRows((string) $user->id, $painted, $occurredAt));
```

`$painted` is what Blade echoes, not the full `SELECT`. The same `$occurredAt` sits on every row of this render, and inside every `event_id`. A retry of this array is the same ids. A later page view stamps a new `$occurredAt` and writes new rows. Omit the stamp from that constructed `event_id` and every repeat view of last night's list collapses to one impression. Omit `event_id` entirely and Cicerone hashes `occurred_at` into a uuid5; a new stamp is a new row.

`event_id` (or `idempotency_key`) is the idempotency id. Omit it and Cicerone hashes `kind`, `user_id`, `item_id`, `occurred_at`, `rank`, `variant`, `experiment_id`, `generated_at`. Rebuild `now()` on retry and you get a new row. A duplicate `event_id` comes back **202** with `accepted` 0 and `event_ids` `[]`. That is a successful retry, not a failed render. Report every non-2xx. The homepage still renders.

## Report the click

Same path. Rank is optional. CTR still needs a prior impression of that `(user_id, item_id)`.

Mint the id **once**, in the browser, at tap time. POST it to Laravel. Do not POST `/track` from the browser.

| What | `event_id` |
| --- | --- |
| One tap | One `crypto.randomUUID()` |
| Retry of that request | The same id |
| A second tap | A new id |

```js
const eventId = crypto.randomUUID();
await fetch('/cicerone/click', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ item_id, user_id, event_id: eventId }),
});
// retry this request with the same eventId
```

```php
public function click(Request $request)
{
    $data = $request->validate([
        'item_id' => ['required', 'string'],
        'user_id' => ['required', 'string'],
        'event_id' => ['required', 'string'],
    ]);
    $occurredAt = now()->utc()->format('Y-m-d\TH:i:s\Z');
    postTrack([[
        'kind' => 'click',
        'user_id' => $data['user_id'],
        'item_id' => $data['item_id'],
        'occurred_at' => $occurredAt,
        'event_id' => $data['event_id'],
    ]]);
    return response()->noContent();
}
```

This example is best-effort: `postTrack` reports a non-2xx; the click action still returns 204. Do not mint `event_id` from `now()` in the controller. That makes every retry a new row.

Do not `POST /events` with `event_type = "click"` unless you mean a training **event**. Track rows never enter `[event_weights]`. The checkout mapper stays on `/events`. This mapper stays on `/track`.

## How Quality counts

CTR is **matched click rows / impression rows**.

- A click matches the **latest earlier** impression of the same `(user_id, item_id)`.
- The join is `occurred_at`. Rank is not in the join.
- The delta must be `0 ≤ Δt ≤ attribution_window_hours` (default 24).
- A click with no such impression is stored. It does not enter the numerator.
- Two matched clicks on one impression count as two. CTR is not capped at 1.

A **conversion** is an `[input]` row Quality attributes to a prior impression (view-through) or a matched click (click-through). Click-through joins the conversion to that matched click, then applies the same window from the click. Conversions **are** capped at impressions. **CVR (click)** and **CVR (view)** are those two attribution paths, not a third noun. Both use impression rows as the denominator.

Quality CTR is a matched-click ratio inside a wall-clock window. It is not causal lift. `[job.eval]` replay is HitRate and friends on last night's lists against later `[input]` events. That is not CTR.

## When this is the wrong tool

- You wanted PostHog, GA, or a pixel on the storefront. Different product. Cicerone never sees the DOM.
- You wanted taps to train the ranker. That is `POST /events` with an `event_type` in `[event_weights]`, not `/track`.
- You wanted request-path ranking. Serve GET stays a lookup.

## After a week

Horizon still prefetches. Blade still POSTs `/track` when the widget is on the page. A tap still POSTs a **click**. Quality still matches `(user_id, item_id)` inside the window.

A fetch is not proof of an impression. The POST from the widget is the host's report that it rendered one.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [evaluation](/evaluation/).

**Normalize.** Required: `kind`, `user_id`, `item_id`, `occurred_at`. `kind` is `impression` or `click` (case folded). Impression requires `rank` ≥ 1. Click rank is optional. `occurred_at` is ISO-8601 with timezone (`Z` or an offset) or Unix epoch seconds. `event_id` or `idempotency_key` is optional; else a uuid5 of those fields plus `variant` / `experiment_id` / `generated_at`. Impression without `rank` is 400: `impression requires rank >= 1`.

**HTTP.** Bearer only if `[serve].auth_token` is set. Same token as GET. Body is one object, an array, or `{"events":[...]}`. 202 writes new rows (`accepted` is that count). Duplicate `event_id` → `accepted` 0. 400 invalid payload. 401 missing or bad Bearer. 413 body too large (1 MiB default, or `events.options.max_body_bytes`). 503 dataset writer lock (`Writer lock is busy` / `Writer lock was lost`). `[track]` off → 404. The example `postTrack` only reports a non-2xx; the Laravel click action still returns 204.

**Storage.** Next to `[output]`: local `track.jsonl`, or `recommendation_track` on db. Object-store JSONL append is refused (`track.enabled requires output kind = "db" or a local dataset path; object-store JSONL append is not atomic`). `events.ha = true` requires db (`track.enabled with events.ha requires output kind = "db"`).

**Quality.** CTR = matched click rows / impression rows (uncapped). Match is latest prior impression, same `(user_id, item_id)`, 0 ≤ Δt ≤ window. Rank is not in the join. `source` / `variant` breakdowns use columns on the impression, filled from the POST when present or joined from recommendation rows. CVR click / CVR view cap conversions at impressions; both denominators are impressions. `min_impressions` default 100 gates experiment Promote for `ctr` / `conversion`; it does not hide Quality. `[job.eval]` is replay, not CTR. Experiments CIs are a mixture bound; that page is [The same customer keeps the same list](/articles/the-same-customer-keeps-the-same-list/). Dashboard `[track]` off hides a track-only snapshot when `served_eval` is missing. Default conversion type is `purchase` when `conversion_event_types` is empty and `primary_metric` is `weighted`, `ctr`, or `conversion`; otherwise it is `primary_metric`.

**Common mistakes.**

| You did | What happens |
| --- | --- |
| `POST /track` from Blade | 202. New `event_id`s are written. Duplicates are skipped. |
| `POST /events` with a tap | Only if the webhook is mounted: ingested as an **event**. Quality does not see a **click**. Otherwise 404. |
| `GET /recommendations` | Lookup. No impression row unless `log_impressions` is on. |
| `log_impressions = true` | One impression row per returned item, background task, new `uuid4` per item every GET. 200 does not mean the write landed. |
| Impression, no `rank` | 400 `impression requires rank >= 1`. |
| Click, no prior impression | Row stored. That tap is not in the CTR numerator. |
| Click before that impression | No match to that later row. An older impression in the window can still match. |
| Click 25h later | Outside the default window. No match. |
| Job with `[track]` writes a `track_eval` object | Live does not run. "As of" is the previous recommendations `generated_at`. Later POSTs wait. |
| No `track_eval` object, dashboard `[track]` on | Quality computes live. |
| Dashboard `[track]` off, no `served_eval` | "No impressions yet." even if a snapshot file exists. |
| `[track]` off on serve | `POST /track` is 404. |

**Failures.**

| What you did | What happens |
| --- | --- |
| `log_impressions` without `[track]` | `ConfigError`: `serve.log_impressions requires track.enabled = true`. |
| Track on S3/object store | `ConfigError`: append is not atomic. |
| `events.ha` + local dataset | `ConfigError`: `track.enabled with events.ha requires output kind = "db"`. |
| Naive `now()` and no `event_id` | Every retry is a new row. |
| Constructed `event_id` without a render stamp | Repeat views of the same list share one impression. |
| `log_impressions` GET retry | New `uuid4` per returned item. New rows. |
| Stored `track_eval` object after a job | Live Quality does not run. Wait for the next job. |

</details>

---
title: A fetch is not an impression
description: Cicerone records CTR from host-reported POST /track rows. GET /recommendations is a lookup. Quality matches a click to a prior (user, item) impression inside the window.
date: 2026-09-17
excerpt: log_impressions counts every returned GET item. The Blade widget POSTs /track when it renders. A tap sent to /events is not a Quality click.
authors:
  - nicholas
---

You want click-through on the homepage widget. You turn on `serve.log_impressions` and have Laravel `Http::get` the serve API. Horizon warms the same URL for last week's buyers. A health check hits it too. You did not measure CTR. You measured fetches.

`GET /recommendations/{user_id}` **returns** a list. `SELECT … ORDER BY rank` **returns** a list. Prefetch returns the same list. A health check returns it again. None of those prove a shopper saw a SKU.

Cicerone cannot see Blade. An impression is a `/track` row the host sends after it paints the widget. The host has to report it.

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

## Returned is not rendered

Four facts, then stop collapsing them.

| What happened | What Cicerone has |
| --- | --- |
| Serve **returned** items on GET (after `limit` / `k` and `category`) | A lookup. Not an impression. |
| Blade **rendered** those items on the homepage | Still nothing, until the host POSTs. |
| Host `POST /track` with `kind` `impression` | An impression row. Quality counts it. The host claimed the SKU was shown. |
| `serve.log_impressions = true` | Serve writes an impression row per **returned** GET item, on a background task, new `uuid4` every GET. |

`log_impressions` is a fetch logger. It does not wait for Blade. Prefetch is a row. A GET retry is a new row. Items the widget never painted are rows. The GET can already be **200** before the write runs.

That is not “the shopper looked at rank 3.” Cicerone never sees the DOM. A host-reported impression is the host saying it showed the SKU. Leave the flag off unless you mean every returned GET item to become that row.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough already said `source` only tells you which list won. The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) post already said impressions and clicks are `POST /track`, not `POST /events`.

## Four nouns

| Noun | Wire | What it is |
| --- | --- | --- |
| **Event** | `POST /events` | Training or incremental row. Route exists only when `[events]` is on and `kind = "webhook"`. Unknown `event_type`s never enter `[event_weights]`. Not a Quality click. |
| **Impression** | `POST /track` `kind=impression` | Host-reported “this SKU was shown.” Rank is required and ≥ 1. Quality counts the row. |
| **Click** | `POST /track` `kind=click` | Host-reported tap. CTR counts it only after a matching impression. |
| **Conversion** | `[input]` + Quality | An `[input]` row (default type `purchase`) attributed to a prior impression (view-through) or a matched click (click-through). Still an input **event**. Conversion is the attribution result. |

A purchase on the Stripe path can still be an **event**. That is training or write-through. It is not an impression.

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

The serve process that accepts `POST /track` needs that table. A SQL-join shop can skip GET. It cannot skip serve if it wants `/track`. Put `[track]` on the job if tonight's run should write `track_eval`. Put it on the dashboard config if you want Quality to compute live before any eval file exists.

`[track]` off means the route is not mounted. That is 404, not a silent drop. Storage, HA, and lock errors are in Reference.

## Report the impression

Laravel is the host here because the earlier posts already used Rails and Node. The [nightly](/articles/a-nightly-table-next-to-your-orders/) `SELECT` does not need to become a GET.

Bearer only if `[serve].auth_token` is set. Same token as GET. One object, an array, or `{"events":[...]}`. Impression without `rank` is 400: `impression requires rank >= 1`. `occurred_at` needs a timezone (`Z` or an offset) or Unix epoch seconds.

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

`$painted` is what Blade echoes, not the full `SELECT`. The same `$occurredAt` is on every row of this render, and inside every `event_id`. A retry of this array is the same ids. A later page view stamps a new `$occurredAt` and writes new rows. Omit the stamp from `event_id` and every repeat view of last night's list collapses to one impression.

`event_id` (or `idempotency_key`) is the idempotency id. Omit it and Cicerone hashes `kind`, `user_id`, `item_id`, `occurred_at`, `rank`, `variant`, `experiment_id`, `generated_at`. Rebuild `now()` on retry and you get a new row. A duplicate `event_id` comes back **202** with `accepted` 0 and `event_ids` `[]`. That is a successful retry, not a failed render. Report every non-2xx. The homepage still renders.

## Report the click

Same path. Rank is optional. CTR still needs a prior impression of that `(user_id, item_id)`.

Mint the id **once**, in the browser, at tap time. Reuse it if this Laravel route is retried. A second tap mints a second id.

```js
const eventId = crypto.randomUUID();
// POST { item_id, user_id, event_id: eventId } — send the same eventId on retry
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

Do not build `event_id` in the controller from `now()`. That request is a new row on every retry.

Do not `POST /events` with `event_type = "click"` unless you mean a training **event**. Track rows never enter `[event_weights]`. The checkout mapper stays on `/events`. This mapper stays on `/track`.

## How Quality counts

Dashboard Quality reads `track_eval` when the last successful job with `[track]` wrote it, and labels that block "As of" the eval `generated_at`. POSTs after that job wait for the next snapshot. Live Quality does not run once that file exists. Before any eval file exists, a dashboard config with `[track]` computes live from the store ("Live from the track store."). Empty copy is "No impressions yet."

CTR is **matched click rows / impression rows**.

- A click matches the **latest earlier** impression of the same `(user_id, item_id)`.
- The join is `occurred_at`. Rank is not in the join.
- The delta must be `0 ≤ Δt ≤ attribution_window_hours` (default 24).
- A click with no such impression is stored. It does not enter the numerator.
- Two matched clicks on one impression count as two. CTR is not capped at 1.

A **conversion** is an `[input]` row Quality attributes to a prior impression (view-through) or a matched click (click-through). Conversions **are** capped at impressions. **CVR (click)** and **CVR (view)** are those two attribution paths, not a third noun.

Quality CTR is a matched-click ratio inside a wall-clock window. It is not causal lift. `[job.eval]` replay is HitRate and friends on last night's lists against later `[input]` events. That is not CTR.

## When this is the wrong tool

- You wanted PostHog, GA, or a pixel on the storefront. Different product. Cicerone never sees the DOM.
- You wanted taps to train the ranker. That is `POST /events` with an `event_type` in `[event_weights]`, not `/track`.
- You wanted request-path ranking. Serve GET stays a lookup.

## After a week

Horizon still prefetches. Those GETs are still fetches. Blade still POSTs `/track` when the widget is on the page. A tap still POSTs a **click**. Quality still matches `(user_id, item_id)` inside the window.

A fetch is not an impression. The POST from the widget is.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [evaluation](/evaluation/).

**Normalize.** Required: `kind`, `user_id`, `item_id`, `occurred_at`. `kind` is `impression` or `click` (case folded). Impression requires `rank` ≥ 1. Click rank is optional. `occurred_at` is ISO-8601 with timezone or Unix epoch seconds. `event_id` or `idempotency_key` is optional; else a uuid5 of those fields plus `variant` / `experiment_id` / `generated_at`.

**HTTP.** 202 writes new rows (`accepted` is that count). Duplicate `event_id` → `accepted` 0. 400 invalid payload. 401 missing or bad Bearer. 413 body too large (1 MiB default, or `events.options.max_body_bytes`). 503 dataset writer lock (`Writer lock is busy` / `Writer lock was lost`). `[track]` off → 404.

**Storage.** Next to `[output]`: local `track.jsonl`, or `recommendation_track` on db. Object-store JSONL append is refused (`track.enabled requires output kind = "db" or a local dataset path; object-store JSONL append is not atomic`). `events.ha = true` requires db (`track.enabled with events.ha requires output kind = "db"`).

**Quality.** CTR = matched click rows / impression rows (uncapped). Match is latest prior impression, same `(user_id, item_id)`, 0 ≤ Δt ≤ window. Rank is not in the join. `source` / `variant` breakdowns use columns on the impression, filled from the POST when present or joined from recommendation rows. CVR click / CVR view cap conversions at impressions. `min_impressions` default 100 gates experiment Promote for `ctr` / `conversion`; it does not hide Quality. `[job.eval]` is replay, not CTR. Experiments CIs are a mixture bound; that page is [The same customer keeps the same list](/articles/the-same-customer-keeps-the-same-list/).

**Common mistakes.**

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

---
title: This afternoon's checkout can move the row
description: Stripe webhook to Cicerone 0.7 online LightFM. Known-ID purchases write through. GET stays a lookup.
date: 2026-08-28
lastUpdated: 2026-08-29
excerpt: A paid Checkout can update an existing recommendation row after a flush. LightFM rewrite still needs known IDs. GET stays a lookup.
authors:
  - nicholas
---

A paid Stripe Checkout can update an **existing** recommendation row. The checkout request never loads a model. The update is asynchronous: Node maps the webhook, `POST`s Cicerone's event contract, and Cicerone holds the purchase in an in-memory queue until a micro-batch flush. That is write-through recommendation state, not request-path ranking. `GET /recommendations` stays a lookup.

[Cicerone](https://cicerone.dev) 0.7 is the job that already fitted [LightFM](https://making.lyst.com/lightfm/docs/home.html) and wrote the table. The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough leaves personalized ranks until 03:00 UTC; that remains the right default. This article is the optional `[events.online]` path: Node verifies the Stripe signature and `POST`s Cicerone's event contract. There is no recommendations SDK.

Skip this path when:

- The row must exist before the checkout response returns. Flush is a queue window (default 60s), not request-path inference.
- The SKUs that matter were listed today. Unknown ids wait for `job.run()`.
- Stripe retries must not add a second purchase. Persist Stripe's `event.id` yourself ([snippet](#persist-stripe-eventid)). That stops ordinary duplicate deliveries; it is not exactly-once across a lost Cicerone response. Cicerone 0.7's in-memory queue is not a durable purchase key.

## What changes

After flush, for users in that micro-batch, in this order:

- popular / latest (and `'__cold_start__'`) can **rewrite**
- if both ids were already in the last job artifact, `[events.online]` is on, and `[experiment]` is off, Cicerone 0.7 can **re-score** those users against the extra interactions while LightFM weights stay frozen
- once `fit_min_events` known-ID events have piled up since the last `fit_partial`, Cicerone can run LightFM `fit_partial` (a **model update**) and **rewrite** affected users again

Flush does not imply `fit_partial`. A re-score is not a model update. The homepage can keep doing `SELECT … ORDER BY rank`. Serve `GET /recommendations/{user_id}` is the same rows.

## What does not change

- This is not live ranking. Serve never loads a model on `GET`.
- 202 means queued, not applied. Flush does not imply `fit_partial`. `fit_partial` does not rewrite every user.
- New user ids and new SKUs do not enter LightFM until `job.run()` writes a new artifact.
- Sequential strategies stay batch. Embedding tables do not grow online.
- Tonight's job remains authoritative: it refits from the full event log and writes a new artifact.
- Cicerone does not durably dedupe Stripe events ([snippet](#persist-stripe-eventid)).

## Architecture

```text
Stripe Checkout
      ↓
webhook verification
      ↓
map Stripe → Cicerone event
      ↓
POST /events
      ↓
202 / in-memory queue (not applied)
      ↓
micro-batch flush
      ↓
popular/latest rewrite
      ↓
conditional frozen re-score
      ↓
conditional LightFM fit_partial
      ↓
rewrite affected users
      ↓
existing recommendation table
      ↓
GET /recommendations
```

![Stripe to queue to flush to optional frozen re-score to optional LightFM to GET lookup. 202 is queued, not applied. The webhook queue is in-memory. Durable webhook idempotency is the application's Stripe event.id.](/images/afternoon-checkout-architecture.png)

A purchase this afternoon can move a row that already exists. A SKU you listed today still waits for `job.run()` unless that id was already in the artifact. `202` means the events are in the in-memory queue, not that the row has changed.

## Stripe payment semantics

`checkout.session.completed` means the Checkout Session completed. On cards and other immediate methods, that event usually arrives with `payment_status === "paid"` — this handler treats that as a successful payment. With delayed methods (SEPA debit, some bank redirects, ACH), Checkout can complete while payment is still pending (`payment_status === "unpaid"`). Stripe sends `checkout.session.async_payment_succeeded` later if that payment succeeds, or `checkout.session.async_payment_failed` if it fails.

If you map every `completed` as a `purchase`, a delayed payment that later fails can still enter the event stream as a purchase. The handler:

- accepts `checkout.session.completed` **only** when `session.payment_status === "paid"`
- accepts `checkout.session.async_payment_succeeded`
- does **not** turn `async_payment_failed` into a purchase

## Stripe → Cicerone event contract

Cicerone IDs are yours. They are not automatically Stripe `cus_` / `prod_` / `price_` values.

| Cicerone | Stripe | Where you set it |
| --- | --- | --- |
| `user_id` | `client_reference_id` | API call: `checkout.sessions.create` |
| `item_id` | Product `metadata.item_id`, else Price `lookup_key` | Dashboard / API call: `products.create` / `prices.create` |
| `event_type` | `"purchase"` | Mapper: hard-coded (must exist in `[event_weights]`) |
| `occurred_at` | `session.created` on paid `completed`; `event.created` on async success | Mapper: Unix seconds (Cicerone 0.7 accepts that) |
| `quantity` | line-item `quantity` | API call: `listLineItems` (`quantity: 5` stays one event) |
| `event_id` | `{event.id}:{item_id}` | Mapper: per-line Cicerone id, not the durable Stripe webhook key |

Whichever catalog string you emit must be the exact `item_id` in the last LightFM artifact. `prod_…` and `price_…` are not aliases for that string. Examples below use `sku-42` on purpose.

If `client_reference_id` is missing, skip the session — do not invent a Stripe customer id. If a line has neither `metadata.item_id` nor `lookup_key`, skip that line. Posting `prod_xxx` against a job that fitted `sku-42` is a silent unknown-id drop: popular and latest can still move; LightFM does not.

`user.id` must be the same string the batch job reads from your orders. Integer primary keys become text in Cicerone; send `String(user.id)` on both paths.

## Durable idempotency

Three different things:

| Layer | What it does | What it does not do |
| --- | --- | --- |
| Stripe delivery retries | Stripe reuses `event.id` across retries of the same webhook. That string is the durable **webhook-level** key you persist. | Does not mean Cicerone applied the purchase only once |
| Cicerone 0.7 webhook queue | Drops a duplicate Cicerone `event_id` (`${stripeEventId}:${itemId}`) while that per-line id is still **pending or in-flight** in this process (current in-memory implementation, not a durable API guarantee) | After flush `ack`, the id is forgotten. A serve-process restart empties the in-memory queue. |
| Durable business-level idempotency | Your store of Stripe `event.id` (or `session.id` once paid) before returning any 2xx | Not provided by this mapper or by `kind = "webhook"`. Not exactly-once if Cicerone accepted the batch and the HTTP response was lost. |

Stripe's `event.id` (`evt_…`) is the webhook-level key. Cicerone's `event_id: ${stripeEventId}:${itemId}` is a **per-line** contract id so two SKUs in one webhook do not collide on `evt_…` alone. It is not a durable purchase key.

How to persist Stripe `event.id` before returning any 2xx: [Persist Stripe `event.id`](#persist-stripe-eventid).

## `occurred_at`

Paid `checkout.session.completed` uses `session.created`; async success uses `event.created`. Both are **ingestion / event-domain** timestamps. Neither is a guaranteed payment-settlement time.

| Path | Value | What it is | What it is not |
| --- | --- | --- | --- |
| Paid `completed` | `session.created` | When the Checkout Session was created | When funds settled, or when the webhook arrived |
| `async_payment_succeeded` | `event.created` | When Stripe created the async-success event | When funds settled, or when Checkout was opened |

Cicerone 0.7 uses `occurred_at` as interaction time (recency / latest). Mixing the two clocks can reorder checkouts. A card session opened at 14:00 and paid at 14:01 is stamped 14:00. A SEPA session opened at 13:00 that succeeds at 16:00 is stamped 16:00. Sorted as interaction time, the later-paid SEPA ranks newer than the earlier card charge. If you need strict temporal ordering, use one consistent clock on both paths, or your own order timestamp.

## Line items and quantity

The Checkout event payload does not include the full line list. `listLineItems()` is a **paginated** Stripe list (default 10 objects per page; the list API accepts a per-page `limit` of 1–100). A single page is not the cart.

`autoPagingToArray({ limit: 100 })` walks those pages. The `100` is stripe-node's **total number of items materialized**, not the Stripe page size. Pagination continues until that many objects are collected or the list ends. It is this mapper's cap, not Cicerone's event contract.

**If a Checkout Session has more than 100 line items, this mapper omits the rest, producing partial purchase ingestion. Raise the `limit` if a session can be larger.**

`quantity: 5` remains **one** Cicerone event with `quantity: 5`. Do not expand it into five purchase events unless you explicitly want five interactions. In Cicerone 0.7, types in `quantity_scaled_events` scale by `log1p(quantity)`.

## Implementation

When you create the session:

```js
await stripe.checkout.sessions.create({
  mode: "payment",
  client_reference_id: String(user.id),
  line_items: [
    { price: priceId, quantity: 1 },
  ],
  success_url: `${origin}/thanks`,
  cancel_url: `${origin}/cart`,
});
```

App Router, Node 18+, `npm install stripe` for the signature. Set `CICERONE_SERVE_TOKEN` to the same value as `[serve].auth_token`. `catalogItemId` must return the exact `item_id` in the last LightFM artifact. Express works if the route sees the **raw** body (`express.raw({ type: "application/json" })`). Parsed JSON makes `constructEvent` fail; that is a Stripe fact, not a Cicerone one.

```js
// app/api/stripe/route.js
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const serveUrl = (process.env.CICERONE_SERVE_URL || "http://localhost:8000").replace(/\/$/, "");
const eventsToken = process.env.CICERONE_SERVE_TOKEN;
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

function catalogItemId(line) {
  const price = line.price;
  if (!price) return null;
  const product = price.product;
  if (product && typeof product === "object" && product.metadata?.item_id) {
    return String(product.metadata.item_id);
  }
  if (price.lookup_key) return String(price.lookup_key);
  return null;
}

function mapSessionToEvents(session, occurredAt, lineItems, stripeEventId) {
  const userId = session.client_reference_id;
  if (!userId) {
    console.warn("skip session %s: no client_reference_id", session.id);
    return [];
  }
  const events = [];
  for (const line of lineItems) {
    const itemId = catalogItemId(line);
    if (!itemId) {
      console.warn("skip line %s on %s: no catalog id", line.id, session.id);
      continue;
    }
    events.push({
      user_id: String(userId),
      item_id: itemId,
      event_type: "purchase",
      quantity: line.quantity ?? 1,
      occurred_at: occurredAt,
      event_id: `${stripeEventId}:${itemId}`,
    });
  }
  return events;
}

export async function POST(request) {
  const raw = await request.text();
  const signature = request.headers.get("stripe-signature");
  if (!webhookSecret) {
    return new Response("set STRIPE_WEBHOOK_SECRET", { status: 500 });
  }

  let event;
  try {
    event = stripe.webhooks.constructEvent(raw, signature, webhookSecret);
  } catch {
    return new Response("invalid payload or signature", { status: 400 });
  }

  const isSync = event.type === "checkout.session.completed";
  const isAsyncSuccess = event.type === "checkout.session.async_payment_succeeded";
  if (!isSync && !isAsyncSuccess) {
    return Response.json({ ok: true });
  }

  const session = event.data.object;
  if (isSync && session.payment_status !== "paid") {
    return Response.json({ ok: true });
  }

  const occurredAt = isSync ? session.created : event.created;

  if (!eventsToken) {
    return new Response("set CICERONE_SERVE_TOKEN", { status: 500 });
  }

  const lineItems = await stripe.checkout.sessions
    .listLineItems(session.id, { expand: ["data.price.product"] })
    .autoPagingToArray({ limit: 100 });

  const events = mapSessionToEvents(session, occurredAt, lineItems, event.id);
  if (events.length === 0) return Response.json({ ok: true });

  const response = await fetch(`${serveUrl}/events`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${eventsToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ events }),
  });

  if (response.status === 429) {
    return new Response(await response.text(), { status: 429 });
  }
  if (!response.ok) {
    return new Response(await response.text(), { status: 502 });
  }
  return Response.json(await response.json(), { status: 202 });
}
```

### Persist Stripe `event.id`

Persist Stripe's webhook-level `event.id` (`evt_…`). That is a different string from Cicerone's per-line `event_id: ${stripeEventId}:${itemId}`. The Cicerone id only keeps two SKUs in one webhook as two rows. In the current 0.7 in-memory webhook source, a duplicate Cicerone `event_id` is dropped only while that id is still pending or in-flight. After flush `ack`, or after a serve restart, the same Stripe `event.id` can be ingested again.

Claim Stripe `event.id` **before** `POST /events`. If Cicerone returns a definite non-OK, delete the claim so Stripe can retry. Persist the Stripe `event.id` before returning any 2xx. That prevents ordinary duplicate deliveries (Stripe retries while the first request is still in flight, or retries after you already returned 2xx).

It does **not** give exactly-once semantics. If Cicerone accepted the batch into the in-memory queue and the HTTP response was lost (timeout, reset, process crash after send), this handler may delete the claim and return 5xx. Stripe then retries, the `INSERT` succeeds again, and a second `POST /events` can enqueue the same purchase.

```sql
CREATE TABLE processed_stripe_events (
  event_id TEXT PRIMARY KEY
);
```

```js
const claimed = await db.query(
  `INSERT INTO processed_stripe_events (event_id)
   VALUES ($1)
   ON CONFLICT (event_id) DO NOTHING
   RETURNING event_id`,
  [event.id],
);
if (claimed.rowCount === 0) {
  return Response.json({ ok: true });
}

// … existing listLineItems + POST /events …

if (response.status === 429 || !response.ok) {
  // Definite rejection: drop the claim so Stripe can retry.
  // If Cicerone accepted the batch and the response was lost, deleting
  // the claim can permit a duplicate on Stripe retry. Not exactly-once.
  await db.query(
    `DELETE FROM processed_stripe_events WHERE event_id = $1`,
    [event.id],
  );
  return new Response(await response.text(), {
    status: response.status === 429 ? 429 : 502,
  });
}
return Response.json(await response.json(), { status: 202 });
```

A Cicerone **400** (malformed JSON, missing fields, bad `occurred_at`) is a contract error. This handler maps every non-OK Cicerone response — including that 400, and 401 — to **502**. That is an application-level retry decision: Stripe must not treat the webhook as acknowledged if Cicerone did not accept the events. Returning 2xx would mark the delivery done.

HTTP from this handler:

| Status | Meaning |
| --- | --- |
| **202** | Cicerone accepted the batch into the in-memory queue (`accepted` is a count; ids are in `event_ids`). Not flushed, not applied to the recommendation table, not a model update. |
| **429** | Cicerone backlog full (`max_pending`, default 10_000, minimum 100). Stripe should retry. |
| **400** | Any `constructEvent()` failure — invalid signature **or** invalid/malformed payload (missing header, parsed/truncated body, wrong secret, clock skew). Not “bad signature” only. Do not process the body. |
| **502** | Cicerone returned non-OK (including its **400** contract errors). Intentional application-level mapping: Stripe must not acknowledge a webhook that was not successfully processed, so this handler returns 502 and Stripe retries. HTTP 502 is not the natural meaning of Cicerone's 400. |
| **500** | Missing `STRIPE_WEBHOOK_SECRET` / `CICERONE_SERVE_TOKEN`. Retryable once the env is set. |

Ignored event types, unpaid `completed`, and an empty mapping return **200** `{ ok: true }` so Stripe does not retry a deliberate skip.

Forward locally with `stripe listen --forward-to localhost:3000/api/stripe --events checkout.session.completed,checkout.session.async_payment_succeeded,checkout.session.async_payment_failed`.

## Queue / flush / LightFM (Cicerone 0.7)

Sequence, in order. Keep the verbs separate: **rewrite** = write recommendation rows; **re-score** = rank known users with **frozen** LightFM weights plus extra interactions; **`fit_partial`** = a **model update** (SGD on LightFM weights). Flush is none of those last two by itself.

1. **202 / queued** — serve took the events into its webhook queue. That queue is in-memory. It is not a durable write, not a row rewrite, and not a model update. The recommendation table is unchanged at this point.
2. **Flush** — after `batch_size` or `batch_window_seconds` (default 60), popular / latest (and `'__cold_start__'`) can **rewrite**. The serve process that applied the batch calls `reader.refresh()` on success, so dataset `GET` on **that** process can see the write then. Other dataset readers wait for `[serve].refresh_interval_seconds` (default 60). A `db` output is a query. Flush does not run `fit_partial`.
3. **Conditional frozen re-score** — personalized / item-KNN / content-fallback rows **rewrite** only when both ids are already in the last job artifact, `[events.online]` is on, and `[experiment]` is off. LightFM weights stay frozen. Known users can be re-scored against the extra interactions. This is not a model update.
4. **Conditional `fit_partial`** — a **model update**, only after `fit_min_events` known-ID events have piled up since the last `fit_partial`. Default is **100**. A single test purchase does not trigger `fit_partial`. Then **rewrite** affected users only. Everyone else keeps last night's personalized rows until they show up in a later flush or the cron runs.

The top-K list can stay put even when the flush ran, or even after a frozen re-score: the SKU might already be in the row.

Item factors can drift globally after a real `fit_partial`, but only this flush's users are rewritten. Online extras on top of the artifact stop at `max_extra_interactions` (default 50_000). Cicerone persists the online artifact **after** source `ack`; if that persist fails, serving rows from the flush stay written and the pending fit is dropped.

`[events.online]` **refuses to start** without an artifact in `[output]`. The batch job needs `[job].save_model_artifact = true`. Unknown `event_type`s are dropped; `purchase` must be in `[event_weights]`. An active `[experiment]` skips online LightFM rewrite on purpose (popular/latest still refresh).

On the **job** that writes the artifact:

```toml
[job]
save_model_artifact = true
```

On the **serve** process (`config/cicerone.serve.toml` or your local copy), next to the `[output]` / `[serve]` you already have. Omit `events.options.auth_token` to reuse `serve.auth_token`; if you set the key, the env var must be present.

```toml
[events]
enabled = true
ha = false
kind = "webhook"

[events.options]
# omit auth_token to reuse serve.auth_token
# max_pending = 10000

[events.incremental]
batch_size = 100
batch_window_seconds = 60
poll_interval_seconds = 1

[events.online]
enabled = true
```

Point Node's `CICERONE_SERVE_TOKEN` at `[serve].auth_token` (same name as [`examples/serve/`](https://github.com/torbido-hq/cicerone/tree/main/examples/serve)).

## Failure modes

| Situation | Expected behavior |
| --- | --- |
| Missing `client_reference_id` | Skip session (handler 200 `{ ok: true }`; Stripe will not retry) |
| Missing catalog ID | Skip that line; other lines still post |
| Invalid Stripe webhook | Handler **400** (any `constructEvent()` failure: signature or payload) |
| Cicerone queue full | Handler **429**; Stripe should retry |
| Cicerone contract error | Handler **502**; Stripe should retry |
| Event accepted | Handler **202**; queued, not applied |
| Unknown user/item | No LightFM personalized rewrite or re-score; popular / latest can still rewrite |
| Below `fit_min_events` | No `fit_partial` (frozen weights; known users can still be re-scored) |
| New SKU | LightFM waits for `job.run()` |
| New user ID | LightFM waits for `job.run()` |
| Serve restart before flush | In-memory queued events can be lost (Stripe already got 2xx) |
| Duplicate Stripe event after acknowledgement | Cicerone can ingest it again unless you persist `event.id` ([snippet](#persist-stripe-eventid)) |
| Cicerone 202, HTTP response lost | Persist-before-POST is not exactly-once; a retry can enqueue the purchase again |

## Testing

Treat **ingest**, **frozen re-score**, and **`fit_partial`** as three checks. A successful flush or re-score may leave the top-K unchanged if the SKU was already in the row.

**Ingest.** One test card is enough. The handler returns 202 with your ids in `event_ids`. After the flush window, the incremental manifest's `generated_at` is newer than the purchase. `online_events_dropped_unknown` is how you see Stripe ids that were not in the artifact.

**Frozen re-score.** Known user and known item, `[events.online]` on, `[experiment]` off. Cicerone 0.7 can **rewrite** that user's personalized / item-KNN / content-fallback rows from extra interactions while weights stay frozen. The top-K can still look unchanged. That is not `fit_partial` and not a model update.

**`fit_partial`.** Default `fit_min_events = 100` (known-ID events since the last `fit_partial`, not “100 in one flush”). A single test purchase will not cross that gate. To test the personalized online path including SGD, either send enough known-ID events to reach the configured threshold, or use a **development** serve config with a deliberately lower `fit_min_events`. Do not lower the production threshold just to see the feature work.

```toml
# development serve config only — default is 100
[events.online]
enabled = true
fit_min_events = 1
```

If 202 came back clean and nothing personalized changed, check the failure table and `[experiment]` before assuming a dead webhook.

## When this is the wrong tool

- You need the row inside the checkout response. Write-through is a queue plus a flush window (default 60s), not request-path inference.
- The interesting catalog is SKUs you listed today. Unknown ids never enter LightFM until `job.run()`.
- You need exactly-once purchases from Stripe retries. Persist Stripe `event.id` yourself ([snippet](#persist-stripe-eventid)) to stop ordinary duplicates. This mapper plus Cicerone 0.7's in-memory webhook source will not give you exactly-once if Cicerone accepted the batch and the response was lost.

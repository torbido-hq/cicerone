---
title: This afternoon's checkout can move the row
description: Stripe webhook to Cicerone 0.7 online LightFM. Known-ID purchases write through. GET stays a lookup.
date: 2026-08-28
excerpt: checkout.session.completed becomes a Cicerone event. LightFM can continue before tonight's cron. The homepage still only reads.
authors:
  - nicholas
---

<img src="https://cicerone.dev/images/docs/cicerone-logo.svg" alt="Cicerone" width="200">

Canonical URL: [https://cicerone.dev/articles/this-afternoons-checkout-can-move-the-row/](https://cicerone.dev/articles/this-afternoons-checkout-can-move-the-row/)

[Cicerone](https://cicerone.dev) is a Docker job that reads who bought what, writes a ranked table, and goes back to sleep. The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough leaves personalized ranks until 03:00 UTC. That is the right default. [LightFM](https://making.lyst.com/lightfm/docs/home.html) is a batch fit, and `GET /recommendations` is a lookup. A Stripe shop already has a second clock, though. `checkout.session.completed` fires when money moves, and from Cicerone 0.7 that event can update the same rows the homepage already reads — after a micro-batch, not inside the checkout request.

I built Cicerone for Torbido, a bottle shop that has not opened yet. The repo's default `features.toml` still has drink columns from that; ignore them. The examples below use `sku-42` on purpose. What the webhook has to get right is four strings the batch job already trained on: `user_id`, `item_id`, `event_type`, and `occurred_at`.

Node is here because that is where the Stripe signature is verified. There is no recommendations SDK. Stripe already forced a webhook; this adds a `POST`.

```text
Stripe   checkout.session.completed (payment_status=paid)
         checkout.session.async_payment_succeeded
  │
  ▼
Node     verify signature, map line items
  │
  ▼
POST /events          202 = accepted into a queue, not applied
  │
  ├─ flush at batch_size or batch_window_seconds
  ├─ popular / latest write-through (always)
  └─ [events.online]  LightFM fit_partial on known IDs,
                      rewrite those users only
  │
  ▼
same table / same GET lookup
```

## What this is not

It is not live ranking. The checkout handler never loads a model. Serve never loads a model on `GET`. The events worker continues the last artifact, writes rows, and goes back to sleep. If you need the row to change in the same request as "add to cart," this is the wrong product.

## What has to already be true

1. A nightly (or on-demand) `job.run()` that sets `[job].save_model_artifact = true`. Online serve **refuses to start** without an artifact in `[output]`.
2. Serve with `[events]` `kind = "webhook"` and `[events.online]` enabled, experiments **off**. While `[experiment]` is on, popular and latest still refresh; LightFM rewrite is skipped so the arms stay isolated.
3. `purchase` (or whatever you send) listed under `[event_weights]` in `features.toml`. Unknown `event_type`s are dropped.

The homepage can keep doing what the [nightly table](/articles/a-nightly-table-next-to-your-orders/) post does: `SELECT … ORDER BY rank`. If you read through serve instead, `GET /recommendations/{user_id}` is the same rows. Dataset output reloads its cache every `[serve].refresh_interval_seconds` (default 60). A `db` output is a query.

## Make Stripe speak the event contract

Cicerone never sees `cus_` or `prod_` unless those are the ids you trained on. They almost never are. The ids have to be yours:

| Cicerone | Stripe | Where you set it |
| --- | --- | --- |
| `user_id` | `client_reference_id` | `checkout.sessions.create` |
| `item_id` | Product `metadata.item_id`, else Price `lookup_key` | Dashboard or `products.create` / `prices.create` |
| `event_type` | `"purchase"` | Hard-coded in the mapper (must exist in `[event_weights]`) |
| `occurred_at` | `session.created` when `completed` is already `paid`; `event.created` on the async path | Unix seconds; Cicerone accepts that |
| `quantity` | line-item `quantity` | Checkout already has it; the repo's default `features.toml` puts `purchase` in `quantity_scaled_events` (`log1p`) |
| `event_id` | `{session.id}:{item_id}` | You mint this; do not use `evt_…` |

The event payload does not include line items. You fetch them. If `client_reference_id` is missing, skip the session — do not invent a Stripe customer id. If a line has neither `metadata.item_id` nor `lookup_key`, skip that line. Posting `prod_xxx` against a job that fitted `sku-42` is a silent unknown-id drop: popular and latest still move, LightFM does not.

**Don't trust `completed` alone.** Delayed payment methods — SEPA debit, some bank redirects — fire `checkout.session.completed` with `payment_status: "unpaid"` while the money is still in flight. Confirming happens later, on `checkout.session.async_payment_succeeded` (or fails on `async_payment_failed`). If you only handle `completed`, a payment that later fails still gets counted as a `purchase`. That's worse than the unknown-id case above, because it's wrong rather than dropped.

The handler below does both: it maps `completed` only when `payment_status === "paid"`, and it maps `async_payment_succeeded`. It ignores `async_payment_failed`.

That's also why `occurred_at` isn't simply `session.created` in every case: for a delayed method, the session can sit open for minutes or longer before it actually resolves. `event.created` (the webhook event's own timestamp) is closer to "money moved"; use it on the async path, and reserve `session.created` for `completed` + `paid`.

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

`user.id` must be the same string the batch job reads from your orders. Integer primary keys become text in Cicerone; send `String(user.id)` on both paths.

## The mapper

App Router, Node 18+, `npm install stripe` for the signature. Set `CICERONE_SERVE_TOKEN` to the same value as `[serve].auth_token`. Express works if the route sees the **raw** body (`express.raw({ type: "application/json" })`). Parsed JSON makes `constructEvent` fail, and that is a Stripe fact rather than a Cicerone one.

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

function mapSessionToEvents(session, occurredAt, lineItems) {
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
      event_id: `${session.id}:${itemId}`,
    });
  }
  return events;
}

export async function POST(request) {
  const raw = await request.text();
  const signature = request.headers.get("stripe-signature");

  let event;
  try {
    event = stripe.webhooks.constructEvent(raw, signature, webhookSecret);
  } catch {
    return new Response("invalid signature", { status: 400 });
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

  const events = mapSessionToEvents(session, occurredAt, lineItems);
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

`event_id` is per SKU, not per `evt_…`, so one Stripe event becomes many Cicerone rows and a retry collides on `{cs_…}:{sku-42}`. Cicerone drops that id only while it is still pending or in-flight. After a successful flush `ack`, the same id is a new ingest — a late retry can inflate popular / latest weights for `quantity_scaled_events`.

`POST /events` returns **202** with `accepted` (a count) and `event_ids`. That means the queue took them, not that LightFM has run. A full backlog (`max_pending`, default 10_000, minimum 100) is **429**; returning that to Stripe is what you want, because Stripe will retry. A **400** from Cicerone is a contract bug in the mapper — bring-up should fail loudly (502 here). Do not 200 a 400 and wonder why the list never moved.

Forward locally with `stripe listen --forward-to localhost:3000/api/stripe --events checkout.session.completed,checkout.session.async_payment_succeeded,checkout.session.async_payment_failed`.

## Serve config

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

Point Node’s `CICERONE_SERVE_TOKEN` at `[serve].auth_token` (same name as [`examples/serve/`](https://github.com/torbido-hq/cicerone/tree/main/examples/serve)).

## What actually moves

Flush when the buffer hits `batch_size` **or** `batch_window_seconds` elapses. Then:

| This afternoon | Still waits for `job.run()` |
| --- | --- |
| Popular / latest for people in the flush, plus `'__cold_start__'` | Brand-new SKUs and brand-new user ids in LightFM |
| Personalized rewrite when **both** ids are already in the last artifact | Sequential strategies |
| `fit_partial` after `fit_min_events` known-ID events (default **100**) | Growing the embedding tables |

Until that 100-event gate, weights stay frozen; the worker can still re-score the affected users against the extra interactions. Item factors can drift globally after a real `fit_partial`, but only this flush's users are rewritten. Everyone else keeps last night's personalized rows until they show up in a later flush or the cron runs. Online extras on top of the artifact stop at `max_extra_interactions` (50_000).

A single test card will not cross the SGD gate. Unknown ids never enter the artifact mid-afternoon. Tonight's job still refits from the full event log and writes a new artifact. The mapper does not replace that job.

## Verifying it

After a test purchase:

1. `POST /events` returned 202, with your ids listed in `event_ids`.
2. After the flush window — and, on a dataset output, after `[serve].refresh_interval_seconds` — the incremental manifest's `generated_at` is newer than the purchase. `online_events_dropped_unknown` is how you see Stripe ids that were not in the artifact.

The top-K list itself may not move: the SKU might already be in the row, popular/latest can refresh while LightFM stays still, and `fit_min_events` defaults to 100. If 202 came back clean and nothing personalized changed, check `[experiment]` next — an active experiment skips the LightFM rewrite on purpose.

## When this is the wrong tool

- You need the row inside the checkout response. Write-through is a queue plus a window (default 60s) plus, on a dataset output, the serve refresh interval.
- The interesting catalog is SKUs you listed today. Unknown ids never enter LightFM until `job.run()`.

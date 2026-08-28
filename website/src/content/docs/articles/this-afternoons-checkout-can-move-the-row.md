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

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough leaves personalized ranks until 03:00 UTC. That is the right default. [LightFM](https://making.lyst.com/lightfm/docs/home.html) is a batch fit, and `GET /recommendations` is a lookup. A Stripe shop already has a second clock, though. `checkout.session.completed` fires when money moves, and from Cicerone 0.7 that event can update the same rows the homepage already reads — after a micro-batch, not inside the checkout request.

I built Cicerone for Torbido, a bottle shop that has not opened yet. The repo's default columns still carry that origin; ignore them, and note that the examples below use plain, domain-neutral ids on purpose — Cicerone doesn't care what you sell. What the webhook has to get right is four strings the batch job already trained on: `user_id`, `item_id`, `event_type`, and `occurred_at`.

Node is here because that is where the Stripe signature is verified. There is no recommendations SDK. Stripe already forced a webhook; this adds a `POST`.

```text
Stripe   checkout.session.completed
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
| `occurred_at` | `session.created` for standard payment methods; see note below for delayed ones | Unix seconds; Cicerone accepts that |
| `quantity` | line-item `quantity` | Checkout already has it; the repo's default `features.toml` puts `purchase` in `quantity_scaled_events` (`log1p`) |
| `event_id` | `{session.id}:{item_id}` | You mint this; do not use `evt_…` |

The event payload does not include line items. You fetch them. If `client_reference_id` is missing, skip the session — do not invent a Stripe customer id. If a line has neither `metadata.item_id` nor `lookup_key`, skip that line. Posting `prod_xxx` against a job that fitted `sku-42` is a silent unknown-id drop: popular and latest still move, LightFM does not.

**Don't trust `completed` alone.** Delayed payment methods — SEPA debit, some bank redirects — fire `checkout.session.completed` with `payment_status: "unpaid"` while the money is still in flight. Confirming happens later, on `checkout.session.async_payment_succeeded` (or fails on `async_payment_failed`). If you only handle `completed`, a payment that later fails still gets counted as a `purchase`. That's worse than the unknown-id case above, because it's wrong rather than dropped. Either gate on `session.payment_status === "paid"` inside the `completed` handler, or subscribe to both events and map both to the same `purchase` mapper, skipping `async_payment_failed` entirely.

That's also why `occurred_at` isn't simply "session.created" in every case: for a delayed method, the session can sit open for minutes or longer before it actually resolves. `event.created` (the webhook event's own timestamp) is closer to "money moved" than `session.created` is; use it for anything routed through the async event, and reserve `session.created` for the synchronous `completed` + `paid` path.

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

App Router, Node 18+, `npm install stripe` for the signature. Express works if the route sees the **raw** body (`express.raw({ type: "application/json" })`). Parsed JSON makes `constructEvent` fail, and that is a Stripe fact rather than a Cicerone one.

```js
// app/api/stripe/route.js
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const serveUrl = (process.env.CICERONE_SERVE_URL || "http://localhost:8000").replace(/\/$/, "");
const eventsToken = process.env.CICERONE_EVENTS_TOKEN; // maps to serve's events.options.auth_token
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
      // per-SKU, not per evt_…: one Stripe event is many Cicerone rows,
      // and a retry should collide on the same {session}:{item} pair.
      // Cicerone dedupes incoming events by event_id, so this key is
      // what makes Stripe's at-least-once delivery safe to replay.
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
    // includes checkout.session.async_payment_failed, which we deliberately no-op
    return Response.json({ ok: true });
  }

  const session = event.data.object;
  if (isSync && session.payment_status !== "paid") {
    // delayed payment method (e.g. SEPA debit): wait for async_payment_succeeded
    return Response.json({ ok: true });
  }

  const occurredAt = isSync ? session.created : event.created;

  const lineItems = await stripe.checkout.sessions
    .listLineItems(session.id, { expand: ["data.price.product"] })
    .autoPagingToArray({ limit: 100 });

  const events = mapSessionToEvents(session, occurredAt, lineItems);
  if (events.length === 0) return Response.json({ ok: true });

  // One POST per session, all line items together, so they land in the
  // same micro-batch instead of straddling a flush boundary.
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

`POST /events` returns **202** with `accepted` and `event_ids`. That means the queue took them, not that LightFM has run. A full backlog (`max_pending`, default 10_000, minimum 100) is **429**; returning that to Stripe is what you want, because Stripe will retry. A **400** from Cicerone is a contract bug in the mapper — bring-up should fail loudly (502 here). Do not 200 a 400 and wonder why Alice never moved.

Forward locally with `stripe listen --forward-to localhost:3000/api/stripe --events checkout.session.completed,checkout.session.async_payment_succeeded,checkout.session.async_payment_failed`.

## Serve config

On the **serve** process (`config/cicerone.serve.toml` or your local copy), next to the `[output]` / `[serve]` you already have:

```toml
[events]
enabled = true
ha = false
kind = "webhook"

[events.options]
auth_token = "${CICERONE_EVENTS_TOKEN}"  # defaults to serve.auth_token if unset
# max_pending = 10000

[events.incremental]
batch_size = 100
batch_window_seconds = 60
poll_interval_seconds = 1

[events.online]
enabled = true
```

## Verifying it

Two things to check after a test purchase, before you trust the wiring:

1. `POST /events` returned 202 in the Node logs, with your ids listed in `event_ids`.
2. After the next `batch_window_seconds` flush, `GET /recommendations/{user_id}` (or a `SELECT` against the output table) shows the rank move, and the run's manifest timestamp is newer than the purchase.

If the row doesn't move but the 202 came back clean, check `[experiment]` first — an active experiment intentionally skips the LightFM rewrite and only refreshes popular/latest.

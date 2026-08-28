---
title: This afternoon's checkout can move the row
description: Stripe webhook to Cicerone 0.7 online LightFM. Known-ID purchases write through. GET stays a lookup.
date: 2026-08-28
excerpt: A paid Checkout can update an existing recommendation row after a flush. LightFM rewrite still needs known IDs. GET stays a lookup.
authors:
  - nicholas
---

<img src="https://cicerone.dev/images/docs/cicerone-logo.svg" alt="Cicerone" width="200">

Canonical URL: [https://cicerone.dev/articles/this-afternoons-checkout-can-move-the-row/](https://cicerone.dev/articles/this-afternoons-checkout-can-move-the-row/)

[Cicerone](https://cicerone.dev) is a Docker job that reads who bought what, writes a ranked table, and goes back to sleep. The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough leaves personalized ranks until 03:00 UTC. That is the right default. [LightFM](https://making.lyst.com/lightfm/docs/home.html) is a batch fit, and `GET /recommendations` is a lookup.

A Stripe shop already has a second clock. Checkout completion is not the same as successful payment. Cards and other immediate methods often arrive as `checkout.session.completed` with `payment_status === "paid"`. Delayed methods — SEPA debit, some bank redirects — can fire `completed` while still `"unpaid"`; the money is confirmed later on `checkout.session.async_payment_succeeded` (or not, on `async_payment_failed`). From Cicerone 0.7 the **paid** path can update an **existing** recommendation row after a micro-batch, not inside the checkout request. It does not grow the catalog, and it does not promise that this afternoon's bottle is already in tonight's personalized list.

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
POST /events          202 = accepted into the serve queue, not applied
  │
  ├─ flush at batch_size or batch_window_seconds
  ├─ popular / latest write-through (always)
  └─ [events.online]  LightFM fit_partial on known IDs,
                      rewrite those users only
  │
  ▼
same table / same GET lookup
```

Three different facts, in order:

1. **202** — serve took the events into its webhook queue. That queue is in-memory. It is not a durable write, and it is not a model update.
2. **Flush** — after `batch_size` or `batch_window_seconds` (default 60), popular / latest (and `'__cold_start__'`) can rewrite. Dataset output may wait another `[serve].refresh_interval_seconds` (default 60) before `GET` sees it. A `db` output is a query.
3. **LightFM** — personalized / item-KNN / content-fallback rows rewrite only when both ids are already in the last job artifact, `[events.online]` is on, `[experiment]` is off, and `fit_min_events` known-ID events have piled up since the last `fit_partial` (default **100**). Until then weights stay frozen; the worker can still re-score affected users against the extra interactions. A single test card will not cross that gate. The top-K list can stay put even when the flush ran: the SKU might already be in the row.

## What this is not

It is not live ranking. The checkout handler never loads a model. Serve never loads a model on `GET`. If you need the row to change in the same request as "add to cart," this is the wrong product.

## What has to already be true

1. A nightly (or on-demand) `job.run()` with `[job].save_model_artifact = true`. Online serve **refuses to start** without an artifact in `[output]`.
2. Serve with `[events]` `kind = "webhook"` and `[events.online]` enabled, experiments **off**.
3. `purchase` (or whatever you send) listed under `[event_weights]` in `features.toml`. Unknown `event_type`s are dropped.

The homepage can keep doing what the [nightly table](/articles/a-nightly-table-next-to-your-orders/) post does: `SELECT … ORDER BY rank`. Serve `GET /recommendations/{user_id}` is the same rows.

## Make Stripe speak the event contract

Cicerone never sees `cus_` or `prod_` unless those are the ids you trained on. They almost never are. The ids have to be yours:

| Cicerone | Stripe | Where you set it |
| --- | --- | --- |
| `user_id` | `client_reference_id` | `checkout.sessions.create` |
| `item_id` | Product `metadata.item_id`, else Price `lookup_key` | Dashboard or `products.create` / `prices.create` |
| `event_type` | `"purchase"` | Hard-coded in the mapper (must exist in `[event_weights]`) |
| `occurred_at` | `session.created` when `completed` is already `paid`; `event.created` on the async path | Unix seconds; Cicerone accepts that |
| `quantity` | line-item `quantity` | One Cicerone event per SKU line; `quantity: 5` stays one row, not five |
| `event_id` | `{event.id}:{item_id}` | Stripe's webhook event id (`evt_…`) plus the catalog id |

Whichever catalog string you emit must be the exact `item_id` in the last LightFM artifact. `prod_…` and `price_…` are not aliases for that string.

The Checkout event payload does not include line items. You fetch them. If `client_reference_id` is missing, skip the session — do not invent a Stripe customer id. If a line has neither `metadata.item_id` nor `lookup_key`, skip that line. Posting `prod_xxx` against a job that fitted `sku-42` is a silent unknown-id drop: popular and latest still move, LightFM does not.

**Don't trust `completed` alone.** If you map every `completed` as a `purchase`, a delayed payment that later fails still trains as a buy. The handler below maps `completed` only when `payment_status === "paid"`, and it maps `async_payment_succeeded`. It ignores `async_payment_failed`.

`occurred_at` is not "when money settled" in every case. On the paid `completed` path, `session.created` is when Checkout was created. On the async path, `event.created` is when Stripe emitted that webhook — when Stripe learned the delayed payment succeeded — not a universal settlement clock. For a delayed method the session can sit open for minutes or longer before that event arrives.

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

`event.id` is Stripe's webhook event id. Stripe reuses it across retries of the same delivery, so those retries mint the same Cicerone `event_id`. `:${itemId}` keeps two SKUs in one webhook as two rows instead of colliding on `evt_…` alone.

That is as far as this mapper goes. Cicerone's webhook source drops a duplicate `event_id` only while that id is still pending or in-flight. After a successful flush `ack`, the same id is a new ingest. A late Stripe retry can then inflate popular / latest weights for `quantity_scaled_events`. 202 does not persist the queue across a serve restart. If a retry after `ack` (or after a crash) must be a no-op, record `event.id` in **your** store before you return 2xx to Stripe. Do not treat the mapper as durable idempotency.

HTTP from this handler:

| Status | Meaning |
| --- | --- |
| **202** | Cicerone accepted the batch into the in-memory queue (`accepted` is a count; ids are in `event_ids`). Not flushed, not fitted. |
| **429** | Cicerone backlog full (`max_pending`, default 10_000, minimum 100). Return it so Stripe retries. |
| **400** | Bad Stripe signature. Stripe should not retry a forged or truncated body. |
| **502** | Cicerone answered non-OK (including its **400** contract errors). The mapper did not ack Stripe, so Stripe retries. A 200 here would hide a bad payload. |
| **500** | Missing `STRIPE_WEBHOOK_SECRET` / `CICERONE_SERVE_TOKEN`. Retryable once the env is set. |

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

Point Node's `CICERONE_SERVE_TOKEN` at `[serve].auth_token` (same name as [`examples/serve/`](https://github.com/torbido-hq/cicerone/tree/main/examples/serve)).

## What actually moves

| This afternoon | Still waits for `job.run()` |
| --- | --- |
| 202 into the queue, then popular / latest for people in the flush, plus `'__cold_start__'` | Brand-new SKUs and brand-new user ids in LightFM |
| Personalized rewrite when **both** ids are in the last artifact, experiments off, and `fit_min_events` (default 100) known-ID events have arrived since the last `fit_partial` | Sequential strategies; growing the embedding tables |

Item factors can drift globally after a real `fit_partial`, but only this flush's users are rewritten. Everyone else keeps last night's personalized rows until they show up in a later flush or the cron runs. Online extras on top of the artifact stop at `max_extra_interactions` (50_000). Tonight's job still refits from the full event log and writes a new artifact. The mapper does not replace that job.

## Verifying it

After a test purchase:

1. This handler returned 202, with your ids listed in `event_ids`.
2. After the flush window — and, on a dataset output, after `[serve].refresh_interval_seconds` — the incremental manifest's `generated_at` is newer than the purchase. `online_events_dropped_unknown` is how you see Stripe ids that were not in the artifact.

If 202 came back clean and nothing personalized changed, that is the table above, not a dead webhook. Check `[experiment]` if you expected a LightFM rewrite: an active experiment skips it on purpose.

## When this is the wrong tool

- You need the row inside the checkout response. Write-through is a queue plus a window (default 60s) plus, on a dataset output, the serve refresh interval.
- The interesting catalog is SKUs you listed today. Unknown ids never enter LightFM until `job.run()`.

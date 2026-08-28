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

I built Cicerone for Torbido, a bottle shop that has not opened yet. The repo’s default columns are drinks; ignore them. What the webhook has to get right is four strings the batch job already trained on: `user_id`, `item_id`, `event_type`, and `occurred_at`.

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

It is not live ranking. The checkout handler never loads a model. Serve never loads a model on `GET`. The events worker continues the last artifact, writes rows, and goes back to sleep. If you need the row to change in the same request as “add to cart,” this is the wrong product.

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
| `occurred_at` | `session.created` | Unix seconds; Cicerone accepts that |
| `quantity` | line-item `quantity` | Checkout already has it; the repo’s default `features.toml` puts `purchase` in `quantity_scaled_events` (`log1p`) |
| `event_id` | `{session.id}:{item_id}` | You mint this; do not use `evt_…` |

The event payload does not include line items. You fetch them. If `client_reference_id` is missing, skip the session — do not invent a Stripe customer id. If a line has neither `metadata.item_id` nor `lookup_key`, skip that line. Posting `prod_xxx` against a job that fitted `ipa-001` is a silent unknown-id drop: popular and latest still move, LightFM does not.

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
const eventsToken = process.env.CICERONE_EVENTS_TOKEN;
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

export async function POST(request) {
  const raw = await request.text();
  const signature = request.headers.get("stripe-signature");
  let event;
  try {
    event = stripe.webhooks.constructEvent(raw, signature, webhookSecret);
  } catch {
    return new Response("invalid signature", { status: 400 });
  }
  if (event.type !== "checkout.session.completed") {
    return Response.json({ ok: true });
  }

  const session = event.data.object;
  const userId = session.client_reference_id;
  if (!userId) {
    console.warn("skip session %s: no client_reference_id", session.id);
    return Response.json({ ok: true });
  }

  const lineItems = await stripe.checkout.sessions
    .listLineItems(session.id, { expand: ["data.price.product"] })
    .autoPagingToArray({ limit: 100 });
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
      occurred_at: session.created,
      event_id: `${session.id}:${itemId}`,
    });
  }
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

`POST /events` returns **202** with `accepted` and `event_ids`. That means the queue took them, not that LightFM has run. A full backlog (`max_pending`, default 10_000, minimum 100) is **429**; returning that to Stripe is what you want, because Stripe will retry. A **400** from Cicerone is a contract bug in the mapper — bring-up should fail loudly (502 here). Do not 200 a 400 and wonder why Alice never moved.

Forward locally with `stripe listen --forward-to localhost:3000/api/stripe`.

One `POST` per session, all line items together, so they land in the same micro-batch. `event_id` is per SKU, not per `evt_…`, because one Stripe event is many Cicerone rows and a retry should collide on `{cs_…}:{ipa-001}`.

## Serve config

On the **serve** process (`config/cicerone.serve.toml` or your local copy), next to the `[output]` / `[serve]` you already have:

```toml
[events]
enabled = true
ha = false
kind = "webhook"

[events.options]
# auth_token = "${EVENTS_AUTH_TOKEN}"  # defaults to serve.auth_token
# max_pending = 10000

[events.incremental]
batch_size = 100
batch_window_seconds = 60
poll_interval_seconds = 1

[events.online]
enabled = true
fit_partial_epochs = 1          # 0 = frozen weights + history refresh only
fit_min_events = 100            # skip SGD until this many known-ID events
max_extra_interactions = 50000  # online-only rows on top of the last job artifact
```

On the **job** that writes the artifact:

```toml
[job]
save_model_artifact = true
```

Webhook ingest is a **single writer**. Do not put three serve replicas behind a load balancer and call it HA; that needs `events.ha = true` and a Postgres or Redis lock, and even then webhook ingress stays sticky. Redis Streams is the fan-out source if you already `XADD`. This post stays on `POST /events` because Stripe already POSTs you.

## What actually moves

Flush when the buffer hits `batch_size` **or** `batch_window_seconds` elapses. Then:

**Always (incremental path).** Popular and latest slices, plus recency boosts, for affected users and `'__cold_start__'`. A first-time buyer can still see the guest list shift. New popular / latest / `incremental` rows get a thin `reasons` payload (`{sources:[{label}]}`) — no history-overlap `similar_items`.

**When both ids are in the last artifact.** The online worker rebuilds that user’s personalized / item-KNN / content-fallback rows and writes them through. `GET` is still a lookup. New catalog ids, and any user the job has never seen, wait for `job.run()`. Sequential models (SASRec / BERT4Rec / HSTU) never `fit_partial`.

**SGD is gated.** `fit_partial` runs only after `fit_min_events` known-ID events have piled up since the last fit (default **100**). Until then, weights stay frozen and the worker still re-scores the affected users against the extra interactions. One quiet Friday does not step the embedding. `fit_partial_epochs = 0` keeps that freeze permanently and only refreshes history.

**Item factors move globally; lists do not.** After a real `fit_partial`, every item vector can drift, but only the flush’s users are rewritten. Everyone else keeps last night’s personalized rows until they appear in a later flush or the cron runs. That is the same class of staleness as [Gorse](https://gorse.io) between worker passes. It is the trade you make for not fitting on the request.

**Cap.** Online-only interactions on top of the artifact stop at `max_extra_interactions` (50_000). The nightly job is the drift backstop, not an optional extra.

| Happens this afternoon | Still waits for `job.run()` |
| --- | --- |
| Popular / latest for people in the flush, plus `'__cold_start__'` | Brand-new SKUs and brand-new user ids in LightFM |
| Personalized rewrite for **known** `(user, item)` | Sequential strategies |
| `fit_partial` after 100 known-ID events | Growing the embedding tables |

Look at `source` in the morning, same as the Rails post. Online does not invent overlap. Fifty checkouts a week is still bestsellers with extra steps.

## Retries

Stripe delivers at-least-once. Cicerone’s webhook dedupe is only while an `event_id` sits in pending or in-flight. After a successful flush `ack`, the same id is a new ingest. Late retries can inflate popular / latest weights for `quantity_scaled_events`. Online LightFM persists the artifact **after** `ack`, so a nack/redelivery does not append the same pairs twice to the model. If that persist fails after retries, the pending fit is dropped; the rows already written stay.

If you need a stronger promise than “Stripe usually retries within the in-flight window,” remember `session.id` in your own table before `POST`, or skip the webhook and let Cicerone poll `order_items` with `kind = "db"` (watermark advances only after a successful flush).

## When this is the wrong tool

- You need the row inside the checkout response. Write-through is a queue plus a window (default 60s) plus, on a dataset output, the serve refresh interval.
- `[experiment]` is on. Popular and latest still move; LightFM rewrite does not, so the arms stay isolated. Resume a single recipe first.
- The interesting catalog is the SKUs you listed *today*. Unknown ids never enter the artifact mid-afternoon.
- `client_reference_id` / catalog metadata were never wired. The mapper will no-op, or worse, you will POST Stripe ids and watch `online_events_dropped_unknown` climb on the incremental manifest.

The [incremental events](/incremental-events/) page is the operator contract (HA, SQS, Redis Streams, metrics). [How it works](/how-it-works/#incremental-vs-full-retrain) is the algorithm split.

## After the webhook

Stripe CLI, one test card, wait the micro-batch window, then `GET /recommendations/{user_id}` or look the id up on the dashboard. The inspector’s event column reads `[input]`, not the webhook queue — overlap highlighting shows up if this handler also writes `order_items` (it should). A Cicerone-only `POST` updates ranks and leaves that pane stale. If `source` is still `popular_fallback` everywhere, the purchase landed and LightFM still had nothing to say. That is thin data, not a dead webhook.

Tonight’s cron still runs. It refits from the full event log, including everything Stripe already sent, and writes a new artifact. The mapper does not replace that job. It only means this afternoon’s bottle can show up on the row before you lock the door.

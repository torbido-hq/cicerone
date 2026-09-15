#!/usr/bin/env node
/**
 * Thin fetch() example for Cicerone serve mode (Node 18+ / browsers).
 *
 *   CICERONE_SERVE_URL=http://localhost:8000 \
 *   CICERONE_SERVE_TOKEN=tutorial-token \
 *   node examples/serve/fetch.mjs
 */

const baseUrl = (process.env.CICERONE_SERVE_URL || "http://localhost:8000").replace(/\/$/, "");
const token = process.env.CICERONE_SERVE_TOKEN;
const userId = process.env.CICERONE_USER_ID || "alice";

function headers() {
  const h = { Accept: "application/json" };
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

const health = await fetch(`${baseUrl}/health`);
console.log("health:", await health.json());

const url = new URL(`${baseUrl}/recommendations/${encodeURIComponent(userId)}`);
url.searchParams.set("limit", "5");

const response = await fetch(url, { headers: headers() });
if (!response.ok) {
  console.error("recommendations failed:", response.status, await response.text());
  process.exit(1);
}
const body = await response.json();
console.log(
  `user=${body.user_id} fallback=${body.fallback} generated_at=${body.generated_at} experiment=${body.experiment_id} variant=${body.variant}`,
);
for (const row of body.items) {
  console.log(`  #${row.rank} ${row.item_id} score=${row.score} source=${row.source}`);
}

if (process.env.CICERONE_POST_TRACK === "1" && body.items?.length) {
  const tracked = await fetch(`${baseUrl}/track`, {
    method: "POST",
    headers: { ...headers(), "Content-Type": "application/json" },
    body: JSON.stringify(
      body.items.map((row) => ({
        kind: "impression",
        user_id: userId,
        item_id: row.item_id,
        rank: row.rank,
        occurred_at: new Date().toISOString(),
        variant: body.variant,
        generated_at: body.generated_at,
      })),
    ),
  });
  if (!tracked.ok) {
    console.error("track failed:", tracked.status, await tracked.text());
    process.exit(1);
  }
  console.log("track:", await tracked.json());
}

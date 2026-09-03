<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Evaluation

Cicerone cannot see the host UI. CTR and conversion of served recommendations
come from **impressions and clicks you send**, plus the purchase (or other
valued) events you already store as `[input]`.

`POST /events` stays the training/incremental contract. Track rows never enter
`[event_weights]` or LightFM.

## Impressions and clicks

```toml
[track]
enabled = true
# attribution_window_hours = 24
# conversion_event_types = ["purchase"]
# min_impressions = 100
```

`POST /track` on the serve process (same Bearer as `GET /recommendations`):

```json
{
  "kind": "impression",
  "user_id": "alice",
  "item_id": "ipa-001",
  "rank": 1,
  "occurred_at": "2026-08-28T12:00:00Z"
}
```

`kind` is `impression` or `click`. Send one row per shown item (or a list /
`{"events":[...]}`). Optional `variant`, `experiment_id`, `generated_at`,
`event_id` (idempotency). Bodies larger than 1 MiB (or
`events.options.max_body_bytes`) return 413.

Storage is next to `[output]`: local `track.jsonl`, or a
`recommendation_track` db table. Object-store JSONL append is refused.
`events.ha = true` requires db output.

SQL-join shops that never call GET still POST `/track` when the widget
renders.

From a Rails render path:

```ruby
require "net/http"
require "json"

uri = URI("#{ENV.fetch("CICERONE_SERVE_URL")}/track")
http = Net::HTTP.new(uri.host, uri.port)
http.use_ssl = uri.scheme == "https"
body = recs.map.with_index(1) do |item, rank|
  {
    kind: "impression",
    user_id: current_user.id,
    item_id: item.item_id,
    rank: rank,
    occurred_at: Time.now.utc.iso8601,
    variant: item.variant,
    generated_at: item.generated_at,
  }
end
request = Net::HTTP::Post.new(uri)
request["Authorization"] = "Bearer #{ENV.fetch("CICERONE_SERVE_TOKEN")}"
request["Content-Type"] = "application/json"
request.body = { events: body }.to_json
http.request(request)
```

Clicks use the same contract with `kind: "click"`.

## Metrics

- **CTR** — clicks that match a prior impression of the same `(user, item)`
  inside the window, divided by impressions.
- **Conversion** — `[input]` events whose type is
  `track.conversion_event_types` (default `purchase`, or
  `[experiment].primary_metric` when that is an event type), attributed to a
  prior impression (view-through) or click (click-through).
- **CVR** — attributed conversions / impressions.

The job writes a compact `track_eval` report (overall, by rank, by `source`,
by variant). The dashboard **Quality** page reads it.

## Auto-impressions

`[serve].log_impressions = true` (requires `[track]`) writes one impression
per **returned** item on `GET /recommendations/{user_id}`. That is a fetch,
not a view. Clicks are always host-reported. Default off.

## Experiments

```toml
[experiment]
primary_metric = "ctr"       # or "conversion", or an event_type / "weighted"
attribution = "click"        # click | impression | user | recommended
```

`user` keeps 0.7.0 intention-to-treat on all events. `click` / `impression`
use the tracking store (`primary_metric` `ctr` / `conversion` require
`track.enabled` and attribution `click` or `impression`). `recommended` counts
only events whose `item_id` was on that user's list (no `/track` required).
Promote stays fail-closed; CTR and conversion also require
`track.min_impressions`.

## Production replay

```toml
[job.eval]
enabled = true
# event_types = ["purchase"]
# ks = [5, 10]
```

At the start of the next job, previous recommendation rows are scored against
later events (`HitRate` / `MAP` / `NDCG` / `Recall` / `MRR` / `Precision` @K,
plus catalog coverage and novelty when prior interactions exist, and hit rate
by `source`). This is not CTR. Recommendation snapshots
(`recommendation_history`) let the window span more than one job.

# Serve API client examples

Minimal examples for calling Cicerone serve mode. Point them at a live
serve process (`cicerone serve` or the serve container — see the README
Serve section and [docs/tutorial.md](../../docs/tutorial.md) step 12).

OpenAPI / ReDoc also embed language samples (`x-codeSamples`, including
Ruby) on `/health`, `/recommendations/{user_id}`, and `POST /events`
(when webhook events are enabled) — open `http://localhost:8000/redoc` or
the checked-in
[`docs/openapi/serve.openapi.json`](../../docs/openapi/serve.openapi.json).

```sh
export CICERONE_SERVE_URL=http://localhost:8000
export CICERONE_SERVE_TOKEN=tutorial-token   # matches [serve].auth_token
# optional: CICERONE_USER_ID=alice

python examples/serve/python_client.py   # needs cicerone-recommender; ServeClient → typed responses
node examples/serve/fetch.mjs            # Node 18+ / browser-style fetch
examples/serve/curl_examples.sh          # curl; needs python (json.tool / json.dumps)
```

Interactive docs while serve is up: `http://localhost:8000/docs` (Swagger)
and `http://localhost:8000/redoc` (code samples).

Prometheus process metrics (no bearer token): `GET /metrics`. Optional
`[serve].metrics_token` → send `X-Metrics-Token`. See the README Serve
section.

`POST /events` is present only when `[events]` webhook ingest is enabled
on that serve process. Tutorial step 13 walks through a local webhook;
see [docs/incremental-events.md](../../docs/incremental-events.md).
The curl script JSON-encodes `CICERONE_USER_ID` with `python` (`json.dumps`).

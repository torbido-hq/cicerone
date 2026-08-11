# Serve API client examples

Minimal examples for calling Cicerone serve mode. Point them at a live
serve container (see the README Serve section and
[docs/tutorial.md](../../docs/tutorial.md) step 12).

OpenAPI / ReDoc also embed language samples (`x-codeSamples`, including
Ruby) on `/health` and `/recommendations/{user_id}` — open
`http://localhost:8000/redoc` or the checked-in
[`docs/openapi/serve.openapi.json`](../../docs/openapi/serve.openapi.json).

```sh
export CICERONE_SERVE_URL=http://localhost:8000
export CICERONE_SERVE_TOKEN=tutorial-token   # matches [serve].auth_token
# optional: CICERONE_USER_ID=alice

python examples/serve/python_client.py   # ServeClient → typed HealthResponse / RecommendationsResponse
node examples/serve/fetch.mjs            # Node 18+ / browser-style fetch
ruby examples/serve/ruby_client.rb       # stdlib Net::HTTP + JSON (matches OpenAPI Ruby sample)
examples/serve/curl_examples.sh          # curl + /openapi.json peek
```

Interactive docs while serve is up: `http://localhost:8000/docs` (Swagger)
and `http://localhost:8000/redoc` (code samples).

Prometheus process metrics (no bearer token): `GET /metrics`. Optional
`[serve].metrics_token` → send `X-Metrics-Token`. See the README Serve
section.

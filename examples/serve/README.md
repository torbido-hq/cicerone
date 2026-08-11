# Serve API client examples

Minimal examples for calling Cicerone serve mode. Point them at a live
serve container (see the README Serve section and
[docs/tutorial.md](../../docs/tutorial.md) step 12).

```sh
export CICERONE_SERVE_URL=http://localhost:8000
export CICERONE_SERVE_TOKEN=tutorial-token   # matches [serve].auth_token
# optional: CICERONE_USER_ID=alice

python examples/serve/python_client.py   # ServeClient → typed HealthResponse / RecommendationsResponse
node examples/serve/fetch.mjs            # Node 18+ / browser-style fetch
ruby examples/serve/ruby_client.rb       # stdlib Net::HTTP + JSON
examples/serve/curl_examples.sh          # curl + /openapi.json peek
```

Interactive docs while serve is up: `http://localhost:8000/docs`.
Checked-in OpenAPI schema: [`docs/openapi/serve.openapi.json`](../../docs/openapi/serve.openapi.json).

Prometheus process metrics (no bearer token): `GET /metrics`. Optional
`[serve].metrics_token` → send `X-Metrics-Token`. See the README Serve
section.

# Serve API client examples

Minimal examples for calling Cicerone serve mode. Point them at a live
serve container (see the README Serve section and
[docs/tutorial.md](../docs/tutorial.md) step 12).

```sh
export CICERONE_SERVE_URL=http://localhost:8000
export CICERONE_SERVE_TOKEN=tutorial-token   # matches [serve].auth_token
# optional: CICERONE_USER_ID=alice

python examples/serve/python_client.py   # uses cicerone.serve_client.ServeClient
node examples/serve/fetch.mjs            # Node 18+ / browser-style fetch
examples/serve/curl_examples.sh          # curl + /openapi.json peek
```

Interactive docs while serve is up: `http://localhost:8000/docs`.
Checked-in OpenAPI schema: [`docs/openapi/serve.openapi.json`](../docs/openapi/serve.openapi.json).

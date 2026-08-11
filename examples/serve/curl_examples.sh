#!/usr/bin/env bash
# curl examples for Cicerone serve mode.
#   export CICERONE_SERVE_URL=http://localhost:8000
#   export CICERONE_SERVE_TOKEN=tutorial-token
#   examples/serve/curl_examples.sh
set -euo pipefail

BASE_URL="${CICERONE_SERVE_URL:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
USER_ID="${CICERONE_USER_ID:-alice}"

auth_header=()
if [[ -n "${CICERONE_SERVE_TOKEN:-}" ]]; then
  auth_header=(-H "Authorization: Bearer ${CICERONE_SERVE_TOKEN}")
fi

echo "## GET /health"
curl -sS "${BASE_URL}/health" | python -m json.tool

echo
echo "## GET /metrics (first lines; no bearer token)"
metrics_header=()
if [[ -n "${CICERONE_METRICS_TOKEN:-}" ]]; then
  metrics_header=(-H "X-Metrics-Token: ${CICERONE_METRICS_TOKEN}")
fi
curl -sS "${metrics_header[@]}" "${BASE_URL}/metrics" | head -n 20

echo
echo "## GET /recommendations/${USER_ID}?limit=5"
curl -sS "${auth_header[@]}" \
  "${BASE_URL}/recommendations/${USER_ID}?limit=5" | python -m json.tool

echo
echo "## GET /openapi.json (first paths only)"
curl -sS "${BASE_URL}/openapi.json" | python -c '
import json, sys
doc = json.load(sys.stdin)
print("title:", doc.get("info", {}).get("title"))
print("paths:", ", ".join(sorted(doc.get("paths", {}))))
'

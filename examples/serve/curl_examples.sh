#!/usr/bin/env bash
# curl examples for Cicerone serve mode.
# Needs python3 or python (`json.tool`, and `json.dumps` so USER_ID with quotes stays valid).
#   export CICERONE_SERVE_URL=http://localhost:8000
#   export CICERONE_SERVE_TOKEN=tutorial-token
#   examples/serve/curl_examples.sh
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "python3 or python required" >&2
  exit 1
fi

BASE_URL="${CICERONE_SERVE_URL:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
USER_ID="${CICERONE_USER_ID:-alice}"

auth_header=()
if [[ -n "${CICERONE_SERVE_TOKEN:-}" ]]; then
  auth_header=(-H "Authorization: Bearer ${CICERONE_SERVE_TOKEN}")
fi

echo "## GET /health"
curl -sS "${BASE_URL}/health" | "$PYTHON" -m json.tool

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
  "${BASE_URL}/recommendations/${USER_ID}?limit=5" | "$PYTHON" -m json.tool

echo
echo "## GET /openapi.json (first paths only)"
curl -sS "${BASE_URL}/openapi.json" | "$PYTHON" -c '
import json, sys
doc = json.load(sys.stdin)
print("title:", doc.get("info", {}).get("title"))
print("paths:", ", ".join(sorted(doc.get("paths", {}))))
'

if [[ "${CICERONE_POST_EVENTS:-}" == "1" ]]; then
  echo
  echo "## POST /events (set CICERONE_POST_EVENTS=1; webhook ingest must be enabled)"
  events_body="$(
    USER_ID="$USER_ID" "$PYTHON" -c '
import json, os
print(json.dumps({
    "user_id": os.environ["USER_ID"],
    "item_id": "ipa-001",
    "event_type": "purchase",
    "quantity": 1,
    "occurred_at": "2026-08-19T12:00:00Z",
}))
'
  )"
  curl -sS "${auth_header[@]}" -X POST \
    -H "Content-Type: application/json" \
    -d "$events_body" \
    "${BASE_URL}/events" | "$PYTHON" -m json.tool
fi

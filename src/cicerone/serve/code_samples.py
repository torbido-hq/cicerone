"""OpenAPI ``x-codeSamples`` snippets for serve mode (ReDoc / exported schema)."""

from __future__ import annotations

from typing import Any

_HEALTH_RUBY = """\
require "json"
require "net/http"
require "uri"

uri = URI("http://localhost:8000/health")
response = Net::HTTP.get_response(uri)
puts JSON.parse(response.body)
"""

_HEALTH_PYTHON = """\
from urllib.request import urlopen
import json

print(json.load(urlopen("http://localhost:8000/health")))
"""

_HEALTH_SHELL = """\
curl -sS http://localhost:8000/health
"""

_RECOMMENDATIONS_RUBY = """\
require "json"
require "net/http"
require "uri"

base = "http://localhost:8000"
token = ENV.fetch("CICERONE_SERVE_TOKEN")
user_id = ENV.fetch("CICERONE_USER_ID", "alice")

uri = URI("#{base}/recommendations/#{URI.encode_www_form_component(user_id)}")
uri.query = URI.encode_www_form(limit: 5)

req = Net::HTTP::Get.new(uri)
req["Accept"] = "application/json"
req["Authorization"] = "Bearer #{token}"

res = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
  http.request(req)
end
abort("HTTP #{res.code}: #{res.body}") unless res.is_a?(Net::HTTPSuccess)

body = JSON.parse(res.body)
puts "user=#{body['user_id']} fallback=#{body['fallback']}"
body["items"].each { |row| puts "  ##{row['rank']} #{row['item_id']} score=#{row['score']}" }
"""

_RECOMMENDATIONS_PYTHON = """\
import os
from cicerone.serve_client import ServeClient

client = ServeClient(
    os.environ.get("CICERONE_SERVE_URL", "http://localhost:8000"),
    token=os.environ["CICERONE_SERVE_TOKEN"],
)
body = client.recommendations(os.environ.get("CICERONE_USER_ID", "alice"), limit=5)
print(body.user_id, body.fallback, body.generated_at)
for row in body.items:
    print(f"  #{row.rank} {row.item_id} score={row.score}")
"""

_RECOMMENDATIONS_JAVASCRIPT = """\
const baseUrl = (process.env.CICERONE_SERVE_URL || "http://localhost:8000").replace(/\\/$/, "");
const token = process.env.CICERONE_SERVE_TOKEN;
const userId = process.env.CICERONE_USER_ID || "alice";

const url = new URL(`${baseUrl}/recommendations/${encodeURIComponent(userId)}`);
url.searchParams.set("limit", "5");

const response = await fetch(url, {
  headers: {
    Accept: "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  },
});
if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
const body = await response.json();
console.log(body.user_id, body.fallback, body.generated_at);
for (const row of body.items) {
  console.log(`  #${row.rank} ${row.item_id} score=${row.score}`);
}
"""

_RECOMMENDATIONS_SHELL = """\
curl -sS \\
  -H "Authorization: Bearer ${CICERONE_SERVE_TOKEN}" \\
  "${CICERONE_SERVE_URL:-http://localhost:8000}/recommendations/${CICERONE_USER_ID:-alice}?limit=5"
"""

HEALTH_CODE_SAMPLES: list[dict[str, str]] = [
    {"lang": "Ruby", "label": "Net::HTTP", "source": _HEALTH_RUBY},
    {"lang": "Python", "label": "urllib", "source": _HEALTH_PYTHON},
    {"lang": "Shell", "label": "curl", "source": _HEALTH_SHELL},
]

RECOMMENDATIONS_CODE_SAMPLES: list[dict[str, str]] = [
    {"lang": "Ruby", "label": "Net::HTTP", "source": _RECOMMENDATIONS_RUBY},
    {"lang": "Python", "label": "ServeClient", "source": _RECOMMENDATIONS_PYTHON},
    {"lang": "JavaScript", "label": "fetch", "source": _RECOMMENDATIONS_JAVASCRIPT},
    {"lang": "Shell", "label": "curl", "source": _RECOMMENDATIONS_SHELL},
]


def attach_code_samples(schema: dict[str, Any]) -> None:
    """Attach ReDoc ``x-codeSamples`` to serve operations (mutates ``schema``)."""
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return
    health = paths.get("/health", {}).get("get")
    if isinstance(health, dict):
        health["x-codeSamples"] = list(HEALTH_CODE_SAMPLES)
    recommendations = paths.get("/recommendations/{user_id}", {}).get("get")
    if isinstance(recommendations, dict):
        recommendations["x-codeSamples"] = list(RECOMMENDATIONS_CODE_SAMPLES)

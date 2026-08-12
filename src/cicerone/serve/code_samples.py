"""OpenAPI ``x-codeSamples`` snippets for serve mode (ReDoc / exported schema)."""

from __future__ import annotations

from typing import Any

# Shared default; every snippet reads CICERONE_SERVE_URL the same way.
_DEFAULT_SERVE_URL = "http://localhost:8000"

_HEALTH_RUBY = f"""\
require "json"
require "net/http"
require "uri"

base = ENV.fetch("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").sub(%r{{/\\z}}, "")
uri = URI("#{{base}}/health")
# /health is unauthenticated (no CICERONE_SERVE_TOKEN).
response = Net::HTTP.get_response(uri)
abort("HTTP #{{response.code}}: #{{response.body}}") unless response.is_a?(Net::HTTPSuccess)
puts JSON.parse(response.body)
"""

_HEALTH_PYTHON = f"""\
import json
import os
from urllib.request import urlopen

base = os.environ.get("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").rstrip("/")
# /health is unauthenticated (no CICERONE_SERVE_TOKEN).
print(json.load(urlopen(f"{{base}}/health")))
"""

_HEALTH_SHELL = f"""\
# /health is unauthenticated (no CICERONE_SERVE_TOKEN).
curl -sS "${{CICERONE_SERVE_URL:-{_DEFAULT_SERVE_URL}}}/health"
"""

_RECOMMENDATIONS_RUBY = f"""\
require "json"
require "net/http"
require "uri"

base = ENV.fetch("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").sub(%r{{/\\z}}, "")
token = ENV.fetch("CICERONE_SERVE_TOKEN")
user_id = ENV.fetch("CICERONE_USER_ID", "alice")

uri = URI("#{{base}}/recommendations/#{{URI.encode_www_form_component(user_id)}}")
uri.query = URI.encode_www_form(limit: 5)

req = Net::HTTP::Get.new(uri)
req["Accept"] = "application/json"
req["Authorization"] = "Bearer #{{token}}"

res = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
  http.request(req)
end
abort("HTTP #{{res.code}}: #{{res.body}}") unless res.is_a?(Net::HTTPSuccess)

body = JSON.parse(res.body)
puts "user=#{{body['user_id']}} fallback=#{{body['fallback']}}"
body["items"].each {{ |row| puts "  ##{{row['rank']}} #{{row['item_id']}} score=#{{row['score']}}" }}
"""

_RECOMMENDATIONS_PYTHON = f"""\
import os
from cicerone.serve_client import ServeClient

client = ServeClient(
    os.environ.get("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}"),
    token=os.environ["CICERONE_SERVE_TOKEN"],
)
body = client.recommendations(os.environ.get("CICERONE_USER_ID", "alice"), limit=5)
print(body.user_id, body.fallback, body.generated_at)
for row in body.items:
    print(f"  #{{row.rank}} {{row.item_id}} score={{row.score}}")
"""

_RECOMMENDATIONS_JAVASCRIPT = f"""\
const baseUrl = (process.env.CICERONE_SERVE_URL || "{_DEFAULT_SERVE_URL}").replace(/\\/$/, "");
const token = process.env.CICERONE_SERVE_TOKEN;
const userId = process.env.CICERONE_USER_ID || "alice";

const url = new URL(`${{baseUrl}}/recommendations/${{encodeURIComponent(userId)}}`);
url.searchParams.set("limit", "5");

const response = await fetch(url, {{
  headers: {{
    Accept: "application/json",
    ...(token ? {{ Authorization: `Bearer ${{token}}` }} : {{}}),
  }},
}});
if (!response.ok) throw new Error(`${{response.status}} ${{await response.text()}}`);
const body = await response.json();
console.log(body.user_id, body.fallback, body.generated_at);
for (const row of body.items) {{
  console.log(`  #${{row.rank}} ${{row.item_id}} score=${{row.score}}`);
}}
"""

_RECOMMENDATIONS_SHELL = f"""\
curl -sS \\
  -H "Authorization: Bearer ${{CICERONE_SERVE_TOKEN}}" \\
  "${{CICERONE_SERVE_URL:-{_DEFAULT_SERVE_URL}}}/recommendations/${{CICERONE_USER_ID:-alice}}?limit=5"
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


def _extend_code_samples(operation: dict[str, Any], samples: list[dict[str, str]]) -> None:
    existing = operation.get("x-codeSamples")
    if not isinstance(existing, list):
        existing = []
    operation["x-codeSamples"] = [*existing, *samples]


def attach_code_samples(schema: dict[str, Any]) -> None:
    """Attach ReDoc ``x-codeSamples`` to serve operations (mutates ``schema``).

    Appends to any samples already present on the operation so route-level
    ``openapi_extra`` additions are preserved.
    """
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return
    health = paths.get("/health", {}).get("get")
    if isinstance(health, dict):
        _extend_code_samples(health, HEALTH_CODE_SAMPLES)
    recommendations = paths.get("/recommendations/{user_id}", {}).get("get")
    if isinstance(recommendations, dict):
        _extend_code_samples(recommendations, RECOMMENDATIONS_CODE_SAMPLES)

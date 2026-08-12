"""OpenAPI ``x-codeSamples`` snippets for serve mode (ReDoc / exported schema)."""

from __future__ import annotations

from typing import Any

HEALTH_PATH = "/health"
RECOMMENDATIONS_PATH = "/recommendations/{user_id}"
RECOMMENDATIONS_PATH_PREFIX = "/recommendations/"

# Shared default; every snippet reads CICERONE_SERVE_URL the same way.
_DEFAULT_SERVE_URL = "http://localhost:8000"

_HEALTH_RUBY = f"""\
require "json"
require "net/http"
require "uri"

base = ENV.fetch("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").sub(%r{{/\\z}}, "")
uri = URI("#{{base}}{HEALTH_PATH}")
# {HEALTH_PATH} is unauthenticated (no CICERONE_SERVE_TOKEN).
response = Net::HTTP.get_response(uri)
abort("HTTP #{{response.code}}: #{{response.body}}") unless response.is_a?(Net::HTTPSuccess)
puts JSON.parse(response.body)
"""

_HEALTH_PYTHON = f"""\
import json
import os
from urllib.request import urlopen

base = os.environ.get("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").rstrip("/")
# {HEALTH_PATH} is unauthenticated (no CICERONE_SERVE_TOKEN).
print(json.load(urlopen(f"{{base}}{HEALTH_PATH}")))
"""

_HEALTH_SHELL = f"""\
# {HEALTH_PATH} is unauthenticated (no CICERONE_SERVE_TOKEN).
curl -sS "${{CICERONE_SERVE_URL:-{_DEFAULT_SERVE_URL}}}{HEALTH_PATH}"
"""

_RECOMMENDATIONS_RUBY = f"""\
require "json"
require "net/http"
require "uri"

base = ENV.fetch("CICERONE_SERVE_URL", "{_DEFAULT_SERVE_URL}").sub(%r{{/\\z}}, "")
token = ENV.fetch("CICERONE_SERVE_TOKEN")
user_id = ENV.fetch("CICERONE_USER_ID", "alice")

uri = URI("#{{base}}{RECOMMENDATIONS_PATH_PREFIX}#{{URI.encode_www_form_component(user_id)}}")
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

# Node 18+ (built-in fetch). Works in CommonJS and ESM.
_RECOMMENDATIONS_JAVASCRIPT = f"""\
const baseUrl = (process.env.CICERONE_SERVE_URL || "{_DEFAULT_SERVE_URL}").replace(/\\/$/, "");
const token = process.env.CICERONE_SERVE_TOKEN;
if (!token) {{
  console.error("Set CICERONE_SERVE_TOKEN to a bearer token");
  process.exit(1);
}}
const userId = process.env.CICERONE_USER_ID || "alice";

async function main() {{
  const url = new URL(`${{baseUrl}}{RECOMMENDATIONS_PATH_PREFIX}${{encodeURIComponent(userId)}}`);
  url.searchParams.set("limit", "5");

  const response = await fetch(url, {{
    headers: {{
      Accept: "application/json",
      Authorization: `Bearer ${{token}}`,
    }},
  }});
  if (!response.ok) throw new Error(`${{response.status}} ${{await response.text()}}`);
  const body = await response.json();
  console.log(body.user_id, body.fallback, body.generated_at);
  for (const row of body.items) {{
    console.log(`  #${{row.rank}} ${{row.item_id}} score=${{row.score}}`);
  }}
}}

main().catch((err) => {{
  console.error(err);
  process.exitCode = 1;
}});
"""

_RECOMMENDATIONS_SHELL = f"""\
BASE_URL="${{CICERONE_SERVE_URL:-{_DEFAULT_SERVE_URL}}}"
BASE_URL="${{BASE_URL%/}}"
TOKEN="${{CICERONE_SERVE_TOKEN:?set CICERONE_SERVE_TOKEN}}"
USER_ID="${{CICERONE_USER_ID:-alice}}"
USER_ID_ENC="$(
  USER_ID="$USER_ID" python3 -c \\
    'import os, urllib.parse; print(urllib.parse.quote(os.environ["USER_ID"], safe=""))'
)"
curl -sS \\
  -H "Authorization: Bearer $TOKEN" \\
  "$BASE_URL{RECOMMENDATIONS_PATH_PREFIX}${{USER_ID_ENC}}?limit=5"
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
    health = paths.get(HEALTH_PATH, {}).get("get")
    if isinstance(health, dict):
        _extend_code_samples(health, HEALTH_CODE_SAMPLES)
    recommendations = paths.get(RECOMMENDATIONS_PATH, {}).get("get")
    if isinstance(recommendations, dict):
        _extend_code_samples(recommendations, RECOMMENDATIONS_CODE_SAMPLES)

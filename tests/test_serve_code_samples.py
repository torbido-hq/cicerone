"""Tests for OpenAPI x-codeSamples helpers."""

from __future__ import annotations

from cicerone.serve.code_samples import (
    DEFAULT_LIMIT,
    DEFAULT_SERVE_URL,
    DEFAULT_USER_ID,
    ENV_SERVE_TOKEN,
    ENV_SERVE_URL,
    ENV_USER_ID,
    EVENTS_CODE_SAMPLES,
    EVENTS_PATH,
    HEALTH_CODE_SAMPLES,
    HEALTH_PATH,
    RECOMMENDATIONS_CODE_SAMPLES,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_PATH_PREFIX,
    attach_code_samples,
)


def _sample_source(schema: dict, path: str, lang: str, method: str = "get") -> str:
    samples = schema["paths"][path][method]["x-codeSamples"]
    return next(s["source"] for s in samples if s["lang"] == lang)


def test_recommendations_path_prefix_derived_from_path():
    assert RECOMMENDATIONS_PATH_PREFIX == "/recommendations/"
    assert RECOMMENDATIONS_PATH.startswith(RECOMMENDATIONS_PATH_PREFIX)
    assert "{user_id}" not in RECOMMENDATIONS_PATH_PREFIX


def test_attach_code_samples_appends_to_existing():
    schema = {
        "paths": {
            HEALTH_PATH: {"get": {"x-codeSamples": [{"lang": "Go", "label": "net/http", "source": "x"}]}},
            RECOMMENDATIONS_PATH: {"get": {}},
            EVENTS_PATH: {"post": {}},
        }
    }
    attach_code_samples(schema)
    health_langs = [s["lang"] for s in schema["paths"][HEALTH_PATH]["get"]["x-codeSamples"]]
    assert health_langs[0] == "Go"
    assert {s["lang"] for s in HEALTH_CODE_SAMPLES}.issubset(health_langs)
    rec_langs = [s["lang"] for s in schema["paths"][RECOMMENDATIONS_PATH]["get"]["x-codeSamples"]]
    assert rec_langs == [s["lang"] for s in RECOMMENDATIONS_CODE_SAMPLES]
    events_langs = [s["lang"] for s in schema["paths"][EVENTS_PATH]["post"]["x-codeSamples"]]
    assert events_langs == [s["lang"] for s in EVENTS_CODE_SAMPLES]


def test_events_javascript_invariants():
    schema = {
        "paths": {
            HEALTH_PATH: {"get": {}},
            RECOMMENDATIONS_PATH: {"get": {}},
            EVENTS_PATH: {"post": {}},
        }
    }
    attach_code_samples(schema)
    js = _sample_source(schema, EVENTS_PATH, "JavaScript", method="post")
    assert ENV_SERVE_TOKEN in js
    assert "if (!token)" in js
    assert "Authorization:" in js and "Bearer" in js
    assert 'method: "POST"' in js
    assert EVENTS_PATH in js
    assert "occurred_at" in js


def test_events_shell_invariants():
    schema = {
        "paths": {
            HEALTH_PATH: {"get": {}},
            RECOMMENDATIONS_PATH: {"get": {}},
            EVENTS_PATH: {"post": {}},
        }
    }
    attach_code_samples(schema)
    shell = _sample_source(schema, EVENTS_PATH, "Shell", method="post")
    assert "curl -fsS -X POST" in shell
    assert f"{ENV_SERVE_TOKEN}:?" in shell
    assert EVENTS_PATH in shell
    assert "occurred_at" in shell
    assert "|| exit 1" in shell


def test_recommendations_javascript_invariants():
    schema = {"paths": {HEALTH_PATH: {"get": {}}, RECOMMENDATIONS_PATH: {"get": {}}}}
    attach_code_samples(schema)
    js = _sample_source(schema, RECOMMENDATIONS_PATH, "JavaScript")
    assert ENV_SERVE_TOKEN in js
    assert "if (!token)" in js
    assert "Authorization:" in js and "Bearer" in js
    assert "async function main()" in js
    assert "encodeURIComponent" in js
    assert DEFAULT_SERVE_URL in js
    assert ".replace(" in js


def test_recommendations_shell_invariants():
    schema = {"paths": {HEALTH_PATH: {"get": {}}, RECOMMENDATIONS_PATH: {"get": {}}}}
    attach_code_samples(schema)
    shell = _sample_source(schema, RECOMMENDATIONS_PATH, "Shell")
    assert "urllib.parse.quote" in shell
    assert f"{ENV_SERVE_TOKEN}:?" in shell
    assert f"{ENV_SERVE_URL}:-" in shell
    assert "${BASE_URL%/}" in shell
    assert RECOMMENDATIONS_PATH_PREFIX in shell
    assert f"{ENV_USER_ID}:-{DEFAULT_USER_ID}" in shell
    assert f"?limit={DEFAULT_LIMIT}" in shell
    assert "curl -fsS" in shell


def test_health_shell_fails_on_http_errors():
    schema = {"paths": {HEALTH_PATH: {"get": {}}, RECOMMENDATIONS_PATH: {"get": {}}}}
    attach_code_samples(schema)
    shell = _sample_source(schema, HEALTH_PATH, "Shell")
    assert "curl -fsS" in shell
    assert "|| exit 1" in shell
    assert ENV_SERVE_URL in shell
    assert DEFAULT_SERVE_URL in shell

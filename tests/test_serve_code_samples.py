"""Tests for OpenAPI x-codeSamples helpers."""

from __future__ import annotations

from cicerone.serve.code_samples import (
    HEALTH_PATH,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_PATH_PREFIX,
    attach_code_samples,
)


def test_recommendations_path_prefix_derived_from_path():
    assert RECOMMENDATIONS_PATH_PREFIX == "/recommendations/"
    assert RECOMMENDATIONS_PATH.startswith(RECOMMENDATIONS_PATH_PREFIX)
    assert "{user_id}" not in RECOMMENDATIONS_PATH_PREFIX


def test_attach_code_samples_appends_to_existing():
    schema = {
        "paths": {
            HEALTH_PATH: {"get": {"x-codeSamples": [{"lang": "Go", "label": "net/http", "source": "x"}]}},
            RECOMMENDATIONS_PATH: {"get": {}},
        }
    }
    attach_code_samples(schema)
    health_langs = [s["lang"] for s in schema["paths"][HEALTH_PATH]["get"]["x-codeSamples"]]
    assert health_langs[0] == "Go"
    assert "Ruby" in health_langs
    rec_langs = [s["lang"] for s in schema["paths"][RECOMMENDATIONS_PATH]["get"]["x-codeSamples"]]
    assert rec_langs[0] == "Ruby"


def test_recommendations_javascript_requires_token_and_avoids_top_level_await():
    schema = {"paths": {HEALTH_PATH: {"get": {}}, RECOMMENDATIONS_PATH: {"get": {}}}}
    attach_code_samples(schema)
    js = next(
        s["source"]
        for s in schema["paths"][RECOMMENDATIONS_PATH]["get"]["x-codeSamples"]
        if s["lang"] == "JavaScript"
    )
    assert "if (!token)" in js
    assert "Authorization: `Bearer ${token}`" in js
    assert "async function main()" in js
    assert "main().catch" in js
    assert "encodeURIComponent(userId)" in js


def test_recommendations_shell_url_encodes_user_id():
    schema = {"paths": {HEALTH_PATH: {"get": {}}, RECOMMENDATIONS_PATH: {"get": {}}}}
    attach_code_samples(schema)
    shell = next(
        s["source"]
        for s in schema["paths"][RECOMMENDATIONS_PATH]["get"]["x-codeSamples"]
        if s["lang"] == "Shell"
    )
    assert "urllib.parse.quote" in shell
    assert "USER_ID_ENC" in shell
    assert "CICERONE_SERVE_TOKEN:?" in shell
    assert "${CICERONE_SERVE_URL:-" in shell
    assert "${BASE_URL%/}" in shell
    assert RECOMMENDATIONS_PATH_PREFIX in shell

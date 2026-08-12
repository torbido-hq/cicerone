"""Tests for OpenAPI x-codeSamples helpers."""

from __future__ import annotations

from cicerone.serve.code_samples import attach_code_samples


def test_attach_code_samples_appends_to_existing():
    schema = {
        "paths": {
            "/health": {"get": {"x-codeSamples": [{"lang": "Go", "label": "net/http", "source": "x"}]}},
            "/recommendations/{user_id}": {"get": {}},
        }
    }
    attach_code_samples(schema)
    health_langs = [s["lang"] for s in schema["paths"]["/health"]["get"]["x-codeSamples"]]
    assert health_langs[0] == "Go"
    assert "Ruby" in health_langs
    rec_langs = [s["lang"] for s in schema["paths"]["/recommendations/{user_id}"]["get"]["x-codeSamples"]]
    assert rec_langs[0] == "Ruby"

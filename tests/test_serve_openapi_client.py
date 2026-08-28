from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from test_serve import _FakeManifest, _FakeReader, _feature_config, _items_df, _recs_df, _settings

from cicerone.export_serve_openapi import build_openapi
from cicerone.export_serve_openapi import main as export_main
from cicerone.serve import SERVE_API_TITLE, SERVE_API_VERSION, create_app
from cicerone.serve.code_samples import (
    ENV_SERVE_TOKEN,
    ENV_SERVE_URL,
    EVENTS_CODE_SAMPLES,
    EVENTS_PATH,
    HEALTH_CODE_SAMPLES,
    HEALTH_PATH,
    RECOMMENDATIONS_CODE_SAMPLES,
    RECOMMENDATIONS_PATH,
)
from cicerone.serve_client import ServeClient, ServeClientError
from cicerone.serve_schemas import HealthResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = REPO_ROOT / "docs" / "openapi" / "serve.openapi.json"


def _app(**kwargs):
    return create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
        **kwargs,
    )


def test_openapi_json_lists_serve_paths_and_schemas():
    client = TestClient(_app())
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == SERVE_API_TITLE
    assert schema["info"]["version"] == SERVE_API_VERSION
    assert "cicerone export-openapi" in schema["info"]["description"]
    assert HEALTH_PATH in schema["paths"]
    assert RECOMMENDATIONS_PATH in schema["paths"]

    components = schema["components"]["schemas"]
    assert "RecommendationsResponse" in components
    assert "experiment_id" in components["RecommendationsResponse"]["properties"]
    assert "variant" in components["RecommendationsResponse"]["properties"]
    assert "RecommendationItem" in components
    assert "HealthResponse" in components
    assert "ErrorDetail" in components

    header_components = schema["components"]["headers"]
    assert "X-Generated-At" in header_components
    assert header_components["X-Generated-At"]["schema"]["type"] == "string"

    rec = schema["paths"][RECOMMENDATIONS_PATH]["get"]
    assert "X-Generated-At" in rec["responses"]["200"].get("headers", {})

    for status_code in ("400", "401", "404"):
        error_response = rec["responses"][status_code]
        error_schema = error_response["content"]["application/json"]["schema"]
        assert error_schema.get("$ref") == "#/components/schemas/ErrorDetail"

    assert schema["components"]["securitySchemes"]

    health_samples = schema["paths"][HEALTH_PATH]["get"]["x-codeSamples"]
    assert {s["lang"] for s in health_samples} >= {s["lang"] for s in HEALTH_CODE_SAMPLES}
    assert all(ENV_SERVE_URL in sample["source"] for sample in health_samples)

    rec_samples = rec["x-codeSamples"]
    assert {s["lang"] for s in rec_samples} >= {s["lang"] for s in RECOMMENDATIONS_CODE_SAMPLES}
    ruby_rec = next(sample for sample in rec_samples if sample["lang"] == "Ruby")
    assert ENV_SERVE_URL in ruby_rec["source"]
    assert ENV_SERVE_TOKEN in ruby_rec["source"]


def test_exported_openapi_includes_events_code_samples():
    schema = build_openapi()
    samples = schema["paths"][EVENTS_PATH]["post"]["x-codeSamples"]
    assert {s["lang"] for s in samples} >= {s["lang"] for s in EVENTS_CODE_SAMPLES}
    assert all("occurred_at" in sample["source"] for sample in samples)


def test_docs_ui_is_available():
    client = TestClient(_app())
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_committed_openapi_matches_generated_schema():
    generated = build_openapi()
    assert OPENAPI_PATH.is_file(), f"missing {OPENAPI_PATH}; run: cicerone export-openapi -o {OPENAPI_PATH}"
    committed = json.loads(OPENAPI_PATH.read_text())
    assert committed == generated


def test_export_serve_openapi_writes_file(tmp_path):
    out = tmp_path / "serve.openapi.json"
    assert export_main(["-o", str(out)]) == 0
    assert json.loads(out.read_text()) == build_openapi()


def test_export_serve_openapi_stdout(capsys):
    assert export_main([]) == 0
    assert json.loads(capsys.readouterr().out) == build_openapi()


def test_generated_at_header_matches_body():
    client = TestClient(_app())
    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] == "2026-08-04T12:00:00+00:00"
    assert response.headers["X-Generated-At"] == body["generated_at"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_serve_url():
    port = _free_port()
    config = uvicorn.Config(_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("uvicorn failed to start for ServeClient test")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_serve_client_health_and_recommendations(live_serve_url):
    client = ServeClient(live_serve_url, token="secret")
    assert client.health() == HealthResponse(status="ok")

    body = client.recommendations("u1", limit=1)
    assert body.user_id == "u1"
    assert body.fallback is False
    assert body.generated_at == "2026-08-04T12:00:00+00:00"
    assert len(body.items) == 1
    assert body.items[0].item_id == "i1"


def test_serve_client_category_and_auth_errors(live_serve_url):
    client = ServeClient(live_serve_url, token="secret")
    body = client.recommendations("u1", category="wine", exclude_unavailable=True)
    assert [row.item_id for row in body.items] == ["i2"]

    unauth = ServeClient(live_serve_url, token="wrong")
    with pytest.raises(ServeClientError) as exc_info:
        unauth.recommendations("u1")
    assert exc_info.value.status_code == 401


def test_serve_client_conflict_limit_k(live_serve_url):
    client = ServeClient(live_serve_url, token="secret")
    with pytest.raises(ServeClientError) as exc_info:
        client.recommendations("u1", limit=1, k=2)
    assert exc_info.value.status_code == 400


def test_serve_api_version_tracks_package_version():
    from cicerone import __version__

    assert __version__ == SERVE_API_VERSION


def test_serve_client_error_detail_fallbacks(monkeypatch):
    import io
    import urllib.error

    import cicerone.serve_client as client_mod

    class _FakeHTTPError(urllib.error.HTTPError):
        def __init__(self, *, code: int, payload: bytes, reason: str = "Boom"):
            super().__init__(
                url="http://example.test/x",
                code=code,
                msg=reason,
                hdrs=None,
                fp=io.BytesIO(payload),
            )

    def _patch(payload: bytes, reason: str = "Boom"):
        def fake_urlopen(_request, timeout=None):
            del timeout
            raise _FakeHTTPError(code=502, payload=payload, reason=reason)

        monkeypatch.setattr(client_mod.urllib.request, "urlopen", fake_urlopen)

    client = ServeClient("http://example.test")

    _patch(b'{"error":"nope"}')
    with pytest.raises(ServeClientError) as missing_detail:
        client.health()
    assert missing_detail.value.detail == '{"error":"nope"}'
    assert "None" not in str(missing_detail.value)

    _patch(b'{"message":"upstream failed"}')
    with pytest.raises(ServeClientError) as message_only:
        client.health()
    assert message_only.value.detail == "upstream failed"

    _patch(b"not-json", reason="Bad Gateway")
    with pytest.raises(ServeClientError) as plain:
        client.health()
    assert plain.value.detail == "not-json"

    _patch(b"", reason="Bad Gateway")
    with pytest.raises(ServeClientError) as empty:
        client.health()
    assert empty.value.detail == "Bad Gateway"

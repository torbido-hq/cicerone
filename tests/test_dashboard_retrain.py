from __future__ import annotations

import urllib.error
from dataclasses import replace

import pytest
from conftest import make_settings

from cicerone import dashboard_retrain
from cicerone.dashboard_retrain import RetrainError, request_retrain, retrain_configured, trigger_base_url


def test_loopback_rewrites_wildcard_bind_hosts():
    assert (
        trigger_base_url(make_settings(trigger_host="0.0.0.0", trigger_port=8080)) == "http://127.0.0.1:8080"
    )
    assert trigger_base_url(make_settings(trigger_host="::", trigger_port=9)) == "http://[::1]:9"
    assert trigger_base_url(make_settings(trigger_host="0:0:0:0:0:0:0:0", trigger_port=9)) == "http://[::1]:9"
    assert trigger_base_url(make_settings(trigger_host="::1", trigger_port=9)) == "http://[::1]:9"
    assert trigger_base_url(make_settings(trigger_host="127.0.0.2", trigger_port=9)) == "http://127.0.0.2:9"
    assert trigger_base_url(make_settings(trigger_host="localhost", trigger_port=9)) == "http://localhost:9"


def test_a_trigger_on_another_host_needs_an_explicit_url():
    with pytest.raises(RetrainError, match="trigger_url"):
        trigger_base_url(make_settings(trigger_host="trigger.internal", trigger_port=8080))


def _with_dashboard_trigger(url: str, token: str | None = None, **kwargs):
    settings = make_settings(**kwargs)
    return replace(settings, dashboard=replace(settings.dashboard, trigger_url=url, trigger_auth_token=token))


def test_explicit_trigger_url_and_token_are_used_without_job_trigger(monkeypatch):
    captured: dict[str, str] = {}

    class _Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["method"] = request.get_method()
        return _Response()

    monkeypatch.setattr(dashboard_retrain._OPENER, "open", _urlopen)
    settings = _with_dashboard_trigger("http://recommender:8080", "dash-token", trigger_enabled=False)
    assert retrain_configured(settings) is True
    assert request_retrain(settings) == "Retrain started"
    assert captured == {
        "url": "http://recommender:8080/trigger/retrain",
        "auth": "Bearer dash-token",
        "method": "POST",
    }


@pytest.mark.parametrize(
    "failure", [urllib.error.URLError(TimeoutError("timed out")), TimeoutError("read timed out")]
)
def test_timeout_is_reported_as_ambiguous(monkeypatch, caplog, failure):
    def _urlopen(request, timeout=0):
        raise failure

    monkeypatch.setattr(dashboard_retrain._OPENER, "open", _urlopen)
    with (
        caplog.at_level("WARNING", logger="cicerone.dashboard_retrain"),
        pytest.raises(RetrainError, match="may still start"),
    ):
        request_retrain(make_settings(trigger_enabled=True))
    assert any("timed out" in record.getMessage() for record in caplog.records)


def test_disabled_trigger_is_refused():
    with pytest.raises(RetrainError, match="not enabled"):
        request_retrain(make_settings(trigger_enabled=False))


def test_unreachable_trigger(monkeypatch, caplog):
    def _urlopen(request, timeout=0):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(dashboard_retrain._OPENER, "open", _urlopen)
    with (
        caplog.at_level("WARNING", logger="cicerone.dashboard_retrain"),
        pytest.raises(RetrainError, match="unreachable"),
    ):
        request_retrain(make_settings(trigger_enabled=True))
    assert any("unreachable: down" in record.getMessage() for record in caplog.records)


def test_unexpected_status_is_a_failure(monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(dashboard_retrain._OPENER, "open", lambda *_a, **_k: _Response())
    with pytest.raises(RetrainError, match="failed"):
        request_retrain(make_settings(trigger_enabled=True, trigger_auth_token=None))


def test_http_error_is_a_failure(monkeypatch):
    def _urlopen(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 500, "nope", hdrs=None, fp=None)

    monkeypatch.setattr(dashboard_retrain._OPENER, "open", _urlopen)
    with pytest.raises(RetrainError, match="failed"):
        request_retrain(make_settings(trigger_enabled=True, trigger_auth_token=None))


def test_a_redirect_is_not_followed_and_the_token_stays_put(caplog):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    leaked: list[str | None] = []

    class _Elsewhere(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            leaked.append(self.headers.get("Authorization"))
            self.send_response(202)
            self.end_headers()

        do_POST = do_GET  # noqa: N815

        def log_message(self, *_args):
            return None

    elsewhere = ThreadingHTTPServer(("127.0.0.1", 0), _Elsewhere)
    target = f"http://localhost:{elsewhere.server_address[1]}/trigger/retrain"

    class _Redirect(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            return None

    trigger = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    for server in (elsewhere, trigger):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        settings = _with_dashboard_trigger(
            f"http://127.0.0.1:{trigger.server_address[1]}", token="secret-token"
        )
        with pytest.raises(RetrainError, match="Retrain trigger failed"):
            request_retrain(settings)
    finally:
        for server in (elsewhere, trigger):
            server.shutdown()
            server.server_close()
    assert leaked == []
    assert "answered 302" in caplog.text


def test_a_connection_dropped_after_the_post_is_reported_as_ambiguous(caplog):
    import socket
    import threading

    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def _accept_then_drop() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)

    worker = threading.Thread(target=_accept_then_drop, daemon=True)
    worker.start()
    try:
        settings = _with_dashboard_trigger(f"http://127.0.0.1:{port}", token="tok")
        with pytest.raises(RetrainError, match="may have started"):
            request_retrain(settings)
    finally:
        worker.join(timeout=5)
        listener.close()
    assert "dropped the connection" in caplog.text


def test_config_and_the_derived_url_agree_on_loopback_hosts():
    from cicerone.config import ConfigError
    from cicerone.config.load import _load_dashboard_trigger_url

    def url_host(host: str) -> str:
        return f"[{host}]" if ":" in host else host

    for host in ("localhost", "LOCALHOST", "127.0.0.1", "127.0.0.2", "::1"):
        assert _load_dashboard_trigger_url(f"http://{url_host(host)}:8080", allow_http=False)
        assert (
            trigger_base_url(make_settings(trigger_host=host, trigger_port=9)) == f"http://{url_host(host)}:9"
        )
    for host in ("recommender", "10.0.0.5", "loopback.example"):
        with pytest.raises(ConfigError, match="trigger_allow_http"):
            _load_dashboard_trigger_url(f"http://{host}:8080", allow_http=False)
        with pytest.raises(RetrainError):
            trigger_base_url(make_settings(trigger_host=host, trigger_port=9))
    # A wildcard bind derives a loopback URL, but is never a loopback target itself.
    for host in ("0.0.0.0", "::"):
        with pytest.raises(ConfigError, match="trigger_allow_http"):
            _load_dashboard_trigger_url(f"http://{url_host(host)}:8080", allow_http=False)

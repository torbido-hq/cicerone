from __future__ import annotations

import urllib.error
from dataclasses import replace

import pytest
from conftest import make_settings

from cicerone.dashboard_retrain import RetrainError, request_retrain, retrain_configured, trigger_base_url


def test_loopback_rewrites_wildcard_bind_hosts():
    assert (
        trigger_base_url(make_settings(trigger_host="0.0.0.0", trigger_port=8080)) == "http://127.0.0.1:8080"
    )
    assert trigger_base_url(make_settings(trigger_host="::", trigger_port=9)) == "http://127.0.0.1:9"
    assert trigger_base_url(make_settings(trigger_host="::1", trigger_port=9)) == "http://[::1]:9"


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
        return _Response()

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", _urlopen)
    settings = _with_dashboard_trigger("http://recommender:8080", "dash-token", trigger_enabled=False)
    assert retrain_configured(settings) is True
    assert request_retrain(settings) == "Retrain started"
    assert captured == {"url": "http://recommender:8080/trigger/retrain", "auth": "Bearer dash-token"}


@pytest.mark.parametrize(
    "failure", [urllib.error.URLError(TimeoutError("timed out")), TimeoutError("read timed out")]
)
def test_timeout_is_reported_as_ambiguous(monkeypatch, caplog, failure):
    def _urlopen(request, timeout=0):
        raise failure

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", _urlopen)
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

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", _urlopen)
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

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", lambda *_a, **_k: _Response())
    with pytest.raises(RetrainError, match="failed"):
        request_retrain(make_settings(trigger_enabled=True, trigger_auth_token=None))


def test_http_error_is_a_failure(monkeypatch):
    def _urlopen(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 500, "nope", hdrs=None, fp=None)

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", _urlopen)
    with pytest.raises(RetrainError, match="failed"):
        request_retrain(make_settings(trigger_enabled=True, trigger_auth_token=None))

from __future__ import annotations

import urllib.error

import pytest
from conftest import make_settings

from cicerone.dashboard_retrain import RetrainError, request_retrain, trigger_base_url


def test_loopback_rewrites_wildcard_bind_hosts():
    assert (
        trigger_base_url(make_settings(trigger_host="0.0.0.0", trigger_port=8080)) == "http://127.0.0.1:8080"
    )
    assert trigger_base_url(make_settings(trigger_host="::", trigger_port=9)) == "http://127.0.0.1:9"
    assert trigger_base_url(make_settings(trigger_host="trigger.internal", trigger_port=8080)) == (
        "http://trigger.internal:8080"
    )


def test_disabled_trigger_is_refused():
    with pytest.raises(RetrainError, match="not enabled"):
        request_retrain(make_settings(trigger_enabled=False))


def test_unreachable_trigger(monkeypatch):
    def _urlopen(request, timeout=0):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("cicerone.dashboard_retrain.urllib.request.urlopen", _urlopen)
    with pytest.raises(RetrainError, match="unreachable"):
        request_retrain(make_settings(trigger_enabled=True))


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

from __future__ import annotations

import pytest

from cicerone.events.registry import build_event_source, register_event_source, registered_event_source_kinds
from cicerone.events.webhook import WebhookEventSource


def test_register_and_build_webhook():
    assert "webhook" in registered_event_source_kinds()
    assert "db" in registered_event_source_kinds()
    assert "s3" in registered_event_source_kinds()
    assert "redis_streams" in registered_event_source_kinds()
    source = build_event_source("webhook", {})
    assert isinstance(source, WebhookEventSource)


def test_config_kind_validation_uses_registry():
    from cicerone.config.events import load_events_settings

    settings = load_events_settings(
        {"enabled": True, "kind": "webhook", "options": {"auth_token": "t"}},
        mode="serve",
        serve_auth_token=None,
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "webhook"


def test_build_unknown_kind():
    with pytest.raises(ValueError, match="Unknown events kind"):
        build_event_source("nope", {})


def test_register_duplicate_kind():
    with pytest.raises(ValueError, match="already registered"):
        register_event_source("webhook", lambda options: WebhookEventSource(options))

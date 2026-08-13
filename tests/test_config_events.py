from __future__ import annotations

import pytest
from support.toml_config import write_toml

from cicerone.config import ConfigError, EventsSettings, load_settings, make_settings
from cicerone.config.events import coerce_events_settings, load_events_settings


def test_make_settings_events_defaults():
    settings = make_settings()
    assert settings.events.enabled is False
    assert settings.events.kind == "webhook"
    assert settings.events.incremental.batch_size == 100
    assert settings.events.incremental.poll_interval_seconds == 1.0
    assert settings.events_enabled is False
    assert settings.events_kind == "webhook"


def test_coerce_events_settings_errors():
    with pytest.raises(TypeError):
        coerce_events_settings("bad")
    with pytest.raises(TypeError):
        coerce_events_settings({"incremental": "bad"})
    with pytest.raises(ValueError):
        coerce_events_settings({"incremental": {"batch_size": 0}})


def test_load_events_section(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        [events.options]
        auth_token = "events-tok"
        [events.incremental]
        batch_size = 5
        batch_window_seconds = 12
        poll_interval_seconds = 0.5
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    settings = load_settings(path)
    assert settings.events.enabled is True
    assert settings.events.options["auth_token"] == "events-tok"
    assert settings.events.incremental.batch_size == 5
    assert settings.events.incremental.batch_window_seconds == 12.0
    assert settings.events.incremental.poll_interval_seconds == 0.5


def test_load_events_unknown_kind(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [events]
        enabled = true
        kind = "kafka"
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="events.kind"):
        load_settings(path)


def test_load_events_webhook_requires_auth_in_serve_mode():
    with pytest.raises(ConfigError, match="auth_token"):
        load_events_settings(
            {"enabled": True, "kind": "webhook", "options": {}},
            mode="serve",
            serve_auth_token=None,
            resolve_env=lambda value, _path: value,
        )


def test_load_events_disabled_allows_unknown_kind():
    settings = load_events_settings(
        {"enabled": False, "kind": "kafka"},
        mode="batch",
        serve_auth_token=None,
        resolve_env=lambda value, _path: value,
    )
    assert settings.enabled is False
    assert settings.kind == "kafka"


def test_coerce_pass_through_events_settings():
    original = EventsSettings(enabled=True, kind="webhook")
    coerced = coerce_events_settings(original)
    assert coerced.enabled is True
    assert coerced.incremental.batch_size == 100

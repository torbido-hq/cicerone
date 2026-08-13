from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from support.toml_config import write_toml
from test_serve import _FakeReader, _recs_df

from cicerone.config import ConfigError, EventsSettings, load_settings, make_settings
from cicerone.events.webhook import WebhookEventSource
from cicerone.serve import create_app


def _settings(**overrides):
    return make_settings(
        **{
            "mode": "serve",
            "serve_auth_token": "secret",
            "events": EventsSettings(enabled=True, kind="webhook"),
            **overrides,
        }
    )


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


def test_post_events_single_and_batch():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    single = client.post(
        "/events",
        headers=headers,
        json={
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-08-13T12:00:00Z",
            "event_id": "e1",
        },
    )
    assert single.status_code == 202
    assert single.json()["accepted"] == 1
    batch = client.post(
        "/events",
        headers=headers,
        json={
            "events": [
                {
                    "user_id": "u1",
                    "item_id": "i2",
                    "event_type": "view",
                    "occurred_at": "2026-08-13T12:01:00Z",
                    "event_id": "e2",
                }
            ]
        },
    )
    assert batch.status_code == 202
    assert batch.json()["accepted"] == 1
    assert source.health().lag == 2


def test_post_events_auth_and_validation():
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=WebhookEventSource({}))
    client = TestClient(app)
    assert client.post("/events", json={"user_id": "u1"}).status_code == 401
    bad = client.post(
        "/events",
        headers={"Authorization": "Bearer secret"},
        json={"user_id": "u1"},
    )
    assert bad.status_code == 400


def test_events_route_absent_when_disabled():
    app = create_app(
        make_settings(mode="serve", serve_auth_token="secret"),
        _FakeReader(_recs_df()),
    )
    client = TestClient(app)
    assert client.post("/events", headers={"Authorization": "Bearer secret"}, json={}).status_code == 404


def test_post_events_list_body_and_invalid_json_shape():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    listed = client.post(
        "/events",
        headers=headers,
        json=[
            {
                "user_id": "u1",
                "item_id": "i3",
                "event_type": "purchase",
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "list-1",
            }
        ],
    )
    assert listed.status_code == 202
    assert listed.json()["accepted"] == 1
    bad_shape = client.post(
        "/events",
        headers={**headers, "Content-Type": "application/json"},
        content=b"null",
    )
    assert bad_shape.status_code == 400

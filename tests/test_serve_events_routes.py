from __future__ import annotations

from fastapi.testclient import TestClient
from test_serve import _FakeReader, _recs_df

from cicerone.config import EventsSettings, make_settings
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
    body = single.json()
    assert body["accepted"] == 1
    assert body["event_ids"] == ["e1"]
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


def test_events_route_absent_for_non_webhook_kind():
    app = create_app(
        _settings(events=EventsSettings(enabled=True, kind="db")),
        _FakeReader(_recs_df()),
    )
    assert (
        TestClient(app).post("/events", headers={"Authorization": "Bearer secret"}, json={}).status_code
        == 404
    )


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
    empty = client.post("/events", headers=headers, json={"events": []})
    assert empty.status_code == 202
    assert empty.json()["accepted"] == 0
    bad_shape = client.post(
        "/events",
        headers={**headers, "Content-Type": "application/json"},
        content=b"null",
    )
    assert bad_shape.status_code == 400


def test_post_events_uses_events_auth_token():
    source = WebhookEventSource({})
    app = create_app(
        _settings(events=EventsSettings(enabled=True, kind="webhook", options={"auth_token": "events"})),
        _FakeReader(_recs_df()),
        event_source=source,
    )
    client = TestClient(app)
    denied = client.post(
        "/events",
        headers={"Authorization": "Bearer secret"},
        json={
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-08-13T12:00:00Z",
        },
    )
    assert denied.status_code == 401
    ok = client.post(
        "/events",
        headers={"Authorization": "Bearer events"},
        json={
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-08-13T12:00:00Z",
            "event_id": "tok-1",
        },
    )
    assert ok.status_code == 202

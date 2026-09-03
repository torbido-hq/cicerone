from __future__ import annotations

from fastapi.testclient import TestClient
from test_serve import _FakeReader, _recs_df

from cicerone.config import EventsSettings, make_settings
from cicerone.events.webhook import WebhookEventSource
from cicerone.serve import create_app
from cicerone.serve_schemas import ValidationErrorDetail


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
    detail = bad.json()["detail"]
    assert isinstance(detail, list)
    assert detail
    assert all(isinstance(item, dict) and "loc" in item and "msg" in item for item in detail)
    ValidationErrorDetail.model_validate({"detail": detail})


def test_post_events_list_body_validation_error_returns_400():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    response = client.post(
        "/events",
        headers={"Authorization": "Bearer secret"},
        json=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
            },
            {
                "user_id": "u2",
                "item_id": "i2",
                "event_type": "purchase",
                "occurred_at": "2026-08-13T12:00:00Z",
            },
        ],
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail
    assert all(isinstance(item, dict) and "loc" in item and "msg" in item for item in detail)
    ValidationErrorDetail.model_validate({"detail": detail})


def test_events_openapi_documents_structured_validation_400():
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=WebhookEventSource({}))
    schema = TestClient(app).get("/openapi.json").json()
    assert "ValidationErrorDetail" in schema["components"]["schemas"]
    assert "ValidationErrorItem" in schema["components"]["schemas"]
    events_400 = schema["paths"]["/events"]["post"]["responses"]["400"]
    content_schema = events_400["content"]["application/json"]["schema"]
    refs = {item.get("$ref") for item in content_schema.get("anyOf", [])}
    assert "#/components/schemas/ErrorDetail" in refs
    assert "#/components/schemas/ValidationErrorDetail" in refs


def test_events_openapi_documents_occurred_at_union():
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=WebhookEventSource({}))
    schema = TestClient(app).get("/openapi.json").json()
    assert "InteractionEvent" in schema["components"]["schemas"]
    assert "EventsIngestRequest" in schema["components"]["schemas"]
    request_body = schema["paths"]["/events"]["post"]["requestBody"]
    assert request_body["required"] is True
    body_schema = request_body["content"]["application/json"]["schema"]
    assert {"$ref": "#/components/schemas/InteractionEvent"} in body_schema["oneOf"]
    occurred_at = schema["components"]["schemas"]["InteractionEvent"]["properties"]["occurred_at"]
    types = {item.get("type") for item in occurred_at.get("anyOf", [])}
    assert types == {"string", "integer", "number"}


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


def test_post_events_normalization_error_returns_400():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    response = client.post(
        "/events",
        headers={"Authorization": "Bearer secret"},
        json=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "occurred_at": "2026-08-13T12:00:00",
            }
        ],
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "occurred_at" in str(detail)


def test_post_events_unix_epoch_occurred_at_accepted():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    response = client.post(
        "/events",
        headers={"Authorization": "Bearer secret"},
        json={
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": 1_724_000_000,
            "event_id": "epoch-1",
        },
    )
    assert response.status_code == 202
    assert response.json()["accepted"] == 1
    assert response.json()["event_ids"] == ["epoch-1"]


def test_post_events_non_json_payload():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    resp = client.post(
        "/events",
        headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"},
        content="not json at all",
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Request body must be JSON"}


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


def test_post_events_backpressure_429():
    source = WebhookEventSource({"max_pending": 100})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    for i in range(100):
        response = client.post(
            "/events",
            headers=headers,
            json={
                "user_id": "u1",
                "item_id": f"i{i}",
                "event_type": "purchase",
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": f"e{i}",
            },
        )
        assert response.status_code == 202
    overflow = client.post(
        "/events",
        headers=headers,
        json={
            "user_id": "u1",
            "item_id": "i-overflow",
            "event_type": "view",
            "occurred_at": "2026-08-13T12:01:00Z",
            "event_id": "e-overflow",
        },
    )
    assert overflow.status_code == 429
    body = overflow.json()
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], str)
    assert "backlog full" in body["detail"]


def test_post_events_rejects_oversized_body():
    source = WebhookEventSource({})
    app = create_app(
        _settings(events=EventsSettings(enabled=True, kind="webhook", options={"max_body_bytes": 64})),
        _FakeReader(_recs_df()),
        event_source=source,
    )
    response = TestClient(app).post(
        "/events",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=b'{"user_id":"' + b"u" * 200 + b'","item_id":"i1","event_type":"view"}',
    )
    assert response.status_code == 413


def test_post_events_rejects_invalid_json():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    response = TestClient(app).post(
        "/events",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=b"not-json",
    )
    assert response.status_code == 400


def test_post_events_rejects_invalid_utf8_body():
    source = WebhookEventSource({})
    app = create_app(_settings(), _FakeReader(_recs_df()), event_source=source)
    response = TestClient(app).post(
        "/events",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=b"\xff\xfe",
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Request body must be JSON"}


def test_post_events_rejects_oversized_content_length():
    source = WebhookEventSource({})
    app = create_app(
        _settings(events=EventsSettings(enabled=True, kind="webhook", options={"max_body_bytes": 64})),
        _FakeReader(_recs_df()),
        event_source=source,
    )
    response = TestClient(app).post(
        "/events",
        headers={
            "Authorization": "Bearer secret",
            "content-type": "application/json",
            "content-length": "9999",
        },
        content=b"{}",
    )
    assert response.status_code == 413

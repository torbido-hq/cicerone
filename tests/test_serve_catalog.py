from __future__ import annotations

from fastapi.testclient import TestClient
from test_serve import _FakeReader, _recs_df, _settings

from cicerone.config import EventsSettings
from cicerone.events.consumed import ConsumedOverlay
from cicerone.io.dataset_catalog import DatasetCatalogStore
from cicerone.io.db_catalog import DatabaseCatalogStore
from cicerone.serve import create_app


def test_catalog_routes_without_store_are_not_implemented():
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=None)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    missing = client.get("/users/u1", headers=headers)
    assert missing.status_code == 501
    assert missing.json()["detail"] == "Catalog CRUD requires a writable dataset or table-backed db input"
    assert client.get("/items/i1", headers=headers).status_code == 501


def test_dataset_catalog_crud_round_trip(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    assert client.put("/users/u1", json={"comment": "alice"}, headers=headers).json()["accepted"] == 1
    user = client.get("/users/u1", headers=headers).json()["row"]
    assert user["user_id"] == "u1"
    assert user["comment"] == "alice"

    assert client.put("/items/i9", json={"comment": "new"}, headers=headers).json()["accepted"] == 1
    item = client.get("/items/i9", headers=headers).json()["row"]
    assert item["item_id"] == "i9"

    posted = client.post(
        "/catalog/events",
        json={
            "events": [
                {
                    "user_id": "u1",
                    "item_id": "i9",
                    "event_type": "purchase",
                    "occurred_at": "2026-09-11T12:00:00Z",
                    "event_id": "e1",
                }
            ]
        },
        headers=headers,
    )
    assert posted.json()["accepted"] == 1
    events = client.get("/catalog/events/u1", headers=headers).json()["events"]
    assert events[0]["item_id"] == "i9"
    assert client.delete("/catalog/events/u1?item_id=i9", headers=headers).json()["accepted"] == 1
    assert client.get("/catalog/events/u1", headers=headers).json()["events"] == []
    assert client.delete("/items/i9", headers=headers).json()["accepted"] == 1
    assert client.get("/items/i9", headers=headers).status_code == 404
    assert client.delete("/users/u1", headers=headers).status_code == 200
    assert client.get("/users/u1", headers=headers).status_code == 404


def test_catalog_events_reject_blank_ids(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store)
    response = TestClient(app).post(
        "/catalog/events",
        json={
            "events": [
                {
                    "user_id": "   ",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "occurred_at": "2026-09-11T12:00:00Z",
                }
            ]
        },
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_catalog_events_update_consumed_overlay(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    overlay = ConsumedOverlay()
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store, consumed=overlay)
    response = TestClient(app).post(
        "/catalog/events",
        json={
            "events": [
                {
                    "user_id": "u1",
                    "item_id": "i9",
                    "event_type": "purchase",
                    "occurred_at": "2026-09-11T12:00:00Z",
                    "event_id": "e1",
                }
            ]
        },
        headers={"Authorization": "Bearer secret"},
    )
    assert response.json()["accepted"] == 1
    assert overlay.item_ids("u1") == {"i9"}


def test_catalog_events_reject_oversized_batch(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    response = TestClient(create_app(_settings(), _FakeReader(_recs_df()), catalog=store)).post(
        "/catalog/events",
        json={
            "events": [
                {
                    "user_id": "u1",
                    "item_id": f"i{n}",
                    "event_type": "view",
                    "occurred_at": "2026-09-11T12:00:00Z",
                }
                for n in range(1001)
            ]
        },
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 422


def test_catalog_events_reject_blank_event_type(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    response = TestClient(create_app(_settings(), _FakeReader(_recs_df()), catalog=store)).post(
        "/catalog/events",
        json={
            "events": [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "   ",
                    "occurred_at": "2026-09-11T12:00:00Z",
                }
            ]
        },
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_catalog_put_rejects_invalid_sql_column():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    response = TestClient(create_app(_settings(), _FakeReader(_recs_df()), catalog=store)).put(
        "/users/u1",
        json={"bad-name": "x"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_catalog_put_rejects_whitespace_path_ids(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    response = TestClient(create_app(_settings(), _FakeReader(_recs_df()), catalog=store)).put(
        "/users/%20",
        json={"comment": "alice"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_catalog_events_reject_oversized_body(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(events=EventsSettings(options={"max_body_bytes": 64})),
        _FakeReader(_recs_df()),
        catalog=store,
    )
    response = TestClient(app).post(
        "/catalog/events",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=b'{"events":[{"user_id":"' + b"u" * 200 + b'","item_id":"i1","event_type":"view"}]}',
    )
    assert response.status_code == 413


def test_catalog_events_replace_overlay_item_for_same_event_id(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    overlay = ConsumedOverlay()
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store, consumed=overlay)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    first = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "occurred_at": "2026-09-11T12:00:00Z",
        "event_id": "e1",
    }
    assert client.post("/catalog/events", json={"events": [first]}, headers=headers).json()["accepted"] == 1
    assert overlay.item_ids("u1") == {"i1"}
    replaced = {**first, "item_id": "i2", "occurred_at": "2026-09-11T13:00:00Z"}
    posted = client.post("/catalog/events", json={"events": [replaced]}, headers=headers)
    assert posted.json()["accepted"] == 1
    assert overlay.item_ids("u1") == {"i2"}


def test_catalog_events_keep_hidden_item_when_sibling_event_still_consumes_it(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    overlay = ConsumedOverlay()
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store, consumed=overlay)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    first = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "occurred_at": "2026-09-11T12:00:00Z",
        "event_id": "e1",
    }
    sibling = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "view",
        "occurred_at": "2026-09-11T12:30:00Z",
        "event_id": "e2",
    }
    assert (
        client.post("/catalog/events", json={"events": [first, sibling]}, headers=headers).json()[
            "accepted"
        ]
        == 2
    )
    assert overlay.item_ids("u1") == {"i1"}
    moved = {**first, "item_id": "i2", "occurred_at": "2026-09-11T13:00:00Z"}
    posted = client.post("/catalog/events", json={"events": [moved]}, headers=headers)
    assert posted.json()["accepted"] == 1
    assert overlay.item_ids("u1") == {"i1", "i2"}


def test_catalog_delete_clears_consumed_overlay(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    overlay = ConsumedOverlay()
    overlay.add("u1", "i9")
    overlay.add("u1", "i2")
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=store, consumed=overlay)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    client.delete("/catalog/events/u1?item_id=i9", headers=headers)
    assert overlay.item_ids("u1") == {"i2"}
    client.delete("/users/u1", headers=headers)
    assert overlay.item_ids("u1") == set()

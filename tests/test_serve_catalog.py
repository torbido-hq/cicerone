from __future__ import annotations

from fastapi.testclient import TestClient
from test_serve import _FakeReader, _recs_df, _settings

from cicerone.io.dataset_catalog import DatasetCatalogStore
from cicerone.serve import create_app


def test_catalog_routes_without_store_are_not_implemented():
    app = create_app(_settings(), _FakeReader(_recs_df()), catalog=None)
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/users/u1", headers=headers).status_code == 501
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

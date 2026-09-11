from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from cicerone.io.catalog import (
    filter_item_row,
    item_row_or_none,
    jsonable_row,
    normalize_event_row,
    require_id,
    user_row_or_none,
)
from cicerone.io.dataset_catalog import DatasetCatalogStore


def test_require_id_rejects_blank():
    with pytest.raises(ValueError, match="user_id"):
        require_id({"user_id": "  "}, "user_id")


def test_normalize_event_row_defaults_quantity():
    row = normalize_event_row(
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-09-11T12:00:00Z",
        }
    )
    assert row["quantity"] == 1


def test_jsonable_row_coerces_numpy_and_nan():
    payload = jsonable_row({"item_id": pd.Series(["i1"]).iloc[0], "score": math.nan})
    assert payload["item_id"] == "i1"
    assert payload["score"] is None


def test_user_row_or_none_missing():
    assert user_row_or_none(pd.DataFrame([{"user_id": "u2"}]), "u1") is None


def test_item_row_or_none_found_and_missing_column():
    frame = pd.DataFrame([{"item_id": "i1", "comment": "sku"}])
    assert item_row_or_none(frame, "i1") == {"item_id": "i1", "comment": "sku"}
    assert item_row_or_none(pd.DataFrame([{"name": "x"}]), "i1") is None
    assert filter_item_row(pd.DataFrame([{"name": "x"}]), "i1").empty


def test_jsonable_row_isoformat():
    payload = jsonable_row({"occurred_at": datetime(2026, 9, 11, 12, 0, tzinfo=UTC)})
    assert payload["occurred_at"].startswith("2026-09-11T12:00:00")


def test_dataset_catalog_round_trip_and_empty_paths(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    assert store.get_user("u1") is None
    assert store.get_item("i1") is None
    assert store.get_events_for_user("u1", 5).empty
    assert store.upsert_events([]) == 0
    assert store.delete_item("missing") == 0
    assert store.delete_user("missing") == 0

    store.upsert_user({"user_id": "u1", "comment": "alice"})
    store.upsert_user({"user_id": "u1", "comment": "updated"})
    assert store.get_user("u1")["comment"] == "updated"

    store.upsert_item({"item_id": "i1", "comment": "sku"})
    assert store.get_item("i1")["item_id"] == "i1"

    first = store.upsert_events(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "occurred_at": "2026-09-11T12:00:00Z",
                "event_id": "e1",
            }
        ]
    )
    again = store.upsert_events(
        [
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "view",
                "occurred_at": "2026-09-11T13:00:00Z",
                "event_id": "e1",
            }
        ]
    )
    assert first == 1
    assert again == 1
    events = store.get_events_for_user("u1", 10)
    assert list(events["item_id"]) == ["i2"]
    assert store.delete_user("u1") >= 1
    assert store.get_user("u1") is None
    assert store.get_events_for_user("u1", 10).empty

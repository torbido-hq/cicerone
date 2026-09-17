from __future__ import annotations

import math
import threading
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import text

from cicerone.config.settings import IOSettings
from cicerone.io.catalog import (
    dedupe_event_rows,
    filter_item_row,
    item_row_or_none,
    jsonable_row,
    normalize_event_row,
    require_id,
    user_row_or_none,
)
from cicerone.io.dataset_catalog import DatasetCatalogStore
from cicerone.io.db_catalog import DatabaseCatalogStore
from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.factory import build_catalog_store, build_user_history_reader


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
    assert row["event_id"]


def test_normalize_event_row_uses_idempotency_key():
    row = normalize_event_row(
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-09-11T12:00:00Z",
            "idempotency_key": "k1",
        }
    )
    assert row["event_id"] == "k1"
    assert "idempotency_key" not in row


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


def test_dedupe_event_rows_last_wins():
    rows = [
        normalize_event_row(
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "view",
                "occurred_at": "2026-09-11T12:00:00Z",
                "event_id": "e1",
            }
        ),
        normalize_event_row(
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "occurred_at": "2026-09-11T13:00:00Z",
                "event_id": "e1",
            }
        ),
    ]
    assert [row["item_id"] for row in dedupe_event_rows(rows)] == ["i2"]


def test_dataset_catalog_dedupes_incoming_event_ids(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    accepted = store.upsert_events(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "view",
                "occurred_at": "2026-09-11T12:00:00Z",
                "event_id": "e1",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "occurred_at": "2026-09-11T13:00:00Z",
                "event_id": "e1",
            },
        ]
    )
    assert accepted == 1
    events = store.get_events_for_user("u1", 10)
    assert list(events["item_id"]) == ["i2"]


def test_database_catalog_shares_memory_engine_with_history():
    settings = IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"})
    history = build_user_history_reader(settings)
    catalog = build_catalog_store(settings)
    assert isinstance(history, DatabaseInputSource)
    assert isinstance(catalog, DatabaseCatalogStore)
    assert history._engine is catalog._engine
    catalog.upsert_user({"user_id": "u1", "comment": "alice"})
    user = history.get_user("u1")
    assert user is not None
    assert user["comment"] == "alice"


def test_database_catalog_dedupes_and_adds_event_id():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    with store._engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events ("
                "user_id TEXT, item_id TEXT, event_type TEXT, occurred_at TEXT, quantity INTEGER)"
            )
        )
    accepted = store.upsert_events(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "view",
                "occurred_at": "2026-09-11T12:00:00Z",
                "event_id": "e1",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "occurred_at": "2026-09-11T13:00:00Z",
                "event_id": "e1",
            },
        ]
    )
    assert accepted == 1
    events = store.get_events_for_user("u1", 10)
    assert list(events["item_id"]) == ["i2"]
    assert list(events["event_id"]) == ["e1"]
    assert (
        store.upsert_events(
            [
                {
                    "user_id": "u1",
                    "item_id": "i2",
                    "event_type": "purchase",
                    "occurred_at": "2026-09-11T13:00:00Z",
                    "event_id": "e1",
                }
            ]
        )
        == 1
    )
    assert len(store.get_events_for_user("u1", 10)) == 1


def test_database_catalog_first_create_is_race_safe():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    errors: list[Exception] = []

    def write(index: int) -> None:
        try:
            store.upsert_user({"user_id": f"u{index}", "comment": str(index)})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    with store._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
    assert count == 8


def test_database_catalog_unique_index_fallback_after_duplicates():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    with store._engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (user_id TEXT, comment TEXT)"))
        conn.execute(text("INSERT INTO users VALUES ('u1', 'a')"))
        conn.execute(text("INSERT INTO users VALUES ('u1', 'b')"))
    store.upsert_user({"user_id": "u1", "comment": "fixed"})
    with store._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
    assert count == 1
    assert store.get_user("u1")["comment"] == "fixed"

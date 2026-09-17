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


def test_jsonable_row_maps_pandas_missing_to_none():
    payload = jsonable_row({"occurred_at": pd.NaT, "comment": pd.NA, "score": math.nan})
    assert payload == {"occurred_at": None, "comment": None, "score": None}


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


def test_dataset_catalog_put_merges_existing_user_columns(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    store.upsert_user({"user_id": "u1", "comment": "alice", "labels": {"vip": True}})
    store.upsert_user({"user_id": "u1", "comment": "updated"})
    user = store.get_user("u1")
    assert user is not None
    assert user["comment"] == "updated"
    assert user["labels"] == {"vip": True}


def test_dataset_catalog_put_keeps_empty_schema_columns(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    pd.DataFrame(columns=["user_id", "comment", "labels"]).to_parquet(tmp_path / "users.parquet", index=False)
    store.upsert_user({"user_id": "u1", "comment": "alice"})
    user = store.get_user("u1")
    assert user is not None
    assert "labels" in user


def test_dataset_catalog_put_merges_existing_item_columns(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    store.upsert_item({"item_id": "i1", "comment": "sku", "labels": {"color": "red"}})
    store.upsert_item({"item_id": "i1", "comment": "updated"})
    item = store.get_item("i1")
    assert item is not None
    assert item["comment"] == "updated"
    assert item["labels"] == {"color": "red"}


def test_dataset_catalog_delete_events_without_item_id_column(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    events = pd.DataFrame([{"user_id": "u1", "event_type": "view"}])
    events.to_parquet(tmp_path / "events.parquet", index=False)
    assert store.delete_events_for_user("u1", item_id="i1") == 0
    assert store.delete_events_for_user("u1") == 1


def test_dataset_catalog_delete_user_without_user_id_column(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    pd.DataFrame([{"name": "alice"}]).to_parquet(tmp_path / "users.parquet", index=False)
    assert store.delete_user("u1") == 0


def test_dataset_catalog_delete_user_blocks_concurrent_event_write(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    store.upsert_user({"user_id": "u1"})
    store.upsert_events(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "view",
                "occurred_at": "2026-09-11T12:00:00Z",
                "event_id": "e1",
            }
        ]
    )
    started = threading.Event()
    proceed = threading.Event()
    original_write = store._write

    def gated_write(filename: str, frame: pd.DataFrame) -> None:
        if filename == "users.parquet" and not started.is_set():
            started.set()
            assert proceed.wait(2)
        original_write(filename, frame)

    store._write = gated_write  # type: ignore[method-assign]
    deleted: dict[str, int] = {}

    def deleter() -> None:
        deleted["n"] = store.delete_user("u1")

    worker = threading.Thread(target=deleter)
    worker.start()
    assert started.wait(2)
    assert store._locks["events.parquet"].locked()
    proceed.set()
    worker.join(2)
    assert deleted["n"] >= 1
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
    assert "_cicerone_shared_engine" not in settings.options


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


def test_database_catalog_rejects_invalid_column_names():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    with pytest.raises(ValueError, match="simple SQL identifier"):
        store.upsert_user({"user_id": "u1", "bad-name": "x"})


def test_dataset_catalog_replace_events_keeps_remaining_pair(tmp_path):
    store = DatasetCatalogStore({"storage_backend": "local", "path": str(tmp_path)})
    first = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "occurred_at": "2026-09-11T12:00:00Z",
        "event_id": "e1",
    }
    sibling = {**first, "event_id": "e2", "event_type": "view"}
    assert store.replace_events([first, sibling]) == (2, [], [("u1", "i1"), ("u1", "i1")])
    accepted, discard, add = store.replace_events([{**first, "item_id": "i2"}])
    assert accepted == 1
    assert discard == []
    assert add == [("u1", "i2")]
    events = store.get_events_for_user("u1", 10)
    assert set(events["item_id"]) == {"i1", "i2"}


def test_database_catalog_replace_events_discards_only_unmatched_pairs():
    store = DatabaseCatalogStore({"database_url": "sqlite+pysqlite://"})
    first = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "occurred_at": "2026-09-11T12:00:00Z",
        "event_id": "e1",
    }
    sibling = {**first, "event_id": "e2", "event_type": "view"}
    assert store.replace_events([first, sibling])[0] == 2
    accepted, discard, add = store.replace_events([{**first, "item_id": "i2"}])
    assert accepted == 1
    assert discard == []
    assert add == [("u1", "i2")]
    only, gone, _ = store.replace_events([{**sibling, "item_id": "i3", "event_id": "e2"}])
    assert only == 1
    assert ("u1", "i1") in gone
    assert store.get_event("e1")["item_id"] == "i2"


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

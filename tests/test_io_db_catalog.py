from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from support.postgres_defaults import resolve_test_database_url

from cicerone.io.db_catalog import DatabaseCatalogStore

TEST_DATABASE_URL = resolve_test_database_url()

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — DB-backed tests run against "
    "a real Postgres in CI (see docker-compose.ci.yml).",
)


@pytest.fixture(autouse=True)
def _clean_tables():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        for table in ("events", "users", "items"):
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    yield
    engine.dispose()


def test_database_catalog_crud_round_trip():
    store = DatabaseCatalogStore({"database_url": TEST_DATABASE_URL})
    store.upsert_user({"user_id": "u1", "comment": "alice"})
    user = store.get_user("u1")
    assert user is not None
    assert user["comment"] == "alice"

    store.upsert_item({"item_id": "i1", "comment": "sku"})
    item = store.get_item("i1")
    assert item is not None
    assert item["item_id"] == "i1"

    assert (
        store.upsert_events(
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
        == 1
    )
    events = store.get_events_for_user("u1", 10)
    assert list(events["item_id"]) == ["i1"]
    assert store.delete_events_for_user("u1", item_id="i1") == 1
    assert store.get_events_for_user("u1", 10).empty
    assert store.delete_item("i1") == 1
    assert store.get_item("i1") is None
    assert store.delete_user("u1") >= 1
    assert store.get_user("u1") is None


def test_database_catalog_missing_tables_are_empty():
    store = DatabaseCatalogStore({"database_url": TEST_DATABASE_URL})
    assert store.get_user("u1") is None
    assert store.get_item("i1") is None
    assert store.get_events_for_user("u1", 5).empty
    assert store.delete_events_for_user("u1") == 0
    assert store.delete_user("missing") == 0

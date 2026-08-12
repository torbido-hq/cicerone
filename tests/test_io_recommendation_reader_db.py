from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from support.postgres_defaults import resolve_test_database_url

from cicerone.config import ConfigError
from cicerone.io.db_store import DatabaseOutputSink
from cicerone.io.recommendation_reader import DbRecommendationReader

TEST_DATABASE_URL = resolve_test_database_url()

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — DB-backed tests run against "
    "a real Postgres in CI (see docker-compose.ci.yml). Set POSTGRES_TEST_HOST=localhost "
    "locally (see CONTRIBUTING.md).",
)


@pytest.fixture(autouse=True)
def _clean_recommendations_table():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS "recommendations"'))
        conn.execute(text('DROP TABLE IF EXISTS "recommendation_items"'))
    yield
    engine.dispose()


def test_db_reader_returns_top_k_sorted_by_rank():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i3", "rank": 3, "score": 0.1, "source": "popular_fallback"},
                {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
                {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
                {"user_id": "u2", "item_id": "i1", "rank": 1, "score": 0.7, "source": "personalized"},
            ]
        )
    )

    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    recs = reader.get_recommendations("u1", k=2)

    assert list(recs["item_id"]) == ["i1", "i2"]


def test_db_reader_missing_items_table_returns_none():
    # Autouse fixture drops recommendation_items; construct with no snapshot written.
    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    assert reader.get_items() is None
    reader.refresh()
    assert reader.get_items() is None


def test_db_reader_keeps_items_on_transient_refresh_error(monkeypatch):
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1", "category": "beer", "published": True}]))
    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    assert list(reader.get_items()["item_id"]) == ["i1"]
    version_before = reader.items_version()

    def boom(*_args, **_kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(pd, "read_sql", boom)
    reader.refresh()

    assert list(reader.get_items()["item_id"]) == ["i1"]
    assert reader.items_version() == version_before


def test_db_reader_items_snapshot_and_cold_start_fallback():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {
                    "user_id": "__cold_start__",
                    "item_id": "i9",
                    "rank": 1,
                    "score": 0.4,
                    "source": "popular_fallback",
                },
                {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            ]
        )
    )
    sink.write_items_snapshot(
        pd.DataFrame([{"item_id": "i1", "category": "beer", "published": True, "in_stock": True}])
    )

    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    items = reader.get_items()
    assert items is not None
    assert list(items["item_id"]) == ["i1"]
    assert list(reader.get_cold_start_fallback(k=1)["item_id"]) == ["i9"]


def test_db_reader_cold_start_fallback_without_sentinel():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {
                    "user_id": "regular_user",
                    "item_id": "i1",
                    "rank": 1,
                    "score": 0.9,
                    "source": "popular_fallback",
                },
                {
                    "user_id": "regular_user",
                    "item_id": "i2",
                    "rank": 2,
                    "score": 0.8,
                    "source": "popular_fallback",
                },
                {
                    "user_id": "other",
                    "item_id": "i9",
                    "rank": 1,
                    "score": 0.1,
                    "source": "personalized",
                },
            ]
        )
    )

    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    cold = reader.get_cold_start_fallback(k=2)

    assert list(cold["item_id"]) == ["i1", "i2"]
    assert set(cold["user_id"].astype(str)) == {"regular_user"}


def test_db_reader_unknown_user_returns_empty():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}]))

    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})

    assert reader.get_recommendations("nobody", k=10).empty


def test_db_reader_refresh_is_a_noop():
    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    reader.refresh()  # must not raise


def test_db_reader_missing_database_url_raises():
    with pytest.raises(ConfigError, match="database_url"):
        DbRecommendationReader({})

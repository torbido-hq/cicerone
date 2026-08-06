"""SQLite-backed smoke coverage for DB recommendation reader SQL paths.

Postgres remains the CI source of truth (``test_io_recommendation_reader_db.py``);
these tests exercise the same reader against an in-memory SQLite URL so coverage
does not depend on ``TEST_DATABASE_URL`` for the new cold-start SQL.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.io.db_store import DatabaseOutputSink
from cicerone.io.recommendation_reader import DbRecommendationReader


def _sqlite_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'cicerone.db'}"


def test_sqlite_db_reader_cold_start_prefers_popular(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {
                    "user_id": "z_user",
                    "item_id": "i1",
                    "rank": 1,
                    "score": 0.9,
                    "source": "popular_fallback",
                },
                {
                    "user_id": "a_user",
                    "item_id": "i9",
                    "rank": 1,
                    "score": 0.4,
                    "source": "latest",
                },
            ]
        )
    )

    reader = DbRecommendationReader({"database_url": url})
    cold = reader.get_cold_start_fallback(k=1)
    assert list(cold["item_id"]) == ["i1"]
    assert list(cold["user_id"]) == ["z_user"]


def test_sqlite_db_reader_get_recommendations_and_items(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
                {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            ]
        )
    )
    sink.write_items_snapshot(
        pd.DataFrame([{"item_id": "i1", "category": "beer", "published": True, "in_stock": True}])
    )

    reader = DbRecommendationReader({"database_url": url})
    assert list(reader.get_recommendations("u1", k=2)["item_id"]) == ["i1", "i2"]
    items = reader.get_items()
    assert items is not None
    assert list(items["item_id"]) == ["i1"]


def test_sqlite_clear_table_for_replace_falls_back_to_delete(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    frame = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    )
    sink.write_recommendations(frame)
    sink.write_recommendations(frame)  # second write must replace, not duplicate
    engine = create_engine(url)
    count = pd.read_sql(text('SELECT COUNT(*) AS n FROM "recommendations"'), engine).iloc[0]["n"]
    assert int(count) == 1

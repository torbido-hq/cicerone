from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.io.db_store import DatabaseOutputSink
from cicerone.io.recommendation_reader import DbRecommendationReader

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set — DB-backed tests run against a real Postgres in CI "
    "(see docker-compose.ci.yml). Set TEST_DATABASE_URL locally to run them.",
)


@pytest.fixture(autouse=True)
def _clean_recommendations_table():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS "recommendations"'))
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


def test_db_reader_unknown_user_returns_empty():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}]))

    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})

    assert reader.get_recommendations("nobody", k=10).empty


def test_db_reader_refresh_is_a_noop():
    reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    reader.refresh()  # must not raise


def test_db_reader_missing_database_url_raises():
    with pytest.raises(RuntimeError, match="database_url"):
        DbRecommendationReader({})

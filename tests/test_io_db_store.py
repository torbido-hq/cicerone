from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set — DB-backed tests run against a real Postgres in CI "
    "(see docker-compose.ci.yml). Set TEST_DATABASE_URL locally to run them.",
)


@pytest.fixture(autouse=True)
def _clean_tables():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        for table in ("events", "users", "items", "recommendations", "recommendation_runs", "custom_events"):
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    yield
    engine.dispose()


def test_database_input_reads_table():
    engine = create_engine(TEST_DATABASE_URL)
    pd.DataFrame([{"user_id": "u1", "item_id": "i1", "event_type": "purchase"}]).to_sql(
        "events", engine, index=False
    )

    source = DatabaseInputSource({"database_url": TEST_DATABASE_URL})
    events = source.read_events()

    assert list(events["user_id"]) == ["u1"]


def test_database_input_reads_custom_query():
    engine = create_engine(TEST_DATABASE_URL)
    pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]).to_sql("custom_events", engine, index=False)

    source = DatabaseInputSource(
        {"database_url": TEST_DATABASE_URL, "events_query": 'SELECT * FROM "custom_events"'}
    )
    events = source.read_events()

    assert list(events["user_id"]) == ["u1"]


def test_database_input_optional_tables_missing_return_none():
    source = DatabaseInputSource({"database_url": TEST_DATABASE_URL})

    assert source.read_users() is None
    assert source.read_items() is None


def test_database_output_writes_and_replaces_recommendations():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})

    first = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    sink.write_recommendations(first)

    second = pd.DataFrame([{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}])
    sink.write_recommendations(second)

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendations"', engine)

    # TRUNCATE before the second write means only the latest snapshot remains.
    assert list(stored["user_id"]) == ["u2"]


def test_database_output_writes_manifest_appends():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})

    sink.write_manifest({"n_events": 1})
    sink.write_manifest({"n_events": 2})

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)

    assert list(stored["n_events"]) == [1, 2]


def test_missing_database_url_raises():
    with pytest.raises(RuntimeError, match="database_url"):
        DatabaseInputSource({})
    with pytest.raises(RuntimeError, match="database_url"):
        DatabaseOutputSink({})

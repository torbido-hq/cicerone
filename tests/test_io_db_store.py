from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from support.postgres_defaults import resolve_test_database_url

from cicerone.config import ConfigError
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink

TEST_DATABASE_URL = resolve_test_database_url()

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — DB-backed tests run against "
    "a real Postgres in CI (see docker-compose.ci.yml). Set POSTGRES_TEST_HOST=localhost "
    "locally (see CONTRIBUTING.md).",
)


@pytest.fixture(autouse=True)
def _clean_tables():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        for table in (
            "events",
            "users",
            "items",
            "recommendations",
            "recommendation_items",
            "recommendation_runs",
            "model_artifacts",
            "custom_events",
            "custom_users",
            "custom_recommendations",
            "custom_manifest_runs",
            "custom_model_artifacts",
        ):
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


def test_database_input_optional_custom_query_missing_table_returns_none():
    source = DatabaseInputSource(
        {"database_url": TEST_DATABASE_URL, "users_query": 'SELECT * FROM "does_not_exist_yet"'}
    )

    assert source.read_users() is None


def test_database_output_writes_and_replaces_recommendations():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})

    first = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    )
    sink.write_recommendations(first)

    second = pd.DataFrame(
        [{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}]
    )
    sink.write_recommendations(second)

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendations"', engine)

    # Clear-before-write means only the latest snapshot remains.
    assert list(stored["user_id"]) == ["u2"]


def test_database_output_replace_recommendations_for_users():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "old", "rank": 1, "score": 0.9, "source": "personalized"},
                {"user_id": "u2", "item_id": "keep", "rank": 1, "score": 0.8, "source": "personalized"},
            ]
        )
    )
    sink.replace_recommendations_for_users(
        pd.DataFrame([{"user_id": "u1", "item_id": "new", "rank": 1, "score": 1.0, "source": "incremental"}]),
        user_ids=["u1"],
    )
    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendations" ORDER BY user_id, item_id', engine)
    assert list(stored[stored["user_id"] == "u2"]["item_id"]) == ["keep"]
    assert list(stored[stored["user_id"] == "u1"]["item_id"]) == ["new"]
    sink.replace_recommendations_for_users(pd.DataFrame(), user_ids=["u1"])
    stored = pd.read_sql('SELECT * FROM "recommendations"', engine)
    assert list(stored["user_id"]) == ["u2"]


def test_database_output_replace_recommendations_rejects_extra_users():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    with pytest.raises(ValueError, match="outside user_ids"):
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u9", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )


def test_database_output_writes_and_replaces_items_snapshot():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1", "category": "beer"}]))
    sink.write_items_snapshot(pd.DataFrame([{"item_id": "i2", "category": "wine"}]))

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendation_items"', engine)
    assert list(stored["item_id"]) == ["i2"]


def test_database_output_writes_manifest_appends():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})

    sink.write_manifest({"n_events": 1})
    sink.write_manifest({"n_events": 2})

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)

    assert list(stored["n_events"]) == [1, 2]


def test_database_output_writes_and_replaces_model_artifact():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})

    sink.write_model_artifact(b"first")
    sink.write_model_artifact(b"second")

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT payload FROM "model_artifacts"', engine)

    assert len(stored) == 1
    payload = stored.iloc[0]["payload"]
    assert bytes(payload) == b"second"


def test_database_output_model_artifact_custom_table_name():
    sink = DatabaseOutputSink(
        {"database_url": TEST_DATABASE_URL, "model_artifact_table": "custom_model_artifacts"}
    )
    sink.write_model_artifact(b"custom")

    engine = create_engine(TEST_DATABASE_URL)
    stored = pd.read_sql('SELECT payload FROM "custom_model_artifacts"', engine)
    assert bytes(stored.iloc[0]["payload"]) == b"custom"


def test_missing_database_url_raises():
    with pytest.raises(ConfigError, match="database_url"):
        DatabaseInputSource({})
    with pytest.raises(ConfigError, match="database_url"):
        DatabaseOutputSink({})


def test_database_output_custom_table_names_are_used():
    options = {
        "database_url": TEST_DATABASE_URL,
        "recommendations_table": "custom_recommendations",
        "manifest_table": "custom_manifest_runs",
    }
    sink = DatabaseOutputSink(options)

    recos = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    )
    sink.write_recommendations(recos)
    sink.write_manifest({"n_events": 1})

    engine = create_engine(TEST_DATABASE_URL)
    stored_recos = pd.read_sql('SELECT * FROM "custom_recommendations"', engine)
    stored_manifest = pd.read_sql('SELECT * FROM "custom_manifest_runs"', engine)

    assert list(stored_recos["user_id"]) == ["u1"]
    assert list(stored_manifest["n_events"]) == [1]


def test_database_input_custom_table_names_are_used():
    engine = create_engine(TEST_DATABASE_URL)
    pd.DataFrame([{"user_id": "u1", "favorite_style": "stout"}]).to_sql("custom_users", engine, index=False)

    source = DatabaseInputSource({"database_url": TEST_DATABASE_URL, "users_table": "custom_users"})

    users = source.read_users()

    assert list(users["user_id"]) == ["u1"]

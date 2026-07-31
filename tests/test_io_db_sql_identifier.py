"""Always-on unit tests for db_store SQL identifier validation.

Kept separate from test_io_db_store.py so they run without TEST_DATABASE_URL.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink, _sql_identifier


def test_sql_identifier_accepts_simple_names():
    assert _sql_identifier("model_artifacts", option="model_artifact_table") == "model_artifacts"
    assert _sql_identifier("_private", option="model_artifact_table") == "_private"


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendations; --',
        "has-dash",
        "has space",
        "123starts_with_digit",
        "",
        None,
    ],
)
def test_sql_identifier_rejects_unsafe_names(bad_name):
    with pytest.raises(ValueError, match="SQL identifier"):
        _sql_identifier(bad_name, option="model_artifact_table")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendations; --',
        "has-dash",
        "has space",
    ],
)
def test_write_model_artifact_rejects_unsafe_table_name_before_db(bad_name):
    sink = DatabaseOutputSink(
        {"database_url": "postgresql+psycopg://u:p@localhost/db", "model_artifact_table": bad_name}
    )
    with pytest.raises(ValueError, match="SQL identifier"):
        sink.write_model_artifact(b"x")


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendations; --',
        "has-dash",
        "has space",
    ],
)
def test_write_recommendations_rejects_unsafe_table_name_before_db(bad_name):
    sink = DatabaseOutputSink(
        {"database_url": "postgresql+psycopg://u:p@localhost/db", "recommendations_table": bad_name}
    )
    with pytest.raises(ValueError, match="SQL identifier"):
        sink.write_recommendations(pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1}]))


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendation_runs; --',
        "has-dash",
        "has space",
    ],
)
def test_write_manifest_rejects_unsafe_table_name_before_db(bad_name):
    sink = DatabaseOutputSink(
        {"database_url": "postgresql+psycopg://u:p@localhost/db", "manifest_table": bad_name}
    )
    with pytest.raises(ValueError, match="SQL identifier"):
        sink.write_manifest({"n_events": 1})


@pytest.mark.parametrize(
    ("method_name", "option"),
    [
        ("read_events", "events_table"),
        ("read_users", "users_table"),
        ("read_items", "items_table"),
    ],
)
@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE events; --',
        "has-dash",
        "has space",
    ],
)
def test_input_source_rejects_unsafe_table_name_before_db(method_name, option, bad_name):
    source = DatabaseInputSource({"database_url": "postgresql+psycopg://u:p@localhost/db", option: bad_name})
    with pytest.raises(ValueError, match="SQL identifier"):
        getattr(source, method_name)()


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendations; --',
        "has-dash",
        "has space",
    ],
)
def test_db_recommendation_reader_rejects_unsafe_table_name(bad_name):
    from cicerone.io.recommendation_reader import DbRecommendationReader

    with pytest.raises(ValueError, match="SQL identifier"):
        DbRecommendationReader(
            {
                "database_url": "postgresql+psycopg://u:p@localhost/db_test",
                "recommendations_table": bad_name,
            }
        )


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendation_runs; --',
        "has-dash",
        "has space",
    ],
)
def test_db_manifest_reader_rejects_unsafe_table_name(bad_name):
    from cicerone.io.manifest_reader import DbManifestReader

    with pytest.raises(ValueError, match="SQL identifier"):
        DbManifestReader(
            {
                "database_url": "postgresql+psycopg://u:p@localhost/db_test",
                "manifest_table": bad_name,
            }
        )

"""Always-on unit tests for db_store SQL identifier validation.

Kept separate from test_io_db_store.py so they run without TEST_DATABASE_URL.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cicerone.io.db_store import (
    DEFAULT_MANIFEST_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
    DatabaseInputSource,
    DatabaseOutputSink,
)
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.io.options import sql_identifier
from cicerone.io.recommendation_reader import DbRecommendationReader


def test_sql_identifier_accepts_simple_names():
    assert sql_identifier("model_artifacts", option="model_artifact_table") == "model_artifacts"
    assert sql_identifier("_private", option="model_artifact_table") == "_private"


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
        sql_identifier(bad_name, option="model_artifact_table")  # type: ignore[arg-type]


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
    with pytest.raises(ValueError, match="SQL identifier"):
        DbRecommendationReader(
            {
                "database_url": "postgresql+psycopg://u:p@localhost/db_test",
                "recommendations_table": bad_name,
            }
        )


@pytest.mark.parametrize(
    "table_name",
    [None, DEFAULT_RECOMMENDATIONS_TABLE, "recommendations_2024", "custom_recos"],
)
def test_db_recommendation_reader_accepts_safe_table_names(table_name):
    options: dict = {"database_url": "postgresql+psycopg://u:p@localhost/db_test"}
    if table_name is not None:
        options["recommendations_table"] = table_name
    reader = DbRecommendationReader(options)
    assert reader._table == (table_name or DEFAULT_RECOMMENDATIONS_TABLE)


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil"; DROP TABLE recommendation_runs; --',
        "has-dash",
        "has space",
    ],
)
def test_db_manifest_reader_rejects_unsafe_table_name(bad_name):
    with pytest.raises(ValueError, match="SQL identifier"):
        DbManifestReader(
            {
                "database_url": "postgresql+psycopg://u:p@localhost/db_test",
                "manifest_table": bad_name,
            }
        )


@pytest.mark.parametrize(
    "table_name",
    [None, DEFAULT_MANIFEST_TABLE, "recommendation_runs_2024", "custom_manifest"],
)
def test_db_manifest_reader_accepts_safe_table_names(table_name):
    options: dict = {"database_url": "postgresql+psycopg://u:p@localhost/db_test"}
    if table_name is not None:
        options["manifest_table"] = table_name
    reader = DbManifestReader(options)
    assert reader._table == (table_name or DEFAULT_MANIFEST_TABLE)

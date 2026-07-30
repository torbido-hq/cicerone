"""Always-on unit tests for db_store SQL identifier validation.

Kept separate from test_io_db_store.py so they run without TEST_DATABASE_URL.
"""

from __future__ import annotations

import pytest

from cicerone.io.db_store import DatabaseOutputSink, _sql_identifier


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

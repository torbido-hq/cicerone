from __future__ import annotations

import logging
import sqlite3

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.config import IOSettings, make_settings
from cicerone.events.store import empty_recommendations_frame, load_recommendations_frame


def test_load_recommendations_missing_file(tmp_path):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert frame.empty


def test_load_recommendations_schema_mismatch_treated_as_empty(tmp_path):
    path = tmp_path / "recommendations.parquet"
    pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert frame.empty


def test_load_recommendations_db_missing_table_treated_as_empty(tmp_path, caplog):
    db_path = tmp_path / "recommendations.db"
    sqlite3.connect(db_path).close()
    settings = make_settings(
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{db_path}"})
    )
    with caplog.at_level(logging.WARNING):
        frame = load_recommendations_frame(settings.output)
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert frame.empty
    assert any("missing" in record.getMessage().lower() for record in caplog.records)


def test_load_recommendations_db_schema_mismatch_treated_as_empty(tmp_path, caplog):
    db_path = tmp_path / "recommendations_schema.db"
    url = f"sqlite+pysqlite:///{db_path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE recommendations (user_id TEXT, item_id TEXT)"))
        conn.execute(text("INSERT INTO recommendations (user_id, item_id) VALUES ('u1', 'i1')"))
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    with caplog.at_level(logging.WARNING):
        frame = load_recommendations_frame(settings.output)
    assert frame.empty
    assert any("schema mismatch" in record.getMessage().lower() for record in caplog.records)


def test_load_recommendations_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendations_frame(IOSettings(kind="other", options={}))

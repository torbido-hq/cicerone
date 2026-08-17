from __future__ import annotations

import logging
import sqlite3

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.config import IOSettings, make_settings
from cicerone.events.store import (
    count_recommendation_users,
    dispose_recommendation_engines,
    empty_recommendations_frame,
    load_recommendations_for_users,
    load_recommendations_frame,
)


@pytest.fixture(autouse=True)
def _dispose_engines():
    yield
    dispose_recommendation_engines()


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


def test_dispose_recommendation_engines_clears_cache(tmp_path):
    db_path = tmp_path / "dispose.db"
    sqlite3.connect(db_path).close()
    url = f"sqlite+pysqlite:///{db_path}"
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    load_recommendations_frame(settings.output)
    dispose_recommendation_engines()
    dispose_recommendation_engines()  # idempotent
    frame = load_recommendations_frame(settings.output)
    assert frame.empty


def test_load_recommendations_for_users_dataset_filters(tmp_path):
    path = tmp_path / "recommendations.parquet"
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_for_users(settings.output, ["u2"])
    assert list(frame["user_id"]) == ["u2"]
    assert count_recommendation_users(settings.output) == 2
    assert load_recommendations_for_users(settings.output, []).empty


def test_load_recommendations_for_users_db(tmp_path):
    db_path = tmp_path / "scoped.db"
    url = f"sqlite+pysqlite:///{db_path}"
    engine = create_engine(url)
    rows = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    )
    rows.to_sql("recommendations", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    frame = load_recommendations_for_users(settings.output, ["u1"])
    assert list(frame["user_id"]) == ["u1"]
    assert count_recommendation_users(settings.output) == 2


def test_load_recommendations_for_users_db_missing_table(tmp_path, caplog):
    db_path = tmp_path / "missing_scoped.db"
    sqlite3.connect(db_path).close()
    settings = make_settings(
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{db_path}"})
    )
    with caplog.at_level(logging.WARNING):
        frame = load_recommendations_for_users(settings.output, ["u1"])
    assert frame.empty
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert any("missing" in record.getMessage().lower() for record in caplog.records)


def test_count_recommendation_users_db_missing_table(tmp_path):
    db_path = tmp_path / "missing_count.db"
    sqlite3.connect(db_path).close()
    settings = make_settings(
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{db_path}"})
    )
    assert count_recommendation_users(settings.output) == 0


def test_count_recommendation_users_db_schema_mismatch(tmp_path, caplog):
    db_path = tmp_path / "count_schema.db"
    url = f"sqlite+pysqlite:///{db_path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE recommendations (item_id TEXT, rank INTEGER)"))
        conn.execute(text("INSERT INTO recommendations (item_id, rank) VALUES ('i1', 1)"))
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    with caplog.at_level(logging.WARNING):
        assert count_recommendation_users(settings.output) == 0
    assert any("schema mismatch" in record.getMessage().lower() for record in caplog.records)


def test_count_recommendation_users_dataset_projects_user_id(tmp_path, monkeypatch):
    path = tmp_path / "recommendations.parquet"
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    calls: list[object] = []
    real_read = __import__("cicerone.io.options", fromlist=["read_parquet"]).read_parquet

    def tracking_read(options, filename, *, s3_client=None, columns=None):  # type: ignore[no-untyped-def]
        calls.append(columns)
        return real_read(options, filename, s3_client=s3_client, columns=columns)

    monkeypatch.setattr("cicerone.events.store.read_parquet", tracking_read)
    assert count_recommendation_users(settings.output) == 2
    assert calls == [["user_id"]]


def test_count_recommendation_users_dataset_schema_mismatch(tmp_path, caplog):
    path = tmp_path / "recommendations.parquet"
    pd.DataFrame([{"item_id": "i1", "rank": 1}]).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    with caplog.at_level(logging.WARNING):
        assert count_recommendation_users(settings.output) == 0
    assert any("schema mismatch" in record.getMessage().lower() for record in caplog.records)


def test_load_recommendations_for_users_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendations_for_users(IOSettings(kind="other", options={}), ["u1"])
    with pytest.raises(ValueError, match="Unsupported output kind"):
        count_recommendation_users(IOSettings(kind="other", options={}))

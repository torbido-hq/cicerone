"""SQLite smoke coverage for DB recommendation reader SQL (no TEST_DATABASE_URL)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.io.db_store import DatabaseOutputSink
from cicerone.io.recommendation_reader import DbRecommendationReader
from cicerone.io.replace_users import RecommendationSchemaError


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


def test_sqlite_db_reader_cold_start_missing_source_column(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"user_id": "z_user", "item_id": "i1", "rank": 1, "score": 0.9}]).to_sql(
        "recommendations", engine, index=False, if_exists="replace"
    )

    reader = DbRecommendationReader({"database_url": url})
    cold = reader.get_cold_start_fallback(k=1)

    assert isinstance(cold, pd.DataFrame)
    assert cold.empty


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


def test_sqlite_replace_recommendations_for_users(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
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
    engine = create_engine(url)
    stored = pd.read_sql(text('SELECT user_id, item_id FROM "recommendations" ORDER BY user_id'), engine)
    assert list(zip(stored["user_id"], stored["item_id"], strict=True)) == [("u1", "new"), ("u2", "keep")]
    assert sink.replace_recommendations_for_users(pd.DataFrame(), user_ids=["u1"]) == 1
    stored = pd.read_sql(text('SELECT user_id FROM "recommendations"'), engine)
    assert list(stored["user_id"]) == ["u2"]
    assert sink.replace_recommendations_for_users(pd.DataFrame(), user_ids=[]) == 0


def test_sqlite_replace_recommendations_creates_table_when_missing(tmp_path, caplog):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    with caplog.at_level(logging.WARNING):
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )
    assert any("delete skipped" in record.getMessage().lower() for record in caplog.records)
    engine = create_engine(url)
    stored = pd.read_sql(text('SELECT user_id, item_id FROM "recommendations"'), engine)
    assert list(zip(stored["user_id"], stored["item_id"], strict=True)) == [("u1", "i1")]


def test_sqlite_replace_recommendations_schema_mismatch(tmp_path, caplog):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE recommendations (item_id TEXT, rank INTEGER, score REAL, source TEXT)")
        )
        conn.execute(text("INSERT INTO recommendations VALUES ('old', 1, 0.1, 'x')"))
    sink = DatabaseOutputSink({"database_url": url})
    with caplog.at_level(logging.WARNING), pytest.raises(RecommendationSchemaError, match="schema mismatch"):
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )
    assert any("delete skipped" in record.getMessage().lower() for record in caplog.records)
    stored = pd.read_sql(text("SELECT item_id FROM recommendations"), engine)
    assert list(stored["item_id"]) == ["old"]


def test_sqlite_db_reader_filters_variant(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "control-item",
                    "rank": 1,
                    "score": 0.9,
                    "source": "personalized",
                    "variant": "control",
                },
                {
                    "user_id": "u1",
                    "item_id": "treatment-item",
                    "rank": 1,
                    "score": 0.8,
                    "source": "personalized",
                    "variant": "treatment",
                },
            ]
        )
    )
    reader = DbRecommendationReader({"database_url": url})
    assert list(reader.get_recommendations("u1", k=10, variant="treatment")["item_id"]) == ["treatment-item"]


def test_sqlite_db_reader_missing_variant_column_falls_back(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    reader = DbRecommendationReader({"database_url": url})
    assert list(reader.get_recommendations("u1", k=10, variant="treatment")["item_id"]) == ["i1"]


def _raise_on_variant_sql(original, variant_queries: dict[str, int]):
    from sqlalchemy.exc import ProgrammingError

    def fake_read_sql(sql, *args, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("variant") is not None or ":variant" in str(sql):
            variant_queries["n"] += 1
            raise ProgrammingError("SELECT", {}, Exception("column variant does not exist"))
        return original(sql, *args, **kwargs)

    return fake_read_sql


def test_sqlite_db_reader_caches_missing_variant_after_query_error(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    reader = DbRecommendationReader({"database_url": url})
    reader._variant_supported = True
    variant_queries = {"n": 0}
    monkeypatch.setattr(
        "cicerone.io.recommendation_reader.pd.read_sql",
        _raise_on_variant_sql(pd.read_sql, variant_queries),
    )
    assert list(reader.get_recommendations("u1", k=10, variant="treatment")["item_id"]) == ["i1"]
    assert reader._variant_supported is False
    assert list(reader.get_recommendations("u1", k=10, variant="treatment")["item_id"]) == ["i1"]
    assert variant_queries["n"] == 1


def test_sqlite_db_reader_cold_start_caches_missing_variant_after_query_error(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [
            {
                "user_id": "z_user",
                "item_id": "i1",
                "rank": 1,
                "score": 0.9,
                "source": "popular_fallback",
            }
        ]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    reader = DbRecommendationReader({"database_url": url})
    reader._variant_supported = True
    variant_queries = {"n": 0}
    monkeypatch.setattr(
        "cicerone.io.recommendation_reader.pd.read_sql",
        _raise_on_variant_sql(pd.read_sql, variant_queries),
    )
    cold = reader.get_cold_start_fallback(k=1, variant="treatment")
    assert list(cold["item_id"]) == ["i1"]
    assert reader._variant_supported is False
    assert list(reader.get_cold_start_fallback(k=1, variant="treatment")["item_id"]) == ["i1"]
    assert variant_queries["n"] == 1


def test_sqlite_db_reader_does_not_cache_unrelated_missing_column(tmp_path, monkeypatch):
    from sqlalchemy.exc import ProgrammingError

    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    reader = DbRecommendationReader({"database_url": url})
    reader._variant_supported = True
    original = pd.read_sql

    def fake_read_sql(sql, *args, **kwargs):
        params = kwargs.get("params") or {}
        if params.get("variant") is not None or ":variant" in str(sql):
            raise ProgrammingError("SELECT", {}, Exception('column "user_id" does not exist'))
        return original(sql, *args, **kwargs)

    monkeypatch.setattr("cicerone.io.recommendation_reader.pd.read_sql", fake_read_sql)
    with pytest.raises(ProgrammingError, match="user_id"):
        reader.get_recommendations("u1", k=10, variant="treatment")
    assert reader._variant_supported is True

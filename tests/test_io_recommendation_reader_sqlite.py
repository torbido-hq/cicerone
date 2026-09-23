"""SQLite smoke coverage for DB recommendation reader SQL (no TEST_DATABASE_URL)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.io.db_store import DatabaseOutputSink, _manifest_column_sql_type
from cicerone.io.recommendation_reader import DbRecommendationReader
from cicerone.io.replace_users import RecommendationSchemaError
from cicerone.locks import LockLostError


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


def test_sqlite_db_reader_get_recommendations_for_users(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
                {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
                {"user_id": "u2", "item_id": "i3", "rank": 1, "score": 0.4, "source": "personalized"},
            ]
        )
    )
    reader = DbRecommendationReader({"database_url": url})
    real_read = pd.read_sql
    rec_reads: list[str] = []

    def counting_read(sql, *args, **kwargs):
        rec_reads.append(str(sql))
        return real_read(sql, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql", counting_read)
    bulk = reader.get_recommendations_for_users(["u1", "u2", "nobody"], k=2)
    assert list(bulk["u1"]["item_id"]) == ["i1", "i2"]
    assert list(bulk["u2"]["item_id"]) == ["i3"]
    assert bulk["nobody"].empty
    assert sum("IN" in query.upper() for query in rec_reads) == 1
    assert any("ROW_NUMBER" in query.upper() for query in rec_reads)


def test_sqlite_db_reader_bulk_limits_rows_in_sql(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    rows = [
        {
            "user_id": user_id,
            "item_id": f"{user_id}-i{rank}",
            "rank": rank,
            "score": 1.0 - rank * 0.1,
            "source": "personalized",
        }
        for user_id in ("u1", "u2")
        for rank in range(1, 6)
    ]
    sink.write_recommendations(pd.DataFrame(rows))
    reader = DbRecommendationReader({"database_url": url})
    real_read = pd.read_sql
    loaded: list[int] = []

    def counting_read(sql, *args, **kwargs):
        frame = real_read(sql, *args, **kwargs)
        if "IN" in str(sql).upper():
            loaded.append(len(frame))
        return frame

    monkeypatch.setattr(pd, "read_sql", counting_read)
    bulk = reader.get_recommendations_for_users(["u1", "u2"], k=2)
    assert list(bulk["u1"]["item_id"]) == ["u1-i1", "u1-i2"]
    assert list(bulk["u2"]["item_id"]) == ["u2-i1", "u2-i2"]
    assert loaded == [4]


def test_sqlite_db_reader_bulk_limits_assigned_variant_in_sql(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    rows = [
        {
            "user_id": "u1",
            "item_id": f"{variant}-i{rank}",
            "rank": rank,
            "score": 1.0 - rank * 0.1,
            "source": "personalized",
            "variant": variant,
        }
        for variant in ("control", "treatment")
        for rank in range(1, 6)
    ]
    sink.write_recommendations(pd.DataFrame(rows))
    reader = DbRecommendationReader({"database_url": url})
    real_read = pd.read_sql
    loaded: list[int] = []

    def counting_read(sql, *args, **kwargs):
        frame = real_read(sql, *args, **kwargs)
        if "IN" in str(sql).upper():
            loaded.append(len(frame))
        return frame

    monkeypatch.setattr(pd, "read_sql", counting_read)
    bulk = reader.get_recommendations_for_users(["u1"], k=2, variant="treatment")
    assert list(bulk["u1"]["item_id"]) == ["treatment-i1", "treatment-i2"]
    assert loaded == [2]


def test_sqlite_db_reader_bulk_limits_fallback_variant_in_sql(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    rows = [
        {
            "user_id": "u1",
            "item_id": f"{variant}-i{rank}",
            "rank": rank,
            "score": 1.0 - rank * 0.1,
            "source": "personalized",
            "variant": variant,
        }
        for variant in ("control", "treatment")
        for rank in range(1, 6)
    ]
    sink.write_recommendations(pd.DataFrame(rows))
    reader = DbRecommendationReader({"database_url": url})
    real_read = pd.read_sql
    loaded: list[int] = []

    def counting_read(sql, *args, **kwargs):
        frame = real_read(sql, *args, **kwargs)
        if "IN" in str(sql).upper():
            loaded.append(len(frame))
        return frame

    monkeypatch.setattr(pd, "read_sql", counting_read)
    bulk = reader.get_recommendations_for_users(["u1"], k=2)
    assert list(bulk["u1"]["item_id"]) == ["control-i1", "control-i2"]
    assert loaded == [2]


def test_sqlite_db_reader_bulk_keeps_rank_order(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    rows = [
        {
            "user_id": "u1",
            "item_id": f"i{rank}",
            "rank": rank,
            "score": 1.0 - rank * 0.1,
            "source": "personalized",
        }
        for rank in (5, 4, 1, 3, 2)
    ]
    sink.write_recommendations(pd.DataFrame(rows))
    reader = DbRecommendationReader({"database_url": url})
    real_read = pd.read_sql
    seen: list[str] = []

    def tracking(sql, *args, **kwargs):
        seen.append(str(sql))
        return real_read(sql, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql", tracking)
    bulk = reader.get_recommendations_for_users(["u1"], k=3)
    assert list(bulk["u1"]["item_id"]) == ["i1", "i2", "i3"]
    assert any("_cicerone_rn" in sql and "ORDER BY" in sql.upper() for sql in seen)


def test_sqlite_db_reader_item_scores_write_replace_and_missing(tmp_path):
    url = _sqlite_url(tmp_path)
    reader = DbRecommendationReader({"database_url": url})
    assert reader.get_item_scores().empty

    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    )
    sink.write_item_scores(
        pd.DataFrame([{"item_id": "i1", "popular_score": 1.0, "latest_score": 0.0, "n_users": 1}])
    )
    sink.write_item_scores(
        pd.DataFrame([{"item_id": "i2", "popular_score": 4.0, "latest_score": 2.0, "n_users": 5}])
    )
    reader.refresh()
    scores, ids = reader.get_item_scores_snapshot()
    assert list(ids) == ["i2"]
    assert list(scores["item_id"]) == ["i2"]
    assert list(reader.get_item_score_ids()) == ["i2"]
    assert float(scores.iloc[0]["popular_score"]) == 4.0
    assert int(scores.iloc[0]["n_users"]) == 5


def test_sqlite_db_reader_item_scores_missing_columns(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}]).to_sql(
        "recommendations", engine, index=False, if_exists="replace"
    )
    pd.DataFrame([{"item_id": "i1", "popular_score": 1.0}]).to_sql(
        "item_scores", engine, index=False, if_exists="replace"
    )
    reader = DbRecommendationReader({"database_url": url})
    assert reader.get_item_scores().empty


def test_sqlite_db_reader_item_scores_keeps_cache_on_bad_schema(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    )
    sink.write_item_scores(
        pd.DataFrame([{"item_id": "i1", "popular_score": 2.5, "latest_score": 1.0, "n_users": 4}])
    )
    reader = DbRecommendationReader({"database_url": url})
    assert list(reader.get_item_scores()["item_id"]) == ["i1"]

    engine = create_engine(url)
    pd.DataFrame([{"item_id": "i1", "popular_score": 1.0}]).to_sql(
        "item_scores", engine, index=False, if_exists="replace"
    )
    reader.refresh()
    kept = reader.get_item_scores()
    assert list(kept["item_id"]) == ["i1"]
    assert float(kept.iloc[0]["popular_score"]) == 2.5

    engine = create_engine(url)
    pd.DataFrame([{"item_id": "i1", "popular_score": "x", "latest_score": 0.0, "n_users": 1}]).to_sql(
        "item_scores", engine, index=False, if_exists="replace"
    )
    reader.refresh()
    assert float(reader.get_item_scores().iloc[0]["popular_score"]) == 2.5

    pd.DataFrame([{"item_id": "i1", "popular_score": 1.0, "latest_score": 0.0, "n_users": -1}]).to_sql(
        "item_scores", engine, index=False, if_exists="replace"
    )
    reader.refresh()
    assert float(reader.get_item_scores().iloc[0]["popular_score"]) == 2.5

    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS "item_scores"'))
    reader.refresh()
    assert float(reader.get_item_scores().iloc[0]["popular_score"]) == 2.5


def test_manifest_column_sql_type_uses_series_dtype():
    assert _manifest_column_sql_type(pd.Series([True, False])) == "BOOLEAN"
    assert _manifest_column_sql_type(pd.Series([1, 4])) == "BIGINT"
    assert _manifest_column_sql_type(pd.Series([1.5, 60.0])) == "FLOAT"
    assert _manifest_column_sql_type(pd.Series(["success", "failed"])) == "TEXT"
    assert _manifest_column_sql_type(pd.Series([None], dtype=object, name="n_item_scores")) == "BIGINT"
    assert _manifest_column_sql_type(pd.Series([None], dtype=object, name="partial_outputs")) == "BOOLEAN"


def test_sqlite_write_manifest_adds_missing_columns(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"n_events": 1, "status": "success"}]).to_sql(
        "recommendation_runs", engine, index=False, if_exists="replace"
    )
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_manifest(
        {
            "n_events": 2,
            "status": "success",
            "n_item_scores": 4,
            "partial_outputs": False,
            "rrf_k": 60.0,
        }
    )
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)
    assert list(stored["n_events"]) == [1, 2]
    assert int(stored.iloc[1]["n_item_scores"]) == 4
    with engine.connect() as conn:
        types = {
            str(row[1]): str(row[2]).upper()
            for row in conn.execute(text("PRAGMA table_info(recommendation_runs)"))
        }
    assert "INT" in types["n_item_scores"]
    assert types["partial_outputs"] in {"BOOLEAN", "BOOL"}
    assert types["rrf_k"] in {"FLOAT", "REAL"}
    sink.write_manifest({"n_events": 3, "status": "success", "n_item_scores": 5})
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)
    assert list(stored["n_events"]) == [1, 2, 3]


def test_sqlite_write_manifest_creates_integer_n_item_scores(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_manifest({"n_events": 1, "status": "failed", "n_item_scores": None})
    engine = create_engine(url)
    with engine.connect() as conn:
        types = {
            str(row[1]): str(row[2]).upper()
            for row in conn.execute(text("PRAGMA table_info(recommendation_runs)"))
        }
    assert "INT" in types["n_item_scores"]
    sink.write_manifest({"n_events": 2, "status": "success", "n_item_scores": 4})
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)
    assert int(stored.iloc[-1]["n_item_scores"]) == 4


def test_sqlite_write_manifest_none_n_item_scores_stays_integer(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"n_events": 1, "status": "failed"}]).to_sql(
        "recommendation_runs", engine, index=False, if_exists="replace"
    )
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_manifest({"n_events": 2, "status": "failed", "n_item_scores": None})
    with engine.connect() as conn:
        types = {
            str(row[1]): str(row[2]).upper()
            for row in conn.execute(text("PRAGMA table_info(recommendation_runs)"))
        }
    assert "INT" in types["n_item_scores"]
    sink.write_manifest({"n_events": 3, "status": "success", "n_item_scores": 4})
    stored = pd.read_sql('SELECT * FROM "recommendation_runs"', engine)
    assert int(stored.iloc[-1]["n_item_scores"]) == 4


def test_sqlite_write_manifest_alters_under_writer_lock(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"n_events": 1, "status": "success"}]).to_sql(
        "recommendation_runs", engine, index=False, if_exists="replace"
    )
    order: list[str] = []

    class _Lock:
        def acquire(self) -> bool:
            order.append("acquire")
            return True

        def release(self) -> None:
            order.append("release")

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    import cicerone.io.db_store as db_store

    real = db_store._add_missing_manifest_columns

    def _wrapped(*args, **kwargs):
        order.append("alter")
        return real(*args, **kwargs)

    monkeypatch.setattr(db_store, "_add_missing_manifest_columns", _wrapped)
    sink = DatabaseOutputSink({"database_url": url}, writer_lock=_Lock())
    sink.write_manifest({"n_events": 2, "status": "success", "n_item_scores": 1})
    assert order.index("acquire") < order.index("alter") < order.index("release")


def test_sqlite_write_item_scores_inspects_under_writer_lock(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"item_id": "i0", "popular_score": 0.5}]).to_sql("item_scores", engine, index=False)
    order: list[str] = []

    class _Lock:
        def acquire(self) -> bool:
            order.append("acquire")
            return True

        def release(self) -> None:
            order.append("release")

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    import cicerone.io.db_store as db_store

    real = db_store._missing_item_scores_columns

    def _wrapped(*args, **kwargs):
        order.append("inspect")
        return real(*args, **kwargs)

    monkeypatch.setattr(db_store, "_missing_item_scores_columns", _wrapped)
    sink = DatabaseOutputSink({"database_url": url}, writer_lock=_Lock())
    sink.write_item_scores(
        pd.DataFrame([{"item_id": "i1", "popular_score": 2.0, "latest_score": 1.0, "n_users": 3}])
    )
    assert order.index("acquire") < order.index("inspect") < order.index("release")


def test_sqlite_write_item_scores_replaces_legacy_table(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"item_id": "i0", "popular_score": 0.5}]).to_sql("item_scores", engine, index=False)

    sink = DatabaseOutputSink({"database_url": url})
    sink.write_item_scores(
        pd.DataFrame([{"item_id": "i1", "popular_score": 2.0, "latest_score": 1.0, "n_users": 3}])
    )

    stored = pd.read_sql('SELECT * FROM "item_scores"', engine)
    assert list(stored.columns) == ["item_id", "popular_score", "latest_score", "n_users"]
    assert stored.to_dict(orient="records") == [
        {"item_id": "i1", "popular_score": 2.0, "latest_score": 1.0, "n_users": 3}
    ]


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


def test_sqlite_write_recommendations_missing_optional_columns(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    sink = DatabaseOutputSink({"database_url": url})
    with pytest.raises(RecommendationSchemaError, match="ALTER TABLE"):
        sink.write_recommendations(
            pd.DataFrame(
                [
                    {
                        "user_id": "u1",
                        "item_id": "i1",
                        "rank": 1,
                        "score": 1.0,
                        "source": "personalized",
                        "variant": "control",
                        "reasons": "[]",
                    }
                ]
            )
        )


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
    collapsed = reader.get_recommendations("u1", k=10)
    assert list(collapsed["item_id"]) == ["control-item"]


def test_sqlite_db_reader_keeps_variant_filter_when_inspect_fails(tmp_path, monkeypatch):
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
    reader._variant_supported = None

    def boom(*_args, **_kwargs):
        raise RuntimeError("inspect down")

    monkeypatch.setattr("cicerone.io.db_recommendation_reader.inspect", boom)
    rows = reader.get_recommendations("u1", k=10, variant="treatment")
    assert list(rows["item_id"]) == ["treatment-item"]
    assert list(rows["variant"]) == ["treatment"]
    assert reader._variant_supported is None


def test_sqlite_db_reader_skips_distinct_when_unassigned(tmp_path, monkeypatch):
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
    seen: list[str] = []
    original = pd.read_sql

    def tracking(sql, *args, **kwargs):
        seen.append(str(sql))
        return original(sql, *args, **kwargs)

    monkeypatch.setattr("cicerone.io.db_recommendation_reader.pd.read_sql", tracking)
    rows = reader.get_recommendations("u1", k=10)
    assert list(rows["item_id"]) == ["control-item"]
    assert not any("DISTINCT" in sql.upper() for sql in seen)
    assert any(":fallback" in sql and "LIMIT" in sql.upper() for sql in seen)


def test_sqlite_db_reader_unassigned_picks_control_before_limit(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    sink.write_recommendations(
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "treatment-item",
                    "rank": 1,
                    "score": 0.9,
                    "source": "personalized",
                    "variant": "treatment",
                },
                {
                    "user_id": "u1",
                    "item_id": "control-item",
                    "rank": 2,
                    "score": 0.8,
                    "source": "personalized",
                    "variant": "control",
                },
            ]
        )
    )
    reader = DbRecommendationReader({"database_url": url})
    rows = reader.get_recommendations("u1", k=1)
    assert list(rows["item_id"]) == ["control-item"]
    assert reader.present_variant_names() == ("control", "treatment")


def test_sqlite_db_reader_missing_variant_column_falls_back(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False, if_exists="replace")
    reader = DbRecommendationReader({"database_url": url})
    assert list(reader.get_recommendations("u1", k=10, variant="treatment")["item_id"]) == ["i1"]
    other = DbRecommendationReader({"database_url": url})
    assert list(other.get_recommendations("u1", k=10)["item_id"]) == ["i1"]


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
        "cicerone.io.db_recommendation_reader.pd.read_sql",
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
        "cicerone.io.db_recommendation_reader.pd.read_sql",
        _raise_on_variant_sql(pd.read_sql, variant_queries),
    )
    cold = reader.get_cold_start_fallback(k=1, variant="treatment")
    assert list(cold["item_id"]) == ["i1"]
    assert reader._variant_supported is False
    assert list(reader.get_cold_start_fallback(k=1, variant="treatment")["item_id"]) == ["i1"]
    assert variant_queries["n"] == 1


def test_sqlite_db_reader_unassigned_probe_error_falls_back(tmp_path, monkeypatch):
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
                }
            ]
        )
    )
    reader = DbRecommendationReader({"database_url": url})
    original = pd.read_sql

    def fake_read_sql(sql, *args, **kwargs):
        if ":fallback" in str(sql):
            raise RuntimeError("prefer leftover down")
        return original(sql, *args, **kwargs)

    monkeypatch.setattr("cicerone.io.db_recommendation_reader.pd.read_sql", fake_read_sql)
    rows = reader.get_recommendations("u1", k=10)
    assert list(rows["item_id"]) == ["control-item"]


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

    monkeypatch.setattr("cicerone.io.db_recommendation_reader.pd.read_sql", fake_read_sql)
    with pytest.raises(ProgrammingError, match="user_id"):
        reader.get_recommendations("u1", k=10, variant="treatment")
    assert reader._variant_supported is True


def test_sqlite_write_manifest_skips_newer_row(tmp_path):
    url = _sqlite_url(tmp_path)
    sink = DatabaseOutputSink({"database_url": url})
    assert (
        sink.write_manifest(
            {
                "generated_at": "2099-01-01T00:00:00+00:00",
                "status": "success",
                "last_incremental_at": "2099-01-01T00:00:00+00:00",
            }
        )
        is True
    )
    assert (
        sink.write_manifest(
            {"generated_at": "2026-01-01T00:00:00+00:00", "status": "failed"},
            skip_if_newer_than="2026-01-01T00:00:00+00:00",
        )
        is False
    )
    engine = create_engine(url)
    stored = pd.read_sql("SELECT generated_at, status FROM recommendation_runs", engine)
    assert list(stored["status"]) == ["success"]


def test_sqlite_db_sink_fence_rejects_writes(tmp_path):
    url = _sqlite_url(tmp_path)
    owned = {"v": True}
    sink = DatabaseOutputSink(
        {"database_url": url},
        fence_check=lambda: owned["v"],
        fence_lost="events apply lock lost before write",
        fence_kind="apply",
    )
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "latest"}])
    sink.write_recommendations(recs)
    owned["v"] = False
    with pytest.raises(LockLostError, match="events apply lock lost"):
        sink.write_manifest({"status": "success"})
    with pytest.raises(LockLostError, match="events apply lock lost"):
        sink.replace_recommendations_for_users(recs, user_ids=["u1"])
    with pytest.raises(LockLostError, match="events apply lock lost"):
        sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1"}]))
    with pytest.raises(LockLostError, match="events apply lock lost"):
        sink.write_model_artifact(b"x")

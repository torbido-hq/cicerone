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
    load_items_catalog_size,
    load_recommendation_guardrail_rows,
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


def test_load_recommendations_keeps_optional_reasons(tmp_path):
    path = tmp_path / "recommendations.parquet"
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "reasons": '{"sources":[{"label":"personalized"}]}',
            }
        ]
    ).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert list(frame.columns) == [*list(empty_recommendations_frame().columns), "reasons"]
    assert "personalized" in str(frame.iloc[0]["reasons"])


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


def test_load_recommendations_for_users_dataset_filters(tmp_path, monkeypatch):
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

    def tracking_read(options, filename, *, s3_client=None, columns=None, filters=None):  # type: ignore[no-untyped-def]
        calls.append({"columns": columns, "filters": filters})
        return real_read(options, filename, s3_client=s3_client, columns=columns, filters=filters)

    monkeypatch.setattr("cicerone.events.store.read_parquet", tracking_read)
    frame = load_recommendations_for_users(settings.output, ["u2"])
    assert list(frame["user_id"]) == ["u2"]
    assert count_recommendation_users(settings.output) == 2
    assert load_recommendations_for_users(settings.output, []).empty
    assert any(call["filters"] == [("user_id", "in", ["u2"])] for call in calls)


def test_load_recommendations_for_users_dataset_filter_fallback(tmp_path, monkeypatch):
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
    real_read = __import__("cicerone.io.options", fromlist=["read_parquet"]).read_parquet

    def failing_filtered_read(options, filename, *, s3_client=None, columns=None, filters=None):  # type: ignore[no-untyped-def]
        if filters is not None:
            raise ValueError("Unsupported filter on user_id column")
        return real_read(options, filename, s3_client=s3_client, columns=columns, filters=filters)

    monkeypatch.setattr("cicerone.events.store.read_parquet", failing_filtered_read)
    frame = load_recommendations_for_users(settings.output, ["u2"])
    assert list(frame["user_id"]) == ["u2"]


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


def test_load_recommendations_for_users_db_empty_user_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "scoped_empty_ids.db"
    url = f"sqlite+pysqlite:///{db_path}"
    engine = create_engine(url)
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_sql("recommendations", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))

    calls = {"n": 0}
    real_read = pd.read_sql_query

    def spy_read_sql_query(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_sql_query", spy_read_sql_query)
    frame = load_recommendations_for_users(settings.output, [])
    assert frame.empty
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert calls["n"] == 0


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


def test_load_recommendations_for_users_db_schema_mismatch(tmp_path, caplog):
    db_path = tmp_path / "mismatched_scoped.db"
    url = f"sqlite+pysqlite:///{db_path}"
    engine = create_engine(url)
    pd.DataFrame(
        [
            {"item_id": "i1", "rank": 1, "score": 0.8, "source": "personalized"},
            {"item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    ).to_sql("recommendations", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    with caplog.at_level(logging.WARNING):
        frame = load_recommendations_for_users(settings.output, ["u1"])
    assert frame.empty
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert any("schema mismatch" in record.getMessage().lower() for record in caplog.records)


def test_load_recommendations_for_users_dataset_schema_mismatch(tmp_path, caplog):
    pd.DataFrame(
        [
            {"item_id": "i1", "rank": 1, "score": 0.8, "source": "personalized"},
            {"item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    with caplog.at_level(logging.WARNING):
        frame = load_recommendations_for_users(settings.output, ["u1"])
    assert frame.empty
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert any("schema mismatch" in record.getMessage().lower() for record in caplog.records)


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

    def tracking_read(options, filename, *, s3_client=None, columns=None, filters=None):  # type: ignore[no-untyped-def]
        calls.append(columns)
        return real_read(options, filename, s3_client=s3_client, columns=columns, filters=filters)

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


def test_load_items_catalog_size_dataset(tmp_path):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    assert load_items_catalog_size(settings.output) is None
    pd.DataFrame({"item_id": ["a", "b", "a"]}).to_parquet(tmp_path / "items_snapshot.parquet", index=False)
    assert load_items_catalog_size(settings.output) == 2
    assert load_items_catalog_size(IOSettings(kind="other", options={})) is None


def test_load_items_catalog_size_sqlite(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'items.db'}"
    engine = create_engine(url)
    pd.DataFrame({"item_id": ["a", "b", "a"]}).to_sql("recommendation_items", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    assert load_items_catalog_size(settings.output) == 2


def test_load_items_catalog_size_sqlite_missing_table(tmp_path):
    db_path = tmp_path / "missing_items.db"
    sqlite3.connect(db_path).close()
    settings = make_settings(
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{db_path}"})
    )
    assert load_items_catalog_size(settings.output) is None


def test_load_recommendation_guardrail_rows_dataset(tmp_path):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    empty = load_recommendation_guardrail_rows(settings.output)
    assert empty is not None
    assert empty.empty
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control",
            }
        ]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)
    frame = load_recommendation_guardrail_rows(settings.output)
    assert frame is not None
    assert list(frame["item_id"]) == ["i1"]
    assert list(frame["variant"]) == ["control"]
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendation_guardrail_rows(IOSettings(kind="other", options={}))


def test_load_recommendation_guardrail_rows_sqlite(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'guard.db'}"
    engine = create_engine(url)
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "treatment",
            }
        ]
    ).to_sql("recommendations", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    frame = load_recommendation_guardrail_rows(settings.output)
    assert frame is not None
    assert list(frame["variant"]) == ["treatment"]


def test_load_recommendation_guardrail_rows_sqlite_without_variant(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'guard_novar.db'}"
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_sql("recommendations", engine, index=False)
    settings = make_settings(output=IOSettings(kind="db", options={"database_url": url}))
    frame = load_recommendation_guardrail_rows(settings.output)
    assert frame is not None
    assert list(frame["item_id"]) == ["i1"]
    assert "variant" not in frame.columns or frame["variant"].isna().all()


def test_load_recommendation_guardrail_rows_sqlite_missing_table(tmp_path):
    db_path = tmp_path / "missing_guard.db"
    sqlite3.connect(db_path).close()
    settings = make_settings(
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{db_path}"})
    )
    frame = load_recommendation_guardrail_rows(settings.output)
    assert frame is not None
    assert frame.empty


def test_recommendation_engine_cache_evicts_oldest(tmp_path):
    from cicerone.events import store as events_store

    events_store.dispose_recommendation_engines()
    for i in range(events_store._MAX_CACHED_ENGINES + 2):
        path = tmp_path / f"e{i}.db"
        sqlite3.connect(path).close()
        events_store._engine_for(f"sqlite+pysqlite:///{path}")
    assert len(events_store._engines) == events_store._MAX_CACHED_ENGINES
    events_store.dispose_recommendation_engines()


def test_load_items_catalog_size_empty_and_read_errors(tmp_path, monkeypatch):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    pd.DataFrame({"item_id": pd.Series(dtype=str)}).to_parquet(
        tmp_path / "items_snapshot.parquet", index=False
    )
    assert load_items_catalog_size(settings.output) is None
    pd.DataFrame({"other": ["x"]}).to_parquet(tmp_path / "items_snapshot.parquet", index=False)
    assert load_items_catalog_size(settings.output) is None
    monkeypatch.setattr(
        "cicerone.events.store._read_parquet_columns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("cicerone.events.store.is_s3_not_found", lambda _exc: False)
    assert load_items_catalog_size(settings.output) is None


def test_read_parquet_columns_falls_back_without_projection(tmp_path, monkeypatch):
    from cicerone.events.store import _read_parquet_columns

    pd.DataFrame({"item_id": ["a"], "extra": [1]}).to_parquet(
        tmp_path / "items_snapshot.parquet", index=False
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    real = __import__("cicerone.io.options", fromlist=["read_parquet"]).read_parquet

    def _maybe_fail(options, filename, *, s3_client=None, columns=None, filters=None):
        if columns is not None:
            raise RuntimeError("no projection")
        return real(options, filename, s3_client=s3_client, columns=columns, filters=filters)

    monkeypatch.setattr("cicerone.events.store.read_parquet", _maybe_fail)
    frame = _read_parquet_columns(settings.output, "items_snapshot.parquet", ("item_id",))
    assert list(frame["item_id"]) == ["a"]


def test_load_recommendations_empty_existing_file(tmp_path):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"]).to_parquet(
        tmp_path / "recommendations.parquet", index=False
    )
    assert load_recommendations_frame(settings.output).empty
    assert count_recommendation_users(settings.output) == 0
    guard = load_recommendation_guardrail_rows(settings.output)
    assert guard is not None
    assert guard.empty


def test_load_recommendations_for_users_filter_fallback(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    real = __import__("cicerone.io.options", fromlist=["read_parquet"]).read_parquet

    def _fail_filters(options, filename, *, s3_client=None, columns=None, filters=None):
        if filters is not None:
            raise RuntimeError("user_id filter unsupported")
        return real(options, filename, s3_client=s3_client, columns=columns, filters=filters)

    monkeypatch.setattr("cicerone.events.store.read_parquet", _fail_filters)
    frame = load_recommendations_for_users(settings.output, ["u2"])
    assert list(frame["user_id"]) == ["u2"]


def test_load_db_recommendations_empty_user_ids(tmp_path):
    from cicerone.events.store import _load_db_recommendations

    url = f"sqlite+pysqlite:///{tmp_path / 'recs.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    assert _load_db_recommendations(output, user_ids=[]).empty


def test_load_items_catalog_size_sqlite_generic_error(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'items.db'}"
    output = IOSettings(kind="db", options={"database_url": url})

    class _Engine:
        def connect(self):
            raise RuntimeError("engine")

    monkeypatch.setattr("cicerone.events.store._engine_for", lambda _url: _Engine())
    monkeypatch.setattr("cicerone.events.store.is_missing_table_error", lambda _exc: False)
    monkeypatch.setattr("cicerone.events.store.is_missing_column_error", lambda _exc: False)
    assert load_items_catalog_size(output) is None

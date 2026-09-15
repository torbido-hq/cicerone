from __future__ import annotations

import pytest

from cicerone.config import IOSettings
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink
from cicerone.io.factory import (
    build_input_source,
    build_manifest_reader,
    build_output_sink,
    build_recommendation_reader,
    build_user_history_reader,
)
from cicerone.io.manifest_reader import DatasetManifestReader, DbManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader, DbRecommendationReader


def test_build_input_source_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert isinstance(build_input_source(settings), DatasetInputSource)
    assert isinstance(build_user_history_reader(settings), DatasetInputSource)


def test_build_output_sink_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    sink = build_output_sink(settings)
    assert isinstance(sink, DatasetOutputSink)
    assert sink._writer_lock is None

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return False

    lock = _Lock()
    locked = build_output_sink(
        settings,
        writer_lock=lock,
        fence_check=lock.owned,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    assert isinstance(locked, DatasetOutputSink)
    assert locked._writer_lock is lock
    assert locked._fence_check is not None
    assert locked._fence_check() is True
    assert locked._fence_kind == "retrain"


def test_build_input_source_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_input_source(settings), DatabaseInputSource)
    assert isinstance(build_user_history_reader(settings), DatabaseInputSource)


def test_build_output_sink_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_output_sink(settings), DatabaseOutputSink)
    fenced = build_output_sink(
        settings,
        fence_check=lambda: True,
        fence_lost="events apply lock lost before write",
        fence_kind="apply",
    )
    assert isinstance(fenced, DatabaseOutputSink)
    assert fenced._fence_check is not None
    assert fenced._fence_check() is True
    assert fenced._fence_kind == "apply"
    locked = object()
    with_lock = build_output_sink(settings, writer_lock=locked)
    assert with_lock._writer_lock is locked
    assert callable(with_lock.recommendations_write)


def test_database_write_honors_writer_lock(tmp_path):
    from cicerone.locks import LockLostError

    class _Lost:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return False

        def is_locked(self) -> bool:
            return True

    import pandas as pd

    url = f"sqlite+pysqlite:///{tmp_path / 'recs.db'}"
    sink = DatabaseOutputSink({"database_url": url}, writer_lock=_Lost())
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        sink.write_recommendations(pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1}]))


def test_database_write_manifest_rechecks_fence_before_append(tmp_path):
    from cicerone.locks import LockLostError

    url = f"sqlite+pysqlite:///{tmp_path / 'manifest.db'}"
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 2

    sink = DatabaseOutputSink(
        {"database_url": url},
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        sink.write_manifest({"n_events": 1, "generated_at": "t"})
    assert exc.value.kind == "retrain"
    assert checks["n"] >= 2


def test_database_write_recommendations_rechecks_fence_after_clear(tmp_path, monkeypatch):
    import pandas as pd
    from sqlalchemy import create_engine, inspect, text

    from cicerone.io import db_store as db_store_mod
    from cicerone.locks import LockLostError

    url = f"sqlite+pysqlite:///{tmp_path / 'recs.db'}"
    held = {"ok": True}

    def fence() -> bool:
        return held["ok"]

    original = db_store_mod._clear_table_for_replace

    def _clear(conn, table):
        original(conn, table)
        held["ok"] = False

    monkeypatch.setattr(db_store_mod, "_clear_table_for_replace", _clear)
    sink = DatabaseOutputSink(
        {"database_url": url},
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        sink.write_recommendations(pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1}]))
    assert exc.value.kind == "retrain"
    engine = create_engine(url)
    if inspect(engine).has_table("recommendations"):
        stored = pd.read_sql(text('SELECT * FROM "recommendations"'), engine)
        assert stored.empty


def test_database_write_items_rechecks_fence_after_clear(tmp_path, monkeypatch):
    import pandas as pd
    from sqlalchemy import create_engine, inspect, text

    from cicerone.io import db_store as db_store_mod
    from cicerone.locks import LockLostError

    url = f"sqlite+pysqlite:///{tmp_path / 'items.db'}"
    held = {"ok": True}

    def fence() -> bool:
        return held["ok"]

    original = db_store_mod._clear_table_for_replace

    def _clear(conn, table):
        original(conn, table)
        held["ok"] = False

    monkeypatch.setattr(db_store_mod, "_clear_table_for_replace", _clear)
    sink = DatabaseOutputSink(
        {"database_url": url},
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1"}]))
    assert exc.value.kind == "retrain"
    engine = create_engine(url)
    if inspect(engine).has_table("recommendation_items"):
        stored = pd.read_sql(text('SELECT * FROM "recommendation_items"'), engine)
        assert stored.empty


def test_build_input_source_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown input kind"):
        build_input_source(settings)


def test_build_output_sink_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown output kind"):
        build_output_sink(settings)


def test_build_recommendation_reader_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert isinstance(build_recommendation_reader(settings), DatasetRecommendationReader)


def test_build_recommendation_reader_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_recommendation_reader(settings), DbRecommendationReader)


def test_build_recommendation_reader_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown recommendation kind"):
        build_recommendation_reader(settings)


def test_build_manifest_reader_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert isinstance(build_manifest_reader(settings), DatasetManifestReader)


def test_build_manifest_reader_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_manifest_reader(settings), DbManifestReader)


def test_build_manifest_reader_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        build_manifest_reader(settings)

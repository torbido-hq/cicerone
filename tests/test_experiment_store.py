from __future__ import annotations

import json
import threading

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.config import ConfigError, IOSettings
from cicerone.experiment.store import ExperimentStore, experiment_state
from cicerone.locks import LockLostError, WriterLockBusyError, held_writer_lock


def test_experiment_store_roundtrip_dataset(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    assert store.read_state() is None
    assert store.read_exposures() == []

    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp"
    assert state["promoted_variant"] == "treatment"

    other = ExperimentStore(output)
    seen = other.read_state()
    assert seen is not None
    assert seen["promoted_variant"] == "treatment"

    store.append_exposures(
        [
            {
                "user_id": "u1",
                "experiment_id": "exp",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "2026-08-25T00:00:00+00:00",
            }
        ]
    )
    store.append_exposures(
        [
            {
                "user_id": "u2",
                "experiment_id": "exp",
                "variant": "treatment",
                "generated_at": "t",
                "exposed_at": "2026-08-25T00:01:00+00:00",
            }
        ]
    )
    rows = store.read_exposures()
    assert [row["user_id"] for row in rows] == ["u1", "u2"]
    store.append_exposures(
        [
            {
                "user_id": "u9",
                "experiment_id": "other",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "2026-08-25T00:02:00+00:00",
            }
        ]
    )
    assert [row["user_id"] for row in store.read_exposures(experiment_id="exp")] == ["u1", "u2"]


def test_experiment_store_roundtrip_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    assert store.read_state() is None
    assert store.read_exposures() == []
    store.write_state(experiment_state("exp", promoted_variant="control"))
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "control"
    store.append_exposures(
        [
            {
                "user_id": "u1",
                "experiment_id": "exp",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "t",
            }
        ]
    )
    assert store.read_exposures()[0]["user_id"] == "u1"
    engine = store._db_engine()
    store.read_state()
    store.append_exposures(
        [
            {
                "user_id": "u2",
                "experiment_id": "exp",
                "variant": "treatment",
                "generated_at": None,
                "exposed_at": "t2",
            }
        ]
    )
    assert store._db_engine() is engine
    assert [row["user_id"] for row in store.read_exposures()] == ["u1", "u2"]


def test_experiment_store_sqlite_replace_clears_other_experiment_ids(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp-a", promoted_variant="control"))
    store.write_state(experiment_state("exp-b", promoted_variant="treatment"))
    frame = pd.read_sql(text("SELECT * FROM experiment_state"), store._db_engine())
    assert len(frame) == 1
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp-b"
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_reads_legacy_table_without_promoted_at(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    engine = store._db_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT)"))
        conn.execute(
            text("INSERT INTO experiment_state (experiment_id, promoted_variant) VALUES ('exp', 'treatment')")
        )
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp"
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_sqlite_replaces_same_experiment_id(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="control"))
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    frame = pd.read_sql(text("SELECT * FROM experiment_state"), store._db_engine())
    assert len(frame) == 1
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_raises_on_invalid_dataset_state(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    (tmp_path / "experiment_state.json").write_text("not-json", encoding="utf-8")
    store = ExperimentStore(output)
    with pytest.raises(json.JSONDecodeError):
        store.read_state()
    (tmp_path / "experiment_state.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        store.read_state()


def test_assignment_overlay_keeps_last_on_invalid_json(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant=None, champion="control", challenger="blend"))
    (tmp_path / "experiment_state.json").write_text("not-json", encoding="utf-8")
    promoted, pair = store.assignment_overlay("exp")
    assert promoted is None
    assert pair == ("control", "blend")


def test_assignment_overlay_reraises_unexpected_read_error(tmp_path, monkeypatch) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant=None, champion="control", challenger="blend"))

    def _boom(self):
        raise RuntimeError("state bug")

    monkeypatch.setattr(ExperimentStore, "read_state", _boom)
    with pytest.raises(RuntimeError, match="state bug"):
        store.assignment_overlay("exp")


def test_append_exposures_rejects_object_store() -> None:
    output = IOSettings(
        kind="dataset",
        options={
            "storage_backend": "s3",
            "bucket": "recs",
            "access_key_id": "id",
            "secret_access_key": "secret",
        },
    )
    store = ExperimentStore(output)
    with pytest.raises(ConfigError, match="not atomic"):
        store.append_exposures(
            [
                {
                    "user_id": "u1",
                    "experiment_id": "exp",
                    "variant": "control",
                    "generated_at": None,
                    "exposed_at": "t",
                }
            ]
        )


def test_require_appendable_exposure_log_allows_db_and_local(tmp_path) -> None:
    from cicerone.experiment.store import require_appendable_exposure_log

    require_appendable_exposure_log(IOSettings(kind="db", options={"database_url": "sqlite://"}))
    require_appendable_exposure_log(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )


def _exposure(user_id: str, suffix: str = "00") -> dict[str, str | None]:
    return {
        "user_id": user_id,
        "experiment_id": "exp",
        "variant": "control",
        "generated_at": None,
        "exposed_at": f"2026-08-25T00:{suffix}:00+00:00",
    }


def test_append_exposures_serializes_concurrent_local_writers(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    started = threading.Barrier(2)

    def _append(user_id: str, suffix: str) -> None:
        started.wait()
        store.append_exposures([_exposure(user_id, suffix)])

    threads = [
        threading.Thread(target=_append, args=("u1", "00")),
        threading.Thread(target=_append, args=("u2", "01")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(row["user_id"] for row in store.read_exposures()) == ["u1", "u2"]


def test_append_exposures_writer_lock_busy(tmp_path, monkeypatch) -> None:
    class _Busy:
        def acquire(self) -> bool:
            return False

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return False

        def is_locked(self) -> bool:
            return True

    monkeypatch.setattr("cicerone.locks.acquire_blocking", lambda _lock, **_kwargs: False)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output, writer_lock=_Busy())
    with pytest.raises(RuntimeError, match="dataset writer lock busy"):
        store.append_exposures([_exposure("u1")])
    assert store.read_exposures() == []


def test_append_exposures_writer_lock_lost(tmp_path) -> None:
    class _Lost:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return False

        def is_locked(self) -> bool:
            return True

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output, writer_lock=_Lost())
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        store.append_exposures([_exposure("u1")])
    assert store.read_exposures() == []


def test_append_exposures_rechecks_owned_before_write(tmp_path) -> None:
    calls = {"n": 0}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            calls["n"] += 1
            return calls["n"] < 2

        def is_locked(self) -> bool:
            return True

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output, writer_lock=_Lock())
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        store.append_exposures([_exposure("u1")])
    assert store.read_exposures() == []


def test_write_state_writer_lock_lost(tmp_path) -> None:
    class _Lost:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return False

        def is_locked(self) -> bool:
            return True

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output, writer_lock=_Lost())
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.read_state() is None


def test_write_state_skips_acquire_when_held_here(tmp_path) -> None:
    acquires = {"n": 0}

    class _Lock:
        def acquire(self) -> bool:
            acquires["n"] += 1
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    lock = _Lock()
    store = ExperimentStore(output, writer_lock=lock)
    with held_writer_lock(lock):
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert acquires["n"] == 1
    assert store.read_state() is not None


def test_write_state_other_thread_does_not_skip_acquire(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("cicerone.locks.acquire_blocking", lambda _lock, **_kwargs: False)

    class _Busy:
        def acquire(self) -> bool:
            return False

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output, writer_lock=_Busy())
    with pytest.raises(WriterLockBusyError, match="dataset writer lock busy"):
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.read_state() is None


def test_write_state_db_honors_fence(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(
        output,
        fence_check=lambda: False,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert store.read_state() is None
    engine = create_engine(url)
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_state'")
        ).fetchall()
    assert tables == []


def test_write_state_db_rolls_back_create_when_fence_lost_after_ddl(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 2

    store = ExperimentStore(
        IOSettings(kind="db", options={"database_url": url}),
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert store.read_state() is None
    engine = create_engine(url)
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_state'")
        ).fetchall()
    assert tables == []


def test_write_state_db_rechecks_fence_before_replace(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 3

    store = ExperimentStore(
        output,
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert checks["n"] >= 3


def test_write_state_db_rechecks_fence_after_delete(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 4

    store = ExperimentStore(
        output,
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert store.read_state() is None
    assert checks["n"] >= 4


def test_write_state_db_rechecks_fence_on_legacy_fallback(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE experiment_state ("
                "experiment_id TEXT PRIMARY KEY, "
                "promoted_variant TEXT, "
                "promoted_at TEXT"
                ")"
            )
        )
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 4

    store = ExperimentStore(
        IOSettings(kind="db", options={"database_url": url}),
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert checks["n"] >= 4


def test_write_state_db_legacy_fallback_completes(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE experiment_state ("
                "experiment_id TEXT PRIMARY KEY, "
                "promoted_variant TEXT, "
                "promoted_at TEXT"
                ")"
            )
        )
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))
    payload = experiment_state("exp", promoted_variant="treatment")
    store.write_state(payload)
    assert store.read_state()["promoted_variant"] == "treatment"


def test_write_state_db_rolls_back_alter_when_fence_lost_after_ddl(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE experiment_state ("
                "experiment_id TEXT PRIMARY KEY, "
                "promoted_variant TEXT, "
                "promoted_at TEXT"
                ")"
            )
        )
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 5

    store = ExperimentStore(
        IOSettings(kind="db", options={"database_url": url}),
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert exc.value.kind == "retrain"
    assert store.read_state() is None
    with engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(experiment_state)"))]
    assert "payload" not in columns


def test_append_exposures_rechecks_after_file_lock(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    fence = {"ok": True}

    @contextmanager
    def _lock(_path):
        fence["ok"] = False
        yield

    monkeypatch.setattr("cicerone.experiment.store.exclusive_file_lock", _lock)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(
        output,
        fence_check=lambda: fence["ok"],
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.append_exposures(
            [{"experiment_id": "exp", "user_id": "u1", "variant": "control", "exposed_at": "t"}]
        )
    assert exc.value.kind == "retrain"
    assert not (tmp_path / "exposures.jsonl").exists()


def test_append_exposures_honors_fence(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(
        output,
        fence_check=lambda: False,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.append_exposures(
            [{"experiment_id": "exp", "user_id": "u1", "variant": "control", "exposed_at": "t"}]
        )
    assert exc.value.kind == "retrain"
    assert not (tmp_path / "exposures.jsonl").exists()


def test_append_exposures_db_rechecks_fence_before_insert(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    checks = {"n": 0}

    def fence() -> bool:
        checks["n"] += 1
        return checks["n"] < 3

    store = ExperimentStore(
        IOSettings(kind="db", options={"database_url": url}),
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.append_exposures(
            [{"experiment_id": "exp", "user_id": "u1", "variant": "control", "exposed_at": "t"}]
        )
    assert exc.value.kind == "retrain"
    assert store.read_exposures() == []
    assert checks["n"] >= 3


def test_append_exposures_db_rolls_back_when_fence_lost_after_insert(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    held = {"ok": True}

    def fence() -> bool:
        return held["ok"]

    original = pd.DataFrame.to_sql

    def _to_sql(self, *args, **kwargs):
        original(self, *args, **kwargs)
        held["ok"] = False

    monkeypatch.setattr(pd.DataFrame, "to_sql", _to_sql)
    store = ExperimentStore(
        IOSettings(kind="db", options={"database_url": url}),
        fence_check=fence,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as exc:
        store.append_exposures(
            [{"experiment_id": "exp", "user_id": "u1", "variant": "control", "exposed_at": "t"}]
        )
    assert exc.value.kind == "retrain"
    assert store.read_exposures() == []


def test_read_state_db_pandas_database_error_is_empty(tmp_path, monkeypatch) -> None:
    from pandas.errors import DatabaseError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise DatabaseError("no such table: experiment_state")

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    assert store.read_state() is None


def test_read_state_db_pandas_database_error_retries_legacy_schema(tmp_path, monkeypatch) -> None:
    from pandas.errors import DatabaseError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))
    calls = {"n": 0}

    def _read(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DatabaseError("no such column: promoted_at")
        return pd.DataFrame([{"experiment_id": "exp", "promoted_variant": "treatment", "payload": None}])

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "treatment"
    assert calls["n"] == 2


def test_read_exposures_db_pandas_database_error_is_empty(tmp_path, monkeypatch) -> None:
    from pandas.errors import DatabaseError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise DatabaseError("no such table: exposures")

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    assert store.read_exposures() == []


def test_read_exposures_db_generic_missing_table_is_empty(tmp_path, monkeypatch) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise SQLAlchemyError("no such table: exposures")

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    monkeypatch.setattr("cicerone.experiment.store.is_missing_table_error", lambda _exc: True)
    assert store.read_exposures() == []


def test_read_exposures_db_unexpected_error_ignores_missing_table_helper(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise RuntimeError("no such table: exposures")

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    monkeypatch.setattr("cicerone.experiment.store.is_missing_table_error", lambda _exc: True)
    with pytest.raises(RuntimeError, match="no such table"):
        store.read_exposures()


def test_read_exposures_db_unexpected_error_reraises(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    with pytest.raises(RuntimeError, match="db down"):
        store.read_exposures()


def test_read_exposures_db_operational_error_reraises(tmp_path, monkeypatch) -> None:
    from sqlalchemy.exc import OperationalError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    store = ExperimentStore(IOSettings(kind="db", options={"database_url": url}))

    def _read(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", _read)
    with pytest.raises(OperationalError, match="connection refused"):
        store.read_exposures()


def test_append_exposures_empty_is_noop(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    ExperimentStore(output).append_exposures([])
    assert not (tmp_path / "exposures.jsonl").exists()


def test_experiment_store_state_roundtrip_s3() -> None:
    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="recs")
        output = IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
                "prefix": "out",
            },
        )
        store = ExperimentStore(output)
        assert store.read_state() is None
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
        state = store.read_state()
        assert state is not None
        assert state["promoted_variant"] == "treatment"


def test_experiment_store_s3_unexpected_read_reraises(monkeypatch) -> None:
    class _Boom:
        def get_object(self, **_kwargs):
            raise RuntimeError("network")

    monkeypatch.setattr("cicerone.io.blob.build_s3_client", lambda _options: _Boom())
    store = ExperimentStore(
        IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
            },
        )
    )
    with pytest.raises(RuntimeError, match="network"):
        store.read_state()
    with pytest.raises(RuntimeError, match="network"):
        store.read_exposures()


def test_experiment_store_s3_access_denied_reraises(monkeypatch) -> None:
    from botocore.exceptions import ClientError

    class _Denied:
        def get_object(self, **_kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"}}, "GetObject")

    monkeypatch.setattr("cicerone.io.blob.build_s3_client", lambda _options: _Denied())
    store = ExperimentStore(
        IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
            },
        )
    )
    with pytest.raises(ClientError, match="AccessDenied"):
        store.read_state()
    with pytest.raises(ClientError, match="AccessDenied"):
        store.read_exposures()


def test_promoted_variant_reuses_cache_when_read_fails(tmp_path, monkeypatch) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.promoted_variant("exp") == "treatment"

    def boom() -> None:
        raise OSError("store down")

    monkeypatch.setattr(store, "read_state", boom)
    assert store.promoted_variant("exp") == "treatment"
    assert store.promoted_variant("other") is None


def test_assignment_overlay_keeps_last_on_backend_io_error(tmp_path, monkeypatch) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="treatment", champion="control", challenger="blend")
    )
    assert store.assignment_overlay("exp") == ("treatment", ("control", "blend"))

    def boom(self):
        raise SQLAlchemyError("engine")

    monkeypatch.setattr(ExperimentStore, "read_state", boom)
    assert store.assignment_overlay("exp") == ("treatment", ("control", "blend"))


def test_promoted_variant_reuses_cache_when_db_read_raises(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.promoted_variant("exp") == "treatment"

    def boom() -> None:
        raise OSError("store down")

    monkeypatch.setattr(store, "_read_state_db", boom)
    assert store.promoted_variant("exp") == "treatment"


def test_read_state_db_reraises_transient_operational_error(tmp_path, monkeypatch) -> None:
    from sqlalchemy.exc import OperationalError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))

    def boom(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", boom)
    with pytest.raises(OperationalError, match="connection refused"):
        store.read_state()
    assert store.assignment_overlay("exp")[0] == "treatment"


def test_read_state_db_reraises_unclassified_programming_error(tmp_path, monkeypatch) -> None:
    from sqlalchemy.exc import ProgrammingError

    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))

    def boom(*_args, **_kwargs):
        raise ProgrammingError("SELECT 1", {}, Exception("permission denied"))

    monkeypatch.setattr("cicerone.experiment.store.pd.read_sql", boom)
    with pytest.raises(ProgrammingError, match="permission denied"):
        store.read_state()


def test_experiment_store_prefers_timestamped_promote_over_null(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="control"))
    engine = store._db_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE experiment_state"))
        conn.execute(
            text(
                "CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT, promoted_at TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'legacy', NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'winner', '2026-09-01T00:00:00+00:00')"
            )
        )
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "winner"


def test_experiment_store_last_state_reuses_write_cache(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="treatment", promoted_at="2026-09-01T00:00:00Z")
    )
    cached = store.last_state("exp")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"
    assert cached["promoted_at"] == "2026-09-01T00:00:00Z"
    assert store.last_state("other") is None


def test_last_state_ignores_cache_when_promoted_variant_queries_other_id(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="treatment", promoted_at="2026-09-02T00:00:00Z")
    )
    assert store.promoted_variant("other") is None
    assert store.last_state("other") is None
    cached = store.last_state("exp")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"


def test_last_state_matches_non_string_experiment_id(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        {
            "experiment_id": 7,
            "promoted_variant": "treatment",
            "promoted_at": "2026-09-02T00:00:00Z",
        }
    )
    cached = store.last_state("7")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"


def test_read_exposures_missing_experiment_id_column_returns_empty(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE recommendation_exposures (user_id TEXT, variant TEXT)"))
        conn.execute(text("INSERT INTO recommendation_exposures (user_id, variant) VALUES ('u1', 'control')"))
    assert ExperimentStore(output).read_exposures(experiment_id="exp") == []


def test_jsonish_numpy_scalar_and_na() -> None:
    import numpy as np
    import pandas as pd

    from cicerone.experiment.store import _jsonish

    assert _jsonish(np.int64(3)) == 3
    assert _jsonish(pd.NA) is None


def test_experiment_state_extra_keys_roundtrip_dataset(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state(
            "exp",
            promoted_variant=None,
            champion="control",
            challenger="blend",
            arms={"control": {"successes": 3, "failures": 1}},
            p_best={"control": 0.8, "blend": 0.2},
            pair_impressions=12,
        )
    )
    state = store.read_state()
    assert state is not None
    assert state["champion"] == "control"
    assert state["challenger"] == "blend"
    assert state["arms"]["control"]["successes"] == 3
    promoted, pair = store.assignment_overlay("exp")
    assert promoted is None
    assert pair == ("control", "blend")


def test_experiment_state_payload_roundtrip_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="control", champion="control", challenger="treatment")
    )
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "control"
    assert state["champion"] == "control"
    assert state["challenger"] == "treatment"
    assert "payload" not in state
    promoted, pair = store.assignment_overlay("exp")
    assert promoted == "control"
    assert pair == ("control", "treatment")


def test_experiment_state_alters_legacy_sqlite_table(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT, promoted_at TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'control', '2026-09-01T00:00:00Z')"
            )
        )
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant=None, champion="a", challenger="b"))
    state = store.read_state()
    assert state is not None
    assert state["champion"] == "a"
    assert state["challenger"] == "b"


def test_hydrate_state_row_ignores_invalid_payload() -> None:
    from cicerone.experiment.store import (
        _hydrate_state_row,
        active_pair_from_state,
        merge_experiment_state,
    )

    assert _hydrate_state_row({"experiment_id": "exp", "payload": "{not-json"}) == {"experiment_id": "exp"}
    assert _hydrate_state_row({"experiment_id": "exp", "payload": '["list"]'}) == {"experiment_id": "exp"}
    hydrated = _hydrate_state_row(
        {"experiment_id": "exp", "promoted_variant": "control", "payload": {"champion": "a", "payload": "x"}}
    )
    assert hydrated["champion"] == "a"
    assert "payload" not in hydrated
    assert active_pair_from_state(None) is None
    assert active_pair_from_state({"champion": "a"}) is None
    merged = merge_experiment_state(
        {"champion": "old", "extra": 1},
        experiment_id="exp",
        promoted_variant=None,
        challenger="new",
    )
    assert merged["champion"] == "old"
    assert merged["challenger"] == "new"
    assert merged["extra"] == 1

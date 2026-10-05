"""Erase one user's output rows without touching the catalog."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone.cli import main
from cicerone.config import ConfigError, IOSettings
from cicerone.experiment.store import ExperimentStore
from cicerone.forget import forget_user
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.db_store import DatabaseOutputSink
from cicerone.locks.hold import held_writer_lock
from cicerone.track.store import TrackStore


def _local(tmp_path) -> IOSettings:
    return IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})


def _recs(*users: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": user, "item_id": f"i-{user}", "rank": 1, "score": 0.5, "source": "personalized"}
            for user in users
        ]
    )


def _seed(output: IOSettings) -> None:
    if output.kind == "dataset":
        DatasetOutputSink(output.options).write_recommendations(_recs("u1", "u2"))
    else:
        DatabaseOutputSink(output.options).write_recommendations(_recs("u1", "u2"))
    track = TrackStore(output)
    track.append_rows(
        [
            {"event_id": "e1", "kind": "impression", "user_id": "u1", "item_id": "i-u1"},
            {"event_id": "e2", "kind": "impression", "user_id": "u2", "item_id": "i-u2"},
        ]
    )
    track.append_history(_recs("u1", "u2"), generated_at="2026-09-29T12:00:00+00:00")
    ExperimentStore(output).append_exposures(
        [
            {"user_id": "u1", "experiment_id": "exp", "variant": "control"},
            {"user_id": "u2", "experiment_id": "exp", "variant": "treatment"},
        ]
    )


def _remaining_users(output: IOSettings) -> set[str]:
    if output.kind == "dataset":
        frame = pd.read_parquet(output.options["path"] + "/recommendations.parquet")
    else:
        from sqlalchemy import create_engine

        engine = create_engine(output.options["database_url"])
        frame = pd.read_sql('SELECT user_id FROM "recommendations"', engine)
        engine.dispose()
    track_users = {row["user_id"] for row in TrackStore(output).read_rows()}
    history = TrackStore(output).read_history()
    exposure_users = {row["user_id"] for row in ExperimentStore(output).read_exposures()}
    return set(frame["user_id"]) | track_users | set(history["user_id"]) | exposure_users


def test_forget_user_on_local_dataset_keeps_other_users(tmp_path) -> None:
    output = _local(tmp_path)
    _seed(output)
    result = forget_user(output, "u1")
    assert result.recommendations == 1
    assert result.track == 1
    assert result.exposures == 1
    assert result.history == 1
    assert _remaining_users(output) == {"u2"}


def test_forget_user_on_sqlite_keeps_other_users(tmp_path) -> None:
    output = IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'out.db'}"})
    _seed(output)
    result = forget_user(output, " u1 ")
    assert result.recommendations == 1
    assert result.track == 1
    assert result.exposures == 1
    assert result.history == 1
    assert _remaining_users(output) == {"u2"}


def test_forget_user_missing_output_is_zero(tmp_path) -> None:
    result = forget_user(_local(tmp_path), "u1")
    assert result == type(result)(0, 0, 0, 0)


def test_forget_user_refuses_object_store() -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "s3", "bucket": "b", "path": "unused"})
    with pytest.raises(ConfigError, match="local dataset"):
        forget_user(output, "u1")


def test_forget_user_requires_a_user_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="user_id"):
        forget_user(_local(tmp_path), "  ")


def test_jsonl_without_a_user_row_leaves_recommendations(tmp_path) -> None:
    output = _local(tmp_path)
    DatasetOutputSink(output.options).write_recommendations(_recs("u1"))
    (tmp_path / "track.jsonl").write_text('{"item_id": "i-u1"}\n[1]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="user_id"):
        forget_user(output, "u1")
    assert list(pd.read_parquet(tmp_path / "recommendations.parquet")["user_id"]) == ["u1"]


def test_corrupt_track_log_leaves_recommendations(tmp_path) -> None:
    output = _local(tmp_path)
    DatasetOutputSink(output.options).write_recommendations(_recs("u1"))
    (tmp_path / "track.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        forget_user(output, "u1")
    frame = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert list(frame["user_id"]) == ["u1"]


class _WriterLock:
    def __init__(self) -> None:
        self.held = False

    def acquire(self) -> bool:
        self.held = True
        return True

    def release(self) -> None:
        self.held = False

    def owned(self, generation: int | None = None) -> bool:
        del generation
        return self.held

    def is_locked(self) -> bool:
        return self.held


def test_forget_user_reuses_a_held_writer_lock(tmp_path) -> None:
    output = _local(tmp_path)
    _seed(output)
    lock = _WriterLock()
    with held_writer_lock(lock):
        result = forget_user(output, "u1", writer_lock=lock)
        assert lock.held
    assert result.recommendations == 1
    assert _remaining_users(output) == {"u2"}


def test_forget_user_rewrites_legacy_history_parquet(tmp_path) -> None:
    output = _local(tmp_path)
    legacy = tmp_path / "recommendation_history.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i-u1", "i-u2"],
            "rank": [1, 1],
            "source": ["personalized", "personalized"],
            "variant": [None, None],
            "generated_at": ["2026-08-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00"],
        }
    ).to_parquet(legacy, index=False)
    assert forget_user(output, "u1").history == 1
    kept = pd.read_parquet(legacy)
    assert list(kept["user_id"]) == ["u2"]


def test_forget_user_drops_a_history_part_that_only_held_that_user(tmp_path) -> None:
    output = _local(tmp_path)
    TrackStore(output).append_history(_recs("u1"), generated_at="2026-09-29T12:00:00+00:00")
    assert forget_user(output, "u1").history == 1
    assert list((tmp_path / "recommendation_history").glob("*.parquet")) == []
    assert list((tmp_path / "recommendation_history").glob(".*.tmp")) == []


def test_history_without_user_id_leaves_recommendations(tmp_path) -> None:
    output = _local(tmp_path)
    DatasetOutputSink(output.options).write_recommendations(_recs("u1"))
    history = tmp_path / "recommendation_history"
    history.mkdir()
    pd.DataFrame({"item_id": ["i-u1"]}).to_parquet(history / "snap.parquet", index=False)
    with pytest.raises(ValueError, match="user_id"):
        forget_user(output, "u1")
    assert list(pd.read_parquet(tmp_path / "recommendations.parquet")["user_id"]) == ["u1"]


def test_sqlite_schema_error_rolls_back_earlier_deletes(tmp_path) -> None:
    output = IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'out.db'}"})
    _seed(output)
    engine = create_engine(output.options["database_url"])
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE recommendation_history"))
        conn.execute(text("CREATE TABLE recommendation_history (item_id TEXT)"))
        conn.execute(text("INSERT INTO recommendation_history (item_id) VALUES ('i-u1')"))
    with pytest.raises(ValueError, match="user_id"):
        forget_user(output, "u1")
    recommendations = pd.read_sql('SELECT user_id FROM "recommendations"', engine)
    track = pd.read_sql('SELECT user_id FROM "recommendation_track"', engine)
    engine.dispose()
    assert set(recommendations["user_id"]) == {"u1", "u2"}
    assert set(track["user_id"]) == {"u1", "u2"}


def test_forget_user_missing_sqlite_tables_are_zero(tmp_path) -> None:
    output = IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'empty.db'}"})
    result = forget_user(output, "u1")
    assert result.recommendations == 0
    assert result.track == 0
    assert result.exposures == 0
    assert result.history == 0


def test_cli_forget_user_prints_counts(tmp_path, capsys) -> None:
    output = tmp_path / "out"
    output.mkdir()
    config = tmp_path / "cicerone.toml"
    config.write_text(
        f"""
        [job]
        mode = "batch"
        cron_schedule = "0 3 * * *"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path / "in"}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{output}"
        """,
        encoding="utf-8",
    )
    _seed(_local(output))
    assert main(["--config", str(config), "forget-user", "u1"]) == 0
    captured = capsys.readouterr()
    assert "recommendations=1" in captured.out
    assert "track=1" in captured.out
    assert _remaining_users(_local(output)) == {"u2"}

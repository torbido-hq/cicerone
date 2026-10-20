"""Erase one user's output rows without touching the catalog."""

from __future__ import annotations

from pathlib import Path

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
    assert list(tmp_path.rglob(".*.tmp")) == []


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


def test_local_stage_failure_leaves_every_file(tmp_path, monkeypatch) -> None:
    output = _local(tmp_path)
    _seed(output)
    track = TrackStore(output)
    track.append_history(_recs("u1", "u2"), generated_at="2026-09-29T13:00:00+00:00")
    before = _local_bytes(tmp_path)
    original = pd.DataFrame.to_parquet
    seen: list[str] = []

    def boom(self: pd.DataFrame, path, *args, **kwargs):
        target = Path(path)
        if target.name.startswith(".") and target.parent.name == "recommendation_history":
            seen.append(target.name)
            if len(seen) > 1:
                raise OSError("disk full")
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        forget_user(output, "u1")
    assert len(seen) == 2
    assert _local_bytes(tmp_path) == before
    assert list(tmp_path.rglob(".*.tmp")) == []


def test_local_commit_failure_restores_every_file(tmp_path, monkeypatch) -> None:
    import cicerone.forget as forget_mod

    output = _local(tmp_path)
    _seed(output)
    before = _local_bytes(tmp_path)
    original = forget_mod.os.replace
    swapped_in: list[str] = []

    def flaky(src, dst):
        if Path(src).name.endswith(".tmp"):
            swapped_in.append(Path(dst).name)
            if len(swapped_in) == 2:
                raise OSError("rename failed")
        return original(src, dst)

    monkeypatch.setattr(forget_mod.os, "replace", flaky)
    with pytest.raises(OSError, match="rename failed"):
        forget_user(output, "u1")
    assert swapped_in == ["recommendations.parquet", "track.jsonl"]
    assert _local_bytes(tmp_path) == before
    assert list(tmp_path.rglob(".*.tmp")) == []
    assert list(tmp_path.rglob(".*.bak")) == []


def test_local_backup_rename_failure_leaves_no_backup(tmp_path, monkeypatch) -> None:
    import cicerone.forget as forget_mod

    output = _local(tmp_path)
    _seed(output)
    before = _local_bytes(tmp_path)
    original = forget_mod.os.replace
    set_aside: list[str] = []

    def flaky(src, dst):
        if Path(dst).name.endswith(".bak"):
            set_aside.append(Path(src).name)
            if len(set_aside) == 2:
                raise OSError("rename failed")
        return original(src, dst)

    monkeypatch.setattr(forget_mod.os, "replace", flaky)
    with pytest.raises(OSError, match="rename failed"):
        forget_user(output, "u1")
    assert _local_bytes(tmp_path) == before
    assert list(tmp_path.rglob(".*.tmp")) == []
    assert list(tmp_path.rglob(".*.bak")) == []


def test_local_restore_failure_keeps_the_original_and_names_it(tmp_path, monkeypatch) -> None:
    import cicerone.forget as forget_mod

    output = _local(tmp_path)
    _seed(output)
    original_recs = (tmp_path / "recommendations.parquet").read_bytes()
    original_track = (tmp_path / "track.jsonl").read_bytes()
    original = forget_mod.os.replace
    swapped_in: list[str] = []

    def flaky(src, dst):
        name = Path(src).name
        if name.endswith(".tmp"):
            swapped_in.append(Path(dst).name)
            if len(swapped_in) == 3:
                raise OSError("rename failed")
        if name.startswith(".track.jsonl.") and name.endswith(".bak"):
            raise OSError("restore failed")
        return original(src, dst)

    monkeypatch.setattr(forget_mod.os, "replace", flaky)
    with pytest.raises(OSError, match="rename failed") as raised:
        forget_user(output, "u1")
    (backup,) = tmp_path.glob(".track.jsonl.*.bak")
    assert backup.read_bytes() == original_track
    assert raised.value.__notes__ == [f"originals not restored: {backup}"]
    # The restore after the failed one still ran.
    assert (tmp_path / "recommendations.parquet").read_bytes() == original_recs


def test_local_backup_cleanup_failure_still_reports_the_erase(tmp_path, monkeypatch, caplog) -> None:
    output = _local(tmp_path)
    _seed(output)
    original = Path.unlink

    def stubborn(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".bak"):
            raise PermissionError("busy")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stubborn)
    with caplog.at_level("WARNING", logger="cicerone.forget"):
        result = forget_user(output, "u1")
    assert result.recommendations == 1
    assert _remaining_users(output) == {"u2"}
    assert any("could not remove backup" in record.getMessage() for record in caplog.records)


def test_local_rewrite_keeps_file_modes(tmp_path) -> None:
    import stat

    output = _local(tmp_path)
    _seed(output)
    modes = {"recommendations.parquet": 0o644, "track.jsonl": 0o640}
    for name, mode in modes.items():
        (tmp_path / name).chmod(mode)
    forget_user(output, "u1")
    for name, mode in modes.items():
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == mode
    assert _remaining_users(output) == {"u2"}


def test_local_staging_ignores_a_planted_symlink(tmp_path) -> None:
    output = _local(tmp_path / "out")
    (tmp_path / "out").mkdir()
    _seed(output)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    for name in ("recommendations.parquet", "track.jsonl", ".exposures.jsonl"):
        (tmp_path / "out" / f".{name}.tmp").symlink_to(victim)
    forget_user(output, "u1")
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert _remaining_users(output) == {"u2"}


def test_forget_user_matches_padded_ids(tmp_path) -> None:
    local = _local(tmp_path / "local")
    DatasetOutputSink(local.options).write_recommendations(_recs(" u1 ", "u2"))
    db = IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'out.db'}"})
    DatabaseOutputSink(db.options).write_recommendations(_recs(" u1 ", "u2"))
    assert forget_user(local, "u1").recommendations == 1
    assert forget_user(db, " u1").recommendations == 1
    assert list(pd.read_parquet(tmp_path / "local" / "recommendations.parquet")["user_id"]) == ["u2"]


def test_forget_user_erases_every_store_on_in_memory_sqlite() -> None:
    from cicerone.io.db_store import create_db_engine

    output = IOSettings(kind="db", options={"database_url": "sqlite://"})
    _seed(output)
    result = forget_user(output, "u1")
    assert (result.recommendations, result.track, result.exposures, result.history) == (1, 1, 1, 1)
    engine = create_db_engine("sqlite://", options=output.options)
    for table in (
        "recommendations",
        "recommendation_track",
        "recommendation_exposures",
        "recommendation_history",
    ):
        assert set(pd.read_sql(f'SELECT user_id FROM "{table}"', engine)["user_id"]) == {"u2"}


def test_delete_rows_reraises_database_errors_that_are_not_missing_tables() -> None:
    from contextlib import nullcontext

    from sqlalchemy.exc import OperationalError

    from cicerone.track.store_db import delete_rows_for_user

    class _Conn:
        def begin_nested(self):
            return nullcontext()

        def execute(self, *_args, **_kwargs):
            raise OperationalError("SELECT", {}, Exception("database is locked"))

    with pytest.raises(OperationalError, match="database is locked"):
        delete_rows_for_user(_Conn(), "recommendations", "u1", fence=lambda: None)


def test_cli_forget_user_reports_storage_errors(tmp_path, monkeypatch, capsys) -> None:
    import cicerone.forget as forget_mod

    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(forget_mod, "forget_user", boom)
    config = _cli_config(tmp_path)
    assert main(["--config", str(config), "forget-user", "u1"]) == 1
    assert "forget-user failed: read-only file system" in capsys.readouterr().err


def test_cli_forget_user_prints_the_note_on_a_lost_writer_lease(tmp_path, monkeypatch, capsys) -> None:
    import cicerone.forget as forget_mod
    from cicerone.locks import LockLostError

    def lose_lease(*_args, **_kwargs):
        exc = LockLostError("writer lease lost")
        exc.add_note("output rows were erased; run forget-user again to finish the Redis track log")
        raise exc

    monkeypatch.setattr(forget_mod, "forget_user", lose_lease)
    config = _cli_config(tmp_path)
    assert main(["--config", str(config), "forget-user", "u1"]) == 2
    assert capsys.readouterr().err.splitlines() == [
        "writer lease lost",
        "output rows were erased; run forget-user again to finish the Redis track log",
    ]


def _local_bytes(root) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock") and not path.name.startswith(".")
    }


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
    exposures = pd.read_sql('SELECT user_id FROM "recommendation_exposures"', engine)
    engine.dispose()
    assert set(recommendations["user_id"]) == {"u1", "u2"}
    assert set(track["user_id"]) == {"u1", "u2"}
    assert set(exposures["user_id"]) == {"u1", "u2"}


def test_forget_user_missing_sqlite_tables_are_zero(tmp_path) -> None:
    output = IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'empty.db'}"})
    result = forget_user(output, "u1")
    assert result.recommendations == 0
    assert result.track == 0
    assert result.exposures == 0
    assert result.history == 0


def _cli_config(tmp_path) -> Path:
    output = tmp_path / "out"
    output.mkdir(exist_ok=True)
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
    return config


def test_cli_forget_user_prints_counts(tmp_path, capsys) -> None:
    config = _cli_config(tmp_path)
    output = tmp_path / "out"
    _seed(_local(output))
    assert main(["--config", str(config), "forget-user", "u1"]) == 0
    captured = capsys.readouterr()
    assert "recommendations=1" in captured.out
    assert "track=1" in captured.out
    assert _remaining_users(_local(output)) == {"u2"}

from __future__ import annotations

import json

import pandas as pd
import pytest

from cicerone.config import ConfigError, IOSettings
from cicerone.track.normalize import TrackNormalizeError, normalize_track
from cicerone.track.store import TrackStore, require_appendable_track_log


def test_store_reexports_prior_constants() -> None:
    from cicerone.track.store import (
        DEFAULT_EVAL_TABLE,
        DEFAULT_HISTORY_TABLE,
        DEFAULT_TRACK_TABLE,
        TRACK_COLUMNS,
        TRACK_LOG_BACKEND_ERROR,
    )

    assert DEFAULT_TRACK_TABLE == "recommendation_track"
    assert DEFAULT_EVAL_TABLE == "recommendation_eval"
    assert DEFAULT_HISTORY_TABLE == "recommendation_history"
    assert "kind" in TRACK_COLUMNS
    assert "output kind" in TRACK_LOG_BACKEND_ERROR


def _row(**overrides):
    payload = {
        "kind": "impression",
        "user_id": "alice",
        "item_id": "ipa-001",
        "rank": 1,
        "occurred_at": "2026-08-28T12:00:00Z",
        "event_id": "imp-1",
    }
    payload.update(overrides)
    return normalize_track(payload).as_row()


def test_normalize_track_requires_kind_and_rank():
    with pytest.raises(TrackNormalizeError, match="missing"):
        normalize_track({"user_id": "a", "item_id": "i", "occurred_at": "2026-08-28T12:00:00Z"})
    with pytest.raises(TrackNormalizeError, match="kind"):
        normalize_track(
            {
                "kind": "view",
                "user_id": "a",
                "item_id": "i",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        )
    with pytest.raises(TrackNormalizeError, match="rank"):
        normalize_track(
            {
                "kind": "impression",
                "user_id": "a",
                "item_id": "i",
                "rank": 0,
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        )
    with pytest.raises(TrackNormalizeError, match="JSON object"):
        normalize_track("nope")
    with pytest.raises(TrackNormalizeError, match="rank"):
        normalize_track(
            {
                "kind": "impression",
                "user_id": "a",
                "item_id": "i",
                "rank": "x",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        )
    with pytest.raises(TrackNormalizeError, match="rank"):
        normalize_track(
            {
                "kind": "impression",
                "user_id": "a",
                "item_id": "i",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        )
    with pytest.raises(TrackNormalizeError, match="non-empty"):
        normalize_track(
            {
                "kind": "impression",
                "user_id": "  ",
                "item_id": "i",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        )


def test_stable_event_id_does_not_collide_on_pipe_in_ids() -> None:
    base = {
        "kind": "impression",
        "rank": 1,
        "occurred_at": "2026-08-28T12:00:00Z",
    }
    left = normalize_track({**base, "user_id": "a|b", "item_id": "c"})
    right = normalize_track({**base, "user_id": "a", "item_id": "b|c"})
    assert left.event_id != right.event_id
    assert normalize_track({**base, "user_id": "a|b", "item_id": "c"}).event_id == left.event_id


def test_track_store_roundtrip_dataset(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = TrackStore(output)
    assert store.read_rows() == []
    assert store.append_rows([]) == 0
    first = store.append_rows([_row(), _row(kind="click", event_id="clk-1", rank=None)])
    assert first == 2
    again = store.append_rows([_row(), _row(kind="click", event_id="clk-1")])
    assert again == 0
    rows = store.read_rows()
    assert {row["event_id"] for row in rows} == {"imp-1", "clk-1"}
    assert [row["event_id"] for row in store.read_rows(kind="click")] == ["clk-1"]
    store.write_eval({"track_eval": {"overall": {"n_impressions": 1}}})
    report = store.read_eval()
    assert report is not None
    assert report["track_eval"]["overall"]["n_impressions"] == 1
    recs = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa-001",
                "rank": 1,
                "source": "personalized",
                "variant": "control",
            }
        ]
    )
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    history = store.read_history()
    assert len(history) == 1
    assert str(history.iloc[0]["user_id"]) == "alice"
    store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
    parts = list((tmp_path / "recommendation_history").glob("*.parquet"))
    assert len(parts) == 2
    assert all(len(pd.read_parquet(part)) == 1 for part in parts)
    assert len(store.read_history()) == 2


def test_track_store_roundtrip_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    assert store.read_rows() == []
    assert store.read_eval() is None
    assert store.append_rows([_row(), _row()]) == 1
    assert store.append_rows([_row()]) == 0
    rows = store.read_rows()
    assert len(rows) == 1
    assert rows[0]["user_id"] == "alice"
    store.write_eval({"ok": True})
    assert store.read_eval() == {"ok": True}
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    assert len(store.read_history()) == 1
    engine = store._db_engine()
    store.append_rows([_row(event_id="imp-2", item_id="ipa-002")])
    assert store._db_engine() is engine
    assert len(store.read_rows()) == 2


def test_track_store_sqlite_unknown_rowcount_does_not_over_accept(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    engine = store._db_engine()
    real_begin = engine.begin

    class _HideRowcount:
        def __init__(self, inner: object) -> None:
            object.__setattr__(self, "_inner", inner)
            self.rowcount = None

        def __iter__(self):
            return iter(self._inner)  # type: ignore[arg-type]

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    class _Conn:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return _HideRowcount(self._inner.execute(*args, **kwargs))  # type: ignore[union-attr]

        def __enter__(self):
            return _Conn(self._inner.__enter__())  # type: ignore[union-attr]

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)  # type: ignore[union-attr]

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    engine.begin = lambda: _Conn(real_begin())  # type: ignore[method-assign]
    assert store.append_rows([_row(event_id="imp-fallback")]) == 1
    assert store.append_rows([_row(event_id="imp-fallback")]) == 0
    mixed = store.append_rows([_row(event_id="imp-fallback"), _row(event_id="imp-2", item_id="ipa-002")])
    assert mixed == 1


def test_track_store_ignores_invalid_eval_json(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    (tmp_path / "track_eval.json").write_text("not-json", encoding="utf-8")
    assert TrackStore(output).read_eval() is None


def test_append_track_rejects_object_store() -> None:
    output = IOSettings(
        kind="dataset",
        options={
            "storage_backend": "s3",
            "bucket": "recs",
            "access_key_id": "id",
            "secret_access_key": "secret",
        },
    )
    with pytest.raises(ConfigError, match="not atomic"):
        TrackStore(output).append_rows([_row()])


def test_require_appendable_track_log_allows_db_and_local(tmp_path) -> None:
    require_appendable_track_log(IOSettings(kind="db", options={"database_url": "sqlite://"}))
    require_appendable_track_log(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )


def test_jsonish_numpy_scalar_and_na() -> None:
    import numpy as np

    from cicerone.track.store import _jsonish

    assert _jsonish(np.int64(3)) == 3
    assert _jsonish(pd.NA) is None


def test_track_jsonl_skips_bad_lines(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    path = tmp_path / "track.jsonl"
    path.write_text("not-json\n" + json.dumps(_row()) + "\n\n", encoding="utf-8")
    rows = TrackStore(output).read_rows()
    assert len(rows) == 1
    assert rows[0]["event_id"] == "imp-1"


def test_track_eval_roundtrip_s3() -> None:
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
        store = TrackStore(output)
        assert store.read_eval() is None
        assert store.read_rows() == []
        store.write_eval({"ok": True})
        assert store.read_eval() == {"ok": True}
        history = store.read_history()
        assert history.empty
        store.append_history(pd.DataFrame(), generated_at="t")
        assert store.read_history().empty
        recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
        store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
        store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
        assert len(store.read_history()) == 2


def test_track_store_sqlite_missing_tables(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'empty.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    assert store.read_rows() == []
    assert store.read_eval() is None
    assert store.read_history().empty


def test_track_history_reads_legacy_single_file(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized", "variant": None}]
    )
    recs["generated_at"] = "2026-08-20T00:00:00+00:00"
    recs.to_parquet(tmp_path / "recommendation_history.parquet", index=False)
    store = TrackStore(output)
    store.append_history(
        pd.DataFrame([{"user_id": "bob", "item_id": "stout", "rank": 1, "source": "personalized"}]),
        generated_at="2026-08-28T03:00:00+00:00",
    )
    users = set(store.read_history()["user_id"].astype(str))
    assert users == {"alice", "bob"}


def test_history_part_name_sanitizes_timestamp() -> None:
    from cicerone.track.store import _history_part_name, _history_stem_before, _unslug_history_stem

    assert _history_part_name("2026-08-28T03:00:00+00:00") == "2026-08-28T03-00-00+00-00.parquet"
    assert _history_part_name("   ") == "snapshot.parquet"
    assert _unslug_history_stem("2026-08-28T03-00-00+00-00") == "2026-08-28T03:00:00+00:00"
    assert _unslug_history_stem("2026-08-28T03-00-00Z") == "2026-08-28T03:00:00Z"
    assert _unslug_history_stem("2026-08-28T03-00-00-05-00") == "2026-08-28T03:00:00-05:00"
    assert _unslug_history_stem("2026-08-28T03-00-00.123+00-00") == "2026-08-28T03:00:00.123+00:00"
    assert _history_stem_before("2026-08-28T03-00-00+00-00", "2026-08-29T00:00:00+00:00")
    assert not _history_stem_before("2026-08-29T03-00-00+00-00", "2026-08-29T00:00:00+00:00")
    from cicerone.track.store import _history_frame

    frame = _history_frame(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]),
        generated_at="t",
    )
    assert list(frame.columns) == ["user_id", "item_id", "rank", "source", "variant", "generated_at"]


def test_jsonish_item_raises() -> None:
    from cicerone.track.store import _jsonish

    class _Boom:
        def item(self):
            raise ValueError("nope")

    boom = _Boom()
    assert _jsonish(boom) is boom


def test_track_eval_ignores_non_object_json(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    (tmp_path / "track_eval.json").write_text("[1]", encoding="utf-8")
    assert TrackStore(output).read_eval() is None


def test_track_store_sqlite_read_errors(tmp_path, monkeypatch) -> None:
    from sqlalchemy import text

    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    store.append_rows([_row()])
    store.write_eval({"ok": True})
    engine = store._db_engine()
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE "recommendation_eval" SET payload = :payload'),
            {"payload": "not-json"},
        )
    assert store.read_eval() is None
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE "recommendation_eval" SET payload = :payload'),
            {"payload": "[1]"},
        )
    assert store.read_eval() is None
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM "recommendation_track"'))
        conn.execute(text('DELETE FROM "recommendation_eval"'))
    assert store.read_rows() == []
    assert store.read_eval() is None

    def _boom(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(pd, "read_sql", _boom)
    assert store.read_rows() == []
    assert store.read_eval() is None
    assert store.read_history().empty


def test_track_jsonl_same_batch_duplicate_event_id(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = TrackStore(output)
    assert store.append_rows([_row(), _row()]) == 1
    assert len(store.read_rows()) == 1


def test_track_jsonl_append_skips_reread_when_warm(tmp_path, monkeypatch) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = TrackStore(output)
    assert store.append_rows([_row()]) == 1
    reads = {"n": 0}
    original = store._read_bytes

    def _count(filename: str) -> bytes | None:
        reads["n"] += 1
        return original(filename)

    monkeypatch.setattr(store, "_read_bytes", _count)
    assert store.append_rows([_row(event_id="imp-2", item_id="ipa-002")]) == 1
    assert reads["n"] == 0
    assert {row["event_id"] for row in store.read_rows()} == {"imp-1", "imp-2"}


def test_track_jsonl_append_without_fcntl(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("cicerone.track.store_dataset.fcntl", None)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = TrackStore(output)
    assert store.append_rows([_row()]) == 1
    assert store.append_rows([_row()]) == 0
    assert store.append_rows([_row(event_id="imp-2", item_id="ipa-002")]) == 1


def test_track_jsonl_assigns_event_id_when_missing(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    row = {
        "kind": "impression",
        "user_id": "alice",
        "item_id": "ipa-001",
        "rank": 1,
        "occurred_at": "2026-08-28T12:00:00Z",
        "event_id": "",
    }
    store = TrackStore(output)
    assert store.append_rows([row, dict(row)]) == 1
    ids = [str(item["event_id"]) for item in store.read_rows()]
    assert len(ids) == 1
    assert all(ids)


def test_track_jsonl_second_store_respects_existing_event_ids(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert TrackStore(output).append_rows([_row()]) == 1
    assert TrackStore(output).append_rows([_row()]) == 0


def test_track_jsonl_dedupes_duplicate_event_ids(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    path = tmp_path / "track.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")
    rows = TrackStore(output).read_rows()
    assert len(rows) == 1
    assert rows[0]["event_id"] == "imp-1"


def test_track_store_sqlite_missing_table_error_helper(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    store.append_rows([_row()])
    monkeypatch.setattr(pd, "read_sql", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr("cicerone.track.store_db.is_missing_table_error", lambda _exc: True)
    assert store.read_rows() == []
    assert store.read_eval() is None
    assert store.read_history().empty


def test_track_history_non_s3_error_reraises(tmp_path, monkeypatch) -> None:
    output = IOSettings(
        kind="dataset",
        options={
            "storage_backend": "s3",
            "bucket": "recs",
            "access_key_id": "test",
            "secret_access_key": "test",
        },
    )
    monkeypatch.setattr(
        "cicerone.io.options.read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("corrupt")),
    )
    monkeypatch.setattr("cicerone.track.store_dataset.is_s3_not_found", lambda _exc: False)
    with pytest.raises(RuntimeError, match="corrupt"):
        TrackStore(output).read_history()


def test_track_read_bytes_s3_generic_error(monkeypatch) -> None:
    import boto3
    from moto import mock_aws

    class _Boom:
        def get_object(self, **_kwargs):
            raise RuntimeError("network")

    monkeypatch.setattr("cicerone.track.store_dataset.build_s3_client", lambda _options: _Boom())
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="recs")
        output = IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
            },
        )
        with pytest.raises(RuntimeError, match="network"):
            TrackStore(output).read_eval()


def test_track_read_rows_filters_experiment_and_since(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = TrackStore(output)
    store.append_rows(
        [
            _row(event_id="a", experiment_id="exp-a", occurred_at="2026-08-28T10:00:00Z"),
            _row(event_id="b", experiment_id="exp-b", occurred_at="2026-08-28T12:00:00Z"),
            _row(event_id="untagged", experiment_id="", occurred_at="2026-08-28T13:00:00Z"),
        ]
    )
    matched = store.read_rows(experiment_id="exp-a")
    assert {row["event_id"] for row in matched} == {"a", "untagged"}
    recent = store.read_rows(since="2026-08-28T12:00:00Z")
    assert {row["event_id"] for row in recent} == {"b", "untagged"}


def test_track_read_rows_filters_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = TrackStore(output)
    store.append_rows(
        [
            _row(event_id="a", experiment_id="exp-a", occurred_at="2026-08-28T10:00:00Z"),
            _row(event_id="b", experiment_id="exp-b", occurred_at="2026-08-28T12:00:00Z"),
            _row(event_id="untagged", experiment_id="", occurred_at="2026-08-28T13:00:00Z"),
        ]
    )
    matched = store.read_rows(experiment_id="exp-a")
    assert {row["event_id"] for row in matched} == {"a", "untagged"}
    recent = store.read_rows(since="2026-08-28T12:00:00Z")
    assert {row["event_id"] for row in recent} == {"b", "untagged"}


def test_track_read_history_generated_ats_skips_other_parts(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
    store = TrackStore(output)
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
    history = store.read_history(generated_ats=["2026-08-29T03:00:00+00:00"])
    assert len(history) == 1
    assert str(history.iloc[0]["generated_at"]) == "2026-08-29T03:00:00+00:00"
    assert store.read_history(generated_ats=[]).empty
    assert len(store.read_history(since="2026-08-29T00:00:00+00:00")) == 1


def test_track_read_history_since_skips_older_part_files(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
    store = TrackStore(output)
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
    old = tmp_path / "recommendation_history" / "2026-08-28T03-00-00+00-00.parquet"
    old.write_bytes(b"not parquet")
    history = store.read_history(since="2026-08-29T00:00:00+00:00")
    assert len(history) == 1
    assert str(history.iloc[0]["generated_at"]) == "2026-08-29T03:00:00+00:00"


def test_track_read_history_s3_since_skips_older_parts() -> None:
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="recs")
        output = IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
            },
        )
        recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
        store = TrackStore(output)
        store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
        store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
        client.put_object(
            Bucket="recs",
            Key="recommendation_history/2026-08-28T03-00-00+00-00.parquet",
            Body=b"not parquet",
        )
        history = store.read_history(since="2026-08-29T00:00:00+00:00")
        assert len(history) == 1
        assert str(history.iloc[0]["generated_at"]) == "2026-08-29T03:00:00+00:00"


def test_track_read_history_sqlite_generated_ats(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'track.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "source": "personalized"}])
    store = TrackStore(output)
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    store.append_history(recs, generated_at="2026-08-29T03:00:00+00:00")
    history = store.read_history(generated_ats=["2026-08-29T03:00:00+00:00"])
    assert len(history) == 1
    assert len(store.read_history(since="2026-08-29T00:00:00+00:00")) == 1

from __future__ import annotations

import json
import threading

import boto3
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from pyarrow.lib import ArrowTypeError
from support.fakes import patch_parquet_reads

from cicerone.config import ConfigError
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.replace_users import RecommendationSchemaError
from cicerone.job_eval import retrain_fence
from cicerone.locks import WriteFence

# --- local backend -----------------------------------------------------------


def test_local_backend_round_trip(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)

    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1", "category": "beer"}]))
    sink.write_manifest({"n_events": 3})
    sink.write_model_artifact(b"fake-artifact-bytes")

    assert not list(tmp_path.glob(".*.tmp"))
    assert sink.read_model_artifact() == b"fake-artifact-bytes"
    source = DatasetInputSource(options)
    (tmp_path / "events.parquet").write_bytes((tmp_path / "recommendations.parquet").read_bytes())
    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]

    manifest_path = tmp_path / "manifest.json"
    assert json.loads(manifest_path.read_text()) == {"n_events": 3}
    assert (tmp_path / "model.artifact").read_bytes() == b"fake-artifact-bytes"
    items_snap = pd.read_parquet(tmp_path / "items_snapshot.parquet")
    assert list(items_snap["item_id"]) == ["i1"]


def test_local_read_model_artifact_missing_returns_none(tmp_path):
    sink = DatasetOutputSink({"storage_backend": "local", "path": str(tmp_path)})
    assert sink.read_model_artifact() is None
    assert sink.model_artifact_fingerprint() is None


def test_local_model_artifact_fingerprint_changes_after_write(tmp_path):
    sink = DatasetOutputSink({"storage_backend": "local", "path": str(tmp_path)})
    sink.write_model_artifact(b"first")
    first = sink.model_artifact_fingerprint()
    assert first is not None
    sink.write_model_artifact(b"second")
    second = sink.model_artifact_fingerprint()
    assert second is not None
    assert second != first


def test_local_replace_model_artifact_if_is_compare_and_swap(tmp_path):
    sink = DatasetOutputSink({"storage_backend": "local", "path": str(tmp_path)})
    sink.write_model_artifact(b"first")
    token = sink.model_artifact_fingerprint()
    assert token is not None
    assert sink.replace_model_artifact_if(b"second", token) is True
    assert sink.read_model_artifact() == b"second"
    assert sink.replace_model_artifact_if(b"third", token) is False
    assert sink.read_model_artifact() == b"second"


def test_local_replace_recommendations_for_users_preserves_others(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
                {"user_id": "u2", "item_id": "keep", "rank": 1, "score": 0.5, "source": "personalized"},
            ]
        )
    )
    assert (
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "new", "rank": 1, "score": 2.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )
        == 2
    )
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert set(stored[stored["user_id"] == "u1"]["item_id"]) == {"new"}
    assert list(stored[stored["user_id"] == "u2"]["item_id"]) == ["keep"]
    assert sink.replace_recommendations_for_users(pd.DataFrame(), user_ids=["u1"]) == 1
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert "u1" not in set(stored["user_id"].astype(str))
    assert list(stored["user_id"]) == ["u2"]
    assert (
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u3", "item_id": "i3", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u3"],
        )
        == 2
    )
    before = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert set(before["user_id"].astype(str)) == {"u2", "u3"}
    assert sink.replace_recommendations_for_users(pd.DataFrame(), user_ids=[]) == 0
    after = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert len(after) == len(before)
    with pytest.raises(ValueError, match="requires user_ids"):
        sink.replace_recommendations_for_users(before, user_ids=[])


def test_local_replace_recommendations_serializes_writers(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "old1", "rank": 1, "score": 1.0, "source": "personalized"},
                {"user_id": "u2", "item_id": "old2", "rank": 1, "score": 1.0, "source": "personalized"},
            ]
        )
    )
    started = threading.Barrier(2)

    def _replace(user_id: str, item_id: str) -> None:
        started.wait()
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": user_id, "item_id": item_id, "rank": 1, "score": 2.0, "source": "incremental"}]
            ),
            user_ids=[user_id],
        )

    threads = [
        threading.Thread(target=_replace, args=("u1", "new1")),
        threading.Thread(target=_replace, args=("u2", "new2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    by_user = {
        str(user_id): list(group["item_id"])
        for user_id, group in stored.groupby(stored["user_id"].astype(str), sort=False)
    }
    assert by_user["u1"] == ["new1"]
    assert by_user["u2"] == ["new2"]


def test_local_write_and_replace_recommendations_share_lock(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "old1", "rank": 1, "score": 1.0, "source": "personalized"},
                {"user_id": "u2", "item_id": "old2", "rank": 1, "score": 1.0, "source": "personalized"},
            ]
        )
    )
    started = threading.Barrier(2)

    def _write_job() -> None:
        started.wait()
        sink.write_recommendations(
            pd.DataFrame(
                [
                    {"user_id": "u1", "item_id": "job1", "rank": 1, "score": 1.0, "source": "personalized"},
                    {"user_id": "u2", "item_id": "job2", "rank": 1, "score": 1.0, "source": "personalized"},
                    {"user_id": "u3", "item_id": "job3", "rank": 1, "score": 1.0, "source": "personalized"},
                ]
            )
        )

    def _replace_u1() -> None:
        started.wait()
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "new1", "rank": 1, "score": 2.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )

    threads = [threading.Thread(target=_write_job), threading.Thread(target=_replace_u1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    by_user = {
        str(user_id): list(group["item_id"])
        for user_id, group in stored.groupby(stored["user_id"].astype(str), sort=False)
    }
    assert set(by_user) == {"u1", "u2", "u3"}
    assert by_user["u2"] == ["job2"]
    assert by_user["u3"] == ["job3"]
    assert by_user["u1"] in (["job1"], ["new1"])


def test_write_recommendations_writer_lock_busy(tmp_path, monkeypatch) -> None:
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
    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Busy()),
    )
    with pytest.raises(RuntimeError, match="dataset writer lock busy"):
        sink.write_recommendations(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
            )
        )
    assert not (tmp_path / "recommendations.parquet").exists()


def test_recommendations_write_reentry_is_per_thread(tmp_path) -> None:
    acquires: list[int] = []

    class _Lock:
        def acquire(self) -> bool:
            acquires.append(threading.get_ident())
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    started = threading.Event()
    release = threading.Event()
    frame = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    )

    def holder() -> None:
        with sink.recommendations_write():
            started.set()
            release.wait(timeout=2)

    def writer() -> None:
        sink.write_recommendations(frame)

    thread = threading.Thread(target=holder)
    thread.start()
    assert started.wait(timeout=2)
    second = threading.Thread(target=writer)
    second.start()
    release.set()
    thread.join(timeout=2)
    second.join(timeout=2)
    assert len(acquires) == 2
    assert acquires[0] != acquires[1]


def test_recommendations_write_skips_acquire_when_lock_held_here(tmp_path) -> None:
    from cicerone.locks import WriteFence

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

    lock = _Lock()
    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(lock),
    )
    frame = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    with WriteFence(lock).hold():
        sink.write_recommendations(frame)
        sink.write_model_artifact(b"artifact")
        sink.write_items_snapshot(frame)
    assert acquires["n"] == 1


def test_nested_replace_rechecks_owned_before_unlocked_write(tmp_path) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return held["v"]

        def is_locked(self) -> bool:
            return True

    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    with sink.recommendations_write():
        held["v"] = False
        with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
            sink.replace_recommendations_for_users(
                pd.DataFrame(
                    [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
                ),
                user_ids=["u1"],
            )
    assert not (tmp_path / "recommendations.parquet").exists()


def test_write_recommendations_rechecks_owned_after_serialize(tmp_path, monkeypatch) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return held["v"]

        def is_locked(self) -> bool:
            return True

    original = pd.DataFrame.to_parquet

    def _to_parquet(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        held["v"] = False
        return result

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _to_parquet)
    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        sink.write_recommendations(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
            )
        )
    assert not (tmp_path / "recommendations.parquet").exists()


def test_write_items_snapshot_rechecks_owned_after_serialize(tmp_path, monkeypatch) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return held["v"]

        def is_locked(self) -> bool:
            return True

    original = pd.DataFrame.to_parquet

    def _to_parquet(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        held["v"] = False
        return result

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _to_parquet)
    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
        sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1", "category": "beer"}]))
    assert not (tmp_path / "items_snapshot.parquet").exists()


def test_write_model_artifact_rechecks_owned_when_nested(tmp_path) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return held["v"]

        def is_locked(self) -> bool:
            return True

    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    with sink.recommendations_write():
        held["v"] = False
        with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
            sink.write_model_artifact(b"stale")
    assert not (tmp_path / "model.artifact").exists()


def test_write_manifest_skips_newer_under_lock(tmp_path) -> None:
    sink = DatasetOutputSink({"storage_backend": "local", "path": str(tmp_path)})
    newer = {"generated_at": "2099-01-01T00:00:00+00:00", "status": "success"}
    (tmp_path / "manifest.json").write_text(json.dumps(newer))
    assert sink.write_manifest({"status": "failed"}, skip_if_newer_than="2026-01-01T00:00:00+00:00") is False
    assert json.loads((tmp_path / "manifest.json").read_text()) == newer


def test_write_manifest_rechecks_owned_when_nested(tmp_path) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return held["v"]

        def is_locked(self) -> bool:
            return True

    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=WriteFence(_Lock()),
    )
    with sink.recommendations_write():
        held["v"] = False
        with pytest.raises(LockLostError, match="dataset writer lock lost before write"):
            sink.write_manifest({"status": "success"})
    assert not (tmp_path / "manifest.json").exists()


def test_write_recommendations_rechecks_fence_after_writer_lock(tmp_path) -> None:
    from cicerone.locks import LockLostError

    held = {"v": True}

    class _Lock:
        def acquire(self) -> bool:
            held["v"] = False
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    sink = DatasetOutputSink(
        {"storage_backend": "local", "path": str(tmp_path)},
        fence=retrain_fence(_Lock(), lambda: held["v"]),
    )
    with pytest.raises(LockLostError, match="retrain lock lost before write") as captured:
        sink.write_recommendations(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
            )
        )
    assert captured.value.kind == "retrain"
    assert not (tmp_path / "recommendations.parquet").exists()


def test_local_replace_recommendations_when_file_missing(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.replace_recommendations_for_users(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]),
        user_ids=["u1"],
    )
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert list(stored["user_id"]) == ["u1"]


def test_local_replace_recommendations_refuses_schema_mismatch(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    pd.DataFrame([{"item_id": "orphan", "rank": 1}]).to_parquet(
        tmp_path / "recommendations.parquet", index=False
    )
    with pytest.raises(RecommendationSchemaError, match="missing user_id"):
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )
    stored = pd.read_parquet(tmp_path / "recommendations.parquet")
    assert list(stored["item_id"]) == ["orphan"]


def test_local_replace_recommendations_rejects_extra_users(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    with pytest.raises(ValueError, match="outside user_ids"):
        sink.replace_recommendations_for_users(
            pd.DataFrame(
                [{"user_id": "u9", "item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]
            ),
            user_ids=["u1"],
        )
    with pytest.raises(ValueError, match="missing user_id"):
        sink.replace_recommendations_for_users(
            pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 1.0, "source": "incremental"}]),
            user_ids=["u1"],
        )


def test_local_backend_optional_inputs_missing_return_none(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    source = DatasetInputSource(options)

    assert source.read_users() is None
    assert source.read_items() is None


def test_local_backend_get_events_for_user_filters_and_limits(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "event_type": "view", "occurred_at": "2026-08-01T00:00:00Z"},
            {
                "user_id": "u1",
                "item_id": "new",
                "event_type": "purchase",
                "occurred_at": "2026-08-21T00:00:00Z",
            },
            {
                "user_id": "u2",
                "item_id": "other",
                "event_type": "view",
                "occurred_at": "2026-08-21T00:00:00Z",
            },
        ]
    )
    events.to_parquet(tmp_path / "events.parquet", index=False)
    source = DatasetInputSource(options)

    rows = source.get_events_for_user("u1", limit=1)

    assert list(rows["item_id"]) == ["new"]
    assert list(source.get_events_for_user("ghost", limit=5)["item_id"]) == []


def test_local_backend_user_lookups_match_a_numeric_user_id_column(tmp_path):
    pd.DataFrame(
        [{"user_id": 1, "item_id": "i1", "event_type": "view", "occurred_at": "2026-08-21T00:00:00Z"}]
    ).to_parquet(tmp_path / "events.parquet", index=False)
    pd.DataFrame([{"user_id": 1, "region_slug": "lazio"}]).to_parquet(tmp_path / "users.parquet", index=False)
    source = DatasetInputSource({"storage_backend": "local", "path": str(tmp_path)})
    assert list(source.get_events_for_user("1", limit=5)["item_id"]) == ["i1"]
    assert source.get_user("1")["region_slug"] == "lazio"
    assert source.get_user("2") is None


def test_local_backend_get_user_returns_row_or_none(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    source = DatasetInputSource(options)
    assert source.get_user("u1") is None

    pd.DataFrame(
        [{"user_id": "u1", "region_slug": "lazio"}, {"user_id": "u2", "region_slug": "toscana"}]
    ).to_parquet(tmp_path / "users.parquet", index=False)
    assert source.get_user("u1")["region_slug"] == "lazio"
    assert source.get_user("ghost") is None


@pytest.mark.parametrize(
    "exc", [TypeError("bad user_id filter arg"), ArrowTypeError("user_id: type mismatch")]
)
def test_local_backend_get_events_for_user_only_falls_back_on_arrow_errors(tmp_path, monkeypatch, exc):
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "view", "occurred_at": "2026-08-21T00:00:00Z"}]
    ).to_parquet(tmp_path / "events.parquet", index=False)
    patch_parquet_reads(monkeypatch, pushdown_error=exc)
    source = DatasetInputSource({"storage_backend": "local", "path": str(tmp_path)})
    if isinstance(exc, ArrowTypeError):
        assert list(source.get_events_for_user("u1", limit=5)["item_id"]) == ["i1"]
    else:
        with pytest.raises(TypeError, match="bad user_id filter arg"):
            source.get_events_for_user("u1", limit=5)


def test_local_backend_get_events_for_user_missing_file_raises(tmp_path):
    source = DatasetInputSource({"storage_backend": "local", "path": str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        source.get_events_for_user("u1", 5)


def test_local_backend_optional_inputs_propagate_corrupt_file_errors(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    (tmp_path / "users.parquet").write_bytes(b"not-a-parquet-file")
    source = DatasetInputSource(options)
    with pytest.raises(ValueError):
        source.read_users()


def test_unknown_storage_backend_raises():
    with pytest.raises(ConfigError, match="Unknown storage_backend"):
        DatasetInputSource({"storage_backend": "ftp"})
    with pytest.raises(ConfigError, match="Unknown storage_backend"):
        DatasetOutputSink({"storage_backend": "ftp"})


def test_local_backend_missing_path_raises():
    with pytest.raises(ConfigError, match="path"):
        DatasetInputSource({"storage_backend": "local"})
    with pytest.raises(ConfigError, match="path"):
        DatasetOutputSink({"storage_backend": "local"})


@pytest.mark.parametrize("missing_key", ["access_key_id", "secret_access_key", "bucket"])
def test_s3_backend_missing_required_option_raises(missing_key):
    options = {
        "storage_backend": "s3",
        "access_key_id": "test",
        "secret_access_key": "test",
        "bucket": "test-bucket",
    }
    del options[missing_key]

    with pytest.raises(ConfigError, match=missing_key):
        DatasetInputSource(options)
    with pytest.raises(ConfigError, match=missing_key):
        DatasetOutputSink(options)


# --- s3 backend (mocked) ------------------------------------------------------


def test_s3_backend_round_trip(s3_options):
    sink = DatasetOutputSink(s3_options)
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.5, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_manifest({"n_events": 1})

    # Re-read sink output as if it were events.parquet.
    source = DatasetInputSource(s3_options)
    client = boto3.client("s3", region_name="us-east-1")
    bucket = s3_options["bucket"]
    body = client.get_object(Bucket=bucket, Key=f"{s3_options['prefix']}/recommendations.parquet")[
        "Body"
    ].read()
    client.put_object(Bucket=bucket, Key=f"{s3_options['prefix']}/events.parquet", Body=body)

    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]


def test_s3_applied_event_keys_read_from_the_parquet_footer(s3_options, monkeypatch):
    sink = DatasetOutputSink(s3_options)
    assert sink.applied_event_keys(["k1"]) == set()
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.5, "source": "personalized"}])
    sink.replace_recommendations_for_users(df, user_ids=["u1"], applied_keys=["k1"])

    def _rows_read(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("applied_event_keys read the recommendation rows")

    monkeypatch.setattr("cicerone.io.dataset_store.read_parquet", _rows_read)
    monkeypatch.setattr("cicerone.io.blob.read_parquet", _rows_read)
    assert sink.applied_event_keys(["k1", "k2"]) == {"k1"}


def test_s3_read_model_artifact_round_trip_and_missing(s3_options):
    sink = DatasetOutputSink(s3_options)
    assert sink.read_model_artifact() is None
    assert sink.model_artifact_fingerprint() is None
    sink.write_model_artifact(b"s3-artifact")
    assert sink.read_model_artifact() == b"s3-artifact"
    assert sink.model_artifact_fingerprint() is not None


def test_s3_replace_model_artifact_if_does_not_write(s3_options):
    sink = DatasetOutputSink(s3_options)
    sink.write_model_artifact(b"first")
    token = sink.model_artifact_fingerprint()
    assert token is not None
    assert sink.replace_model_artifact_if(b"second", token) is False
    assert sink.read_model_artifact() == b"first"


def test_s3_read_model_artifact_closes_streaming_body(s3_options, mocker):
    sink = DatasetOutputSink(s3_options)
    sink.write_model_artifact(b"payload")
    body = mocker.Mock()
    body.read.return_value = b"payload"
    client = mocker.Mock()
    client.get_object.return_value = {"Body": body}
    mocker.patch("cicerone.io.blob.build_s3_client", return_value=client)
    assert sink.read_model_artifact() == b"payload"
    body.close.assert_called_once()


def test_s3_backend_optional_inputs_missing_return_none(s3_options):
    source = DatasetInputSource(s3_options)
    assert source.read_users() is None
    assert source.read_items() is None


def test_s3_backend_optional_users_not_found_logs_warning_and_returns_none(s3_options, mocker, caplog):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Object not found"}},
        "GetObject",
    )
    mocker.patch("cicerone.io.blob.read_parquet", side_effect=client_error)

    with caplog.at_level("WARNING", logger="cicerone.io.dataset_store"):
        result = source.read_users()

    assert result is None
    assert any("users" in record.getMessage().lower() for record in caplog.records)


def test_s3_backend_optional_users_client_error_is_propagated_for_non_not_found_codes(s3_options, mocker):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
        "GetObject",
    )
    mocker.patch("cicerone.io.blob.read_parquet", side_effect=client_error)

    with pytest.raises(ClientError):
        source.read_users()


def test_s3_backend_optional_items_not_found_logs_warning_and_returns_none(s3_options, mocker, caplog):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "GetObject",
    )
    mocker.patch("cicerone.io.blob.read_parquet", side_effect=client_error)

    with caplog.at_level("WARNING", logger="cicerone.io.dataset_store"):
        result = source.read_items()

    assert result is None
    assert any("items" in record.getMessage().lower() for record in caplog.records)


def test_s3_backend_no_prefix_writes_flat_key(s3_options):
    flat_options = {
        "storage_backend": "s3",
        "access_key_id": "test",
        "secret_access_key": "test",
        "bucket": s3_options["bucket"],
        "prefix": "",
    }
    sink = DatasetOutputSink(flat_options)
    sink.write_manifest({"ok": True})

    client = boto3.client("s3", region_name="us-east-1")
    body = client.get_object(Bucket=flat_options["bucket"], Key="manifest.json")["Body"].read()
    assert json.loads(body) == {"ok": True}


def test_s3_applied_event_keys_fallback_reads_only_the_footer(s3_options, monkeypatch):
    from cicerone.io import options as io_options

    sink = DatasetOutputSink(s3_options)
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(5000)], "item_id": "i1", "rank": 1, "score": 0.5})
    df["source"] = "personalized"
    sink.replace_recommendations_for_users(df, user_ids=list(df["user_id"]), applied_keys=["k1"])

    def _pyarrow_fails(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("pyarrow S3 unavailable")

    ranges: list[str] = []
    real_client = io_options.build_s3_client

    def _client(options):  # type: ignore[no-untyped-def]
        client = real_client(options)
        get_object = client.get_object

        def _get_object(**kwargs):  # type: ignore[no-untyped-def]
            ranges.append(kwargs.get("Range", "full"))
            return get_object(**kwargs)

        client.get_object = _get_object
        return client

    monkeypatch.setattr(io_options, "_s3_filesystem", _pyarrow_fails)
    monkeypatch.setattr(io_options, "build_s3_client", _client)
    assert sink.applied_event_keys(["k1", "k2"]) == {"k1"}
    assert ranges and "full" not in ranges

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from cicerone import job
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings
from cicerone.config.constants import DEFAULT_SERVE_MAX_K
from cicerone.job import _recommendation_user_count, _target_user_ids
from cicerone.model import RRF_K
from cicerone.publish import PublishError
from cicerone.track.store import TrackStore

REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _write_config(
    tmp_path,
    input_dir,
    output_dir,
    top_k: int = 10,
    extra_job: str = "",
    extra: str = "",
    extra_output: str = "",
) -> str:
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = {top_k}
        feature_config_path = "{REPO_FEATURES_CONFIG}"
        {extra_job}

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{input_dir}"

        [output]
        kind = "dataset"
        {extra_output}
        [output.options]
        storage_backend = "local"
        path = "{output_dir}"

        {extra}
        """
    )
    return str(config_path)


def test_job_run_end_to_end_with_local_dataset_backend(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    job.run()

    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert set(recommendations["user_id"]) == {"u1", "u2", COLD_START_USER_ID}

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["n_events"] == 4
    assert manifest["n_target_users"] == 2
    assert manifest["top_k"] == 2
    assert manifest["automl_enabled"] is False
    assert manifest["automl_metrics"] == ""
    assert manifest["triggered_by"] == "manual"
    assert manifest["lock_backend"] == "in_process"
    assert manifest["artifact_written"] is False
    assert manifest["artifact_schema_version"] is None
    assert not (output_dir / "model.artifact").exists()
    scores = pd.read_parquet(output_dir / "item_scores.parquet")
    assert set(scores["item_id"]) == {"i1", "i2", "i3"}
    assert manifest["n_item_scores"] == 3
    popular = pd.read_parquet(output_dir / "popular.parquet")
    assert not popular.empty
    assert (output_dir / "latest.parquet").is_file()
    assert (output_dir / "item_neighbors.parquet").is_file()
    assert (output_dir / "surfaces_stamp.json").is_file()


def test_job_materializes_neighbors_for_session_overfetch(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    ).to_parquet(input_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    ).to_parquet(input_dir / "items.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=1))
    seen: list[int] = []
    popular_k: list[int] = []
    latest_k: list[int] = []
    real = job.neighbors_from_events
    real_popular = job.popular_from_events
    real_latest = job.latest_from_items

    def _capture(events, k_neighbors):
        seen.append(k_neighbors)
        return real(events, k_neighbors)

    def _popular(events, k):
        popular_k.append(k)
        return real_popular(events, k)

    def _latest(items, k, date_columns=None):
        latest_k.append(k)
        return real_latest(items, k, date_columns)

    monkeypatch.setattr("cicerone.job.neighbors_from_events", _capture)
    monkeypatch.setattr("cicerone.job.popular_from_events", _popular)
    monkeypatch.setattr("cicerone.job.latest_from_items", _latest)
    job.run()
    expected = DEFAULT_SERVE_MAX_K * 5
    assert seen == [expected]
    assert popular_k == [expected]
    assert latest_k == [expected]


def test_job_uses_one_weighting_timestamp(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    ).to_parquet(input_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    ).to_parquet(input_dir / "items.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=1))
    seen: dict[str, object] = {}
    real_dataset = job.build_dataset
    real_scores = job.build_item_scores

    def _dataset(*args, **kwargs):
        seen["dataset"] = kwargs.get("now")
        return real_dataset(*args, **kwargs)

    def _scores(*args, **kwargs):
        seen["scores"] = kwargs.get("now")
        return real_scores(*args, **kwargs)

    monkeypatch.setattr("cicerone.job.build_dataset", _dataset)
    monkeypatch.setattr("cicerone.job.build_item_scores", _scores)
    job.run()
    assert seen["dataset"] is not None
    assert seen["scores"] == seen["dataset"]


def test_target_user_ids_skip_missing_values():
    events = pd.DataFrame({"user_id": ["u1", float("nan"), pd.NA, "u2"]})
    users = pd.DataFrame({"user_id": ["u3", float("nan"), None]})
    assert _target_user_ids(events, users) == ["u1", "u2", "u3"]
    assert _target_user_ids(events, None) == ["u1", "u2"]
    assert "nan" not in _target_user_ids(events, users)


def test_job_publishes_recommendations_after_write(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    captured: list[pd.DataFrame] = []
    closed = {"n": 0}

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            captured.append(df.copy())

        def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    job.run()
    assert len(captured) == 1
    assert {"u1", "u2"}.issubset(set(captured[0]["user_id"].astype(str)))
    assert closed["n"] == 1


def test_job_reraises_lock_lost_on_publisher_close(tmp_path, monkeypatch):
    from cicerone.locks import LockLostError

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise LockLostError("retrain lock lost before write", kind="retrain")

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with pytest.raises(LockLostError, match="retrain lock lost"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"


def test_job_writes_failure_manifest_when_close_loses_lock_after_error(tmp_path, monkeypatch):
    from cicerone.locks import LockLostError

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise LockLostError("retrain lock lost before write", kind="retrain")

    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, tmp_path, tmp_path))
    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with pytest.raises(LockLostError, match="retrain lock lost"):
        job.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "events.parquet" in manifest["error"]


def test_job_keeps_original_error_when_close_raises_after_failure(tmp_path, monkeypatch, caplog):
    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise RuntimeError("close bug")

    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, tmp_path, tmp_path))
    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with (
        caplog.at_level("ERROR", logger="cicerone.job"),
        pytest.raises(Exception, match="events.parquet"),
    ):
        job.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "events.parquet" in manifest["error"]
    assert "close bug" not in manifest["error"]
    assert any(
        "Failed to close recommendation publisher (RuntimeError: close bug)" in record.getMessage()
        for record in caplog.records
    )


def test_job_succeeds_when_publisher_close_raises_publish_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise PublishError("flush on close")

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"


def test_job_rewrites_manifest_when_close_raises_unexpected_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise RuntimeError("close bug")

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with pytest.raises(RuntimeError, match="close bug"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "close bug" in manifest["error"]


def test_job_failed_rewrite_uses_fresh_generated_at(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))
    written: list[tuple[dict, str | None]] = []
    real = job.write_job_manifest

    def _capture(sink, manifest, *, skip_if_newer_than=None):
        written.append((dict(manifest), skip_if_newer_than))
        return real(sink, manifest, skip_if_newer_than=skip_if_newer_than)

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise RuntimeError("close bug")

    monkeypatch.setattr("cicerone.job.write_job_manifest", _capture)
    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with pytest.raises(RuntimeError, match="close bug"):
        job.run()
    success = next(item for item in written if item[0].get("status") == "success")
    failed = written[-1]
    assert failed[0]["status"] == "failed"
    assert failed[1] == success[0]["generated_at"]
    assert failed[0]["generated_at"] > success[0]["generated_at"]


def test_job_succeeds_when_publish_fails_after_write(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            raise PublishError("broker down")

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert (output_dir / "recommendations.parquet").exists()


def test_job_rewrites_manifest_when_publish_raises_unexpected_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            raise RuntimeError("broken sidecar")

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    with pytest.raises(RuntimeError, match="broken sidecar"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "broken sidecar" in manifest["error"]
    assert manifest.get("partial_outputs") is False
    assert (output_dir / "recommendations.parquet").exists()


def test_job_rewrites_manifest_when_sidecar_generation_raises_unexpected_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            raise AssertionError("publish should not run after generation check failure")

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    monkeypatch.setattr(
        "cicerone.job.sidecar_generation_current",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("generation check bug")),
    )
    with pytest.raises(RuntimeError, match="generation check bug"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "generation check bug" in manifest["error"]
    assert manifest.get("partial_outputs") is False
    assert (output_dir / "recommendations.parquet").exists()


def test_job_skips_publish_when_generation_unconfirmed(tmp_path, monkeypatch, caplog):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=2))
    published = {"n": 0}

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df
            published["n"] += 1

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    monkeypatch.setattr("cicerone.job.sidecar_generation_current", lambda *_args, **_kwargs: None)
    with caplog.at_level("INFO"):
        job.run()
    assert published["n"] == 0
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "success"
    assert any("could not confirm sidecar generation" in record.getMessage() for record in caplog.records)
    assert not any("superseded" in record.getMessage() for record in caplog.records)


def test_job_raises_when_fence_lost_before_publish(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    published: list[pd.DataFrame] = []

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())

    def fence() -> bool:
        return not (output_dir / "manifest.json").exists()

    from cicerone.locks import LockLostError

    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job.run(fence_check=fence)
    assert published == []
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "success"


def test_job_raises_when_fence_lost_after_connect(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    published: list[pd.DataFrame] = []
    lost_after_connect = {"lost": False}

    class _Pub:
        def connect(self) -> None:
            lost_after_connect["lost"] = True

        def publish(self, df: pd.DataFrame) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())

    def fence() -> bool:
        return not lost_after_connect["lost"]

    from cicerone.locks import LockLostError

    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job.run(fence_check=fence)
    assert published == []
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "success"


def test_job_skips_publish_when_manifest_generation_changes(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    published: list[pd.DataFrame] = []

    class _Pub:
        def connect(self) -> None:
            path = output_dir / "manifest.json"
            payload = json.loads(path.read_text())
            payload["generated_at"] = "2099-01-01T00:00:00+00:00"
            path.write_text(json.dumps(payload))

        def publish(self, df: pd.DataFrame) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    job.run()
    assert published == []
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "success"


def test_job_run_writes_track_and_served_eval(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)
    extra = """
        [track]
        enabled = true
        [job.eval]
        enabled = true
        """
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    job.run()
    previous_generated_at = json.loads((output_dir / "manifest.json").read_text())["generated_at"]
    from cicerone.config import IOSettings
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    recs = pd.read_parquet(output_dir / "recommendations.parquet")
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(output_dir)})
    store = TrackStore(output)
    row = recs.iloc[0]
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": str(row["user_id"]),
                    "item_id": str(row["item_id"]),
                    "rank": 1,
                    "occurred_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "event_id": "imp-job-1",
                }
            ).as_row()
        ]
    )
    job.run()
    report = json.loads((output_dir / "track_eval.json").read_text())
    assert "track_eval" in report
    assert report["generated_at"] == previous_generated_at
    history_dir = output_dir / "recommendation_history"
    assert history_dir.is_dir()
    assert list(history_dir.glob("*.parquet"))
    assert report["track_eval"]["overall"]["n_impressions"] >= 1


def test_score_previous_run_reads_history_when_track_disabled(tmp_path, monkeypatch):
    from cicerone.config import EvalSettings, IOSettings, TrackSettings, make_settings
    from cicerone.job import _score_previous_run

    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        track=TrackSettings(enabled=False),
        eval=EvalSettings(enabled=True, event_types=("purchase",), ks=(1,)),
    )
    recs = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    history = recs.copy()
    history["generated_at"] = "2026-08-28T03:00:00+00:00"
    calls: list[set[str] | None] = []

    monkeypatch.setattr("cicerone.job_eval.load_recommendations_frame", lambda _output: recs)

    def _read_history(self, *, generated_ats=None, since=None):
        calls.append(generated_ats)
        return history

    monkeypatch.setattr("cicerone.job.TrackStore.read_history", _read_history)
    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda self: [])
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": pd.Timestamp("2026-08-28T04:00:00+00:00"),
            }
        ]
    )
    _track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert calls == [{"2026-08-28T03:00:00+00:00"}]
    assert served is not None


def test_score_previous_run_ignores_preloaded_track_when_disabled(tmp_path, monkeypatch):
    from cicerone.config import EvalSettings, IOSettings, TrackSettings, make_settings
    from cicerone.job import _score_previous_run

    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        track=TrackSettings(enabled=False),
        eval=EvalSettings(enabled=True, event_types=("purchase",), ks=(1,)),
    )
    recs = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    history = recs.copy()
    history["generated_at"] = "2026-08-28T03:00:00+00:00"
    calls: list[set[str] | None] = []

    monkeypatch.setattr("cicerone.job_eval.load_recommendations_frame", lambda _output: recs)

    def _read_history(self, *, generated_ats=None, since=None):
        calls.append(generated_ats)
        return history

    monkeypatch.setattr("cicerone.job.TrackStore.read_history", _read_history)

    def _boom(*_args, **_kwargs):
        raise AssertionError("read_rows should not run when track is disabled")

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", _boom)
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": pd.Timestamp("2026-08-28T04:00:00+00:00"),
            }
        ]
    )
    preloaded = [
        {
            "event_id": "extra",
            "generated_at": "2026-08-27T03:00:00+00:00",
            "user_id": "u1",
            "item_id": "i1",
        }
    ]
    _track, served = _score_previous_run(
        settings,
        events,
        {"generated_at": "2026-08-28T03:00:00+00:00"},
        preloaded_track=preloaded,
        preloaded_recs=recs,
    )
    assert calls == [{"2026-08-28T03:00:00+00:00"}]
    assert served is not None
    assert _track is None


def test_replay_assignments_prefers_first_impression_then_hash(tmp_path):
    from cicerone.blending import COLD_START_USER_ID
    from cicerone.config import IOSettings, make_settings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _replay_assignments

    recs = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "i1", "variant": "control"},
            {"user_id": "alice", "item_id": "i2", "variant": "treatment"},
            {"user_id": "bob", "item_id": "i1", "variant": "control"},
            {"user_id": "bob", "item_id": "i2", "variant": "treatment"},
            {"user_id": COLD_START_USER_ID, "item_id": "i1", "variant": "control"},
        ]
    )
    assert _replay_assignments(make_settings(), pd.DataFrame(), []) is None
    assert _replay_assignments(make_settings(), recs.drop(columns=["variant"]), []) is None
    single = recs[recs["variant"] == "control"]
    assert _replay_assignments(make_settings(), single, []) is None

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    assigned = _replay_assignments(
        settings,
        recs,
        [
            {"kind": "click", "user_id": "alice", "variant": "treatment"},
            {
                "kind": "impression",
                "user_id": "alice",
                "variant": "treatment",
                "occurred_at": "2026-08-28T13:00:00Z",
                "event_id": "later",
            },
            {
                "kind": "impression",
                "user_id": "alice",
                "variant": "treatment",
                "occurred_at": "not-a-time",
                "event_id": "bad",
            },
            {
                "kind": "impression",
                "user_id": "alice",
                "variant": "control",
                "occurred_at": "2026-08-28T12:00:00Z",
                "event_id": "first",
            },
            {"kind": "impression", "user_id": "", "variant": "control"},
            {"kind": "impression", "user_id": "alice", "variant": "unknown"},
        ],
    )
    assert assigned is not None
    assert assigned["alice"] == "control"
    assert assigned["bob"] in {"control", "treatment"}
    assert COLD_START_USER_ID not in assigned

    from cicerone.config.constants import ALLOCATION_THOMPSON

    thompson = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.33),
                VariantSettings(name="treatment", traffic=0.33),
                VariantSettings(name="blend", traffic=0.34),
            ),
        ),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        track=TrackSettings(enabled=True),
    )
    hashed = _replay_assignments(thompson, recs[recs["user_id"] != COLD_START_USER_ID], [])
    assert hashed is not None
    assert set(hashed.values()) <= {"control", "treatment"}
    assert "blend" not in hashed.values()

    fallback = _replay_assignments(
        make_settings(),
        recs[recs["user_id"] != COLD_START_USER_ID],
        [],
    )
    assert fallback is not None
    assert set(fallback.values()) <= {"control", "treatment"}
    assert set(fallback) == {"alice", "bob"}


def test_job_run_swallows_eval_persistence_errors(tmp_path, monkeypatch, caplog):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    ).to_parquet(input_dir / "items.parquet", index=False)
    extra = """
        [track]
        enabled = true
        [job.eval]
        enabled = true
        """
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.write_eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("eval")),
    )
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.append_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("history")),
    )
    with caplog.at_level("ERROR", logger="cicerone.job_eval"):
        job.run()
    assert (output_dir / "recommendations.parquet").exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Failed to write track eval (OSError: eval)" in message for message in messages)
    assert any(
        "Failed to append recommendation history (OSError: history)" in message for message in messages
    )
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "success"


def test_job_prefers_close_lock_loss_over_persist_error(tmp_path, monkeypatch):
    from cicerone.locks import LockLostError

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    extra = """
        [track]
        enabled = true
        [job.eval]
        enabled = true
        """
    monkeypatch.setenv(
        "CICERONE_CONFIG_PATH",
        _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra),
    )

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            del df

        def close(self) -> None:
            raise LockLostError("retrain lock lost before write", kind="retrain")

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.write_eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval store bug")),
    )
    with pytest.raises(LockLostError, match="retrain lock lost"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert (output_dir / "recommendations.parquet").exists()


def test_job_rewrites_manifest_when_persist_raises_unexpected_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    extra = """
        [track]
        enabled = true
        [job.eval]
        enabled = true
        """
    monkeypatch.setenv(
        "CICERONE_CONFIG_PATH",
        _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra),
    )
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.write_eval",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval store bug")),
    )
    with pytest.raises(RuntimeError, match="eval store bug"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "eval store bug" in manifest["error"]
    assert (output_dir / "recommendations.parquet").exists()


def test_job_skips_failed_rewrite_when_newer_manifest_exists(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    extra = """
        [track]
        enabled = true
        [job.eval]
        enabled = true
        """
    monkeypatch.setenv(
        "CICERONE_CONFIG_PATH",
        _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra),
    )
    newer = {
        "triggered_by": "job",
        "status": "success",
        "generated_at": "2099-01-01T00:00:00+00:00",
    }

    def _newer_then_fail(*args, **kwargs):
        del args, kwargs
        (output_dir / "manifest.json").write_text(json.dumps(newer))
        raise RuntimeError("eval store bug")

    monkeypatch.setattr("cicerone.track.store.TrackStore.write_eval", _newer_then_fail)
    with pytest.raises(RuntimeError, match="eval store bug"):
        job.run()
    assert json.loads((output_dir / "manifest.json").read_text()) == newer


def test_read_input_swallows_manifest_reader_construction(monkeypatch):
    class _Source:
        def read_events(self) -> pd.DataFrame:
            return pd.DataFrame({"user_id": ["u1"]})

        def read_users(self) -> pd.DataFrame | None:
            return None

        def read_items(self) -> pd.DataFrame | None:
            return None

        def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
            return pd.DataFrame()

        def get_user(self, user_id: str) -> dict | None:
            return None

    monkeypatch.setattr(
        "cicerone.job_eval.build_manifest_reader",
        lambda _output: (_ for _ in ()).throw(ValueError("bad url")),
    )
    events, users, items, manifest = job._read_input(
        _Source(),
        IOSettings(kind="db", options={"database_url": "postgresql+psycopg://bad"}),
    )
    assert list(events["user_id"]) == ["u1"]
    assert users is None
    assert items is None
    assert manifest is None


def test_try_load_logs_exception_type_and_message(caplog):
    from cicerone.job_eval import try_load

    def _boom() -> None:
        raise OSError("recs")

    with caplog.at_level("ERROR", logger="cicerone.job_eval"):
        assert try_load("load previous recommendations for eval", _boom, None) is None
    assert any(
        "Failed to load previous recommendations for eval (OSError: recs)" in record.getMessage()
        for record in caplog.records
    )


def test_try_load_does_not_swallow_unexpected_errors():
    from cicerone.job_eval import try_load

    with pytest.raises(RuntimeError, match="bug"):
        try_load(
            "load previous recommendations for eval",
            lambda: (_ for _ in ()).throw(RuntimeError("bug")),
            None,
        )


def test_try_load_reraises_lock_errors():
    from cicerone.job_eval import try_load
    from cicerone.locks import LockLostError, WriterLockBusyError

    with pytest.raises(LockLostError, match="retrain lock lost"):
        try_load(
            "write track eval",
            lambda: (_ for _ in ()).throw(LockLostError("retrain lock lost", kind="retrain")),
            None,
        )
    with pytest.raises(WriterLockBusyError, match="busy"):
        try_load(
            "write track eval",
            lambda: (_ for _ in ()).throw(WriterLockBusyError("dataset writer lock busy")),
            None,
        )


def test_persist_track_outputs_reraises_lock_errors(monkeypatch):
    from cicerone.locks import WriterLockBusyError

    def _busy(*_args, **_kwargs):
        raise WriterLockBusyError("dataset writer lock busy")

    monkeypatch.setattr("cicerone.job_eval.held_writer_lock", _busy)
    with pytest.raises(WriterLockBusyError, match="dataset writer lock busy"):
        job._persist_track_outputs(
            TrackStore(
                IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
                writer_lock=object(),
            ),
            kind="dataset",
            eval_report={"generated_at": "t"},
            recommendations=None,
            generated_at="t",
        )


def test_persist_track_outputs_reraises_write_eval_lock_loss(monkeypatch):
    from cicerone.locks import LockLostError

    def _lost(self, report):
        del report
        raise LockLostError("retrain lock lost before write", kind="retrain")

    monkeypatch.setattr(TrackStore, "write_eval", _lost)
    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job._persist_track_outputs(
            TrackStore(IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"})),
            kind="dataset",
            eval_report={"generated_at": "t"},
            recommendations=None,
            generated_at="t",
        )


def test_refresh_pending_thompson_keeps_live_promotion(tmp_path):
    from cicerone.experiment.store import ExperimentStore, experiment_state

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment", champion="control"))
    pending = experiment_state("exp", promoted_variant=None, champion="treatment")
    merged = job._refresh_pending_thompson(store, pending)
    assert merged["promoted_variant"] == "treatment"
    assert merged["champion"] == "treatment"


def test_persist_track_outputs_serializes_db_writes(monkeypatch):
    active = 0
    max_active = 0
    order: list[str] = []

    def _write_eval(self, report):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        order.append("eval")
        active -= 1

    def _append_history(self, recommendations, *, generated_at):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        order.append("hist")
        active -= 1

    monkeypatch.setattr(TrackStore, "write_eval", _write_eval)
    monkeypatch.setattr(TrackStore, "append_history", _append_history)
    job._persist_track_outputs(
        TrackStore(IOSettings(kind="db", options={"database_url": "sqlite://"})),
        kind="db",
        eval_report={"generated_at": "t"},
        recommendations=pd.DataFrame([{"user_id": "u1"}]),
        generated_at="t",
    )
    assert order == ["eval", "hist"]
    assert max_active == 1


def test_persist_track_outputs_holds_lock_for_db(monkeypatch):
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

    monkeypatch.setattr(TrackStore, "write_eval", lambda self, report: None)
    monkeypatch.setattr(TrackStore, "append_history", lambda self, recommendations, *, generated_at: None)
    job._persist_track_outputs(
        TrackStore(
            IOSettings(kind="db", options={"database_url": "sqlite://"}),
            writer_lock=_Lock(),
        ),
        kind="db",
        eval_report={"generated_at": "t"},
        recommendations=pd.DataFrame([{"user_id": "u1"}]),
        generated_at="t",
    )
    assert acquires["n"] == 1


def test_recommendation_user_count_excludes_cold_start():
    frame = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1"},
            {"user_id": COLD_START_USER_ID, "item_id": "c1"},
            {"user_id": "u1", "item_id": "i2"},
        ]
    )
    assert _recommendation_user_count(frame) == 1
    assert _recommendation_user_count(pd.DataFrame()) == 0


def test_job_run_writes_model_artifact_when_enabled(tmp_path, monkeypatch):
    from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, load_artifact, recommend_from_artifact

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(
        tmp_path, input_dir, output_dir, top_k=2, extra_job="save_model_artifact = true"
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    job.run()

    artifact_path = output_dir / "model.artifact"
    assert artifact_path.exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["artifact_written"] is True
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION

    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    loaded = load_artifact(artifact_path)
    from_artifact = recommend_from_artifact(loaded, sorted(recommendations["user_id"].unique()), top_k=2)
    pd.testing.assert_frame_equal(
        recommendations.sort_values(["user_id", "rank"]).reset_index(drop=True),
        from_artifact.sort_values(["user_id", "rank"]).reset_index(drop=True),
    )


def test_job_run_signs_model_artifact_when_hmac_key_set(tmp_path, monkeypatch):
    import zipfile

    from cicerone.artifact import load_artifact

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    ).to_parquet(input_dir / "items.parquet", index=False)
    key = "0123456789abcdef"
    config_path = _write_config(
        tmp_path,
        input_dir,
        output_dir,
        top_k=2,
        extra_job="save_model_artifact = true",
        extra_output=f'artifact_hmac_key = "{key}"',
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    job.run()
    artifact_path = output_dir / "model.artifact"
    with zipfile.ZipFile(artifact_path) as zf:
        assert "hmac.sha256" in zf.namelist()
    loaded = load_artifact(artifact_path, hmac_key=key)
    assert loaded.models
    with pytest.raises(ValueError, match="HMAC"):
        load_artifact(artifact_path, hmac_key="fedcba9876543210")


def test_job_run_with_automl_enabled_selects_and_records_best_candidate(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    interactions = {"u1": ["i1", "i2"], "u2": ["i2", "i3"], "u3": ["i1", "i3"]}
    for day_offset in range(0, 21, 3):
        occurred_at = now - pd.Timedelta(days=day_offset)
        for user, item_ids in interactions.items():
            for item_id in item_ids:
                rows.append(
                    {
                        "user_id": user,
                        "item_id": item_id,
                        "event_type": "purchase",
                        "quantity": 1,
                        "occurred_at": occurred_at,
                    }
                )
    events = pd.DataFrame(rows)
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = 2
        feature_config_path = "{REPO_FEATURES_CONFIG}"

        [job.automl]
        enabled = true
        n_splits = 1
        test_days = 7
        primary_metric = "MAP"

        [[job.automl.candidates]]
        models = ["popular"]

        [[job.automl.candidates]]
        models = ["latest"]

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{input_dir}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{output_dir}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    job.run()

    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert not recommendations.empty

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["automl_enabled"] is True
    assert manifest["models"] in ("popular", "latest")
    assert manifest["automl_metrics"] != ""
    # Priority-mode candidates → empty model_weights, not stale fusion values.
    assert manifest["model_weights"] == ""
    assert manifest["rrf_k"] == RRF_K
    automl_metrics = manifest["automl_metrics"].split(",")
    assert any(metric.startswith("MAP@") for metric in automl_metrics)
    assert any(metric.startswith("NDCG@") for metric in automl_metrics)
    assert any(metric.startswith("Recall@") for metric in automl_metrics)


def test_job_run_with_automl_fusion_candidate_reports_effective_weights(tmp_path, monkeypatch):
    # Single fusion candidate → manifest model_weights/rrf_k are deterministic.
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    interactions = {"u1": ["i1", "i2"], "u2": ["i2", "i3"], "u3": ["i1", "i3"]}
    for day_offset in range(0, 21, 3):
        occurred_at = now - pd.Timedelta(days=day_offset)
        for user, item_ids in interactions.items():
            for item_id in item_ids:
                rows.append(
                    {
                        "user_id": user,
                        "item_id": item_id,
                        "event_type": "purchase",
                        "quantity": 1,
                        "occurred_at": occurred_at,
                    }
                )
    events = pd.DataFrame(rows)
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = 2
        feature_config_path = "{REPO_FEATURES_CONFIG}"

        [job.automl]
        enabled = true
        n_splits = 1
        test_days = 7
        primary_metric = "MAP"

        [[job.automl.candidates]]
        models = ["popular", "latest"]
        rrf_k = 30

        [job.automl.candidates.weights]
        popular = 1.0
        latest = 0.5

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{input_dir}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{output_dir}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    job.run()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["automl_enabled"] is True
    assert manifest["models"] == "popular,latest"
    assert manifest["model_weights"] == "popular=1.0,latest=0.5"
    assert manifest["rrf_k"] == 30.0


def test_job_run_with_manual_fusion_configuration_reports_manifest_fields(tmp_path, monkeypatch):
    # AutoML off: TOML job.models / model_weights / rrf_k must reach the manifest.
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = 2
        feature_config_path = "{REPO_FEATURES_CONFIG}"
        models = ["popular", "latest"]
        rrf_k = 30

        [job.model_weights]
        popular = 1.0
        latest = 0.5

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{input_dir}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{output_dir}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    job.run()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["automl_enabled"] is False
    assert manifest["models"] == "popular,latest"
    assert manifest["model_weights"] == "popular=1.0,latest=0.5"
    assert manifest["rrf_k"] == 30.0


def test_job_run_raises_on_failure(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, tmp_path, tmp_path)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    with pytest.raises(Exception, match="events.parquet"):
        job.run()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "events.parquet" in manifest["error"]


def test_job_succeeds_when_publisher_connect_fails_after_write(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.utcnow()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    ).to_parquet(input_dir / "events.parquet", index=False)
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    class _Pub:
        def connect(self) -> None:
            raise PublishError("broker down")

        def publish(self, df: pd.DataFrame) -> None:
            raise AssertionError("publish should not run after connect failure")

        def close(self) -> None:
            return None

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings, **_kwargs: _Pub())
    job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert (output_dir / "recommendations.parquet").exists()


def test_job_run_truncates_an_overly_long_error_message(tmp_path, monkeypatch):
    # Manifest error is persisted/shown as-is — must stay bounded.
    config_path = _write_config(tmp_path, tmp_path, tmp_path)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    long_message = "x" * 1000
    monkeypatch.setattr(
        job,
        "build_input_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(long_message)),
    )

    with pytest.raises(RuntimeError):
        job.run()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["error"]) < len(long_message)
    assert manifest["error"].endswith("... (truncated)")


def test_job_run_records_custom_triggered_by(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    job.run(triggered_by="webhook")

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["triggered_by"] == "webhook"
    assert manifest["lock_backend"] == "in_process"


def test_job_run_records_configured_lock_backend(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(
        tmp_path,
        input_dir,
        output_dir,
        extra_job='[job.trigger]\nlock_backend = "redis"\nredis_url = "redis://localhost:6379/0"\n',
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    monkeypatch.setattr("cicerone.job.build_dataset_writer_lock", lambda _settings: None)
    monkeypatch.setattr("cicerone.job.build_output_writer_lock", lambda _settings: None)

    class _Held:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    monkeypatch.setattr("cicerone.job.build_lock_backend", lambda _settings: _Held())
    monkeypatch.setattr("cicerone.job.acquire_blocking", lambda _lock, **_kwargs: True)

    job.run(triggered_by="cron")

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["lock_backend"] == "redis"


def test_job_marks_partial_outputs_when_recommendation_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir, extra_job="save_model_artifact = true")
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.io.dataset_store import DatasetOutputSink

    original_write = DatasetOutputSink.write_recommendations

    def boom(self, df):
        raise RuntimeError("disk full")

    monkeypatch.setattr(DatasetOutputSink, "write_recommendations", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        job.run()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is True
    assert manifest["artifact_written"] is True
    del original_write


def test_write_job_manifest_accepts_legacy_signature():
    written: dict[str, object] = {}

    class _LegacySink:
        def write_manifest(self, manifest):
            written["manifest"] = manifest

    assert job.write_job_manifest(_LegacySink(), {"status": "failed"}, skip_if_newer_than="x") is True
    assert written["manifest"] == {"status": "failed"}


def test_direct_job_acquires_retrain_lock_when_distributed(monkeypatch):
    from cicerone.locks import WriterLockBusyError

    class _Busy:
        def acquire(self) -> bool:
            return False

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return False

        def is_locked(self) -> bool:
            return True

    monkeypatch.setattr("cicerone.job.has_distributed_lock", lambda _settings: True)
    monkeypatch.setattr("cicerone.job.build_lock_backend", lambda _settings: _Busy())
    monkeypatch.setattr("cicerone.job.acquire_blocking", lambda _lock, **_kwargs: False)
    monkeypatch.setattr("cicerone.job.load_settings", lambda: object())
    with pytest.raises(WriterLockBusyError, match="retrain lock busy"):
        job.run()


def test_direct_job_releases_retrain_lock_after_run(monkeypatch):
    released = {"n": 0}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            released["n"] += 1

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    monkeypatch.setattr("cicerone.job.has_distributed_lock", lambda _settings: True)
    monkeypatch.setattr("cicerone.job.build_lock_backend", lambda _settings: _Lock())
    monkeypatch.setattr("cicerone.job.acquire_blocking", lambda _lock, **_kwargs: True)
    monkeypatch.setattr("cicerone.job.load_settings", lambda: object())
    monkeypatch.setattr("cicerone.job._run_job", lambda *_args, **_kwargs: None)
    job.run()
    assert released["n"] == 1


def test_scheduler_job_skips_direct_retrain_lock(monkeypatch):
    built = {"n": 0}

    def _build(_settings):
        built["n"] += 1
        raise AssertionError("should not build")

    monkeypatch.setattr("cicerone.job.has_distributed_lock", lambda _settings: True)
    monkeypatch.setattr("cicerone.job.build_lock_backend", _build)
    monkeypatch.setattr("cicerone.job.load_settings", lambda: object())
    monkeypatch.setattr("cicerone.job._run_job", lambda *_args, **_kwargs: None)
    job.run(fence_check=lambda: True)
    assert built["n"] == 0


def test_write_job_manifest_reraises_implementation_type_error():
    class _Sink:
        def write_manifest(self, manifest, *, skip_if_newer_than=None):
            raise TypeError("bad payload")

    with pytest.raises(TypeError, match="bad payload"):
        job.write_job_manifest(_Sink(), {"status": "failed"}, skip_if_newer_than="x")


def test_job_holds_writer_lock_for_artifact_and_items(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir, extra_job="save_model_artifact = true")
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.io.dataset_store import DatasetOutputSink

    depths: list[tuple[str, int]] = []
    original_artifact = DatasetOutputSink.write_model_artifact
    original_items = DatasetOutputSink.write_items_snapshot

    def capture_artifact(self, payload):
        depths.append(("artifact", self._recs_write_depth()))
        return original_artifact(self, payload)

    def capture_items(self, df):
        depths.append(("items", self._recs_write_depth()))
        return original_items(self, df)

    original_scores = job.build_item_scores
    original_build = job.build_output_sink
    sinks: list = []

    def capture_sink(*args, **kwargs):
        sink = original_build(*args, **kwargs)
        sinks.append(sink)
        return sink

    def capture_scores(*args, **kwargs):
        depths.append(("scores", sinks[-1]._recs_write_depth()))
        return original_scores(*args, **kwargs)

    monkeypatch.setattr(DatasetOutputSink, "write_model_artifact", capture_artifact)
    monkeypatch.setattr(DatasetOutputSink, "write_items_snapshot", capture_items)
    monkeypatch.setattr("cicerone.job.build_output_sink", capture_sink)
    monkeypatch.setattr("cicerone.job.build_item_scores", capture_scores)

    job.run()

    assert depths[0] == ("scores", 0)
    assert ("artifact", 1) in depths
    assert ("items", 1) in depths


def test_job_skips_item_scores_when_sink_lacks_writer(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    ).to_parquet(input_dir / "events.parquet", index=False)
    pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    ).to_parquet(input_dir / "items.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=1))
    real = job.build_output_sink

    def _legacy(*args, **kwargs):
        sink = real(*args, **kwargs)
        sink.write_item_scores = None
        return sink

    monkeypatch.setattr("cicerone.job.build_output_sink", _legacy)
    built = {"n": 0}
    original_scores = job.build_item_scores

    def _count(*args, **kwargs):
        built["n"] += 1
        return original_scores(*args, **kwargs)

    monkeypatch.setattr("cicerone.job.build_item_scores", _count)
    job.run()
    assert built["n"] == 0
    assert not (output_dir / "item_scores.parquet").exists()
    assert (output_dir / "recommendations.parquet").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["n_item_scores"] is None


def test_job_skips_failure_manifest_when_incremental_is_newer(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    incremental = {
        "triggered_by": "incremental",
        "status": "success",
        "generated_at": "2099-01-01T00:00:00+00:00",
        "last_incremental_at": "2099-01-01T00:00:00+00:00",
    }
    (output_dir / "manifest.json").write_text(json.dumps(incremental))

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.io.dataset_store import DatasetOutputSink

    def boom(self, df):
        raise RuntimeError("disk full")

    monkeypatch.setattr(DatasetOutputSink, "write_recommendations", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        job.run()

    assert json.loads((output_dir / "manifest.json").read_text()) == incremental


def test_job_keeps_prior_recommendations_when_item_scores_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    job.run()
    first = pd.read_parquet(output_dir / "recommendations.parquet")

    from cicerone.io.dataset_store import DatasetOutputSink

    def boom(self, df):
        raise RuntimeError("scores unavailable")

    monkeypatch.setattr(DatasetOutputSink, "write_item_scores", boom)
    with pytest.raises(RuntimeError, match="scores unavailable"):
        job.run()

    kept = pd.read_parquet(output_dir / "recommendations.parquet")
    pd.testing.assert_frame_equal(first, kept)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is True


def test_job_recs_fail_without_published_outputs_is_not_partial(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    ).to_parquet(input_dir / "events.parquet", index=False)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, input_dir, output_dir, top_k=1))
    real = job.build_output_sink

    def _legacy(*args, **kwargs):
        sink = real(*args, **kwargs)
        sink.write_item_scores = None
        sink.write_surfaces = None
        return sink

    from cicerone.io.dataset_store import DatasetOutputSink

    def boom(self, df):
        raise RuntimeError("disk full")

    monkeypatch.setattr("cicerone.job.build_output_sink", _legacy)
    monkeypatch.setattr(DatasetOutputSink, "write_recommendations", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        job.run()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is False


def test_job_preserves_success_when_manifest_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.io.dataset_store import DatasetOutputSink

    def boom(self, manifest, *, skip_if_newer_than=None):
        del skip_if_newer_than
        raise RuntimeError("manifest unavailable")

    monkeypatch.setattr(DatasetOutputSink, "write_manifest", boom)

    with pytest.raises(RuntimeError, match="manifest unavailable"):
        job.run()

    assert (output_dir / "recommendations.parquet").exists()


def test_job_skips_writes_when_fence_lost_before_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.locks import LockLostError

    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job.run(fence_check=lambda: False)

    assert not (output_dir / "recommendations.parquet").exists()
    assert not (output_dir / "manifest.json").exists()


def test_job_marks_partial_outputs_when_fence_lost_after_write(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.locks import LockLostError

    calls = {"n": 0}

    def fence() -> bool:
        calls["n"] += 1
        return not (output_dir / "recommendations.parquet").exists()

    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job.run(fence_check=fence)

    assert (output_dir / "recommendations.parquet").exists()
    assert not (output_dir / "manifest.json").exists()
    assert calls["n"] >= 2


def test_run_guard_skips_job_writes_when_owned_is_false(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    items = pd.DataFrame(
        [{"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True}]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    config_path = _write_config(tmp_path, input_dir, output_dir)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    from cicerone.trigger import RunGuard

    released = threading.Event()

    class DeadLock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            released.set()

        def owned(self) -> bool:
            return False

    guard = RunGuard(debounce_seconds=0, run_fn=job.run, lock_backend=DeadLock())
    assert guard.trigger("webhook") is True
    assert released.wait(timeout=30)
    assert not (output_dir / "recommendations.parquet").exists()
    assert not (output_dir / "manifest.json").exists()


def test_job_run_writes_both_experiment_variants(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)

    extra = """
        [experiment]
        enabled = true
        id = "rrf-vs-priority"

        [[experiment.variants]]
        name = "control"
        traffic = 0.5

        [[experiment.variants]]
        name = "treatment"
        traffic = 0.5
        combiner = "rrf"
        """
    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2, extra=extra)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    job.run()

    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert "variant" in recommendations.columns
    assert set(recommendations["variant"].astype(str)) == {"control", "treatment"}
    for variant in ("control", "treatment"):
        users = set(recommendations.loc[recommendations["variant"] == variant, "user_id"].astype(str))
        assert {"u1", "u2"} <= users

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["experiment_id"] == "rrf-vs-priority"
    variants = json.loads(manifest["experiment_variants"])
    assert [item["name"] for item in variants] == ["control", "treatment"]


def _thompson_job_extra() -> str:
    return """
        [track]
        enabled = true
        [experiment]
        enabled = true
        id = "ranking-cvr"
        primary_metric = "conversion"
        attribution = "click"
        allocation = "thompson"
        [[experiment.variants]]
        name = "control"
        traffic = 0.34
        [[experiment.variants]]
        name = "treatment"
        traffic = 0.33
        [[experiment.variants]]
        name = "blend"
        traffic = 0.33
        combiner = "rrf"
        """


def test_job_thompson_fail_closed_empty_track_writes_all_variants(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)
    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: True)
    config_path = _write_config(tmp_path, input_dir, output_dir, extra=_thompson_job_extra())
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    job.run()
    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert set(recommendations["variant"].astype(str)) == {"control", "treatment", "blend"}


def test_job_thompson_writes_active_pair_and_keeps_it(tmp_path, monkeypatch):
    from cicerone.config import IOSettings
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.experiment.thompson import ArmCounts, ThompsonAllocation

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(output_dir)})
    ExperimentStore(output).write_state(
        experiment_state("ranking-cvr", promoted_variant=None, champion="control", challenger="blend")
    )

    def _allocate(**kwargs):
        names = list(kwargs["names"])
        return ThompsonAllocation(
            champion="control",
            challenger="blend",
            arms={name: ArmCounts(0, 0) for name in names},
            p_best={name: 0.5 for name in names},
            pair_impressions=int((kwargs.get("previous") or {}).get("pair_impressions") or 0),
            window_started_at="2026-09-04T00:00:00+00:00",
            rotated=False,
        )

    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: True)
    monkeypatch.setattr("cicerone.job.allocate_thompson", _allocate)
    config_path = _write_config(tmp_path, input_dir, output_dir, extra=_thompson_job_extra())
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    job.run()
    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert set(recommendations["variant"].astype(str)) == {"control", "blend"}
    state = ExperimentStore(output).read_state()
    assert state is not None
    assert state["champion"] == "control"
    assert state["challenger"] == "blend"
    job.run()
    again = pd.read_parquet(output_dir / "recommendations.parquet")
    assert set(again["variant"].astype(str)) == {"control", "blend"}


def test_job_thompson_keeps_previous_pair_when_recs_write_fails(tmp_path, monkeypatch):
    from cicerone.config import IOSettings
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.experiment.thompson import ArmCounts, ThompsonAllocation
    from cicerone.io.dataset_store import DatasetOutputSink

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
        ]
    )
    events.to_parquet(input_dir / "events.parquet", index=False)
    items.to_parquet(input_dir / "items.parquet", index=False)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(output_dir)})
    ExperimentStore(output).write_state(
        experiment_state("ranking-cvr", promoted_variant=None, champion="control", challenger="blend")
    )

    def _allocate(**kwargs):
        names = list(kwargs["names"])
        return ThompsonAllocation(
            champion="control",
            challenger="treatment",
            arms={name: ArmCounts(0, 0) for name in names},
            p_best={name: 0.5 for name in names},
            pair_impressions=0,
            window_started_at="2026-09-04T00:00:00+00:00",
            rotated=True,
        )

    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: True)
    monkeypatch.setattr("cicerone.job.allocate_thompson", _allocate)
    monkeypatch.setattr(
        DatasetOutputSink,
        "write_recommendations",
        lambda self, df: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    config_path = _write_config(tmp_path, input_dir, output_dir, extra=_thompson_job_extra())
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    with pytest.raises(RuntimeError, match="disk full"):
        job.run()
    state = ExperimentStore(output).read_state()
    assert state is not None
    assert state["champion"] == "control"
    assert state["challenger"] == "blend"


def test_select_thompson_recipes_fail_closed_paths(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    assert _select_thompson_recipes(settings, recipes[:1], pd.DataFrame()).recipes == recipes[:1]

    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: (_ for _ in ()).throw(OSError("state")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()).recipes == recipes

    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: {"experiment_id": "other", "champion": "control", "challenger": "treatment"},
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()).recipes == recipes

    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: {
            "experiment_id": "ranking-cvr",
            "champion": "control",
            "challenger": "treatment",
        },
    )
    monkeypatch.setattr(
        "cicerone.job.TrackStore.read_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("track")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()).recipes == recipes

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "cicerone.job.allocate_thompson",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("mab")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()).recipes == recipes
    monkeypatch.setattr(
        "cicerone.job.allocate_thompson",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mab bug")),
    )
    with pytest.raises(RuntimeError, match="mab bug"):
        _select_thompson_recipes(settings, recipes, pd.DataFrame())


def test_select_thompson_recipes_fail_closed_on_state_read_does_not_clear_promote(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: (_ for _ in ()).throw(OSError("state")),
    )
    monkeypatch.setattr(
        "cicerone.job.TrackStore.read_rows",
        lambda *args, **kwargs: [
            {
                "user_id": "u-1",
                "item_id": "i-1",
                "kind": "impression",
                "occurred_at": "2026-09-01T00:00:00Z",
                "variant": "control",
            }
        ],
    )
    writer = MagicMock()
    monkeypatch.setattr("cicerone.job.ExperimentStore.write_state", writer)
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()).recipes == recipes
    writer.assert_not_called()


def test_select_thompson_recipes_survives_recs_and_catalog_errors(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.experiment.thompson import ArmCounts, ThompsonAllocation
    from cicerone.feature_config import BlendingConfig
    from cicerone.io.recommendation_schema import VARIANT_COLUMN
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state(
            "ranking-cvr",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            allocation="thompson",
            pair_impressions=20,
        )
    )

    def _allocate(**kwargs):
        names = list(kwargs["names"])
        return ThompsonAllocation(
            champion="control",
            challenger="treatment",
            arms={name: ArmCounts(0, 0) for name in names},
            p_best={name: 0.5 for name in names},
            pair_impressions=20,
            window_started_at="2026-09-04T00:00:00+00:00",
            rotated=False,
        )

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr("cicerone.job.allocate_thompson", _allocate)
    monkeypatch.setattr(
        "cicerone.job.load_recommendations_frame",
        lambda output: (_ for _ in ()).throw(OSError("recs gone")),
    )
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert [recipe.name for recipe in selected.recipes] == ["control", "treatment"]
    assert selected.state is not None

    recs = pd.DataFrame(
        {
            "user_id": ["u-1", "u-1"],
            "item_id": ["i-1", "i-2"],
            "score": [1.0, 0.9],
            VARIANT_COLUMN: ["control", "treatment"],
        }
    )
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda output: recs)
    monkeypatch.setattr(
        "cicerone.job.load_items_catalog_size",
        lambda output: (_ for _ in ()).throw(OSError("catalog gone")),
    )
    again = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert [recipe.name for recipe in again.recipes] == ["control", "treatment"]
    assert again.state is not None


def test_select_thompson_recipes_reraises_unexpected_catalog_error(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state(
            "ranking-cvr",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            allocation="thompson",
            pair_impressions=20,
        )
    )
    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "cicerone.job.load_recommendations_frame",
        lambda output: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "cicerone.job.load_items_catalog_size",
        lambda output: (_ for _ in ()).throw(RuntimeError("catalog bug")),
    )
    with pytest.raises(RuntimeError, match="catalog bug"):
        _select_thompson_recipes(settings, recipes, pd.DataFrame())


def test_select_thompson_recipes_reads_in_memory_sqlite_on_caller_thread(monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.experiment.thompson import ArmCounts, ThompsonAllocation
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    output = IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"})
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=output,
    )
    seeded = ExperimentStore(output)
    seeded.write_state(
        experiment_state(
            "ranking-cvr",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            allocation="thompson",
        )
    )
    monkeypatch.setattr("cicerone.job.ExperimentStore", lambda _output: seeded)
    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *args, **kwargs: [])

    def _allocate(**kwargs):
        names = list(kwargs["names"])
        return ThompsonAllocation(
            champion="control",
            challenger="treatment",
            arms={name: ArmCounts(0, 0) for name in names},
            p_best={name: 0.5 for name in names},
            pair_impressions=0,
            window_started_at="2026-09-04T00:00:00+00:00",
            rotated=False,
        )

    monkeypatch.setattr("cicerone.job.allocate_thompson", _allocate)
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert selected.state is not None
    assert selected.state["champion"] == "control"


def test_select_thompson_recipes_does_not_write_state(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.experiment.thompson import ArmCounts, ThompsonAllocation
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state(
            "ranking-cvr",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            allocation="thompson",
        )
    )

    def _allocate(**kwargs):
        names = list(kwargs["names"])
        return ThompsonAllocation(
            champion="control",
            challenger="treatment",
            arms={name: ArmCounts(0, 0) for name in names},
            p_best={name: 0.5 for name in names},
            pair_impressions=0,
            window_started_at="2026-09-04T00:00:00+00:00",
            rotated=False,
        )

    monkeypatch.setattr("cicerone.job.allocate_thompson", _allocate)
    writer = MagicMock()
    monkeypatch.setattr("cicerone.job.ExperimentStore.write_state", writer)
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    writer.assert_not_called()
    assert selected.state is not None
    assert selected.state["champion"] == "control"


def test_select_thompson_recipes_fail_closed_without_variant(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    monkeypatch.setattr(
        "cicerone.job.TrackStore.read_rows",
        lambda *args, **kwargs: [
            {
                "user_id": "u-1",
                "item_id": "i-1",
                "kind": "impression",
                "occurred_at": "2026-09-01T00:00:00Z",
            }
        ],
    )
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert selected.recipes == recipes
    assert selected.state is None


def test_select_thompson_recipes_skips_trials_when_window_started_missing(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.experiment.thompson import ArmCounts
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: {
            "experiment_id": "ranking-cvr",
            "champion": "control",
            "challenger": "treatment",
            "arms": {
                "control": {"successes": 4, "failures": 6},
                "treatment": {"successes": 1, "failures": 2},
            },
        },
    )
    monkeypatch.setattr(
        "cicerone.job.TrackStore.read_rows",
        lambda *args, **kwargs: [
            {
                "user_id": "u-1",
                "item_id": "i-1",
                "kind": "impression",
                "variant": "control",
                "experiment_id": "ranking-cvr",
                "occurred_at": "2026-09-01T00:00:00Z",
            }
        ],
    )
    seen: dict[str, object] = {}

    def _capture(**kwargs):
        from cicerone.experiment.thompson import allocate_thompson as _real

        seen["window_trials"] = kwargs.get("window_trials")
        return _real(**kwargs)

    monkeypatch.setattr("cicerone.job.allocate_thompson", _capture)
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    trials = seen.get("window_trials")
    assert isinstance(trials, dict)
    assert trials["control"] == ArmCounts(0, 0)
    assert selected.state is not None


def test_select_thompson_recipes_uses_preloaded_track(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.feature_config import BlendingConfig
    from cicerone.job import _select_thompson_recipes

    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("read_rows should not run when preloaded")

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", _boom)
    selected = _select_thompson_recipes(
        settings,
        recipes,
        pd.DataFrame(),
        preloaded_track=[],
        preloaded_recs=pd.DataFrame(),
    )
    assert calls["n"] == 0
    assert selected.recipes == recipes


def test_load_shared_eval_inputs_drops_partial_preload(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])

    def _boom(*_args, **_kwargs):
        raise OSError("track down")

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", _boom)
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda _output: recs)
    track, loaded = _load_shared_eval_inputs(settings)
    assert track is None
    assert loaded is recs


def test_load_shared_eval_inputs_keeps_track_when_recs_fail(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    def _boom(*_args, **_kwargs):
        raise OSError("recs down")

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *_args, **_kwargs: [{"event_id": "a"}])
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", _boom)
    track, loaded = _load_shared_eval_inputs(settings)
    assert track == [{"event_id": "a"}]
    assert loaded is None


def test_load_shared_eval_inputs_keeps_empty_recs(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    empty = pd.DataFrame()
    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *_args, **_kwargs: [{"event_id": "a"}])
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda _output: empty)
    track, recs = _load_shared_eval_inputs(settings)
    assert track == [{"event_id": "a"}]
    assert recs is empty


def test_load_shared_eval_inputs_loads_dataset_in_parallel(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    seen: dict[str, object] = {}
    empty = pd.DataFrame()

    def _pair(*_args, **kwargs):
        seen["parallel"] = kwargs.get("parallel")
        return [{"event_id": "a"}], empty

    monkeypatch.setattr("cicerone.job._try_load_pair", _pair)
    track, recs = _load_shared_eval_inputs(settings)
    assert seen["parallel"] is True
    assert track == [{"event_id": "a"}]
    assert recs is empty


def test_load_shared_eval_inputs_loads_db_serially(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'job.db'}"}),
    )
    seen: dict[str, object] = {}

    def _pair(*_args, **kwargs):
        seen["parallel"] = kwargs.get("parallel")
        return [{"event_id": "a"}], pd.DataFrame()

    monkeypatch.setattr("cicerone.job._try_load_pair", _pair)
    _load_shared_eval_inputs(settings)
    assert seen["parallel"] is False


def test_load_shared_eval_inputs_does_not_bound_to_thompson_window(tmp_path, monkeypatch):
    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.experiment.store import ExperimentStore, experiment_state
    from cicerone.job import _load_shared_eval_inputs

    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            allocation="thompson",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state(
            "ranking-cvr",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            window_started_at="2026-09-04T00:00:00+00:00",
        )
    )
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    seen: dict[str, object] = {}

    def _read(self, **kwargs):
        seen.update(kwargs)
        return [{"event_id": "old"}]

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", _read)
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda _output: recs)
    track, loaded = _load_shared_eval_inputs(settings)
    assert seen.get("since") is None
    assert seen.get("experiment_id") is None
    assert track == [{"event_id": "old"}]
    assert loaded is recs


def test_load_shared_eval_inputs_scopes_track_when_eval_disabled(tmp_path, monkeypatch):
    from dataclasses import replace

    from conftest import make_settings

    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
    from cicerone.job import _load_shared_eval_inputs

    settings = replace(
        make_settings(
            experiment=ExperimentSettings(
                enabled=True,
                id="ranking-cvr",
                allocation="thompson",
                variants=(
                    VariantSettings(name="control", traffic=0.5),
                    VariantSettings(name="treatment", traffic=0.5),
                ),
            ),
            track=TrackSettings(enabled=True),
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        ),
        track=TrackSettings(enabled=False),
    )
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    seen: dict[str, object] = {}

    def _read(self, **kwargs):
        seen.update(kwargs)
        return [{"event_id": "a", "experiment_id": "ranking-cvr"}]

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", _read)
    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda _output: recs)
    track, loaded = _load_shared_eval_inputs(settings)
    assert seen.get("experiment_id") == "ranking-cvr"
    assert track == [{"event_id": "a", "experiment_id": "ranking-cvr"}]
    assert loaded is recs

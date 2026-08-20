from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from cicerone import job
from cicerone.model import RRF_K

REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _write_config(tmp_path, input_dir, output_dir, top_k: int = 10, extra_job: str = "") -> str:
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
        [output.options]
        storage_backend = "local"
        path = "{output_dir}"
        """
    )
    return str(config_path)


def test_job_run_end_to_end_with_local_dataset_backend(tmp_path, monkeypatch):
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

    config_path = _write_config(tmp_path, input_dir, output_dir, top_k=2)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    job.run()

    recommendations = pd.read_parquet(output_dir / "recommendations.parquet")
    assert set(recommendations["user_id"]) == {"u1", "u2"}

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


def test_job_run_writes_model_artifact_when_enabled(tmp_path, monkeypatch):
    from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, load_artifact, recommend_from_artifact

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


def test_job_run_with_automl_enabled_selects_and_records_best_candidate(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.utcnow()
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

    now = pd.Timestamp.utcnow()
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

    now = pd.Timestamp.utcnow()
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

    now = pd.Timestamp.utcnow()
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

    now = pd.Timestamp.utcnow()
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

    job.run(triggered_by="cron")

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["lock_backend"] == "redis"


def test_job_marks_partial_outputs_when_recommendation_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.utcnow()
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


def test_job_preserves_success_when_manifest_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.utcnow()
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

    def boom(self, manifest):
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

    now = pd.Timestamp.utcnow()
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
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is False
    assert "retrain lock lost" in manifest["error"]


def test_job_marks_partial_outputs_when_fence_lost_after_write(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.utcnow()
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
        return calls["n"] < 2

    with pytest.raises(LockLostError, match="retrain lock lost before write"):
        job.run(fence_check=fence)

    assert (output_dir / "recommendations.parquet").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is True
    assert calls["n"] == 2


def test_run_guard_skips_job_writes_when_owned_is_false(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    now = pd.Timestamp.utcnow()
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
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "retrain lock lost" in manifest["error"]

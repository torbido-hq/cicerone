from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from cicerone import job
from cicerone.blending import COLD_START_USER_ID
from cicerone.job import _recommendation_user_count, _target_user_ids
from cicerone.model import RRF_K

REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _write_config(
    tmp_path, input_dir, output_dir, top_k: int = 10, extra_job: str = "", extra: str = ""
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
        def publish(self, df: pd.DataFrame) -> None:
            captured.append(df.copy())

        def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr("cicerone.job.build_publisher", lambda _settings: _Pub())
    job.run()
    assert len(captured) == 1
    assert {"u1", "u2"}.issubset(set(captured[0]["user_id"].astype(str)))
    assert closed["n"] == 1


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

    monkeypatch.setattr("cicerone.job.load_recommendations_frame", lambda _output: recs)

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


def test_job_run_swallows_eval_persistence_errors(tmp_path, monkeypatch):
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
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval")),
    )
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.append_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("history")),
    )
    job.run()
    assert (output_dir / "recommendations.parquet").exists()


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


def test_job_run_records_publisher_init_failure(tmp_path, monkeypatch):
    from cicerone.config import ConfigError

    config_path = _write_config(tmp_path, tmp_path, tmp_path)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "cicerone.job.build_publisher",
        lambda _settings: (_ for _ in ()).throw(
            ConfigError("publish.options.bootstrap_servers is unreachable")
        ),
    )
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        job.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "bootstrap_servers" in manifest["error"]


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
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is False
    assert "retrain lock lost" in manifest["error"]


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
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["partial_outputs"] is True
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
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "retrain lock lost" in manifest["error"]


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
    assert _select_thompson_recipes(settings, recipes[:1], pd.DataFrame()) == recipes[:1]

    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: (_ for _ in ()).throw(RuntimeError("state")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()) == recipes

    monkeypatch.setattr(
        "cicerone.job.ExperimentStore.read_state",
        lambda self: {"experiment_id": "other", "champion": "control", "challenger": "treatment"},
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()) == recipes

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
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("track")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()) == recipes

    monkeypatch.setattr("cicerone.job.TrackStore.read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "cicerone.job.allocate_thompson",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mab")),
    )
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()) == recipes


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
        lambda self: (_ for _ in ()).throw(RuntimeError("state")),
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
    assert _select_thompson_recipes(settings, recipes, pd.DataFrame()) == recipes
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
        lambda output: (_ for _ in ()).throw(RuntimeError("recs gone")),
    )
    selected = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert [recipe.name for recipe in selected] == ["control", "treatment"]

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
        lambda output: (_ for _ in ()).throw(RuntimeError("catalog gone")),
    )
    again = _select_thompson_recipes(settings, recipes, pd.DataFrame())
    assert [recipe.name for recipe in again] == ["control", "treatment"]

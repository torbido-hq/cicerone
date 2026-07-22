from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cicerone import job

REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _write_config(tmp_path, input_dir, output_dir, top_k: int = 10) -> str:
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = {top_k}
        feature_config_path = "{REPO_FEATURES_CONFIG}"

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


def test_job_run_raises_on_failure(tmp_path, monkeypatch):
    # no events.parquet present in tmp_path -> should fail
    config_path = _write_config(tmp_path, tmp_path, tmp_path)
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config_path)

    with pytest.raises(Exception, match="events.parquet"):
        job.run()

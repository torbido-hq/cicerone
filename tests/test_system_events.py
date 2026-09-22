"""System-style check: webhook ``POST /events`` write-through after a batch job.

Seeds the shared local parquet catalog, runs ``job.run``, then mounts the same
serve / dashboard apps as production — including ``start_events_runtime`` —
so a queued purchase is not visible until the worker flushes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from support.system_spec import (
    DASHBOARD_AUTH,
    SERVE_HEADERS,
    dashboard_client,
    mount_serve_app,
    run_system_job,
    sample_system_catalog,
    seed_dataset_catalog,
    stop_serve_events,
    write_system_config,
)

from cicerone.config import load_settings
from cicerone.events.updater_merge import INCREMENTAL_SOURCE
from cicerone.io.factory import build_manifest_reader

LIVE_USER = "u-live"
LIVE_ITEM = "i2"
LIVE_EVENT_ID = "system-spec-events-1"


@dataclass(frozen=True)
class TrainedEventsSystem:
    config_path: Path
    output_path: Path


@pytest.fixture(scope="module")
def trained_system(tmp_path_factory: pytest.TempPathFactory) -> TrainedEventsSystem:
    root = tmp_path_factory.mktemp("system-spec-events")
    input_path = root / "in"
    output_path = root / "out"
    output_path.mkdir()
    events, users, items = sample_system_catalog()
    seed_dataset_catalog(input_path, events, users, items)
    config_path = write_system_config(
        root / "cicerone.toml",
        kind="dataset",
        input_path=input_path,
        output_path=output_path,
        events_webhook=True,
    )
    run_system_job(config_path, triggered_by="system-spec")
    return TrainedEventsSystem(config_path=config_path, output_path=output_path)


def _settings(trained: TrainedEventsSystem):
    return load_settings(str(trained.config_path))


def test_system_events_webhook_write_through_serve_and_dashboard(
    trained_system: TrainedEventsSystem,
) -> None:
    """202 is a queue ack; serve and dashboard change only after the worker flush."""
    settings = _settings(trained_system)
    dash = dashboard_client(settings, trained_system.config_path)
    empty = dash.get("/dashboard", auth=DASHBOARD_AUTH)
    assert empty.status_code == 200
    assert "Incremental events" in empty.text
    assert "Source: webhook" in empty.text
    assert "No incremental flushes recorded in recent manifests yet." in empty.text

    app = mount_serve_app(settings, start_events_worker=False)
    try:
        from fastapi.testclient import TestClient

        serve = TestClient(app)
        before = serve.get(f"/recommendations/{LIVE_USER}", headers=SERVE_HEADERS)
        assert before.status_code == 200
        assert before.json()["fallback"] is True
        assert INCREMENTAL_SOURCE not in {row["source"] for row in before.json()["items"]}

        queued = serve.post(
            "/events",
            headers=SERVE_HEADERS,
            json={
                "user_id": LIVE_USER,
                "item_id": LIVE_ITEM,
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-09-22T12:00:00Z",
                "event_id": LIVE_EVENT_ID,
            },
        )
        assert queued.status_code == 202
        assert queued.json()["accepted"] == 1
        assert queued.json()["event_ids"] == [LIVE_EVENT_ID]

        still_cold = serve.get(f"/recommendations/{LIVE_USER}", headers=SERVE_HEADERS)
        assert still_cold.status_code == 200
        assert still_cold.json()["fallback"] is True
        assert still_cold.json()["items"] == before.json()["items"]

        still_empty = dash.get("/dashboard", auth=DASHBOARD_AUTH)
        assert still_empty.status_code == 200
        assert "No incremental flushes recorded in recent manifests yet." in still_empty.text

        worker = app.state.events_worker
        assert worker is not None
        assert worker.tick() == 1

        after = serve.get(f"/recommendations/{LIVE_USER}", headers=SERVE_HEADERS)
        assert after.status_code == 200
        served = after.json()
        assert served["fallback"] is False
        incremental = [row for row in served["items"] if row["item_id"] == LIVE_ITEM]
        assert incremental
        assert incremental[0]["source"] == INCREMENTAL_SOURCE
        served_ids = {row["item_id"] for row in served["items"]}
        assert "i3" not in served_ids
        assert "i4" not in served_ids
    finally:
        stop_serve_events(app)

    latest = build_manifest_reader(settings.output).read_latest()
    assert latest is not None
    assert latest["triggered_by"] == "incremental"
    assert latest["status"] == "success"
    assert int(latest["n_events"]) == 1
    assert int(latest["incremental_events_applied"]) == 1

    on_disk = pd.read_parquet(trained_system.output_path / "recommendations.parquet")
    live = on_disk.loc[on_disk["user_id"].astype(str) == LIVE_USER]
    assert not live.empty
    assert LIVE_ITEM in set(live["item_id"].astype(str))
    assert INCREMENTAL_SOURCE in set(live["source"].astype(str))

    flushed = dash.get("/dashboard", auth=DASHBOARD_AUTH)
    assert flushed.status_code == 200
    assert "Incremental events" in flushed.text
    assert "Source: webhook" in flushed.text
    assert "No incremental flushes recorded in recent manifests yet." not in flushed.text
    assert "Events applied" in flushed.text
    assert "<dd>1</dd>" in flushed.text
    assert "success" in flushed.text

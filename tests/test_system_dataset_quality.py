"""System-spec follow-up (dataset): track ingest → second job → Quality.

Isolated from ``test_system_dataset`` so the second ``job.run`` cannot change
the first module's trained catalog mid-suite.

Dataset-specific contracts: ``POST /track`` appends ``track.jsonl`` under the
output path; the conversion purchase is appended to input ``events.parquet``;
the second job *overwrites* ``manifest.json`` (``read_recent`` stays 1).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from support.system_spec import (
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    append_dataset_events,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    run_system_job,
    sample_system_catalog,
    seed_dataset_catalog,
    write_system_config,
)

from cicerone.config import load_settings
from cicerone.io.factory import build_manifest_reader
from cicerone.io.manifest_reader import DatasetManifestReader
from cicerone.track.store import TrackStore
from cicerone.track.store_common import EVAL_FILENAME, HISTORY_DIR, TRACK_FILENAME

_SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
_DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)


@dataclass(frozen=True)
class QualitySystem:
    config_path: Path
    input_path: Path
    output_path: Path
    events: pd.DataFrame


@pytest.fixture(scope="module")
def quality_system(tmp_path_factory: pytest.TempPathFactory) -> Iterator[QualitySystem]:
    root = tmp_path_factory.mktemp("system-dataset-quality")
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
    )
    run_system_job(config_path, triggered_by="system-spec")
    yield QualitySystem(
        config_path=config_path,
        input_path=input_path,
        output_path=output_path,
        events=events,
    )


def _parse_track_eval(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(str(raw or ""))
    if not isinstance(parsed, dict):
        raise AssertionError(f"expected track_eval object, got {type(parsed).__name__}")
    return parsed


def test_system_track_eval_quality_loop(quality_system: QualitySystem) -> None:
    settings = load_settings(str(quality_system.config_path))
    assert settings.output.kind == "dataset"
    serve = TestClient(mount_serve_app(settings))
    served = serve.get("/recommendations/u1", headers=_SERVE_HEADERS)
    assert served.status_code == 200
    body = served.json()
    items = body["items"]
    assert items
    generated_at = body["generated_at"]
    first_item = str(items[0]["item_id"])
    occurred = "2026-09-01T13:00:00Z"
    impressions = [
        {
            "kind": "impression",
            "user_id": "u1",
            "item_id": str(row["item_id"]),
            "rank": int(row["rank"]),
            "occurred_at": occurred,
            "event_id": f"sys-imp-{row['rank']}",
            "generated_at": generated_at,
        }
        for row in items
    ]
    click = {
        "kind": "click",
        "user_id": "u1",
        "item_id": first_item,
        "rank": int(items[0]["rank"]),
        "occurred_at": "2026-09-01T13:05:00Z",
        "event_id": "sys-clk-1",
        "generated_at": generated_at,
    }
    tracked = serve.post("/track", headers=_SERVE_HEADERS, json={"events": [*impressions, click]})
    assert tracked.status_code == 202
    assert tracked.json()["accepted"] == len(impressions) + 1

    track_path = quality_system.output_path / TRACK_FILENAME
    assert track_path.is_file()
    assert not (quality_system.input_path / TRACK_FILENAME).exists()
    tracked_rows = [json.loads(line) for line in track_path.read_text().splitlines() if line.strip()]
    expected_ids = {event["event_id"] for event in impressions} | {"sys-clk-1"}
    assert {row["event_id"] for row in tracked_rows} == expected_ids

    conversion = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": first_item,
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": pd.Timestamp("2026-09-01T14:00:00Z"),
            }
        ]
    )
    append_dataset_events(quality_system.input_path, conversion)
    events_on_disk = pd.read_parquet(quality_system.input_path / "events.parquet")
    assert len(events_on_disk) == len(quality_system.events) + 1
    added = events_on_disk.iloc[len(quality_system.events) :]
    assert list(added["user_id"].astype(str)) == ["u1"]
    assert list(added["item_id"].astype(str)) == [first_item]
    assert list(added["event_type"].astype(str)) == ["purchase"]
    assert not (quality_system.output_path / "events.parquet").exists()

    first_manifest = json.loads((quality_system.output_path / "manifest.json").read_text())
    assert first_manifest["triggered_by"] == "system-spec"

    run_system_job(quality_system.config_path, triggered_by="system-spec-eval")

    store = TrackStore(settings.output)
    assert {row["event_id"] for row in store.read_rows()} == expected_ids

    reader = build_manifest_reader(settings.output)
    assert isinstance(reader, DatasetManifestReader)
    recent = reader.read_recent(limit=5)
    assert len(recent) == 1
    latest = recent[0]
    on_disk = json.loads((quality_system.output_path / "manifest.json").read_text())
    assert latest == on_disk
    assert latest["triggered_by"] == "system-spec-eval"
    assert latest["status"] == "success"
    assert int(latest["n_events"]) == len(quality_system.events) + 1

    track_eval = _parse_track_eval(latest.get("track_eval"))
    overall = track_eval["overall"]
    assert int(overall["n_impressions"]) == len(impressions)
    assert int(overall["n_clicks"]) == 1
    assert overall["ctr"] == pytest.approx(1 / len(impressions))

    eval_path = quality_system.output_path / EVAL_FILENAME
    assert eval_path.is_file()
    eval_file = json.loads(eval_path.read_text())
    assert int(eval_file["track_eval"]["overall"]["n_impressions"]) == len(impressions)

    history_parts = list((quality_system.output_path / HISTORY_DIR).glob("*.parquet"))
    assert history_parts
    history = store.read_history()
    assert not history.empty
    assert "u1" in set(history["user_id"].astype(str))
    assert first_item in set(history["item_id"].astype(str))

    dashboard = TestClient(
        mount_dashboard_app(settings, dashboard_users(), config_path=quality_system.config_path)
    )
    quality = dashboard.get("/dashboard/quality", auth=_DASHBOARD_AUTH)
    assert quality.status_code == 200
    assert "Could not load quality metrics." not in quality.text
    assert "No impressions yet." not in quality.text
    expected_ctr = f"{(1 / len(impressions)) * 100:.2f}%"
    assert re.search(rf"Impressions</dt><dd[^>]*>{len(impressions)}</dd>", quality.text)
    assert re.search(r"Clicks</dt><dd[^>]*>1</dd>", quality.text)
    assert re.search(rf"CTR</dt>\s*<dd[^>]*>{re.escape(expected_ctr)}", quality.text)

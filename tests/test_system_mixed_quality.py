"""Mixed I/O Quality: db catalog + dataset track/eval/manifest overwrite.

Isolated from ``test_system_mixed``. Conversion is appended to Postgres
``events``; ``POST /track`` writes ``track.jsonl`` under the dataset output.
The second job overwrites ``manifest.json`` (``read_recent`` stays 1).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from support.system_db import (
    DASHBOARD_AUTH,
    OUTPUT_DB_TABLES,
    SERVE_HEADERS,
    SKIP_NO_TEST_DB,
    TEST_DATABASE_URL,
    dashboard_client,
    parse_track_eval,
    postgres_ready,
    run_system_job,
    sample_system_catalog,
    seed_catalog,
    serve_client,
    write_system_config,
)

from cicerone.config import load_settings
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.factory import build_manifest_reader
from cicerone.io.manifest_reader import DatasetManifestReader
from cicerone.track.store import TrackStore
from cicerone.track.store_common import EVAL_FILENAME, HISTORY_DIR, TRACK_FILENAME


@dataclass(frozen=True)
class QualitySystem:
    engine: Engine
    config_path: Path
    output_path: Path
    events: pd.DataFrame


@pytest.fixture(scope="module")
def quality_system(
    db_engine: Engine, clean_schema: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[QualitySystem]:
    events, users, items = sample_system_catalog()
    seed_catalog(db_engine, events, users, items)
    output_path = tmp_path_factory.mktemp("system-mixed-quality") / "out"
    output_path.mkdir()
    config_path = write_system_config(
        output_path.parent / "cicerone.toml",
        input_kind="db",
        output_kind="dataset",
        database_url=TEST_DATABASE_URL,
        output_path=output_path,
    )
    run_system_job(config_path, triggered_by="system-spec")
    yield QualitySystem(
        engine=db_engine,
        config_path=config_path,
        output_path=output_path,
        events=events,
    )


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=SKIP_NO_TEST_DB)
def test_system_mixed_track_eval_quality_loop(quality_system: QualitySystem) -> None:
    settings = load_settings(str(quality_system.config_path))
    assert settings.input.kind == "db"
    assert settings.output.kind == "dataset"

    serve = serve_client(settings)
    served = serve.get("/recommendations/u1", headers=SERVE_HEADERS)
    assert served.status_code == 200
    body = served.json()
    items = body["items"]
    assert items
    generated_at = body["generated_at"]
    first_item = str(items[0]["item_id"])
    impressions = [
        {
            "kind": "impression",
            "user_id": "u1",
            "item_id": str(row["item_id"]),
            "rank": int(row["rank"]),
            "occurred_at": "2026-09-01T13:00:00Z",
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
    tracked = serve.post("/track", headers=SERVE_HEADERS, json={"events": [*impressions, click]})
    assert tracked.status_code == 202
    assert tracked.json()["accepted"] == len(impressions) + 1

    track_path = quality_system.output_path / TRACK_FILENAME
    assert track_path.is_file()
    tracked_rows = [json.loads(line) for line in track_path.read_text().splitlines() if line.strip()]
    expected_ids = {event["event_id"] for event in impressions} | {"sys-clk-1"}
    assert {row["event_id"] for row in tracked_rows} == expected_ids
    assert OUTPUT_DB_TABLES.isdisjoint(inspect(quality_system.engine).get_table_names())

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
    postgres_ready(conversion).to_sql(
        DEFAULT_EVENTS_TABLE,
        quality_system.engine,
        if_exists="append",
        index=False,
    )
    events_sql = text(f'SELECT COUNT(*) AS n FROM "{DEFAULT_EVENTS_TABLE}"')
    n_events = int(pd.read_sql(events_sql, quality_system.engine).iloc[0]["n"])
    assert n_events == len(quality_system.events) + 1
    assert not (quality_system.output_path / "events.parquet").exists()

    first_manifest = json.loads((quality_system.output_path / "manifest.json").read_text())
    assert first_manifest["triggered_by"] == "system-spec"

    run_system_job(quality_system.config_path, triggered_by="system-spec-eval")
    assert OUTPUT_DB_TABLES.isdisjoint(inspect(quality_system.engine).get_table_names())

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
    assert int(latest["n_events"]) == len(quality_system.events) + 1

    overall = parse_track_eval(latest.get("track_eval"))["overall"]
    assert int(overall["n_impressions"]) == len(impressions)
    assert int(overall["n_clicks"]) == 1
    assert overall["ctr"] == pytest.approx(1 / len(impressions))

    eval_file = json.loads((quality_system.output_path / EVAL_FILENAME).read_text())
    assert int(eval_file["track_eval"]["overall"]["n_impressions"]) == len(impressions)
    assert list((quality_system.output_path / HISTORY_DIR).glob("*.parquet"))

    quality = dashboard_client(settings, quality_system.config_path).get(
        "/dashboard/quality", auth=DASHBOARD_AUTH
    )
    assert quality.status_code == 200
    expected_ctr = f"{(1 / len(impressions)) * 100:.2f}%"
    assert re.search(rf"Impressions</dt><dd[^>]*>{len(impressions)}</dd>", quality.text)
    assert re.search(r"Clicks</dt><dd[^>]*>1</dd>", quality.text)
    assert re.search(rf"CTR</dt><dd[^>]*>{re.escape(expected_ctr)}</dd>", quality.text)

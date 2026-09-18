"""System-spec follow-up: track ingest → second job → dashboard Quality.

Isolated from ``test_system_db`` so the second ``job.run`` cannot change
the first module's trained catalog mid-suite.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from support.postgres_defaults import resolve_test_database_url
from support.system_db import (
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    postgres_ready,
    reset_schema,
    sample_system_catalog,
    seed_catalog,
    write_system_config,
)

from cicerone import job
from cicerone.config import load_settings
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.track.store import TrackStore

TEST_DATABASE_URL = resolve_test_database_url()

_SKIP_NO_TEST_DB = (
    "TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — start compose postgres "
    "(`docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres`) "
    "and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, "
    "or run via docker-compose.ci.yml"
)

_SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
_DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)


@dataclass(frozen=True)
class QualitySystem:
    engine: Engine
    config_path: Path
    events: pd.DataFrame


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    if not TEST_DATABASE_URL:
        pytest.skip(_SKIP_NO_TEST_DB)
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def quality_system(db_engine: Engine, tmp_path_factory: pytest.TempPathFactory) -> Iterator[QualitySystem]:
    reset_schema(db_engine)
    events, users, items = sample_system_catalog()
    seed_catalog(db_engine, events, users, items)
    config_path = write_system_config(
        tmp_path_factory.mktemp("system-quality") / "cicerone.toml",
        database_url=TEST_DATABASE_URL,
    )
    previous = os.environ.get("CICERONE_CONFIG_PATH")
    os.environ["CICERONE_CONFIG_PATH"] = str(config_path)
    try:
        job.run(triggered_by="system-spec")
        yield QualitySystem(engine=db_engine, config_path=config_path, events=events)
    finally:
        reset_schema(db_engine)
        if previous is None:
            os.environ.pop("CICERONE_CONFIG_PATH", None)
        else:
            os.environ["CICERONE_CONFIG_PATH"] = previous


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=_SKIP_NO_TEST_DB)
def test_system_track_eval_quality_loop(quality_system: QualitySystem) -> None:
    settings = load_settings(str(quality_system.config_path))
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
    postgres_ready(conversion).to_sql(DEFAULT_EVENTS_TABLE, quality_system.engine, if_exists="append", index=False)

    os.environ["CICERONE_CONFIG_PATH"] = str(quality_system.config_path)
    job.run(triggered_by="system-spec-eval")

    store = TrackStore(settings.output)
    rows = store.read_rows()
    assert {row["event_id"] for row in rows} >= {event["event_id"] for event in impressions} | {"sys-clk-1"}

    latest = DbManifestReader({"database_url": TEST_DATABASE_URL}).read_latest()
    assert latest is not None
    assert latest["triggered_by"] == "system-spec-eval"
    assert latest["status"] == "success"
    assert int(latest["n_events"]) == len(quality_system.events) + 1
    track_eval = json.loads(str(latest.get("track_eval") or "{}") or "{}")
    if not track_eval:
        stored = store.read_eval() or {}
        raw = stored.get("track_eval")
        track_eval = raw if isinstance(raw, dict) else {}
    overall = track_eval.get("overall") if isinstance(track_eval, dict) else None
    assert isinstance(overall, dict)
    assert int(overall.get("n_impressions") or 0) >= len(impressions)
    assert int(overall.get("n_clicks") or 0) >= 1
    assert float(overall.get("ctr") or 0) > 0

    history = store.read_history()
    assert history is not None
    assert not history.empty

    dashboard = TestClient(
        mount_dashboard_app(settings, dashboard_users(), config_path=quality_system.config_path)
    )
    quality = dashboard.get("/dashboard/quality", auth=_DASHBOARD_AUTH)
    assert quality.status_code == 200
    assert "Could not load quality metrics." not in quality.text
    assert "No impressions yet." not in quality.text
    assert "Impressions" in quality.text
    assert "CTR" in quality.text
    assert "%" in quality.text

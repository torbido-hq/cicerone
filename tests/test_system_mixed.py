"""System-style check: Postgres catalog in, local parquet recommendations out.

Input and output kinds are independent (README). This spec seeds events in
Postgres, runs the job with dataset output, and asserts serve/dashboard read
the parquet tree — not recommendation tables on the same database.
"""

from __future__ import annotations

import json
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
    run_system_job,
    sample_system_catalog,
    seed_catalog,
    serve_client,
    write_system_config,
)
from support.system_spec import DATASET_OUTPUT_FILES, available_ids_from_files

from cicerone.artifact import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    loads_artifact,
    recommend_from_artifact,
)
from cicerone.config import load_settings
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE, DEFAULT_ITEMS_TABLE, DEFAULT_USERS_TABLE
from cicerone.io.factory import build_manifest_reader, build_output_sink, build_recommendation_reader
from cicerone.io.manifest_reader import DatasetManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader
from cicerone.io.recommendation_reader_common import ITEMS_SNAPSHOT_FILENAME


@dataclass(frozen=True)
class TrainedSystem:
    engine: Engine
    config_path: Path
    output_path: Path
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    rec_reader: DatasetRecommendationReader
    manifest_reader: DatasetManifestReader


@pytest.fixture(scope="module")
def trained_system(
    db_engine: Engine,
    clean_schema: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainedSystem:
    events, users, items = sample_system_catalog()
    seed_catalog(db_engine, events, users, items)
    output_path = tmp_path_factory.mktemp("system-mixed") / "out"
    output_path.mkdir()
    config_path = write_system_config(
        output_path.parent / "cicerone.toml",
        input_kind="db",
        output_kind="dataset",
        database_url=TEST_DATABASE_URL,
        output_path=output_path,
    )
    run_system_job(config_path, triggered_by="system-spec")
    settings = load_settings(str(config_path))
    rec_reader = build_recommendation_reader(settings.output)
    manifest_reader = build_manifest_reader(settings.output)
    assert isinstance(rec_reader, DatasetRecommendationReader)
    assert isinstance(manifest_reader, DatasetManifestReader)
    return TrainedSystem(
        engine=db_engine,
        config_path=config_path,
        output_path=output_path,
        events=events,
        users=users,
        items=items,
        rec_reader=rec_reader,
        manifest_reader=manifest_reader,
    )


def _settings(trained: TrainedSystem):
    return load_settings(str(trained.config_path))


def _output_recs(trained: TrainedSystem) -> pd.DataFrame:
    return pd.read_parquet(trained.output_path / "recommendations.parquet")


def _output_items(trained: TrainedSystem) -> pd.DataFrame:
    return pd.read_parquet(trained.output_path / ITEMS_SNAPSHOT_FILENAME)


def _output_manifest(trained: TrainedSystem) -> dict:
    return json.loads((trained.output_path / "manifest.json").read_text())


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=SKIP_NO_TEST_DB)
def test_system_job_mixed_db_in_dataset_out(trained_system: TrainedSystem) -> None:
    settings = _settings(trained_system)
    assert settings.input.kind == "db"
    assert settings.output.kind == "dataset"
    assert settings.input.options["database_url"] == TEST_DATABASE_URL
    assert Path(settings.output.options["path"]) == trained_system.output_path

    tables = set(inspect(trained_system.engine).get_table_names())
    assert {DEFAULT_EVENTS_TABLE, DEFAULT_USERS_TABLE, DEFAULT_ITEMS_TABLE} <= tables
    assert tables.isdisjoint(OUTPUT_DB_TABLES)
    for name in DATASET_OUTPUT_FILES:
        assert (trained_system.output_path / name).is_file()
    assert not (trained_system.output_path / "events.parquet").exists()

    events_sql = text(f'SELECT COUNT(*) AS n FROM "{DEFAULT_EVENTS_TABLE}"')
    n_events = int(pd.read_sql(events_sql, trained_system.engine).iloc[0]["n"])
    assert n_events == len(trained_system.events)

    recs = _output_recs(trained_system)
    expected_users = set(trained_system.events["user_id"]) | set(trained_system.users["user_id"])
    assert set(recs["user_id"].astype(str)) == expected_users
    u1_file = recs.loc[recs["user_id"].astype(str) == "u1"].sort_values("rank")
    served = trained_system.rec_reader.get_recommendations("u1", k=2)
    assert list(served["item_id"].astype(str)) == list(u1_file["item_id"].astype(str).head(2))

    snapshot = _output_items(trained_system)
    assert bool(snapshot.loc[snapshot["item_id"].astype(str) == "i3", "in_stock"].iloc[0]) is False
    assert bool(snapshot.loc[snapshot["item_id"].astype(str) == "i4", "published"].iloc[0]) is False

    on_disk = _output_manifest(trained_system)
    assert trained_system.manifest_reader.read_latest() == on_disk
    assert trained_system.manifest_reader.read_recent(limit=5) == [on_disk]
    assert on_disk["triggered_by"] == "system-spec"
    assert int(on_disk["n_events"]) == len(trained_system.events)
    assert bool(on_disk["artifact_written"]) is True
    assert int(on_disk["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION

    payload = build_output_sink(settings.output).read_model_artifact()
    assert payload is not None
    assert (trained_system.output_path / ARTIFACT_FILENAME).read_bytes() == payload
    loaded = loads_artifact(payload)
    assert not recommend_from_artifact(loaded, ["u1"], top_k=3).empty


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=SKIP_NO_TEST_DB)
def test_system_serve_http_reads_dataset_output(trained_system: TrainedSystem) -> None:
    settings = _settings(trained_system)
    expected_u1 = available_ids_from_files(trained_system.output_path, "u1", settings=settings)
    expected_u4 = available_ids_from_files(trained_system.output_path, "u4", settings=settings)
    generated_at = _output_manifest(trained_system)["generated_at"]
    client = serve_client(settings)

    response = client.get("/recommendations/u1", headers=SERVE_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is False
    assert [row["item_id"] for row in body["items"]] == expected_u1
    assert body["generated_at"] == generated_at
    assert response.headers.get("X-Generated-At") == generated_at

    warm = client.get("/recommendations/u4", headers=SERVE_HEADERS)
    assert warm.status_code == 200
    assert warm.json()["fallback"] is False
    assert [row["item_id"] for row in warm.json()["items"]] == expected_u4

    unknown = client.get("/recommendations/u-unknown", headers=SERVE_HEADERS)
    assert unknown.status_code == 200
    assert unknown.json()["fallback"] is True
    assert unknown.json()["items"]

    beer = client.get("/recommendations/u1?category=beer", headers=SERVE_HEADERS)
    beer_ids = {row["item_id"] for row in beer.json()["items"]}
    assert beer_ids
    assert beer_ids <= {"i1", "i2"}

    filtered = client.get(
        "/recommendations/u1?exclude_unavailable=true&limit=10",
        headers=SERVE_HEADERS,
    )
    filtered_ids = {row["item_id"] for row in filtered.json()["items"]}
    assert filtered_ids
    assert "i3" not in filtered_ids
    assert "i4" not in filtered_ids


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=SKIP_NO_TEST_DB)
def test_system_dashboard_http_matches_dataset_output(trained_system: TrainedSystem) -> None:
    settings = _settings(trained_system)
    serve_ids = available_ids_from_files(trained_system.output_path, "u1", settings=settings)
    manifest = _output_manifest(trained_system)
    client = dashboard_client(settings, trained_system.config_path)
    status = client.get("/dashboard", auth=DASHBOARD_AUTH)
    assert status.status_code == 200
    assert manifest["triggered_by"] in status.text
    assert str(manifest["n_events"]) in status.text
    lookup = client.get("/dashboard", params={"user_id": "u1"}, auth=DASHBOARD_AUTH)
    assert lookup.status_code == 200
    for item_id in serve_ids:
        assert item_id in lookup.text

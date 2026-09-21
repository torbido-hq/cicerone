"""System-style end-to-end check against local parquet (dataset I/O).

Same operator journeys as ``test_system_db``, but assertions are on the
files the dataset backend actually writes: input parquet stays in ``in/``,
output is ``recommendations.parquet`` / ``items_snapshot.parquet`` /
``manifest.json`` / ``model.artifact``, and ``read_recent`` is that one
manifest (latest-only). No live database required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from support.system_spec import (
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    available_recommendation_ids,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    run_system_job,
    sample_system_catalog,
    seed_dataset_catalog,
    write_system_config,
)

from cicerone.artifact import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    loads_artifact,
    recommend_from_artifact,
)
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import load_settings
from cicerone.feature_config import load_feature_config
from cicerone.io.factory import build_manifest_reader, build_output_sink, build_recommendation_reader
from cicerone.io.manifest_reader import DatasetManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader
from cicerone.io.recommendation_reader_common import ITEMS_SNAPSHOT_FILENAME

_SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
_DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)
_OUTPUT_FILES = (
    "recommendations.parquet",
    ITEMS_SNAPSHOT_FILENAME,
    "manifest.json",
    ARTIFACT_FILENAME,
)


@dataclass(frozen=True)
class TrainedSystem:
    config_path: Path
    input_path: Path
    output_path: Path
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    rec_reader: DatasetRecommendationReader
    manifest_reader: DatasetManifestReader


@pytest.fixture(scope="module")
def trained_system(tmp_path_factory: pytest.TempPathFactory) -> TrainedSystem:
    root = tmp_path_factory.mktemp("system-dataset")
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
    settings = load_settings(str(config_path))
    rec_reader = build_recommendation_reader(settings.output)
    manifest_reader = build_manifest_reader(settings.output)
    assert isinstance(rec_reader, DatasetRecommendationReader)
    assert isinstance(manifest_reader, DatasetManifestReader)
    return TrainedSystem(
        config_path=config_path,
        input_path=input_path,
        output_path=output_path,
        events=events,
        users=users,
        items=items,
        rec_reader=rec_reader,
        manifest_reader=manifest_reader,
    )


def _settings(trained: TrainedSystem):
    return load_settings(str(trained.config_path))


def _serve_client(trained: TrainedSystem) -> TestClient:
    return TestClient(mount_serve_app(_settings(trained)))


def _dashboard_client(trained: TrainedSystem) -> TestClient:
    app = mount_dashboard_app(_settings(trained), dashboard_users(), config_path=trained.config_path)
    return TestClient(app)


def _output_recs(trained: TrainedSystem) -> pd.DataFrame:
    return pd.read_parquet(trained.output_path / "recommendations.parquet")


def _output_items(trained: TrainedSystem) -> pd.DataFrame:
    return pd.read_parquet(trained.output_path / ITEMS_SNAPSHOT_FILENAME)


def _output_manifest(trained: TrainedSystem) -> dict:
    return json.loads((trained.output_path / "manifest.json").read_text())


def _available_ids_from_files(
    trained: TrainedSystem,
    user_id: str,
    *,
    k: int | None = None,
) -> list[str]:
    settings = _settings(trained)
    feature_config = load_feature_config(settings.feature_config_path)
    recs = _output_recs(trained)
    user_rows = recs.loc[recs["user_id"].astype(str) == user_id]
    return available_recommendation_ids(
        user_rows,
        _output_items(trained),
        availability_filters=feature_config.item_availability_filters,
        category_column=settings.serve.category_column,
        k=settings.serve.default_k if k is None else k,
    )


def test_system_job_dataset_round_trip_with_artifact_and_readers(trained_system: TrainedSystem) -> None:
    """Local parquet catalog → job.run → the files and readers serve/dashboard use."""
    settings = _settings(trained_system)
    assert settings.input.kind == "dataset"
    assert settings.output.kind == "dataset"
    assert Path(settings.input.options["path"]) == trained_system.input_path
    assert Path(settings.output.options["path"]) == trained_system.output_path
    assert trained_system.input_path != trained_system.output_path

    for name in ("events.parquet", "users.parquet", "items.parquet"):
        assert (trained_system.input_path / name).is_file()
        assert not (trained_system.output_path / name).exists()
    for name in _OUTPUT_FILES:
        assert (trained_system.output_path / name).is_file()
        assert not (trained_system.input_path / name).exists()

    seeded_events = pd.read_parquet(trained_system.input_path / "events.parquet")
    assert len(seeded_events) == len(trained_system.events)
    assert set(seeded_events["user_id"].astype(str)) == set(trained_system.events["user_id"])

    recs = _output_recs(trained_system)
    expected_users = set(trained_system.events["user_id"]) | set(trained_system.users["user_id"])
    rec_users = set(recs["user_id"].astype(str))
    assert expected_users <= rec_users
    assert rec_users <= expected_users | {COLD_START_USER_ID}
    assert set(recs.columns) >= {"user_id", "item_id", "rank", "score", "source"}

    snapshot = _output_items(trained_system)
    assert set(snapshot["item_id"].astype(str)) == {"i1", "i2", "i3", "i4"}
    wine = snapshot.loc[snapshot["item_id"].astype(str) == "i3"].iloc[0]
    assert bool(wine["in_stock"]) is False
    unpublished = snapshot.loc[snapshot["item_id"].astype(str) == "i4"].iloc[0]
    assert bool(unpublished["published"]) is False

    u1_file = recs.loc[recs["user_id"].astype(str) == "u1"].sort_values("rank")
    served = trained_system.rec_reader.get_recommendations("u1", k=2)
    assert list(served["item_id"].astype(str)) == list(u1_file["item_id"].astype(str).head(2))
    assert list(served["rank"]) == [1, 2]

    on_disk = _output_manifest(trained_system)
    latest = trained_system.manifest_reader.read_latest()
    assert latest == on_disk
    assert on_disk["status"] == "success"
    assert on_disk["triggered_by"] == "system-spec"
    assert int(on_disk["n_events"]) == len(trained_system.events)
    assert bool(on_disk["artifact_written"]) is True
    assert int(on_disk["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION
    recent = trained_system.manifest_reader.read_recent(limit=5)
    assert recent == [on_disk]

    payload = build_output_sink(settings.output).read_model_artifact()
    assert payload is not None
    assert (trained_system.output_path / ARTIFACT_FILENAME).read_bytes() == payload
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models
    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    artifact_users = set(from_artifact["user_id"].astype(str))
    assert {"u1", "u2"} <= artifact_users
    assert artifact_users <= {"u1", "u2", COLD_START_USER_ID}
    assert not from_artifact.empty


def test_system_serve_http_reads_job_output(trained_system: TrainedSystem) -> None:
    expected_u1 = _available_ids_from_files(trained_system, "u1")
    expected_u4 = _available_ids_from_files(trained_system, "u4")
    manifest = _output_manifest(trained_system)
    generated_at = manifest["generated_at"]
    client = _serve_client(trained_system)

    response = client.get("/recommendations/u1", headers=_SERVE_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["fallback"] is False
    assert [row["item_id"] for row in body["items"]] == expected_u1
    assert response.headers.get("X-Generated-At") == generated_at
    assert body["generated_at"] == generated_at

    warm = client.get("/recommendations/u4", headers=_SERVE_HEADERS)
    assert warm.status_code == 200
    assert warm.json()["fallback"] is False
    assert [row["item_id"] for row in warm.json()["items"]] == expected_u4

    unknown = client.get("/recommendations/u-unknown", headers=_SERVE_HEADERS)
    assert unknown.status_code == 200
    assert unknown.json()["fallback"] is True
    unknown_ids = [row["item_id"] for row in unknown.json()["items"]]
    assert unknown_ids
    snapshot = _output_items(trained_system)
    rec_ids = set(_output_recs(trained_system)["item_id"].astype(str))
    available = snapshot.loc[
        snapshot["published"].astype(bool) & snapshot["in_stock"].astype(bool), "item_id"
    ].astype(str)
    assert set(unknown_ids) <= set(available) & rec_ids

    beer = client.get("/recommendations/u1?category=beer", headers=_SERVE_HEADERS)
    assert beer.status_code == 200
    beer_ids = [row["item_id"] for row in beer.json()["items"]]
    beer_catalog = set(snapshot.loc[snapshot["category"].astype(str) == "beer", "item_id"].astype(str))
    assert beer_ids
    assert set(beer_ids) <= beer_catalog & {"i1", "i2"}

    filtered = client.get(
        "/recommendations/u1?exclude_unavailable=true&limit=10",
        headers=_SERVE_HEADERS,
    )
    assert filtered.status_code == 200
    filtered_ids = [row["item_id"] for row in filtered.json()["items"]]
    assert filtered_ids
    assert "i3" not in filtered_ids
    assert "i4" not in filtered_ids
    assert set(filtered_ids) <= set(available)


def test_system_dashboard_http_matches_serve(trained_system: TrainedSystem) -> None:
    serve_ids = _available_ids_from_files(trained_system, "u1")
    manifest = _output_manifest(trained_system)

    client = _dashboard_client(trained_system)
    status = client.get("/dashboard", auth=_DASHBOARD_AUTH)
    assert status.status_code == 200
    assert manifest["status"] in status.text
    assert manifest["triggered_by"] in status.text
    assert str(manifest["n_events"]) in status.text

    lookup = client.get("/dashboard", params={"user_id": "u1"}, auth=_DASHBOARD_AUTH)
    assert lookup.status_code == 200
    assert "Recommendations for" in lookup.text
    assert "u1" in lookup.text
    for item_id in serve_ids:
        assert item_id in lookup.text

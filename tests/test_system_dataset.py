"""System-style end-to-end check against local parquet (dataset I/O).

Same journeys as ``test_system_db``: seed events/users/items, run the full
batch job with dataset input/output + a model artifact, then hit the same
serve and dashboard HTTP apps production uses.

Dataset ``read_recent`` is latest-only (0–1). Artifact is the output sink
file, not a Postgres table. No live database required.
"""

from __future__ import annotations

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

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.config import load_settings
from cicerone.feature_config import load_feature_config
from cicerone.io.base import ManifestReader, RecommendationReader
from cicerone.io.factory import build_manifest_reader, build_output_sink, build_recommendation_reader

_SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
_DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)


@dataclass(frozen=True)
class TrainedSystem:
    config_path: Path
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    rec_reader: RecommendationReader
    manifest_reader: ManifestReader


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
    return TrainedSystem(
        config_path=config_path,
        events=events,
        users=users,
        items=items,
        rec_reader=build_recommendation_reader(settings.output),
        manifest_reader=build_manifest_reader(settings.output),
    )


def _settings(trained: TrainedSystem):
    return load_settings(str(trained.config_path))


def _serve_client(trained: TrainedSystem) -> TestClient:
    return TestClient(mount_serve_app(_settings(trained)))


def _dashboard_client(trained: TrainedSystem) -> TestClient:
    app = mount_dashboard_app(_settings(trained), dashboard_users(), config_path=trained.config_path)
    return TestClient(app)


def test_system_job_dataset_round_trip_with_artifact_and_readers(trained_system: TrainedSystem) -> None:
    """Local parquet catalog → job.run → recommendations/manifest/artifact readers."""
    expected_users = set(trained_system.events["user_id"]) | set(trained_system.users["user_id"])
    rec_reader = trained_system.rec_reader
    for user_id in sorted(expected_users):
        served_all = rec_reader.get_recommendations(user_id, k=10)
        assert not served_all.empty, f"expected recommendations for {user_id}"
        assert set(served_all.columns) >= {"user_id", "item_id", "rank", "score", "source"}
        assert served_all["rank"].min() >= 1

    served = rec_reader.get_recommendations("u1", k=2)
    assert len(served) == 2
    assert set(served["user_id"]) == {"u1"}
    assert list(served["rank"]) == sorted(served["rank"].tolist())

    latest = trained_system.manifest_reader.read_latest()
    assert latest is not None
    assert latest["status"] == "success"
    assert latest["triggered_by"] == "system-spec"
    assert int(latest["n_events"]) == len(trained_system.events)
    assert bool(latest["artifact_written"]) is True
    assert int(latest["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION
    recent = trained_system.manifest_reader.read_recent(limit=5)
    assert len(recent) == 1

    payload = build_output_sink(_settings(trained_system).output).read_model_artifact()
    assert payload is not None
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models

    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    assert not from_artifact.empty
    assert set(from_artifact["user_id"]) <= {"u1", "u2"}


def test_system_serve_http_reads_job_output(trained_system: TrainedSystem) -> None:
    settings = _settings(trained_system)
    feature_config = load_feature_config(settings.feature_config_path)
    raw = trained_system.rec_reader.get_recommendations("u1", k=10)
    expected_ids = available_recommendation_ids(
        raw,
        trained_system.items,
        availability_filters=feature_config.item_availability_filters,
        category_column=settings.serve.category_column,
        k=settings.serve.default_k,
    )
    latest = trained_system.manifest_reader.read_latest()
    assert latest is not None
    client = _serve_client(trained_system)

    response = client.get("/recommendations/u1", headers=_SERVE_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["fallback"] is False
    served_ids = [row["item_id"] for row in body["items"]]
    assert served_ids == expected_ids
    generated_at = latest.get("generated_at")
    if generated_at:
        assert response.headers.get("X-Generated-At") == str(generated_at)
        assert body["generated_at"] == str(generated_at)

    warm = client.get("/recommendations/u4", headers=_SERVE_HEADERS)
    assert warm.status_code == 200
    assert warm.json()["items"]
    assert warm.json()["fallback"] is False

    unknown = client.get("/recommendations/u-unknown", headers=_SERVE_HEADERS)
    assert unknown.status_code == 200
    assert unknown.json()["fallback"] is True
    assert unknown.json()["items"]

    beer = client.get("/recommendations/u1?category=beer", headers=_SERVE_HEADERS)
    assert beer.status_code == 200
    beer_ids = {row["item_id"] for row in beer.json()["items"]}
    assert beer_ids <= {"i1", "i2"}

    filtered = client.get(
        "/recommendations/u1?exclude_unavailable=true&limit=10",
        headers=_SERVE_HEADERS,
    )
    assert filtered.status_code == 200
    filtered_ids = {row["item_id"] for row in filtered.json()["items"]}
    assert "i3" not in filtered_ids
    assert "i4" not in filtered_ids


def test_system_dashboard_http_matches_serve(trained_system: TrainedSystem) -> None:
    serve = _serve_client(trained_system).get("/recommendations/u1", headers=_SERVE_HEADERS)
    assert serve.status_code == 200
    serve_ids = [row["item_id"] for row in serve.json()["items"]]

    client = _dashboard_client(trained_system)
    status = client.get("/dashboard", auth=_DASHBOARD_AUTH)
    assert status.status_code == 200
    assert "success" in status.text
    assert "system-spec" in status.text
    assert str(len(trained_system.events)) in status.text

    lookup = client.get("/dashboard", params={"user_id": "u1"}, auth=_DASHBOARD_AUTH)
    assert lookup.status_code == 200
    assert "Recommendations for" in lookup.text
    assert "u1" in lookup.text
    for item_id in serve_ids:
        assert item_id in lookup.text

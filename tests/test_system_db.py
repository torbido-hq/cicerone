"""System-style end-to-end check against a real Postgres (Rails system-spec analogue).

Seeds events/users/items, runs the full batch job with db input/output + a
model artifact, then hits the same serve and dashboard HTTP apps production
uses — FastAPI TestClient over the SQLAlchemy stores.

Requires a test DB URL via ``TEST_DATABASE_URL`` or ``POSTGRES_TEST_HOST``
(see ``support.postgres_defaults`` / CONTRIBUTING.md). Schema resets are
gated by ``support.system_db.reset_schema``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from support.postgres_defaults import resolve_test_database_url
from support.system_db import (
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    reset_schema,
    sample_system_catalog,
    seed_catalog,
    write_system_config,
)

from cicerone import job
from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.config import load_settings
from cicerone.io.db_store import DEFAULT_MODEL_ARTIFACT_TABLE
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.io.recommendation_reader import DbRecommendationReader

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
class TrainedSystem:
    engine: Engine
    config_path: Path
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    rec_reader: DbRecommendationReader
    manifest_reader: DbManifestReader


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """One Engine for the whole test session — avoids per-test connect/dispose."""
    if not TEST_DATABASE_URL:
        pytest.skip(_SKIP_NO_TEST_DB)
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def clean_schema(db_engine: Engine) -> Iterator[None]:
    """Reset schema once around the system-spec."""
    reset_schema(db_engine)
    yield
    reset_schema(db_engine)


@pytest.fixture(scope="module")
def trained_system(
    db_engine: Engine,
    clean_schema: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainedSystem:
    events, users, items = sample_system_catalog()
    seed_catalog(db_engine, events, users, items)
    config_path = write_system_config(
        tmp_path_factory.mktemp("system-spec") / "cicerone.toml",
        database_url=TEST_DATABASE_URL,
    )
    previous = os.environ.get("CICERONE_CONFIG_PATH")
    os.environ["CICERONE_CONFIG_PATH"] = str(config_path)
    try:
        job.run(triggered_by="system-spec")
    finally:
        if previous is None:
            os.environ.pop("CICERONE_CONFIG_PATH", None)
        else:
            os.environ["CICERONE_CONFIG_PATH"] = previous
    return TrainedSystem(
        engine=db_engine,
        config_path=config_path,
        events=events,
        users=users,
        items=items,
        rec_reader=DbRecommendationReader({"database_url": TEST_DATABASE_URL}),
        manifest_reader=DbManifestReader({"database_url": TEST_DATABASE_URL}),
    )


def _settings(trained: TrainedSystem):
    return load_settings(str(trained.config_path))


def _serve_client(trained: TrainedSystem) -> TestClient:
    return TestClient(mount_serve_app(_settings(trained)))


def _dashboard_client(trained: TrainedSystem) -> TestClient:
    app = mount_dashboard_app(_settings(trained), dashboard_users(), config_path=trained.config_path)
    return TestClient(app)


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=_SKIP_NO_TEST_DB)
def test_system_job_db_round_trip_with_artifact_and_readers(trained_system: TrainedSystem) -> None:
    """Postgres catalog → job.run → recommendations/manifest/artifact readers."""
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

    artifacts = pd.read_sql(
        text(f'SELECT payload FROM "{DEFAULT_MODEL_ARTIFACT_TABLE}"'),
        trained_system.engine,
    )
    assert len(artifacts) == 1
    payload = bytes(artifacts.iloc[0]["payload"])
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models

    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    assert not from_artifact.empty
    assert set(from_artifact["user_id"]) <= {"u1", "u2"}


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=_SKIP_NO_TEST_DB)
def test_system_serve_http_reads_job_output(trained_system: TrainedSystem) -> None:
    reader_ids = list(trained_system.rec_reader.get_recommendations("u1", k=10)["item_id"].astype(str))
    latest = trained_system.manifest_reader.read_latest()
    assert latest is not None
    client = _serve_client(trained_system)

    response = client.get("/recommendations/u1", headers=_SERVE_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["fallback"] is False
    served_ids = [row["item_id"] for row in body["items"]]
    assert served_ids == reader_ids[: len(served_ids)]
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


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=_SKIP_NO_TEST_DB)
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

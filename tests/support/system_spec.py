"""Shared helpers for system-style end-to-end specs (db, dataset, events).

Catalog, TOML, HTTP mounts, and dataset parquet seed live here so the
scenario modules stay focused on the journeys.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone import job
from cicerone.artifact import ARTIFACT_FILENAME
from cicerone.io.recommendation_reader_common import ITEMS_SNAPSHOT_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_FEATURES_CONFIG = REPO_ROOT / "config" / "features.toml"
SYSTEM_SERVE_TOKEN = "system-spec-secret"
SYSTEM_DASHBOARD_USER = "alice"
SYSTEM_DASHBOARD_PASSWORD = "s3cret"
SKIP_NO_TEST_DB = (
    "TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — start compose postgres "
    "(`docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres`) "
    "and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, "
    "or run via docker-compose.ci.yml"
)
SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)
DATASET_OUTPUT_FILES = (
    "recommendations.parquet",
    ITEMS_SNAPSHOT_FILENAME,
    "manifest.json",
    ARTIFACT_FILENAME,
)


def sample_system_catalog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared events/users/items used by both I/O backends."""
    now = pd.Timestamp("2026-09-01T12:00:00Z")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 3, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i2", "event_type": "cart_add", "quantity": 1, "occurred_at": now},
        ]
    )
    users = pd.DataFrame(
        [
            {"user_id": "u1", "favorite_styles": ["ipa", "stout"], "region_slug": "lazio"},
            {"user_id": "u2", "favorite_styles": ["lager"], "region_slug": "toscana"},
            {"user_id": "u3", "favorite_styles": [], "region_slug": None},
            {"user_id": "u4", "favorite_styles": ["ipa"], "region_slug": "lazio"},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": False},
            {"item_id": "i4", "category": "wine", "producer_id": "p3", "published": False, "in_stock": True},
        ]
    )
    return events, users, items


def seed_dataset_catalog(
    input_path: Path,
    events: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> None:
    """Write the shared catalog as local parquet the dataset input source reads."""
    input_path.mkdir(parents=True, exist_ok=True)
    events.to_parquet(input_path / "events.parquet", index=False)
    users.to_parquet(input_path / "users.parquet", index=False)
    items.to_parquet(input_path / "items.parquet", index=False)


def append_dataset_events(input_path: Path, extra: pd.DataFrame) -> None:
    """Append rows to the local dataset events.parquet (quality conversion)."""
    path = input_path / "events.parquet"
    existing = pd.read_parquet(path)
    pd.concat([existing, extra], ignore_index=True).to_parquet(path, index=False)


def _io_toml(role: str, kind: str, *, database_url: str | None, path: Path | str | None) -> str:
    if kind == "db":
        if not database_url:
            raise ValueError(f"database_url is required when {role} kind='db'")
        return f"""
        [{role}]
        kind = "db"
        [{role}.options]
        database_url = "{database_url}"
        """
    if kind == "dataset":
        option = "input_path" if role == "input" else "output_path"
        if path is None:
            raise ValueError(f"{option} is required when {role} kind='dataset'")
        return f"""
        [{role}]
        kind = "dataset"
        [{role}.options]
        storage_backend = "local"
        path = "{Path(path)}"
        """
    raise ValueError(f"Unknown system-spec {role} kind: {kind!r}")


def write_system_config(
    path: Path,
    *,
    kind: str = "db",
    input_kind: str | None = None,
    output_kind: str | None = None,
    database_url: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    feature_config_path: Path | str = REPO_FEATURES_CONFIG,
    serve_token: str = SYSTEM_SERVE_TOKEN,
    events_webhook: bool = False,
) -> Path:
    """Write the shared system-spec TOML. ``input_kind`` / ``output_kind`` override ``kind``."""
    io_blocks = _io_toml(
        "input",
        input_kind or kind,
        database_url=database_url,
        path=input_path,
    ) + _io_toml(
        "output",
        output_kind or kind,
        database_url=database_url,
        path=output_path,
    )
    events_block = ""
    if events_webhook:
        events_block = """
        [events]
        enabled = true
        kind = "webhook"

        [events.incremental]
        batch_size = 1
        batch_window_seconds = 60
        poll_interval_seconds = 60
        """

    path.write_text(
        f"""
        [job]
        top_k = 3
        feature_config_path = "{feature_config_path}"
        models = ["collaborative", "popular"]
        save_model_artifact = true

        [job.eval]
        enabled = true

        {io_blocks}

        [serve]
        auth_token = "{serve_token}"
        category_column = "category"
        default_k = 3

        [dashboard]
        enabled = true

        [track]
        enabled = true
        {events_block}
        """
    )
    return path


def run_system_job(config_path: Path, *, triggered_by: str) -> None:
    previous = os.environ.get("CICERONE_CONFIG_PATH")
    os.environ["CICERONE_CONFIG_PATH"] = str(config_path)
    try:
        job.run(triggered_by=triggered_by)
    finally:
        if previous is None:
            os.environ.pop("CICERONE_CONFIG_PATH", None)
        else:
            os.environ["CICERONE_CONFIG_PATH"] = previous


def available_recommendation_ids(
    recs: pd.DataFrame,
    items: pd.DataFrame | None,
    *,
    availability_filters: Sequence[str],
    category_column: str = "category",
    k: int | None = None,
) -> list[str]:
    """Item ids after the same availability filter serve applies by default."""
    from cicerone.serve.item_filters import available_item_ids, filter_recommendations

    filtered = filter_recommendations(
        recs,
        items=items,
        available_ids=available_item_ids(items, availability_filters) if items is not None else None,
        category=None,
        category_column=category_column,
        exclude_unavailable=True,
    )
    ids = list(filtered["item_id"].astype(str))
    return ids if k is None else ids[:k]


def dashboard_users(
    username: str = SYSTEM_DASHBOARD_USER,
    password: str = SYSTEM_DASHBOARD_PASSWORD,
) -> dict[str, str]:
    import bcrypt

    return {username: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")}


def mount_serve_app(settings: Any):
    from cicerone.feature_config import load_feature_config
    from cicerone.io.factory import build_manifest_reader, build_recommendation_reader
    from cicerone.serve import create_app
    from cicerone.serve.bootstrap_events import start_events_runtime
    from cicerone.serve.item_filters import configure_reader_item_filters

    reader = build_recommendation_reader(settings.output)
    feature_path = Path(settings.feature_config_path)
    feature_config = load_feature_config(feature_path) if feature_path.is_file() else None
    availability = list(feature_config.item_availability_filters) if feature_config else []
    configure_reader_item_filters(
        reader,
        category_column=settings.serve.category_column,
        availability_filters=availability,
    )
    events_runtime = start_events_runtime(settings, feature_config=feature_config, reader=reader)
    app = create_app(
        settings,
        reader,
        manifest_reader=build_manifest_reader(settings.output),
        feature_config=feature_config,
        event_source=events_runtime.webhook_source,
        events_worker=events_runtime.worker,
    )
    app.state.events_runtime = events_runtime
    return app


def mount_dashboard_app(
    settings: Any,
    users: dict[str, str],
    *,
    config_path: str | Path | None = None,
):
    from cicerone.dashboard import create_app
    from cicerone.io.factory import (
        build_manifest_reader,
        build_recommendation_reader,
        build_user_history_reader,
    )

    return create_app(
        settings,
        build_manifest_reader(settings.output),
        users,
        build_recommendation_reader(settings.output),
        build_user_history_reader(settings.input),
        config_path=None if config_path is None else str(config_path),
    )


def stop_serve_events(app: Any) -> None:
    runtime = getattr(app.state, "events_runtime", None)
    if runtime is not None:
        runtime.stop()


def serve_client(settings: Any):
    from fastapi.testclient import TestClient

    return TestClient(mount_serve_app(settings))


def dashboard_client(settings: Any, config_path: Path | str | None = None):
    from fastapi.testclient import TestClient

    return TestClient(mount_dashboard_app(settings, dashboard_users(), config_path=config_path))


def parse_track_eval(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(str(raw or "") or "{}")
    if not isinstance(parsed, dict):
        raise AssertionError(f"expected track_eval object, got {type(parsed).__name__}")
    return parsed


def available_ids_from_files(
    output_path: Path,
    user_id: str,
    *,
    settings: Any,
    k: int | None = None,
) -> list[str]:
    from cicerone.feature_config import load_feature_config

    feature_config = load_feature_config(settings.feature_config_path)
    recs = pd.read_parquet(output_path / "recommendations.parquet")
    items = pd.read_parquet(output_path / ITEMS_SNAPSHOT_FILENAME)
    user_rows = recs.loc[recs["user_id"].astype(str) == user_id]
    return available_recommendation_ids(
        user_rows,
        items,
        availability_filters=feature_config.item_availability_filters,
        category_column=settings.serve.category_column,
        k=settings.serve.default_k if k is None else k,
    )

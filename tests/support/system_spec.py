"""Shared helpers for system-style end-to-end specs (db and local dataset).

Catalog, TOML, HTTP mounts, and dataset parquet seed live here so the
scenario modules stay focused on the journeys.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone import job

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_FEATURES_CONFIG = REPO_ROOT / "config" / "features.toml"
SYSTEM_SERVE_TOKEN = "system-spec-secret"
SYSTEM_DASHBOARD_USER = "alice"
SYSTEM_DASHBOARD_PASSWORD = "s3cret"


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


def write_system_config(
    path: Path,
    *,
    kind: str = "db",
    database_url: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    feature_config_path: Path | str = REPO_FEATURES_CONFIG,
    serve_token: str = SYSTEM_SERVE_TOKEN,
) -> Path:
    """Write the shared system-spec TOML (artifact, serve, dashboard, track, eval)."""
    if kind == "db":
        if not database_url:
            raise ValueError("database_url is required when kind='db'")
        io_blocks = f"""
        [input]
        kind = "db"
        [input.options]
        database_url = "{database_url}"

        [output]
        kind = "db"
        [output.options]
        database_url = "{database_url}"
        """
    elif kind == "dataset":
        if input_path is None or output_path is None:
            raise ValueError("input_path and output_path are required when kind='dataset'")
        io_blocks = f"""
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{Path(input_path)}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{Path(output_path)}"
        """
    else:
        raise ValueError(f"Unknown system-spec kind: {kind!r}")

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
    return create_app(
        settings,
        reader,
        manifest_reader=build_manifest_reader(settings.output),
        feature_config=feature_config,
    )


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

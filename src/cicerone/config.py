"""Configuration for the cicerone recommender job.

Everything is read from environment variables (set in docker-compose.yml /
.env). No secrets are hardcoded and nothing is read from the local host —
this module only exists inside the container.

Input and output are each independently configurable as either:
  - "dataset": static parquet files, on an S3-compatible store (R2, AWS S3,
               MinIO, ...) or on the local filesystem, or
  - "db":      a database table/query (e.g. reading straight from a torbido
               read-replica, or writing recommendations into a Postgres
               table torbido's own app can query).

This means input and output don't have to match: e.g. read events straight
from a Postgres replica while writing recommendations back out as parquet
to R2, or any other combination.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class DatasetLocation:
    """A static-file data location: S3-compatible object storage or local disk."""

    backend: str  # "s3" | "local"
    # backend == "s3"
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    bucket: str | None = None
    prefix: str = ""
    # backend == "local"
    path: str | None = None


@dataclass(frozen=True)
class DatabaseLocation:
    """A database data location (SQLAlchemy connection string)."""

    url: str
    events_table: str = "events"
    users_table: str = "users"
    items_table: str = "items"
    recommendations_table: str = "recommendations"
    manifest_table: str = "recommendation_runs"
    # Optional raw SQL overrides — use these to read straight from torbido's
    # own schema instead of requiring it to materialize events/users/items
    # tables verbatim (e.g. JOIN orders+order_items+reviews into one query).
    events_query: str | None = None
    users_query: str | None = None
    items_query: str | None = None


@dataclass(frozen=True)
class IOSettings:
    kind: str  # "dataset" | "db"
    dataset: DatasetLocation | None
    database: DatabaseLocation | None


@dataclass(frozen=True)
class Settings:
    input: IOSettings
    output: IOSettings
    feature_config_path: str
    top_k: int
    half_life_days: float


def _load_dataset_location(prefix: str) -> DatasetLocation:
    backend = os.environ.get(f"{prefix}_STORAGE_BACKEND", "s3").lower()
    if backend == "local":
        return DatasetLocation(
            backend="local",
            path=_require(f"{prefix}_LOCAL_PATH"),
        )
    if backend == "s3":
        return DatasetLocation(
            backend="s3",
            endpoint_url=_require(f"{prefix}_S3_ENDPOINT_URL"),
            access_key_id=_require(f"{prefix}_S3_ACCESS_KEY_ID"),
            secret_access_key=_require(f"{prefix}_S3_SECRET_ACCESS_KEY"),
            bucket=_require(f"{prefix}_S3_BUCKET"),
            prefix=os.environ.get(f"{prefix}_S3_PREFIX", ""),
        )
    raise RuntimeError(f"Unknown {prefix}_STORAGE_BACKEND: {backend!r} (expected 's3' or 'local')")


def _load_database_location(prefix: str) -> DatabaseLocation:
    return DatabaseLocation(
        url=_require(f"{prefix}_DATABASE_URL"),
        events_table=os.environ.get(f"{prefix}_EVENTS_TABLE", "events"),
        users_table=os.environ.get(f"{prefix}_USERS_TABLE", "users"),
        items_table=os.environ.get(f"{prefix}_ITEMS_TABLE", "items"),
        recommendations_table=os.environ.get(f"{prefix}_RECOMMENDATIONS_TABLE", "recommendations"),
        manifest_table=os.environ.get(f"{prefix}_MANIFEST_TABLE", "recommendation_runs"),
        events_query=os.environ.get(f"{prefix}_EVENTS_QUERY"),
        users_query=os.environ.get(f"{prefix}_USERS_QUERY"),
        items_query=os.environ.get(f"{prefix}_ITEMS_QUERY"),
    )


def _load_io_settings(prefix: str) -> IOSettings:
    kind = os.environ.get(f"{prefix}_KIND", "dataset").lower()
    if kind == "dataset":
        return IOSettings(kind="dataset", dataset=_load_dataset_location(prefix), database=None)
    if kind == "db":
        return IOSettings(kind="db", dataset=None, database=_load_database_location(prefix))
    raise RuntimeError(f"Unknown {prefix}_KIND: {kind!r} (expected 'dataset' or 'db')")


def load_settings() -> Settings:
    return Settings(
        input=_load_io_settings("INPUT"),
        output=_load_io_settings("OUTPUT"),
        feature_config_path=os.environ.get("FEATURE_CONFIG_PATH", "/app/config/features.yml"),
        top_k=int(os.environ.get("TOP_K", "10")),
        half_life_days=float(os.environ.get("INTERACTION_HALF_LIFE_DAYS", "90")),
    )

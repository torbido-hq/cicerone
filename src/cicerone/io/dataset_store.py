"""Static-file input/output backend: S3-compatible object storage (R2, AWS
S3, MinIO, ...) or the local filesystem. Files are always parquet, except
the manifest which is written as JSON.

Configured generically via an "options" dict (see cicerone.config.IOSettings)
built from the [input.options] / [output.options] tables in cicerone.toml:

  storage_backend   "s3" | "local" (default: "local")
  # storage_backend == "s3"
  access_key_id, secret_access_key, bucket   required
  endpoint_url                                optional (needed for R2/MinIO/etc, not AWS S3)
  prefix                                      optional, default ""
  # storage_backend == "local"
  path                                        required
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config

logger = logging.getLogger(__name__)


def _require(options: dict[str, Any], key: str, backend: str) -> Any:
    value = options.get(key)
    if not value:
        raise RuntimeError(f"Missing required option '{key}' for dataset storage_backend={backend!r}")
    return value


def _s3_client(options: dict[str, Any]):
    return boto3.client(
        "s3",
        endpoint_url=options.get("endpoint_url"),
        aws_access_key_id=_require(options, "access_key_id", "s3"),
        aws_secret_access_key=_require(options, "secret_access_key", "s3"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def _full_key(options: dict[str, Any], filename: str) -> str:
    prefix = str(options.get("prefix", "")).strip("/")
    return f"{prefix}/{filename}" if prefix else filename


class DatasetInputSource:
    """Reads events/users/items parquet files from an S3-compatible store or local disk."""

    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")
        if self._backend not in ("s3", "local"):
            raise ValueError(f"Unknown storage_backend: {self._backend!r} (expected 's3' or 'local')")

    def _read(self, filename: str) -> pd.DataFrame:
        if self._backend == "local":
            path = Path(_require(self._options, "path", "local")) / filename
            logger.info("Reading %s", path)
            return pd.read_parquet(path)

        bucket = _require(self._options, "bucket", "s3")
        key = _full_key(self._options, filename)
        logger.info("Reading s3://%s/%s", bucket, key)
        client = _s3_client(self._options)
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def read_events(self) -> pd.DataFrame:
        return self._read("events.parquet")

    def read_users(self) -> pd.DataFrame | None:
        try:
            return self._read("users.parquet")
        except Exception:  # noqa: BLE001 - optional input, missing file is expected
            logger.warning("Optional input 'users.parquet' not found — continuing without user features.")
            return None

    def read_items(self) -> pd.DataFrame | None:
        try:
            return self._read("items.parquet")
        except Exception:  # noqa: BLE001 - optional input, missing file is expected
            logger.warning("Optional input 'items.parquet' not found — continuing without item features.")
            return None


class DatasetOutputSink:
    """Writes recommendations.parquet + manifest.json to an S3-compatible store or local disk."""

    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")
        if self._backend not in ("s3", "local"):
            raise ValueError(f"Unknown storage_backend: {self._backend!r} (expected 's3' or 'local')")

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        if self._backend == "local":
            path = Path(_require(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Writing %s", path)
            path.write_bytes(payload)
            return

        bucket = _require(self._options, "bucket", "s3")
        key = _full_key(self._options, filename)
        logger.info("Writing s3://%s/%s", bucket, key)
        client = _s3_client(self._options)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._write_bytes("recommendations.parquet", buffer.getvalue(), "application/octet-stream")

    def write_manifest(self, manifest: dict) -> None:
        self._write_bytes("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

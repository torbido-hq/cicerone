"""Static-file input/output backend: S3-compatible object storage (R2, AWS
S3, MinIO, ...) or the local filesystem. Files are always parquet, except
the manifest which is written as JSON.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

from cicerone.config import DatasetLocation

logger = logging.getLogger(__name__)


def _s3_client(location: DatasetLocation):
    return boto3.client(
        "s3",
        endpoint_url=location.endpoint_url,
        aws_access_key_id=location.access_key_id,
        aws_secret_access_key=location.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def _full_key(location: DatasetLocation, filename: str) -> str:
    prefix = location.prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


class DatasetInputSource:
    """Reads events/users/items parquet files from an S3-compatible store or local disk."""

    def __init__(self, location: DatasetLocation):
        self._loc = location

    def _read(self, filename: str) -> pd.DataFrame:
        if self._loc.backend == "local":
            path = Path(self._loc.path) / filename
            logger.info("Reading %s", path)
            return pd.read_parquet(path)

        key = _full_key(self._loc, filename)
        logger.info("Reading s3://%s/%s", self._loc.bucket, key)
        client = _s3_client(self._loc)
        obj = client.get_object(Bucket=self._loc.bucket, Key=key)
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

    def __init__(self, location: DatasetLocation):
        self._loc = location

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        if self._loc.backend == "local":
            path = Path(self._loc.path) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Writing %s", path)
            path.write_bytes(payload)
            return

        key = _full_key(self._loc, filename)
        logger.info("Writing s3://%s/%s", self._loc.bucket, key)
        client = _s3_client(self._loc)
        client.put_object(Bucket=self._loc.bucket, Key=key, Body=payload, ContentType=content_type)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._write_bytes("recommendations.parquet", buffer.getvalue(), "application/octet-stream")

    def write_manifest(self, manifest: dict) -> None:
        self._write_bytes("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

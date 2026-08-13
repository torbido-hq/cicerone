"""Static-file input/output: S3-compatible object storage or local filesystem.

Options (from [input.options] / [output.options]):

  storage_backend   "s3" | "local" (default "local")
  # s3: access_key_id, secret_access_key, bucket (required); endpoint_url, prefix
  # local: path (required)
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    read_parquet,
    require_option,
    validate_storage_options,
)

logger = logging.getLogger(__name__)


class DatasetInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)

    def _read(self, filename: str) -> pd.DataFrame:
        return read_parquet(self._options, filename)

    def read_events(self) -> pd.DataFrame:
        return self._read("events.parquet")

    def _read_optional(self, filename: str, label: str) -> pd.DataFrame | None:
        try:
            return self._read(filename)
        except FileNotFoundError:
            logger.warning("Optional input %r not found — continuing without %s features.", filename, label)
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                logger.warning(
                    "Optional input %r not found — continuing without %s features.", filename, label
                )
                return None
            raise

    def read_users(self) -> pd.DataFrame | None:
        return self._read_optional("users.parquet", "user")

    def read_items(self) -> pd.DataFrame | None:
        return self._read_optional("items.parquet", "item")


class DatasetOutputSink:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Writing %s", path)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return

        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        logger.info("Writing s3://%s/%s", bucket, key)
        client = build_s3_client(self._options)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._write_bytes("recommendations.parquet", buffer.getvalue(), "application/octet-stream")

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._write_bytes("items_snapshot.parquet", buffer.getvalue(), "application/octet-stream")

    def write_manifest(self, manifest: dict) -> None:
        self._write_bytes("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    def write_model_artifact(self, payload: bytes) -> None:
        from cicerone.artifact import ARTIFACT_FILENAME

        self._write_bytes(ARTIFACT_FILENAME, payload, "application/octet-stream")

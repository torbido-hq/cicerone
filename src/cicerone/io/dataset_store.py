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
from collections.abc import Sequence
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
from cicerone.io.recommendation_schema import USER_COLUMN
from cicerone.io.replace_users import RecommendationSchemaError, normalize_replace_user_ids
from cicerone.io.user_lookup import filter_rows_for_user, newest_events

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

    def _read_for_user(self, filename: str, user_id: str) -> pd.DataFrame:
        try:
            frame = read_parquet(self._options, filename, filters=[("user_id", "==", user_id)])
        except FileNotFoundError:
            raise
        except Exception as exc:
            if is_s3_not_found(exc):
                raise
            message = str(exc).lower()
            if "user_id" in message or "fieldref" in message or "filter" in message:
                logger.warning("Filtered %s read failed; falling back to full-file load: %s", filename, exc)
                frame = read_parquet(self._options, filename)
            else:
                raise
        if USER_COLUMN not in frame.columns:
            frame = read_parquet(self._options, filename)
        return filter_rows_for_user(frame, user_id)

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        return newest_events(self._read_for_user("events.parquet", user_id), limit)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        try:
            frame = self._read_for_user("users.parquet", user_id)
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()


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

    def replace_recommendations_for_users(self, df: pd.DataFrame, *, user_ids: Sequence[str]) -> int:
        # Read-modify-write; concurrent replicas need events.ha (DB sink is transactional).
        ids = normalize_replace_user_ids(df, user_ids)
        if not ids:
            return 0
        try:
            existing = read_parquet(self._options, "recommendations.parquet")
        except FileNotFoundError:
            existing = pd.DataFrame()
        except Exception as exc:
            if is_s3_not_found(exc):
                existing = pd.DataFrame()
            else:
                raise
        if existing.empty:
            remaining = existing
        elif USER_COLUMN not in existing.columns:
            raise RecommendationSchemaError(
                f"Recommendations schema mismatch (missing {USER_COLUMN}); refusing replace"
            )
        else:
            remaining = existing[~existing[USER_COLUMN].astype(str).isin(ids)]
        parts = [frame for frame in (remaining, df) if not frame.empty]
        merged = pd.concat(parts, ignore_index=True) if parts else df
        self.write_recommendations(merged)
        if merged.empty or USER_COLUMN not in merged.columns:
            return 0
        return int(merged[USER_COLUMN].astype(str).nunique())

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._write_bytes("items_snapshot.parquet", buffer.getvalue(), "application/octet-stream")

    def write_manifest(self, manifest: dict) -> None:
        self._write_bytes("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    def _read_bytes(self, filename: str) -> bytes | None:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def write_model_artifact(self, payload: bytes) -> None:
        from cicerone.artifact import ARTIFACT_FILENAME

        self._write_bytes(ARTIFACT_FILENAME, payload, "application/octet-stream")

    def read_model_artifact(self) -> bytes | None:
        from cicerone.artifact import ARTIFACT_FILENAME

        return self._read_bytes(ARTIFACT_FILENAME)

    def model_artifact_fingerprint(self) -> str | None:
        from cicerone.artifact import ARTIFACT_FILENAME

        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / ARTIFACT_FILENAME
            try:
                stat = path.stat()
            except FileNotFoundError:
                return None
            return f"local:{stat.st_mtime_ns}:{stat.st_size}"

        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, ARTIFACT_FILENAME)
        client = build_s3_client(self._options)
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        etag = str(head.get("ETag") or "").strip('"')
        return f"s3:{etag}:{head.get('ContentLength', '')}"

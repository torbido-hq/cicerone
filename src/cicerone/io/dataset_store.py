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
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyarrow.lib import ArrowException, ArrowNotImplementedError

from cicerone.io.applied_events import APPLIED_EVENTS_ATTR, merge_applied_keys, unexpired_keys
from cicerone.io.blob import (
    S3_READ_ERRORS,
    STORAGE_READ_ERRORS,
    is_storage_not_found,
    read_parquet_if_exists,
    read_storage_bytes,
    write_storage_bytes,
)
from cicerone.io.filenames import (
    EVENTS_FILENAME,
    ITEMS_FILENAME,
    MANIFEST_FILENAME,
    RECOMMENDATIONS_FILENAME,
    USERS_FILENAME,
)
from cicerone.io.options import (
    build_s3_client,
    exclusive_file_lock,
    is_s3_not_found,
    object_key,
    read_parquet,
    read_parquet_attrs,
    require_option,
    validate_storage_options,
)
from cicerone.io.recommendation_schema import USER_COLUMN
from cicerone.io.replace_users import RecommendationSchemaError, normalize_replace_user_ids
from cicerone.io.user_lookup import filter_rows_for_user, newest_events
from cicerone.locks.hold import WriteFence

logger = logging.getLogger(__name__)


def _existing_manifest_newer(raw: bytes | None, started_at: str) -> bool:
    if not raw:
        return False
    try:
        existing = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(existing, dict):
        return False
    for key in ("generated_at", "last_incremental_at"):
        value = existing.get(key)
        if isinstance(value, str) and value > started_at:
            return True
    return False


class DatasetInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)

    def read_events(self) -> pd.DataFrame:
        return read_parquet(self._options, EVENTS_FILENAME)

    def _read_optional(self, filename: str, label: str) -> pd.DataFrame | None:
        frame = read_parquet_if_exists(self._options, filename)
        if frame is None:
            logger.warning("Optional input %r not found — continuing without %s features.", filename, label)
        return frame

    def read_users(self) -> pd.DataFrame | None:
        return self._read_optional(USERS_FILENAME, "user")

    def read_items(self) -> pd.DataFrame | None:
        return self._read_optional(ITEMS_FILENAME, "item")

    def _read_for_user(self, filename: str, user_id: str) -> pd.DataFrame:
        try:
            frame = read_parquet(self._options, filename, filters=[("user_id", "==", user_id)])
        except FileNotFoundError:
            raise
        except (OSError, ValueError, ArrowException) as exc:
            message = str(exc).lower()
            # A numeric user_id column has no kernel for the string filter value.
            if (
                isinstance(exc, ArrowNotImplementedError)
                or "user_id" in message
                or "fieldref" in message
                or "filter" in message
            ):
                logger.warning("Filtered %s read failed; falling back to full-file load: %s", filename, exc)
                frame = read_parquet(self._options, filename)
            else:
                raise
        if USER_COLUMN not in frame.columns:
            frame = read_parquet(self._options, filename)
        return filter_rows_for_user(frame, user_id)

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        return newest_events(self._read_for_user(EVENTS_FILENAME, user_id), limit)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        try:
            frame = self._read_for_user(USERS_FILENAME, user_id)
        except STORAGE_READ_ERRORS as exc:
            if is_storage_not_found(exc):
                return None
            raise
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()


class DatasetOutputSink:
    def __init__(
        self,
        options: dict[str, Any],
        *,
        fence: WriteFence | None = None,
    ):
        self._options = options
        self._backend = validate_storage_options(options)
        self._fence = fence or WriteFence()
        self._recs_write_held = threading.local()

    @contextmanager
    def _local_file_lock(self, filename: str) -> Iterator[None]:
        if self._backend != "local":
            yield
            return
        path = Path(require_option(self._options, "path", "local")) / filename
        with exclusive_file_lock(path):
            yield

    @contextmanager
    def _recommendations_lock(self) -> Iterator[None]:
        if self._fence.held_here():
            with self._local_file_lock(".recommendations.lock"):
                self._ensure_writer_still_held()
                yield
            return
        with (
            self._local_file_lock(".recommendations.lock"),
            self._fence.hold(),
        ):
            yield

    def _recs_write_depth(self) -> int:
        return int(getattr(self._recs_write_held, "depth", 0))

    @contextmanager
    def recommendations_write(self) -> Iterator[None]:
        with self._recommendations_lock():
            self._recs_write_held.depth = self._recs_write_depth() + 1
            try:
                yield
            finally:
                self._recs_write_held.depth = self._recs_write_depth() - 1

    @contextmanager
    def _maybe_recommendations_lock(self) -> Iterator[None]:
        if self._recs_write_depth():
            yield
            return
        with self._recommendations_lock():
            yield

    @contextmanager
    def _artifact_lock(self) -> Iterator[None]:
        with self._local_file_lock(".model-artifact.lock"):
            yield

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            logger.info("Writing %s", path)
        else:
            bucket = require_option(self._options, "bucket", "s3")
            logger.info("Writing s3://%s/%s", bucket, object_key(self._options, filename))
        write_storage_bytes(self._options, filename, payload, content_type)

    def ensure_writer_held(self) -> None:
        self._ensure_writer_still_held()

    def _ensure_writer_still_held(self) -> None:
        self._fence.ensure()

    def _write_recommendations_unlocked(self, df: pd.DataFrame) -> None:
        self._ensure_writer_still_held()
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        self._ensure_writer_still_held()
        self._write_bytes(RECOMMENDATIONS_FILENAME, buffer.getvalue(), "application/octet-stream")

    def write_recommendations(self, df: pd.DataFrame) -> None:
        if APPLIED_EVENTS_ATTR in df.attrs:
            # A full write replaces every row, so no earlier incremental key describes it.
            df = df.copy(deep=False)
            df.attrs = {key: value for key, value in df.attrs.items() if key != APPLIED_EVENTS_ATTR}
        with self._maybe_recommendations_lock():
            self._write_recommendations_unlocked(df)

    def _read_recommendations_or_empty(self) -> pd.DataFrame:
        frame = read_parquet_if_exists(self._options, RECOMMENDATIONS_FILENAME)
        return pd.DataFrame() if frame is None else frame

    def applied_event_keys(self, keys: Sequence[str]) -> set[str]:
        wanted = {str(key) for key in keys}
        if not wanted:
            return set()
        try:
            attrs = read_parquet_attrs(self._options, RECOMMENDATIONS_FILENAME)
        except STORAGE_READ_ERRORS as exc:
            if is_storage_not_found(exc):
                return set()
            raise
        recorded = attrs.get(APPLIED_EVENTS_ATTR) or {}
        return wanted & unexpired_keys(recorded, datetime.now(UTC))

    def replace_recommendations_for_users(
        self, df: pd.DataFrame, *, user_ids: Sequence[str], applied_keys: Sequence[str] = ()
    ) -> int:
        # Read-modify-write under host fcntl plus the optional dataset-append lease.
        ids = normalize_replace_user_ids(df, user_ids)
        if not ids:
            return 0
        with self._maybe_recommendations_lock():
            existing = self._read_recommendations_or_empty()
            recorded = existing.attrs.get(APPLIED_EVENTS_ATTR) or {}
            if existing.empty:
                remaining = existing
            elif USER_COLUMN not in existing.columns:
                raise RecommendationSchemaError(
                    f"Recommendations schema mismatch (missing {USER_COLUMN}); refusing replace"
                )
            else:
                remaining = existing[~existing[USER_COLUMN].astype(str).isin(ids)]
            parts = [frame for frame in (remaining, df) if not frame.empty]
            merged = pd.concat(parts, ignore_index=True) if parts else df.copy()
            # The record rides in the same parquet write as the rows, so it lands only with them.
            merged.attrs = {}
            if applied_keys or recorded:
                merged.attrs[APPLIED_EVENTS_ATTR] = merge_applied_keys(
                    recorded, applied_keys, datetime.now(UTC)
                )
            self._write_recommendations_unlocked(merged)
            if merged.empty or USER_COLUMN not in merged.columns:
                return 0
            return int(merged[USER_COLUMN].astype(str).nunique())

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        with self._maybe_recommendations_lock():
            self._ensure_writer_still_held()
            self._write_bytes("items_snapshot.parquet", buffer.getvalue(), "application/octet-stream")

    def write_manifest(self, manifest: dict, *, skip_if_newer_than: str | None = None) -> bool:
        payload = json.dumps(manifest, indent=2).encode("utf-8")
        with self._maybe_recommendations_lock():
            self._ensure_writer_still_held()
            if skip_if_newer_than is not None and _existing_manifest_newer(
                self._read_bytes(MANIFEST_FILENAME), skip_if_newer_than
            ):
                return False
            self._ensure_writer_still_held()
            self._write_bytes(MANIFEST_FILENAME, payload, "application/json")
            return True

    def _read_bytes(self, filename: str) -> bytes | None:
        return read_storage_bytes(self._options, filename)

    def write_model_artifact(self, payload: bytes) -> None:
        from cicerone.artifact import ARTIFACT_FILENAME

        with self._maybe_recommendations_lock(), self._artifact_lock():
            self._ensure_writer_still_held()
            self._write_bytes(ARTIFACT_FILENAME, payload, "application/octet-stream")

    def replace_model_artifact_if(self, payload: bytes, expected_fingerprint: str) -> bool:
        from cicerone.artifact import ARTIFACT_FILENAME

        if self._backend != "local":
            logger.warning("Skipping model artifact replace: S3 is not compare-and-swap")
            return False
        with self._maybe_recommendations_lock(), self._artifact_lock():
            if self.model_artifact_fingerprint() != expected_fingerprint:
                return False
            self._ensure_writer_still_held()
            self._write_bytes(ARTIFACT_FILENAME, payload, "application/octet-stream")
            return True

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
        except S3_READ_ERRORS as exc:
            if is_s3_not_found(exc):
                return None
            raise
        etag = str(head.get("ETag") or "").strip('"')
        return f"s3:{etag}:{head.get('ContentLength', '')}"

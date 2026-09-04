"""Dataset/object-store backend for TrackStore."""

from __future__ import annotations

import json
import threading

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    require_option,
    validate_storage_options,
)
from cicerone.track.store_common import HISTORY_DIR, HISTORY_FILENAME, TRACK_FILENAME, _history_stem_before


class TrackDatasetBackend:
    _options: dict[str, Any]
    _known_ids: set[str] | None
    _track_size: int | None
    _append_lock: threading.Lock

    def _read_legacy_history(self) -> list[pd.DataFrame]:
        from cicerone.io.options import read_parquet

        try:
            frame = read_parquet(self._options, HISTORY_FILENAME)
        except FileNotFoundError:
            return []
        except Exception as exc:
            if is_s3_not_found(exc):
                return []
            raise
        return [] if frame.empty else [frame]

    def _read_history_parts(
        self,
        *,
        generated_ats: set[str] | None,
        since: str | None,
    ) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        backend = validate_storage_options(self._options)
        if backend == "local":
            root = Path(require_option(self._options, "path", "local")) / HISTORY_DIR
            if not root.is_dir():
                return []
            if generated_ats is not None:
                paths = sorted(root / _history_part_name(stamp) for stamp in generated_ats)
                paths = [path for path in paths if path.is_file()]
            else:
                paths = sorted(root.glob("*.parquet"))
                if since:
                    paths = [path for path in paths if not _history_stem_before(path.stem, since)]
            for path in paths:
                frame = pd.read_parquet(path)
                if not frame.empty:
                    frames.append(frame)
            return frames
        bucket = require_option(self._options, "bucket", "s3")
        prefix = object_key(self._options, f"{HISTORY_DIR}/")
        client = build_s3_client(self._options)
        try:
            keys = _list_s3_parquet_keys(client, bucket, prefix)
        except Exception as exc:
            if is_s3_not_found(exc):
                return []
            raise
        if generated_ats is not None:
            wanted_names = {_history_part_name(stamp) for stamp in generated_ats}
            keys = [key for key in keys if Path(key).name in wanted_names]
        elif since:
            keys = [key for key in keys if not _history_stem_before(Path(key).stem, since)]
        keys = sorted(keys)
        for key in keys:
            obj = client.get_object(Bucket=bucket, Key=key)
            frame = pd.read_parquet(BytesIO(obj["Body"].read()))
            if not frame.empty:
                frames.append(frame)
        return frames

    def _read_rows_dataset(self) -> list[dict[str, Any]]:
        raw = self._read_bytes(TRACK_FILENAME)
        if raw is None:
            if self._known_ids is None:
                self._known_ids = set()
            return []
        rows: list[dict[str, Any]] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        if self._known_ids is None:
            self._known_ids = {str(row.get("event_id") or "") for row in rows}
            self._known_ids.discard("")
        return rows

    def _track_file_size(self) -> int:
        path = Path(require_option(self._options, "path", "local")) / TRACK_FILENAME
        if not path.exists():
            return 0
        return path.stat().st_size

    def _refresh_known_ids(self) -> set[str]:
        size = self._track_file_size()
        if self._known_ids is not None and self._track_size == size:
            return self._known_ids
        self._known_ids = None
        self._read_rows_dataset()
        assert self._known_ids is not None
        self._track_size = size
        return self._known_ids

    @contextmanager
    def _dataset_append_lock(self) -> Iterator[None]:
        path = Path(require_option(self._options, "path", "local")) / ".track.jsonl.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock, path.open("a") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_bytes(self, filename: str) -> bytes | None:
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            if not path.exists():
                return None
            return path.read_bytes()
        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        return obj["Body"].read()

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return
        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)

    def _append_bytes(self, filename: str, payload: bytes) -> None:
        path = Path(require_option(self._options, "path", "local")) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(payload)


def _history_part_name(generated_at: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-+." else "-" for ch in generated_at.strip())
    slug = slug.strip("-.") or "snapshot"
    return f"{slug}.parquet"


def _list_s3_parquet_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if key.endswith(".parquet"):
                keys.append(key)
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")
        if not token:
            return keys

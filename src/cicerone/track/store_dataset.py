"""Dataset/object-store backend for TrackStore."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from cicerone.config.settings import IOSettings
from cicerone.io.blob import append_storage_bytes, read_storage_bytes, write_storage_bytes
from cicerone.io.options import (
    build_s3_client,
    exclusive_file_lock,
    is_s3_not_found,
    object_key,
    read_s3_body,
    require_option,
    validate_storage_options,
)
from cicerone.locks import WriteFence
from cicerone.track.store_common import (
    EVAL_FILENAME,
    HISTORY_COLUMNS,
    HISTORY_DIR,
    HISTORY_FILENAME,
    TRACK_FILENAME,
    filter_history,
    history_part_matches,
    history_stem_before,
    require_appendable_track_log,
    row_matches,
    unique_matching_rows,
)

logger = logging.getLogger(__name__)

_HISTORY_READ_WORKERS = 8
_S3_READ_ERRORS = (BotoCoreError, ClientError)


def _collect_frames(loaders: list[Callable[[], pd.DataFrame | None]]) -> list[pd.DataFrame]:
    if not loaders:
        return []
    if len(loaders) == 1:
        frame = loaders[0]()
        return [] if frame is None or frame.empty else [frame]
    workers = min(_HISTORY_READ_WORKERS, len(loaders))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(lambda load: load(), loaders))
    return [frame for frame in loaded if frame is not None and not frame.empty]


class TrackDatasetBackend:
    """``track.jsonl``, ``track_eval.json`` and history parquet parts on the dataset store."""

    def __init__(self, output: IOSettings, fence: WriteFence):
        self._output = output
        self._options = output.options
        self._fence = fence
        self._known_ids: set[str] | None = None
        self._track_size: int | None = None
        self._append_lock = threading.Lock()

    def append_accepted(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        require_appendable_track_log(self._output)
        with self._dataset_append_lock(), self._fence.lease():
            known = self._refresh_known_ids()
            received_at = datetime.now(UTC).isoformat()
            fresh: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                event_id = str(row.get("event_id") or "")
                if event_id and (event_id in known or event_id in seen):
                    continue
                fresh.append({**row, "received_at": received_at})
                if event_id:
                    seen.add(event_id)
            if not fresh:
                return []
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh).encode("utf-8")
            self._fence.ensure()
            self._append_bytes(TRACK_FILENAME, encoded)
            known.update(seen)
            self._track_size = (self._track_size or 0) + len(encoded)
            return fresh

    def read_rows(
        self, *, kind: str | None, experiment_id: str | None, since: str | None
    ) -> list[dict[str, Any]]:
        # ``since`` is applied while reading the file.
        rows = self._read_rows_dataset(since=since)
        return unique_matching_rows(rows, kind=kind, experiment_id=experiment_id, since=None)

    def eval_writer(self, payload: dict[str, Any]) -> Callable[[], None]:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        return lambda: self._write_bytes(EVAL_FILENAME, encoded, "application/json")

    def read_eval(self) -> dict[str, Any] | None:
        raw = self._read_bytes(EVAL_FILENAME)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid track_eval.json; ignoring")
            return None
        return parsed if isinstance(parsed, dict) else None

    def history_writer(self, frame: pd.DataFrame, generated_at: str) -> Callable[[], None]:
        buf = BytesIO()
        frame.to_parquet(buf, index=False)
        part = f"{HISTORY_DIR}/{history_part_name(generated_at)}"
        payload = buf.getvalue()
        return lambda: self._write_bytes(part, payload, "application/octet-stream")

    def read_history(self, *, generated_ats: set[str] | None, since: str | None) -> pd.DataFrame:
        frames = []
        if generated_ats is None and since is None:
            frames.extend(self._read_legacy_history())
        frames.extend(self._read_history_parts(generated_ats=generated_ats, since=since))
        if not frames:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return filter_history(pd.concat(frames, ignore_index=True), generated_ats=generated_ats, since=since)

    def _read_legacy_history(self) -> list[pd.DataFrame]:
        from cicerone.io.options import read_parquet

        try:
            frame = read_parquet(self._options, HISTORY_FILENAME)
        except FileNotFoundError:
            return []
        except _S3_READ_ERRORS as exc:
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
        backend = validate_storage_options(self._options)
        if backend == "local":
            root = Path(require_option(self._options, "path", "local")) / HISTORY_DIR
            if not root.is_dir():
                return []
            if generated_ats is not None:
                paths = _local_history_paths(root, generated_ats)
            else:
                paths = sorted(root.glob("*.parquet"))
                if since:
                    paths = [path for path in paths if not history_stem_before(path.stem, since)]
            return _collect_frames([_bind_local_frame(path) for path in paths])
        bucket = require_option(self._options, "bucket", "s3")
        prefix = object_key(self._options, f"{HISTORY_DIR}/")
        client = build_s3_client(self._options)
        if generated_ats is not None:
            return _s3_history_frames(client, bucket, prefix, self._options, generated_ats)
        try:
            keys = _list_s3_parquet_keys(client, bucket, prefix)
        except _S3_READ_ERRORS as exc:
            if is_s3_not_found(exc):
                return []
            raise
        if since:
            keys = [key for key in keys if not history_stem_before(Path(key).stem, since)]
        return _collect_frames([_bind_s3_frame(client, bucket, key) for key in sorted(keys)])

    def _iter_track_lines(self) -> Iterator[bytes]:
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / TRACK_FILENAME
            if not path.exists():
                return
            with path.open("rb") as handle:
                yield from handle
            return
        raw = self._read_bytes(TRACK_FILENAME)
        if raw is None:
            return
        yield from raw.splitlines()

    def _read_rows_dataset(self, *, since: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        known: set[str] | None = set() if self._known_ids is None else None
        for raw_line in self._iter_track_lines():
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if known is not None:
                event_id = str(parsed.get("event_id") or "")
                if event_id:
                    known.add(event_id)
            if since and not row_matches(parsed, kind=None, experiment_id=None, since=since):
                continue
            rows.append(parsed)
        if known is not None:
            self._known_ids = known
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
        with self._append_lock, exclusive_file_lock(path):
            yield

    def _read_bytes(self, filename: str) -> bytes | None:
        return read_storage_bytes(self._options, filename)

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        write_storage_bytes(self._options, filename, payload, content_type)

    def _append_bytes(self, filename: str, payload: bytes) -> None:
        append_storage_bytes(self._options, filename, payload)


def history_part_name(generated_at: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-+." else "-" for ch in generated_at.strip())
    slug = slug.strip("-.") or "snapshot"
    return f"{slug}.parquet"


def _local_history_paths(root: Path, generated_ats: set[str]) -> list[Path]:
    exact = [root / history_part_name(stamp) for stamp in generated_ats]
    if exact and all(path.is_file() for path in exact):
        return sorted(exact)
    wanted_names = {history_part_name(stamp) for stamp in generated_ats}
    return [
        path
        for path in sorted(root.glob("*.parquet"))
        if path.name in wanted_names or history_part_matches(path.stem, generated_ats)
    ]


def _bind_local_frame(path: Path) -> Callable[[], pd.DataFrame | None]:
    def _load() -> pd.DataFrame | None:
        return _local_parquet_frame(path)

    return _load


def _bind_s3_frame(client: Any, bucket: str, key: str) -> Callable[[], pd.DataFrame | None]:
    def _load() -> pd.DataFrame | None:
        return _s3_parquet_frame(client, bucket, key)

    return _load


def _local_parquet_frame(path: Path) -> pd.DataFrame | None:
    frame = pd.read_parquet(path)
    return None if frame.empty else frame


def _s3_parquet_frame(client: Any, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except _S3_READ_ERRORS as exc:
        if is_s3_not_found(exc):
            return None
        raise
    return pd.read_parquet(BytesIO(read_s3_body(obj)))


def _s3_history_frames(
    client: Any,
    bucket: str,
    prefix: str,
    options: dict[str, Any],
    generated_ats: set[str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    loaded: set[str] = set()
    missing: set[str] = set()
    exact = [
        (stamp, object_key(options, f"{HISTORY_DIR}/{history_part_name(stamp)}")) for stamp in generated_ats
    ]

    def _one(item: tuple[str, str]) -> tuple[str, str, pd.DataFrame | None]:
        stamp, key = item
        return stamp, key, _s3_parquet_frame(client, bucket, key)

    if exact:
        workers = min(_HISTORY_READ_WORKERS, len(exact))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, exact))
        for stamp, key, frame in results:
            if frame is None:
                missing.add(stamp)
                continue
            loaded.add(key)
            if not frame.empty:
                frames.append(frame)
    if not missing:
        return frames
    try:
        keys = _list_s3_parquet_keys(client, bucket, prefix)
    except _S3_READ_ERRORS as exc:
        if is_s3_not_found(exc):
            return frames
        raise
    missing_names = {history_part_name(stamp) for stamp in missing}
    extra = [
        key
        for key in keys
        if key not in loaded
        and (Path(key).name in missing_names or history_part_matches(Path(key).stem, missing))
    ]
    frames.extend(_collect_frames([_bind_s3_frame(client, bucket, key) for key in extra]))
    return frames


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

"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import BytesIO
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from cicerone.config.settings import IOSettings, Settings
from cicerone.deprecation import renamed_attribute
from cicerone.events.errors import RedisError
from cicerone.locks import LockBackend, WriteFence
from cicerone.track.store_common import (
    DEFAULT_EVAL_TABLE,  # noqa: F401
    DEFAULT_HISTORY_TABLE,  # noqa: F401
    DEFAULT_TRACK_TABLE,  # noqa: F401
    EVAL_FILENAME,
    HISTORY_COLUMNS,
    HISTORY_DIR,
    HISTORY_FILENAME,  # noqa: F401
    TRACK_COLUMNS,  # noqa: F401
    TRACK_FILENAME,
    TRACK_LOG_BACKEND_ERROR,  # noqa: F401
    TRACK_LOG_HA_ERROR,
    filter_history,
    history_frame,
    iso_utc,
    jsonish,
    require_appendable_track_log,
    row_matches,
    row_with_event_id,
)
from cicerone.track.store_dataset import TrackDatasetBackend, history_part_name
from cicerone.track.store_db import TrackDbBackend
from cicerone.track.store_redis import RedisTrackLog, default_redis_key_prefix

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EVAL_TABLE",
    "DEFAULT_HISTORY_TABLE",
    "DEFAULT_TRACK_TABLE",
    "EVAL_FILENAME",
    "HISTORY_COLUMNS",
    "HISTORY_DIR",
    "HISTORY_FILENAME",
    "TRACK_COLUMNS",
    "TRACK_FILENAME",
    "TRACK_LOG_BACKEND_ERROR",
    "TRACK_LOG_HA_ERROR",
    "TrackStore",
    "history_frame",
    "history_part_name",
    "jsonish",
    "require_appendable_track_log",
]


class TrackStore(TrackDbBackend, TrackDatasetBackend):
    """Output-store side channel for track rows, eval JSON, and rec snapshots."""

    def __init__(
        self,
        output: IOSettings,
        *,
        writer_lock: LockBackend | None = None,
        fence_check: Callable[[], bool] | None = None,
        fence_lost: str = "lock lost before write",
        fence_kind: str = "lock",
        fence: WriteFence | None = None,
        redis_url: str | None = None,
        redis_client: Any = None,
        redis_key_prefix: str | None = None,
    ):
        self._output = output
        self._kind = output.kind
        self._options = output.options
        self._engine: Engine | None = None
        self._engine_lock = threading.Lock()
        self._known_ids: set[str] | None = None
        self._track_size: int | None = None
        self._append_lock = threading.Lock()
        self._fence = WriteFence.resolve(
            fence,
            writer_lock=writer_lock,
            fence_check=fence_check,
            fence_lost=fence_lost,
            fence_kind=fence_kind,
        )
        self._redis_url = redis_url or None
        self._redis_client = redis_client
        self._redis_key_prefix = redis_key_prefix or default_redis_key_prefix(output)
        self._redis_log: RedisTrackLog | None = None

    @property
    def fence(self) -> WriteFence:
        return self._fence

    def _ensure_fence(self) -> None:
        self._fence.ensure()

    @contextmanager
    def _writer_lease(self) -> Iterator[None]:
        with self._fence.lease():
            yield

    def _track_log(self) -> RedisTrackLog | None:
        if self._redis_url is None and self._redis_client is None:
            return None
        if self._redis_log is None:
            self._redis_log = RedisTrackLog(
                self._redis_url or "",
                client=self._redis_client,
                key_prefix=self._redis_key_prefix,
            )
        return self._redis_log

    def log_redis_location(self) -> None:
        """Log the Redis key and its size, so a wrong prefix shows up as an unexpected key or an empty log."""
        redis_log = self._track_log()
        if redis_log is None:
            return
        try:
            rows = redis_log.size()
        except RedisError:
            logger.warning("Track log: redis key %s (unreachable at startup)", redis_log.key)
            return
        logger.info("Track log: redis key %s (%d rows)", redis_log.key, rows)

    def append_accepted_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        payload = [row_with_event_id(row) for row in rows]
        redis_log = self._track_log()
        if redis_log is not None:
            with self._append_lock, self._writer_lease():
                return redis_log.append(payload)
        if self._kind == "db":
            with self._append_lock, self._writer_lease():
                return self._append_rows_db(payload)
        require_appendable_track_log(self._output)
        with self._dataset_append_lock(), self._writer_lease():
            known = self._refresh_known_ids()
            fresh: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in payload:
                event_id = str(row.get("event_id") or "")
                if event_id and (event_id in known or event_id in seen):
                    continue
                fresh.append(row)
                if event_id:
                    seen.add(event_id)
            if not fresh:
                return []
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh).encode("utf-8")
            self._ensure_fence()
            self._append_bytes(TRACK_FILENAME, encoded)
            known.update(seen)
            self._track_size = (self._track_size or 0) + len(encoded)
            return fresh

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return len(self.append_accepted_rows(rows))

    def delete_user_rows(self, user_id: str, *, conn: Any) -> int:
        """Delete one user's track rows on the caller's DB transaction (``forget-user``)."""
        return self._delete_user_rows_db(user_id, conn=conn)

    def ping_redis(self) -> None:
        redis_log = self._track_log()
        if redis_log is not None:
            redis_log.ping()

    def delete_redis_rows(self, user_id: str) -> int:
        """Delete one user's rows from the Redis track log; zero when the log is not on Redis."""
        redis_log = self._track_log()
        if redis_log is None:
            return 0
        with self._writer_lease():
            return redis_log.delete_user(user_id)

    def delete_history_for_user(self, user_id: str, *, conn: Any) -> int:
        """Delete one user's history rows on the caller's DB transaction (``forget-user``)."""
        return self._delete_history_db(user_id, conn=conn)

    def read_rows(
        self,
        *,
        kind: str | None = None,
        experiment_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        since = iso_utc(since)
        if self._kind == "db":
            rows = self._read_rows_db(kind=kind, experiment_id=experiment_id, since=since)
        else:
            rows = self._read_rows_dataset(since=since)
        redis_log = self._track_log()
        if redis_log is not None:
            # [output] rows stay readable: written before the switch, or by an instance not yet switched.
            rows = [*rows, *redis_log.read()]
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            # Filter first: an [output] copy that fails the filter must not hide a matching Redis row.
            if not row_matches(row, kind=kind, experiment_id=experiment_id, since=since):
                continue
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            unique.append(row)
        return unique

    def write_eval(self, report: Mapping[str, Any]) -> None:
        payload = dict(report)
        encoded = None if self._kind == "db" else json.dumps(payload, indent=2).encode("utf-8")

        with self._writer_lease():
            if self._kind == "db":
                self._write_eval_db(payload)
                return
            assert encoded is not None
            self._write_bytes(EVAL_FILENAME, encoded, "application/json")

    def read_eval(self) -> dict[str, Any] | None:
        if self._kind == "db":
            return self._read_eval_db()
        raw = self._read_bytes(EVAL_FILENAME)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid track_eval.json; ignoring")
            return None
        return parsed if isinstance(parsed, dict) else None

    def append_history(self, recommendations: pd.DataFrame, *, generated_at: str) -> None:
        if recommendations.empty:
            return
        frame = history_frame(recommendations, generated_at)
        part = None
        payload = None
        if self._kind != "db":
            buf = BytesIO()
            frame.to_parquet(buf, index=False)
            part = f"{HISTORY_DIR}/{history_part_name(generated_at)}"
            payload = buf.getvalue()

        with self._writer_lease():
            if self._kind == "db":
                self._append_history_db(frame)
                return
            assert part is not None and payload is not None
            self._write_bytes(part, payload, "application/octet-stream")

    def read_history(
        self,
        *,
        generated_ats: Collection[str] | None = None,
        since: str | None = None,
    ) -> pd.DataFrame:
        wanted = {str(stamp) for stamp in generated_ats} if generated_ats is not None else None
        if wanted is not None:
            wanted.discard("")
            if not wanted:
                return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        since = iso_utc(since)
        if self._kind == "db":
            frame = self._read_history_db(generated_ats=wanted)
            return filter_history(frame, generated_ats=wanted, since=since)
        frames = []
        if wanted is None and since is None:
            frames.extend(self._read_legacy_history())
        frames.extend(self._read_history_parts(generated_ats=wanted, since=since))
        if not frames:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return filter_history(pd.concat(frames, ignore_index=True), generated_ats=wanted, since=since)


_RENAMED = {
    "_history_frame": "cicerone.track.store.history_frame",
    "_history_part_name": "cicerone.track.store.history_part_name",
    "_jsonish": "cicerone.track.store.jsonish",
}
__all__ += list(_RENAMED)


def __getattr__(name: str) -> Any:
    return renamed_attribute(__name__, _RENAMED, name)


def open_track_store(settings: Settings, **kwargs: Any) -> TrackStore:
    """Track log on Redis when ``track.redis_url`` is set; history and eval stay on output."""
    return TrackStore(
        settings.output,
        redis_url=settings.track.redis_url,
        redis_key_prefix=settings.track.redis_key_prefix,
        **kwargs,
    )

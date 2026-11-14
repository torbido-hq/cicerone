"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol

import pandas as pd

from cicerone.config.settings import IOSettings, Settings
from cicerone.events.errors import RedisError
from cicerone.locks import WriteFence
from cicerone.track.normalize import assign_missing_event_id
from cicerone.track.store_common import (
    empty_history_frame,
    history_frame,
    iso_utc,
    row_matches,
    unique_event_rows,
)
from cicerone.track.store_dataset import TrackDatasetBackend
from cicerone.track.store_db import TrackDbBackend
from cicerone.track.store_redis import RedisTrackLog, default_redis_key_prefix

logger = logging.getLogger(__name__)


class TrackBackend(Protocol):
    """Writers encode eagerly; the returned call runs under the writer lease."""

    def append_accepted(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def read_rows(
        self, *, kind: str | None, experiment_id: str | None, since: str | None
    ) -> list[dict[str, Any]]: ...

    def eval_writer(self, payload: dict[str, Any]) -> Callable[[], None]: ...

    def read_eval(self) -> dict[str, Any] | None: ...

    def history_writer(self, frame: pd.DataFrame, generated_at: str) -> Callable[[], None]: ...

    def read_history(self, *, generated_ats: set[str] | None, since: str | None) -> pd.DataFrame: ...


def build_track_backend(output: IOSettings, fence: WriteFence) -> TrackBackend:
    if output.kind == "db":
        return TrackDbBackend(output.options, fence)
    return TrackDatasetBackend(output, fence)


class TrackStore:
    """Output-store side channel for track rows, eval JSON, and rec snapshots."""

    def __init__(
        self,
        output: IOSettings,
        *,
        fence: WriteFence | None = None,
        redis_url: str | None = None,
        redis_key_prefix: str | None = None,
    ):
        self._fence = fence or WriteFence()
        self.backend = build_track_backend(output, self._fence)
        self._redis_url = redis_url or None
        self._redis_key_prefix = redis_key_prefix or default_redis_key_prefix(output)
        self._redis_log: RedisTrackLog | None = None

    @property
    def fence(self) -> WriteFence:
        return self._fence

    def _track_log(self) -> RedisTrackLog | None:
        if self._redis_url is None:
            return None
        if self._redis_log is None:
            self._redis_log = RedisTrackLog(self._redis_url, key_prefix=self._redis_key_prefix)
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
        payload = [assign_missing_event_id(row) for row in rows]
        redis_log = self._track_log()
        if redis_log is None:
            return self.backend.append_accepted(payload)
        with self._fence.lease():
            return redis_log.append(payload)

    def ping_redis(self) -> None:
        redis_log = self._track_log()
        if redis_log is not None:
            redis_log.ping()

    def delete_redis_rows(self, user_id: str) -> int:
        """Delete one user's rows from the Redis track log; zero when the log is not on Redis."""
        redis_log = self._track_log()
        if redis_log is None:
            return 0
        with self._fence.lease():
            return redis_log.delete_user(user_id)

    def read_rows(
        self,
        *,
        kind: str | None = None,
        experiment_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        since = iso_utc(since)
        rows = self.backend.read_rows(kind=kind, experiment_id=experiment_id, since=since)
        redis_log = self._track_log()
        if redis_log is None:
            return rows
        # [output] rows stay readable: written before the switch, or by an instance not yet switched.
        # Filter first: an [output] copy that fails the filter must not hide a matching Redis row.
        return unique_event_rows(
            row
            for row in [*rows, *redis_log.read()]
            if row_matches(row, kind=kind, experiment_id=experiment_id, since=since)
        )

    def write_eval(self, report: Mapping[str, Any]) -> None:
        write = self.backend.eval_writer(dict(report))
        with self._fence.lease():
            write()

    def read_eval(self) -> dict[str, Any] | None:
        return self.backend.read_eval()

    def append_history(self, recommendations: pd.DataFrame, *, generated_at: str) -> None:
        if recommendations.empty:
            return
        write = self.backend.history_writer(history_frame(recommendations, generated_at), generated_at)
        with self._fence.lease():
            write()

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
                return empty_history_frame()
        return self.backend.read_history(generated_ats=wanted, since=iso_utc(since))


def open_track_store(settings: Settings, *, fence: WriteFence | None = None) -> TrackStore:
    """Track log on Redis when ``track.redis_url`` is set; history and eval stay on output."""
    return TrackStore(
        settings.output,
        fence=fence,
        redis_url=settings.track.redis_url,
        redis_key_prefix=settings.track.redis_key_prefix,
    )

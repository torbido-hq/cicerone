"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol

import pandas as pd

from cicerone.config.settings import IOSettings
from cicerone.deprecation import renamed_attribute
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
    history_frame,
    iso_utc,
    jsonish,
    require_appendable_track_log,
    row_with_event_id,
)
from cicerone.track.store_dataset import TrackDatasetBackend, history_part_name
from cicerone.track.store_db import TrackDbBackend

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
    "TrackBackend",
    "TrackStore",
    "build_track_backend",
    "history_frame",
    "history_part_name",
    "jsonish",
    "require_appendable_track_log",
]


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
        writer_lock: LockBackend | None = None,
        fence_check: Callable[[], bool] | None = None,
        fence_lost: str = "lock lost before write",
        fence_kind: str = "lock",
        fence: WriteFence | None = None,
    ):
        self._fence = WriteFence.resolve(
            fence,
            writer_lock=writer_lock,
            fence_check=fence_check,
            fence_lost=fence_lost,
            fence_kind=fence_kind,
        )
        self.backend = build_track_backend(output, self._fence)

    @property
    def fence(self) -> WriteFence:
        return self._fence

    def append_accepted_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        return self.backend.append_accepted([row_with_event_id(row) for row in rows])

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return len(self.append_accepted_rows(rows))

    def read_rows(
        self,
        *,
        kind: str | None = None,
        experiment_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.backend.read_rows(kind=kind, experiment_id=experiment_id, since=iso_utc(since))

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
                return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return self.backend.read_history(generated_ats=wanted, since=iso_utc(since))


_RENAMED = {
    "_history_frame": "cicerone.track.store.history_frame",
    "_history_part_name": "cicerone.track.store.history_part_name",
    "_jsonish": "cicerone.track.store.jsonish",
}
__all__ += list(_RENAMED)


def __getattr__(name: str) -> Any:
    return renamed_attribute(__name__, _RENAMED, name)

"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Collection, Mapping, Sequence
from io import BytesIO
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from cicerone.config.settings import IOSettings
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
    _filter_history,
    _history_frame,
    _history_stem_before,  # noqa: F401
    _iso_utc,
    _jsonish,
    _row_matches,
    _row_with_event_id,
    _unslug_history_stem,  # noqa: F401
    require_appendable_track_log,
)
from cicerone.track.store_dataset import TrackDatasetBackend, _history_part_name
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
    "TrackStore",
    "_history_frame",
    "_history_part_name",
    "_history_stem_before",
    "_jsonish",
    "_unslug_history_stem",
    "require_appendable_track_log",
]


class TrackStore(TrackDbBackend, TrackDatasetBackend):
    """Output-store side channel for track rows, eval JSON, and rec snapshots."""

    def __init__(self, output: IOSettings):
        self._output = output
        self._kind = output.kind
        self._options = output.options
        self._engine: Engine | None = None
        self._known_ids: set[str] | None = None
        self._track_size: int | None = None
        self._append_lock = threading.Lock()

    def append_accepted_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        payload = [_row_with_event_id(row) for row in rows]
        if self._kind == "db":
            return self._append_rows_db(payload)
        require_appendable_track_log(self._output)
        with self._dataset_append_lock():
            known = self._refresh_known_ids()
            fresh: list[dict[str, Any]] = []
            for row in payload:
                event_id = str(row.get("event_id") or "")
                if event_id and event_id in known:
                    continue
                if event_id:
                    known.add(event_id)
                fresh.append(row)
            if not fresh:
                return []
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh).encode("utf-8")
            self._append_bytes(TRACK_FILENAME, encoded)
            self._track_size = (self._track_size or 0) + len(encoded)
            return fresh

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return len(self.append_accepted_rows(rows))

    def read_rows(
        self,
        *,
        kind: str | None = None,
        experiment_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        since = _iso_utc(since)
        if self._kind == "db":
            rows = self._read_rows_db(kind=kind, experiment_id=experiment_id)
        else:
            rows = self._read_rows_dataset()
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            if not _row_matches(row, kind=kind, experiment_id=experiment_id, since=since):
                continue
            unique.append(row)
        return unique

    def write_eval(self, report: Mapping[str, Any]) -> None:
        payload = dict(report)
        if self._kind == "db":
            self._write_eval_db(payload)
            return
        encoded = json.dumps(payload, indent=2).encode("utf-8")
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
        frame = _history_frame(recommendations, generated_at)
        if self._kind == "db":
            self._append_history_db(frame)
            return
        buf = BytesIO()
        frame.to_parquet(buf, index=False)
        part = f"{HISTORY_DIR}/{_history_part_name(generated_at)}"
        self._write_bytes(part, buf.getvalue(), "application/octet-stream")

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
        since = _iso_utc(since)
        if self._kind == "db":
            frame = self._read_history_db(generated_ats=wanted)
            return _filter_history(frame, generated_ats=wanted, since=since)
        frames = []
        if wanted is None and since is None:
            frames.extend(self._read_legacy_history())
        frames.extend(self._read_history_parts(generated_ats=wanted, since=since))
        if not frames:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return _filter_history(pd.concat(frames, ignore_index=True), generated_ats=wanted, since=since)

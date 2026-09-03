"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from cicerone.config.settings import IOSettings
from cicerone.track.store_common import (
    EVAL_FILENAME,
    HISTORY_COLUMNS,
    HISTORY_DIR,
    TRACK_FILENAME,
    TRACK_LOG_HA_ERROR,
    _history_frame,
    _jsonish,
    _row_with_event_id,
    require_appendable_track_log,
)
from cicerone.track.store_dataset import TrackDatasetBackend, _history_part_name
from cicerone.track.store_db import TrackDbBackend

logger = logging.getLogger(__name__)

__all__ = [
    "TRACK_LOG_HA_ERROR",
    "TrackStore",
    "_history_frame",
    "_history_part_name",
    "_jsonish",
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

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
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
                return 0
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh).encode("utf-8")
            self._append_bytes(TRACK_FILENAME, encoded)
            self._track_size = (self._track_size or 0) + len(encoded)
            return len(fresh)

    def read_rows(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self._read_rows_db() if self._kind == "db" else self._read_rows_dataset()
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            if kind is not None and str(row.get("kind") or "") != kind:
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

    def read_history(self) -> pd.DataFrame:
        if self._kind == "db":
            return self._read_history_db()
        frames = self._read_legacy_history() + self._read_history_parts()
        if not frames:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return pd.concat(frames, ignore_index=True)

"""Dataset backend for ExperimentStore (``experiment_state.json``, ``exposures.jsonl``)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cicerone.config.settings import IOSettings
from cicerone.io.blob import (
    append_storage_bytes,
    encode_jsonl,
    iter_jsonl_objects,
    json_file_writer,
    read_json_object,
    read_storage_bytes,
    require_db_or_local_dataset,
)
from cicerone.io.options import exclusive_file_lock, require_option
from cicerone.locks import WriteFence

logger = logging.getLogger(__name__)

STATE_FILENAME = "experiment_state.json"
EXPOSURES_FILENAME = "exposures.jsonl"
EXPOSURE_LOG_BACKEND_ERROR = (
    'experiment.log_exposures requires output kind = "db" or a local dataset path; '
    "object-store JSONL append is not atomic"
)


class ExperimentDatasetBackend:
    """Promote state as one JSON object, exposures as appended JSON lines."""

    def __init__(self, output: IOSettings, fence: WriteFence):
        self._output = output
        self._options = output.options
        self._fence = fence

    def read_state(self) -> dict[str, Any] | None:
        try:
            return read_json_object(self._options, STATE_FILENAME)
        except json.JSONDecodeError:
            logger.warning("Invalid experiment_state.json")
            raise

    def state_writer(self, state: dict[str, Any]) -> Callable[[], None]:
        return json_file_writer(self._options, STATE_FILENAME, state)

    def exposure_writer(self, rows: Sequence[Mapping[str, Any]]) -> Callable[[], None]:
        require_db_or_local_dataset(self._output, EXPOSURE_LOG_BACKEND_ERROR)
        encoded = encode_jsonl(rows)
        path = Path(require_option(self._options, "path", "local")) / ".exposures.jsonl.lock"

        def _append() -> None:
            with exclusive_file_lock(path):
                self._fence.ensure()
                append_storage_bytes(self._options, EXPOSURES_FILENAME, encoded)

        return _append

    def read_exposures(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        raw = read_storage_bytes(self._options, EXPOSURES_FILENAME)
        if raw is None:
            return []
        return [
            row
            for row in iter_jsonl_objects(raw.splitlines())
            if not experiment_id or str(row.get("experiment_id") or "") == experiment_id
        ]

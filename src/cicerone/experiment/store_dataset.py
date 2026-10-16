"""Dataset backend for ExperimentStore (``experiment_state.json``, ``exposures.jsonl``)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.blob import append_storage_bytes, read_storage_bytes, write_storage_bytes
from cicerone.io.options import exclusive_file_lock, require_option, storage_backend
from cicerone.locks import WriteFence

logger = logging.getLogger(__name__)

STATE_FILENAME = "experiment_state.json"
EXPOSURES_FILENAME = "exposures.jsonl"
EXPOSURE_LOG_BACKEND_ERROR = (
    'experiment.log_exposures requires output kind = "db" or a local dataset path; '
    "object-store JSONL append is not atomic"
)


def require_appendable_exposure_log(output: IOSettings) -> None:
    """Reject object-store JSONL append (read-modify-write is racy)."""
    if output.kind == "db":
        return
    if output.kind == "dataset" and storage_backend(output.options) == "local":
        return
    raise ConfigError(EXPOSURE_LOG_BACKEND_ERROR)


class ExperimentDatasetBackend:
    """Promote state as one JSON object, exposures as appended JSON lines."""

    def __init__(self, output: IOSettings, fence: WriteFence):
        self._output = output
        self._options = output.options
        self._fence = fence

    def read_state(self) -> dict[str, Any] | None:
        raw = read_storage_bytes(self._options, STATE_FILENAME)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid experiment_state.json")
            raise
        if not isinstance(parsed, dict):
            raise ValueError("experiment_state.json is not an object")
        return parsed

    def state_writer(self, state: dict[str, Any]) -> Callable[[], None]:
        encoded = json.dumps(state, indent=2).encode("utf-8")
        return lambda: write_storage_bytes(self._options, STATE_FILENAME, encoded, "application/json")

    def exposure_writer(self, rows: Sequence[Mapping[str, Any]]) -> Callable[[], None]:
        require_appendable_exposure_log(self._output)
        encoded = "".join(json.dumps(dict(row), separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
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
        return rows

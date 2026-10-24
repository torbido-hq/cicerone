"""Persist experiment promote state and optional serve-time exposures."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError

from cicerone.config.settings import IOSettings
from cicerone.experiment.store_dataset import (
    EXPOSURE_LOG_BACKEND_ERROR,
    EXPOSURES_FILENAME,
    STATE_FILENAME,
    ExperimentDatasetBackend,
    require_appendable_exposure_log,
)
from cicerone.experiment.store_db import (
    DEFAULT_EXPOSURES_TABLE,
    DEFAULT_STATE_TABLE,
    EXPOSURE_COLUMNS,
    STATE_CORE_KEYS,
    ExperimentDbBackend,
)
from cicerone.io.db_errors import SQL_READ_ERRORS
from cicerone.locks import WriteFence

logger = logging.getLogger(__name__)

_OVERLAY_READ_ERRORS = (OSError, ValueError, BotoCoreError, *SQL_READ_ERRORS)

EXPOSURE_LOG_HA_ERROR = 'experiment.log_exposures with events.ha requires output kind = "db"'

__all__ = [
    "DEFAULT_EXPOSURES_TABLE",
    "DEFAULT_STATE_TABLE",
    "EXPOSURES_FILENAME",
    "EXPOSURE_COLUMNS",
    "EXPOSURE_LOG_BACKEND_ERROR",
    "EXPOSURE_LOG_HA_ERROR",
    "STATE_CORE_KEYS",
    "STATE_FILENAME",
    "ExperimentBackend",
    "ExperimentStore",
    "active_pair_from_state",
    "build_experiment_backend",
    "experiment_state",
    "merge_experiment_state",
    "require_appendable_exposure_log",
]


class ExperimentBackend(Protocol):
    """Writers validate and encode eagerly; the returned call does the I/O under the writer fence.

    ``read_exposures`` returns only the given experiment's rows.
    """

    def read_state(self) -> dict[str, Any] | None: ...

    def state_writer(self, state: dict[str, Any]) -> Callable[[], None]: ...

    def exposure_writer(self, rows: Sequence[Mapping[str, Any]]) -> Callable[[], None]: ...

    def read_exposures(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]: ...


def build_experiment_backend(output: IOSettings, fence: WriteFence) -> ExperimentBackend:
    if output.kind == "db":
        return ExperimentDbBackend(output.options, fence)
    return ExperimentDatasetBackend(output, fence)


def experiment_state(
    experiment_id: str,
    *,
    promoted_variant: str | None,
    promoted_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "promoted_variant": promoted_variant,
        "promoted_at": promoted_at
        if promoted_at is not None
        else (datetime.now(UTC).isoformat() if promoted_variant else None),
    }
    for key, value in extra.items():
        if key not in STATE_CORE_KEYS and key != "payload":
            payload[key] = value
    return payload


def merge_experiment_state(
    previous: Mapping[str, Any] | None,
    *,
    experiment_id: str,
    promoted_variant: str | None,
    promoted_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    merged = dict(previous or {})
    merged.update(
        experiment_state(
            experiment_id,
            promoted_variant=promoted_variant,
            promoted_at=promoted_at,
            **extra,
        )
    )
    return merged


def active_pair_from_state(state: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not state:
        return None
    champion = str(state.get("champion") or "")
    challenger = str(state.get("challenger") or "")
    if champion and challenger:
        return champion, challenger
    return None


class ExperimentStore:
    """Output-store side channel for promote state and exposure logs."""

    def __init__(
        self,
        output: IOSettings,
        *,
        fence: WriteFence | None = None,
    ):
        self._promote_lock = threading.Lock()
        self._promote_loaded = False
        self._promote_experiment_id: str | None = None
        self._promote_value: str | None = None
        self._promote_state: dict[str, Any] | None = None
        self._fence = fence or WriteFence()
        self.backend = build_experiment_backend(output, self._fence)

    def read_state(self) -> dict[str, Any] | None:
        state = self.backend.read_state()
        self._remember_state(state)
        return state

    def last_state(self, experiment_id: str) -> dict[str, Any] | None:
        """Last successful promote-state row for ``experiment_id``, if any."""
        wanted = str(experiment_id)
        with self._promote_lock:
            if not self._promote_loaded or self._promote_state is None:
                return None
            if str(self._promote_state.get("experiment_id") or "") != wanted:
                return None
            return dict(self._promote_state)

    def _remember_state(self, state: dict[str, Any] | None) -> None:
        with self._promote_lock:
            self._promote_loaded = True
            if state and state.get("experiment_id"):
                self._promote_experiment_id = str(state.get("experiment_id") or "")
                promoted = state.get("promoted_variant")
                self._promote_value = str(promoted) if promoted else None
                self._promote_state = dict(state)
            else:
                self._promote_experiment_id = None
                self._promote_value = None
                self._promote_state = None

    def promoted_variant(self, experiment_id: str) -> str | None:
        """Return the live promote winner; reuse the last successful read on failure."""
        return self.assignment_overlay(experiment_id)[0]

    def assignment_overlay(self, experiment_id: str) -> tuple[str | None, tuple[str, str] | None]:
        """Return ``(promoted_variant, active_pair)``; reuse the last successful read on failure."""
        wanted = str(experiment_id)
        try:
            state = self.read_state()
        except _OVERLAY_READ_ERRORS:
            logger.exception("Failed to read experiment promote state")
            with self._promote_lock:
                if self._promote_loaded and self._promote_experiment_id == wanted:
                    return self._promote_value, active_pair_from_state(self._promote_state)
            return None, None
        if state and str(state.get("experiment_id") or "") == wanted:
            promoted = state.get("promoted_variant")
            return (str(promoted) if promoted else None), active_pair_from_state(state)
        return None, None

    def write_state(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        write = self.backend.state_writer(payload)

        def _persist() -> None:
            self._fence.ensure()
            write()

        if self._fence.held_here():
            _persist()
        else:
            with self._fence.hold():
                _persist()
        promoted = payload.get("promoted_variant")
        with self._promote_lock:
            self._promote_loaded = True
            self._promote_experiment_id = str(payload.get("experiment_id") or "")
            self._promote_value = str(promoted) if promoted else None
            self._promote_state = dict(payload)

    def append_exposures(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        write = self.backend.exposure_writer(rows)

        def _persist() -> None:
            self._fence.ensure()
            write()

        if self._fence.held_here():
            _persist()
            return
        with self._fence.hold():
            _persist()

    def read_exposures(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.backend.read_exposures(experiment_id=experiment_id)
        if not experiment_id:
            return rows
        # Backends filter too; this keeps the string match the db filter may not (a legacy numeric column).
        return [row for row in rows if str(row.get("experiment_id") or "") == experiment_id]

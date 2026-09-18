"""Sidecar publish helpers shared by the job and incremental updater."""

from __future__ import annotations

from cicerone.config.settings import IOSettings
from cicerone.io.factory import build_manifest_reader
from cicerone.job_eval import OPTIONAL_IO_ERRORS, log_caught
from cicerone.locks import LockLostError, WriterLockBusyError


def sidecar_generation_current(output: IOSettings, generated_at: str) -> bool:
    try:
        latest = build_manifest_reader(output).read_latest()
    except (LockLostError, WriterLockBusyError):
        raise
    except OPTIONAL_IO_ERRORS as exc:
        log_caught("Failed to read manifest generation before sidecar publish", exc)
        return False
    current = latest.get("generated_at") if latest else None
    return current is not None and str(current) == generated_at

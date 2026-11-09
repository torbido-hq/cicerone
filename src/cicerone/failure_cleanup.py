"""Cleanup that runs while another error propagates (leaf module: publish/ and events/ both use it)."""

from __future__ import annotations

import logging
from collections.abc import Callable


def cleanup_after_failure(cleanup: Callable[[], object], message: str, log: logging.Logger) -> None:
    # Any cleanup error is logged so the error already propagating wins.
    try:
        cleanup()
    except Exception:
        log.exception(message)

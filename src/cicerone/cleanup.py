"""Run a cleanup step and log any error it raises, so an error already propagating wins.

A leaf module (no cicerone imports) so publish/ and events/ can use it without an import cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable


def cleanup_logging_errors(
    cleanup: Callable[[], object], message: str, log: logging.Logger
) -> Exception | None:
    """Return the logged error, or None when the cleanup succeeded."""
    try:
        cleanup()
    except Exception as exc:
        log.exception(message)
        return exc
    return None

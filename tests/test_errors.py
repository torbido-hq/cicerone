from __future__ import annotations

import logging
import re

import pytest
from pyarrow.lib import ArrowTypeError

from cicerone.errors import OPTIONAL_EVAL_ERRORS, OPTIONAL_IO_ERRORS, log_caught, try_load
from cicerone.locks import LockLostError, WriterLockBusyError

_LOG = logging.getLogger("cicerone.test_errors")


def _raise(exc: BaseException):
    def fn() -> None:
        raise exc

    return fn


def test_log_caught_defaults_to_errors_logger(caplog):
    with caplog.at_level("ERROR"):
        log_caught("Failed to read", ValueError("bad"))
    assert [(record.name, record.getMessage()) for record in caplog.records] == [
        ("cicerone.errors", "Failed to read (ValueError: bad)")
    ]


def test_try_load_logs_type_and_message_on_given_logger(caplog):
    with caplog.at_level("ERROR"):
        assert try_load("load previous recs", _raise(OSError("recs")), None, log=_LOG) is None
    assert [(record.name, record.getMessage()) for record in caplog.records] == [
        ("cicerone.test_errors", "Failed to load previous recs (OSError: recs)")
    ]


def test_try_load_treats_arrow_type_errors_as_optional():
    assert try_load("load recs", _raise(ArrowTypeError("schema mismatch")), None, log=_LOG) is None


@pytest.mark.parametrize(
    ("exc", "errors"),
    [
        (RuntimeError("bug"), OPTIONAL_IO_ERRORS),
        (TypeError("bug"), OPTIONAL_IO_ERRORS),
        (TypeError("bug"), OPTIONAL_EVAL_ERRORS),
        (LockLostError("retrain lock lost", kind="retrain"), OPTIONAL_IO_ERRORS),
        (WriterLockBusyError("dataset writer lock busy"), OPTIONAL_IO_ERRORS),
    ],
)
def test_try_load_propagates_unexpected_errors(exc, errors):
    with pytest.raises(type(exc), match=re.escape(str(exc))):
        try_load("write track eval", _raise(exc), None, log=_LOG, errors=errors)

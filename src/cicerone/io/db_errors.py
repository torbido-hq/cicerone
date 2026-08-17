"""Shared SQLAlchemy error classification for recommendation I/O."""

from __future__ import annotations

from sqlalchemy.exc import ProgrammingError


def db_error_message(exc: BaseException) -> str:
    return str(getattr(exc, "orig", exc)).lower()


def is_missing_column_error(exc: BaseException) -> bool:
    message = db_error_message(exc)
    return "no such column" in message or ("column" in message and "does not exist" in message)


def is_missing_table_error(exc: BaseException) -> bool:
    message = db_error_message(exc)
    return isinstance(exc, ProgrammingError) or "does not exist" in message or "no such table" in message

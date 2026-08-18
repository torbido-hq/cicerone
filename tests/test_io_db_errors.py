from __future__ import annotations

from sqlalchemy.exc import OperationalError, ProgrammingError

from cicerone.io.db_errors import db_error_message, is_missing_column_error, is_missing_table_error


def test_is_missing_column_error():
    exc = OperationalError("stmt", {}, Exception("no such column: user_id"))
    assert is_missing_column_error(exc)
    exc = ProgrammingError("stmt", {}, Exception('column "user_id" does not exist'))
    assert is_missing_column_error(exc)
    assert not is_missing_column_error(OperationalError("stmt", {}, Exception("connection refused")))


def test_is_missing_table_error():
    exc = OperationalError("stmt", {}, Exception("no such table: recommendations"))
    assert is_missing_table_error(exc)
    assert is_missing_table_error(ProgrammingError("stmt", {}, Exception("boom")))
    assert not is_missing_table_error(OperationalError("stmt", {}, Exception("disk full")))


def test_db_error_message_uses_orig():
    exc = OperationalError("stmt", {}, Exception("ORIG MSG"))
    assert db_error_message(exc) == "orig msg"

"""Always-on SQLite coverage for bounded model-artifact reads."""

from __future__ import annotations

import pytest

from cicerone.io.db_store import DatabaseOutputSink


def test_database_read_model_artifact_rejects_oversize(tmp_path):
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"123456")
    with pytest.raises(ValueError, match="max is 4"):
        sink.read_model_artifact(max_bytes=4)


def test_database_read_model_artifact_rejects_nonpositive_max_bytes(tmp_path):
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    with pytest.raises(ValueError, match="max_bytes"):
        sink.read_model_artifact(max_bytes=0)


def test_database_read_model_artifact_returns_payload_within_limit(tmp_path):
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"ok")
    assert sink.read_model_artifact(max_bytes=8) == b"ok"


def test_database_read_model_artifact_uses_one_select(tmp_path):
    from sqlalchemy import event

    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"123456")
    statements: list[str] = []

    def _capture(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(sink._engine, "before_cursor_execute", _capture)
    try:
        with pytest.raises(ValueError, match="max is 4"):
            sink.read_model_artifact(max_bytes=4)
    finally:
        event.remove(sink._engine, "before_cursor_execute", _capture)
    payload_selects = [sql for sql in statements if "payload" in sql.lower()]
    assert len(payload_selects) == 1
    assert "substr" in payload_selects[0].lower()

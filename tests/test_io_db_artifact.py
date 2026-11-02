"""Always-on SQLite coverage for bounded model-artifact reads."""

from __future__ import annotations

import pytest

from cicerone.io.db_store import DatabaseOutputSink


def test_database_read_model_artifact_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr("cicerone.io.db_store.DEFAULT_MAX_ARTIFACT_BYTES", 4)
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"123456")
    with pytest.raises(ValueError, match="max is 4"):
        sink.read_model_artifact()


def test_database_read_model_artifact_returns_payload_within_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("cicerone.io.db_store.DEFAULT_MAX_ARTIFACT_BYTES", 8)
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"ok")
    assert sink.read_model_artifact() == b"ok"


def test_database_read_model_artifact_uses_one_select(tmp_path, monkeypatch):
    from sqlalchemy import event

    monkeypatch.setattr("cicerone.io.db_store.DEFAULT_MAX_ARTIFACT_BYTES", 4)

    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"123456")
    statements: list[tuple[str, object]] = []

    def _capture(
        _conn: object,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append((statement, parameters))

    event.listen(sink._engine, "before_cursor_execute", _capture)
    try:
        with pytest.raises(ValueError, match="max is 4"):
            sink.read_model_artifact()
    finally:
        event.remove(sink._engine, "before_cursor_execute", _capture)
    payload_selects = [(sql, params) for sql, params in statements if "payload" in sql.lower()]
    assert len(payload_selects) == 1
    sql, params = payload_selects[0]
    assert "substr" in sql.lower()
    assert params == (5,)


def test_database_replace_model_artifact_if_matches_the_sqlite_fingerprint(tmp_path):
    sink = DatabaseOutputSink({"database_url": f"sqlite:///{tmp_path}/artifacts.sqlite"})
    sink.write_model_artifact(b"first")
    token = sink.model_artifact_fingerprint()
    assert token is not None
    assert sink.replace_model_artifact_if(b"second", token) is True
    assert sink.read_model_artifact() == b"second"
    assert sink.replace_model_artifact_if(b"third", token) is False

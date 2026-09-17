"""Unit tests for shared I/O helpers in cicerone.io.options."""

from __future__ import annotations

import errno

import pytest
from botocore.exceptions import ClientError

from cicerone.config import ConfigError
from cicerone.io.options import (
    S3_NOT_FOUND_CODES,
    close_s3_body,
    exclusive_file_lock,
    is_s3_not_found,
    object_key,
    read_s3_body,
    readonly_select,
    storage_backend,
    validate_storage_options,
)


class _FakeS3Body:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.closed = False
        self.reads: list[int | None] = []

    def read(self, size: int | None = None) -> bytes:
        self.reads.append(size)
        if self._fail:
            raise RuntimeError("read failed")
        if size is None:
            return self._payload
        return self._payload[:size]

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "options,filename,expected",
    [
        ({}, "file.json", "file.json"),
        ({"prefix": ""}, "file.json", "file.json"),
        ({"prefix": "artifacts"}, "file.json", "artifacts/file.json"),
        ({"prefix": "artifacts/"}, "file.json", "artifacts/file.json"),
        ({"prefix": "artifacts/subdir/"}, "file.json", "artifacts/subdir/file.json"),
        ({"prefix": "/artifacts/"}, "file.json", "artifacts/file.json"),
    ],
)
def test_object_key_handles_empty_and_trailing_prefix(options, filename, expected):
    assert object_key(options, filename) == expected


@pytest.mark.parametrize("error_code", sorted(S3_NOT_FOUND_CODES))
def test_is_s3_not_found_true_for_not_found_codes(error_code):
    exc = ClientError(
        error_response={"Error": {"Code": error_code, "Message": "Not found"}},
        operation_name="HeadObject",
    )
    assert is_s3_not_found(exc) is True


def test_is_s3_not_found_false_for_other_client_error():
    exc = ClientError(
        error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
        operation_name="HeadObject",
    )
    assert is_s3_not_found(exc) is False


@pytest.mark.parametrize("exc", [ValueError("boom"), Exception("boom")])
def test_is_s3_not_found_false_for_non_client_error_exceptions(exc):
    assert is_s3_not_found(exc) is False


def test_read_s3_body_closes_after_read():
    body = _FakeS3Body(b"payload")
    assert read_s3_body({"Body": body}) == b"payload"
    assert body.closed is True


def test_read_s3_body_closes_when_read_fails():
    body = _FakeS3Body(b"payload", fail=True)
    with pytest.raises(RuntimeError, match="read failed"):
        read_s3_body({"Body": body})
    assert body.closed is True


def test_close_s3_body_ignores_missing_close():
    close_s3_body(object())


def test_read_s3_body_rejects_content_length_and_closes():
    body = _FakeS3Body(b"123456789")
    with pytest.raises(ValueError, match="max is 4"):
        read_s3_body({"Body": body, "ContentLength": 9}, max_bytes=4)
    assert body.closed is True


def test_read_s3_body_rejects_oversize_read_and_closes():
    body = _FakeS3Body(b"123456789")
    with pytest.raises(ValueError, match="max is 4"):
        read_s3_body({"Body": body}, max_bytes=4)
    assert body.reads == [5]
    assert body.closed is True


def test_read_s3_body_rejects_nonpositive_max_bytes():
    body = _FakeS3Body(b"x")
    with pytest.raises(ValueError, match="max_bytes"):
        read_s3_body({"Body": body}, max_bytes=0)
    assert body.closed is True


def test_read_parquet_s3_closes_body(mocker):
    import pandas as pd

    from cicerone.io.options import read_parquet

    body = _FakeS3Body(b"parquet-bytes")
    client = mocker.Mock()
    client.get_object.return_value = {"Body": body}
    mocker.patch("cicerone.io.options.pd.read_parquet", return_value=pd.DataFrame({"x": [1]}))
    frame = read_parquet(
        {
            "storage_backend": "s3",
            "access_key_id": "id",
            "secret_access_key": "secret",
            "bucket": "bucket",
        },
        "data.parquet",
        s3_client=client,
    )
    assert list(frame.columns) == ["x"]
    assert body.closed is True


def test_validate_storage_options_resolves_from_options():
    assert validate_storage_options({"storage_backend": "local", "path": "/tmp"}) == "local"


def test_validate_storage_options_rejects_explicit_backend_mismatch():
    with pytest.raises(ConfigError, match="does not match"):
        validate_storage_options({"storage_backend": "local", "path": "/tmp"}, backend="s3")


def test_storage_backend_rejects_unknown():
    with pytest.raises(ConfigError, match="Unknown storage_backend"):
        storage_backend({"storage_backend": "gcs"})


def test_validate_storage_options_rejects_unknown_backend():
    with pytest.raises(ConfigError, match="Unknown storage_backend"):
        validate_storage_options({"storage_backend": "gcs", "path": "/tmp"})


def test_readonly_select_accepts_simple_select():
    assert readonly_select("SELECT * FROM events;", option="q") == "SELECT * FROM events"


def test_readonly_select_accepts_lower_function():
    assert (
        readonly_select("SELECT lower(user_id) FROM events", option="q")
        == "SELECT lower(user_id) FROM events"
    )


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM events",
        "SELECT 1; DROP TABLE events",
        "SELECT * FROM events INTO dump",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_read_binary_file('/etc/passwd')",
        "SELECT pg_write_file('/tmp/x', 'x')",
        "SELECT pg_write_binary_file('/tmp/x', 'x')",
        "SELECT pg_ls_dir('/')",
        "SELECT pg_ls_logdir()",
        "SELECT pg_ls_waldir()",
        "SELECT pg_stat_file('/etc/passwd')",
        "SELECT pg_reload_conf()",
        "SELECT pg_execute_server_program('id')",
        "SELECT pg_terminate_backend(1)",
        "SELECT pg_cancel_backend(1)",
        "SELECT set_config('x', 'y', false)",
        "SELECT dblink('host=x', 'SELECT 1')",
        "SELECT dblink_exec('conn', 'DROP TABLE events')",
        "SELECT lo_get(1)",
        "SELECT lo_put(1, 0, 'x')",
        "SELECT lo_from_bytea(0, 'x')",
        "SELECT lo_unlink(1)",
        "",
    ],
)
def test_readonly_select_rejects_writes(query):
    with pytest.raises(ValueError, match="q"):
        readonly_select(query, option="q")


def test_exclusive_file_lock_serializes_without_fcntl(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr("cicerone.io.options.fcntl", None)
    path = tmp_path / "writers.lock"
    seen: list[int] = []
    hold = threading.Event()
    entered = threading.Event()

    def _hold() -> None:
        with exclusive_file_lock(path):
            entered.set()
            hold.wait(timeout=2)
            seen.append(1)

    first = threading.Thread(target=_hold)
    first.start()
    assert entered.wait(timeout=2)
    second_started = threading.Event()

    def _second() -> None:
        second_started.set()
        with exclusive_file_lock(path):
            seen.append(2)

    second = threading.Thread(target=_second)
    second.start()
    assert second_started.wait(timeout=2)
    assert seen == []
    hold.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert seen == [1, 2]


def test_exclusive_file_lock_times_out(tmp_path):
    import threading

    from cicerone.locks import WriterLockBusyError

    path = tmp_path / "writers.lock"
    entered = threading.Event()
    hold = threading.Event()

    def _hold() -> None:
        with exclusive_file_lock(path):
            entered.set()
            hold.wait(timeout=2)

    first = threading.Thread(target=_hold)
    first.start()
    assert entered.wait(timeout=2)
    with (
        pytest.raises(WriterLockBusyError, match="dataset writer lock busy"),
        exclusive_file_lock(path, timeout_seconds=0.1),
    ):
        pass
    hold.set()
    first.join(timeout=2)


def test_exclusive_file_lock_flock_times_out(tmp_path, monkeypatch):
    from cicerone.locks import WriterLockBusyError

    class _Fcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        def flock(self, _fd: int, op: int) -> None:
            if op != self.LOCK_UN:
                raise OSError(errno.EAGAIN, "busy")

    monkeypatch.setattr("cicerone.io.options.fcntl", _Fcntl())
    with (
        pytest.raises(WriterLockBusyError, match="dataset writer lock busy"),
        exclusive_file_lock(tmp_path / "writers.lock", timeout_seconds=0.0),
    ):
        pass


def test_exclusive_file_lock_reraises_unsupported_flock(tmp_path, monkeypatch):
    class _Fcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        def flock(self, _fd: int, op: int) -> None:
            if op != self.LOCK_UN:
                raise OSError(errno.ENOTSUP, "not supported")

    monkeypatch.setattr("cicerone.io.options.fcntl", _Fcntl())
    with (
        pytest.raises(OSError, match="not supported"),
        exclusive_file_lock(tmp_path / "writers.lock", timeout_seconds=0.1),
    ):
        pass


def test_exclusive_file_lock_uses_msvcrt_when_fcntl_missing(tmp_path, monkeypatch):
    locked: list[int] = []

    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def locking(self, _fd: int, mode: int, _nbytes: int) -> None:
            locked.append(mode)

    monkeypatch.setattr("cicerone.io.options.fcntl", None)
    monkeypatch.setattr("cicerone.io.options.msvcrt", _Msvcrt())
    path = tmp_path / "writers.lock"
    with exclusive_file_lock(path):
        assert locked == [_Msvcrt.LK_NBLCK]
        assert path.read_bytes()[:1] == b"\0"
    assert locked == [_Msvcrt.LK_NBLCK, _Msvcrt.LK_UNLCK]


def test_exclusive_file_lock_msvcrt_times_out(tmp_path, monkeypatch):
    from cicerone.locks import WriterLockBusyError

    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def locking(self, _fd: int, mode: int, _nbytes: int) -> None:
            if mode != self.LK_UNLCK:
                raise OSError(errno.EACCES, "busy")

    monkeypatch.setattr("cicerone.io.options.fcntl", None)
    monkeypatch.setattr("cicerone.io.options.msvcrt", _Msvcrt())
    with (
        pytest.raises(WriterLockBusyError, match="dataset writer lock busy"),
        exclusive_file_lock(tmp_path / "writers.lock", timeout_seconds=0.0),
    ):
        pass


def test_exclusive_file_lock_reraises_unsupported_msvcrt(tmp_path, monkeypatch):
    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def locking(self, _fd: int, mode: int, _nbytes: int) -> None:
            if mode != self.LK_UNLCK:
                raise OSError(errno.ENOTSUP, "not supported")

    monkeypatch.setattr("cicerone.io.options.fcntl", None)
    monkeypatch.setattr("cicerone.io.options.msvcrt", _Msvcrt())
    with (
        pytest.raises(OSError, match="not supported"),
        exclusive_file_lock(tmp_path / "writers.lock", timeout_seconds=0.1),
    ):
        pass

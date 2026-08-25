"""Unit tests for shared I/O helpers in cicerone.io.options."""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError

from cicerone.config import ConfigError
from cicerone.io.options import (
    S3_NOT_FOUND_CODES,
    _S3RangeFile,
    is_s3_not_found,
    object_key,
    storage_backend,
    validate_storage_options,
)


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


def test_s3_range_file_seek_and_closed_read():
    class _Client:
        def get_object(self, **kwargs):
            raise AssertionError("should not GET on seek")

    f = _S3RangeFile(_Client(), "b", "k", 10)
    assert f.seekable()
    assert f.readable()
    assert not f.writable()
    assert f.seek(2) == 2
    assert f.seek(1, io.SEEK_CUR) == 3
    assert f.seek(-1, io.SEEK_END) == 9
    assert f.tell() == 9
    with pytest.raises(ValueError, match="invalid whence"):
        f.seek(0, 99)
    with pytest.raises(ValueError, match="negative"):
        f.seek(-1)
    f.close()
    with pytest.raises(ValueError, match="closed"):
        f.read(1)


def test_s3_range_file_read_empty_and_readinto():
    class _Client:
        def get_object(self, **kwargs):
            raise AssertionError("empty object should not GET")

    empty = _S3RangeFile(_Client(), "b", "k", 0)
    assert empty.read() == b""
    assert empty.read(None) == b""
    assert empty.read(0) == b""
    buf = bytearray(4)
    assert empty.readinto(buf) == 0

    payload = b"abcdefghij"

    class _RangeClient:
        def get_object(self, **kwargs):
            spec = str(kwargs["Range"]).removeprefix("bytes=")
            start_s, end_s = spec.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            body = io.BytesIO(payload[start : end + 1])
            return {"Body": body}

    ranged = _S3RangeFile(_RangeClient(), "b", "k", len(payload))
    assert ranged.read(4) == b"abcd"
    into = bytearray(3)
    assert ranged.readinto(into) == 3
    assert bytes(into) == b"efg"

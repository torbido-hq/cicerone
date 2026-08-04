"""Unit tests for shared I/O helpers in cicerone.io.options."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from cicerone.io.options import S3_NOT_FOUND_CODES, is_s3_not_found, object_key


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

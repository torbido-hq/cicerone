"""Shared helpers for I/O backends configured via an options dict."""

from __future__ import annotations

import re
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
S3_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NotFound"})


def require_option(options: dict[str, Any], key: str, backend: str) -> Any:
    value = options.get(key)
    if value is None:
        raise RuntimeError(f"Missing required option '{key}' for backend {backend!r}")
    return value


def object_key(options: dict[str, Any], filename: str) -> str:
    prefix = str(options.get("prefix", "")).strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def sql_identifier(name: str, *, option: str) -> str:
    if not isinstance(name, str) or not _SQL_IDENTIFIER.fullmatch(name):
        raise ValueError(
            f"{option} must be a simple SQL identifier matching [A-Za-z_][A-Za-z0-9_]*, got {name!r}"
        )
    return name


def is_s3_not_found(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code")
    return code in S3_NOT_FOUND_CODES


def build_s3_client(options: dict[str, Any]):
    return boto3.client(
        "s3",
        endpoint_url=options.get("endpoint_url"),
        aws_access_key_id=require_option(options, "access_key_id", "s3"),
        aws_secret_access_key=require_option(options, "secret_access_key", "s3"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )

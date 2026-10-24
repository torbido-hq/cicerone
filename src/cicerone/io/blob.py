"""Local/S3 byte helpers shared by dataset, track, and experiment stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from cicerone.config.constants import DEFAULT_MAX_STORAGE_READ_BYTES
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    read_capped,
    read_parquet,
    read_s3_body,
    require_max_bytes,
    require_option,
    validate_storage_options,
)

S3_READ_ERRORS: tuple[type[BaseException], ...] = (BotoCoreError, ClientError)
STORAGE_READ_ERRORS: tuple[type[BaseException], ...] = (FileNotFoundError, *S3_READ_ERRORS)


def is_storage_not_found(exc: BaseException) -> bool:
    return isinstance(exc, FileNotFoundError) or is_s3_not_found(exc)


def read_parquet_if_exists(options: dict[str, Any], filename: str, **kwargs: Any) -> pd.DataFrame | None:
    try:
        return read_parquet(options, filename, **kwargs)
    except STORAGE_READ_ERRORS as exc:
        if is_storage_not_found(exc):
            return None
        raise


def read_storage_bytes(
    options: dict[str, Any],
    filename: str,
    *,
    max_bytes: int = DEFAULT_MAX_STORAGE_READ_BYTES,
) -> bytes | None:
    require_max_bytes(max_bytes)
    backend = validate_storage_options(options)
    if backend == "local":
        path = Path(require_option(options, "path", "local")) / filename
        try:
            with path.open("rb") as handle:
                return read_capped(handle.read, max_bytes)
        except FileNotFoundError:
            return None
    bucket = require_option(options, "bucket", "s3")
    key = object_key(options, filename)
    client = build_s3_client(options)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except S3_READ_ERRORS as exc:
        if is_s3_not_found(exc):
            return None
        raise
    return read_s3_body(response, max_bytes=max_bytes)


def write_storage_bytes(options: dict[str, Any], filename: str, payload: bytes, content_type: str) -> None:
    backend = validate_storage_options(options)
    if backend == "local":
        path = Path(require_option(options, "path", "local")) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return
    bucket = require_option(options, "bucket", "s3")
    key = object_key(options, filename)
    client = build_s3_client(options)
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def append_storage_bytes(options: dict[str, Any], filename: str, payload: bytes) -> None:
    path = Path(require_option(options, "path", "local")) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)

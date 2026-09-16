"""Local/S3 byte helpers shared by dataset, track, and experiment stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    require_option,
    validate_storage_options,
)


def read_storage_bytes(options: dict[str, Any], filename: str) -> bytes | None:
    backend = validate_storage_options(options)
    if backend == "local":
        path = Path(require_option(options, "path", "local")) / filename
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
    bucket = require_option(options, "bucket", "s3")
    key = object_key(options, filename)
    client = build_s3_client(options)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if is_s3_not_found(exc):
            return None
        raise
    body = response["Body"]
    try:
        return body.read()
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()


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

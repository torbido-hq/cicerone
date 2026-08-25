"""Shared helpers for I/O backends configured via an options dict."""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError

from cicerone.config import ConfigError

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
S3_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
STORAGE_BACKENDS = frozenset({"s3", "local"})


def require_option(options: dict[str, Any], key: str, backend: str) -> Any:
    value = options.get(key)
    if value is None:
        raise ConfigError(f"Missing required option '{key}' for backend {backend!r}")
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


def storage_backend(options: dict[str, Any]) -> str:
    backend = options.get("storage_backend", "local")
    if backend not in STORAGE_BACKENDS:
        raise ConfigError(f"Unknown storage_backend: {backend!r} (expected 's3' or 'local')")
    return str(backend)


def validate_storage_options(options: dict[str, Any], backend: str | None = None) -> str:
    """Validate ``storage_backend`` and required options; return the resolved backend.

    Always resolves from ``options``. If ``backend`` is passed, it must match
    ``options['storage_backend']`` (or the default ``local``).

    Raises ``ConfigError`` for unknown backends, backend mismatches, or missing
    required options.
    """
    resolved = storage_backend(options)
    if backend is not None and str(backend) != resolved:
        raise ConfigError(f"explicit backend {backend!r} does not match options storage_backend {resolved!r}")
    if resolved == "local":
        require_option(options, "path", "local")
    else:
        require_option(options, "access_key_id", "s3")
        require_option(options, "secret_access_key", "s3")
        require_option(options, "bucket", "s3")
    return resolved


class _S3RangeFile(io.RawIOBase):
    """Random-access S3 object that issues Range GETs instead of downloading the body."""

    def __init__(self, client: Any, bucket: str, key: str, size: int) -> None:
        super().__init__()
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError(f"negative seek value {pos}")
        self._pos = pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        if self.closed:
            raise ValueError("read of closed file")
        if size is not None and size < 0:
            size = -1
        if self._pos >= self._size or size == 0:
            return b""
        remaining = self._size - self._pos
        length = remaining if size is None or size < 0 else min(size, remaining)
        start = self._pos
        end = start + length - 1
        obj = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={start}-{end}",
        )
        data = obj["Body"].read()
        self._pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n


def _read_s3_parquet_filtered(
    client: Any,
    bucket: str,
    key: str,
    *,
    columns: Sequence[str] | None,
    filters: Sequence[Any],
) -> pd.DataFrame:
    size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    table = pq.read_table(
        _S3RangeFile(client, bucket, key, size),
        columns=list(columns) if columns is not None else None,
        filters=list(filters),
        pre_buffer=False,
    )
    return table.to_pandas()


def read_parquet(
    options: dict[str, Any],
    filename: str,
    *,
    s3_client: Any | None = None,
    columns: Sequence[str] | None = None,
    filters: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Read a parquet object from local path or S3 using ``storage_backend`` options.

    When ``columns`` is set, only those columns are loaded (projection pushdown
    where the parquet engine supports it). ``filters`` use row-group predicate
    pushdown; on S3 that is ranged ``GetObject``, not a full body download.
    """
    backend = validate_storage_options(options)
    read_kwargs: dict[str, Any] = {}
    if columns is not None:
        read_kwargs["columns"] = list(columns)
    if filters is not None:
        read_kwargs["filters"] = list(filters)
    if backend == "local":
        path = Path(require_option(options, "path", "local")) / filename
        logger.info("Reading %s", path)
        return pd.read_parquet(path, **read_kwargs)

    bucket = require_option(options, "bucket", "s3")
    key = object_key(options, filename)
    logger.info("Reading s3://%s/%s", bucket, key)
    client = s3_client if s3_client is not None else build_s3_client(options)
    if filters is not None:
        return _read_s3_parquet_filtered(client, bucket, key, columns=columns, filters=filters)
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), **read_kwargs)

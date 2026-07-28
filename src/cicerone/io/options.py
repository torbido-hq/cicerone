"""Small shared helpers for I/O backends configured via an "options" dict
(see cicerone.config.IOSettings). Centralized here so every backend reports
missing required options / builds S3 clients the same way.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config


def require_option(options: dict[str, Any], key: str, backend: str) -> Any:
    value = options.get(key)
    if value is None:
        raise RuntimeError(f"Missing required option '{key}' for backend {backend!r}")
    return value


def build_s3_client(options: dict[str, Any]):
    return boto3.client(
        "s3",
        endpoint_url=options.get("endpoint_url"),
        aws_access_key_id=require_option(options, "access_key_id", "s3"),
        aws_secret_access_key=require_option(options, "secret_access_key", "s3"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )

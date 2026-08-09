"""Postgres lock URL resolution for ``[job.trigger].lock_backend = "postgres"``."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.config.settings import Settings

POSTGRES_LOCK_URL_REQUIRED = (
    'job.trigger.lock_backend = "postgres" needs a database URL: set '
    '[job.trigger].postgres_url, or use [output].kind = "db" with '
    "[output.options].database_url"
)


def resolve_postgres_lock_url_parts(
    *,
    postgres_url: str | None,
    output_kind: str,
    output_options: dict[str, Any],
) -> str | None:
    """Explicit trigger URL wins; otherwise reuse ``[output]`` when ``kind = "db"``."""
    if postgres_url:
        return postgres_url
    if output_kind == "db":
        url = output_options.get("database_url")
        return str(url) if url else None
    return None


def require_postgres_lock_url_parts(
    *,
    postgres_url: str | None,
    output_kind: str,
    output_options: dict[str, Any],
) -> str:
    """Validate at config load; returns the resolved URL."""
    url = resolve_postgres_lock_url_parts(
        postgres_url=postgres_url,
        output_kind=output_kind,
        output_options=output_options,
    )
    if not url:
        raise ConfigError(POSTGRES_LOCK_URL_REQUIRED)
    return url


def resolve_postgres_lock_url(settings: Settings) -> str | None:
    return resolve_postgres_lock_url_parts(
        postgres_url=settings.trigger.postgres_url,
        output_kind=settings.output.kind,
        output_options=settings.output.options,
    )

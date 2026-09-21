"""Redis clients with a health-checked connection pool."""

from __future__ import annotations

from typing import Any

REDIS_HEALTH_CHECK_INTERVAL_SECONDS = 30


def redis_from_url(url: str, **kwargs: Any) -> Any:
    import redis

    kwargs.setdefault("health_check_interval", REDIS_HEALTH_CHECK_INTERVAL_SECONDS)
    return redis.Redis.from_url(url, **kwargs)

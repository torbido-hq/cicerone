from __future__ import annotations

import sys

from cicerone.redis_client import REDIS_HEALTH_CHECK_INTERVAL, redis_from_url


def test_redis_from_url_sets_health_check_interval(monkeypatch):
    seen: dict[str, object] = {}

    class _Redis:
        @staticmethod
        def from_url(url: str, **kwargs: object) -> object:
            seen["url"] = url
            seen["kwargs"] = kwargs
            return object()

    fake = type(sys)("redis")
    fake.Redis = _Redis
    monkeypatch.setitem(sys.modules, "redis", fake)
    client = redis_from_url("redis://localhost:6379/0", decode_responses=True)
    assert client is not None
    assert seen["url"] == "redis://localhost:6379/0"
    assert seen["kwargs"] == {
        "decode_responses": True,
        "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL,
    }

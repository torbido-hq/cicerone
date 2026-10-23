from __future__ import annotations

import threading
import time

import pytest

from cicerone.refreshed_value import RefreshedValue


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.calls


def test_get_loads_lazily_and_refresh_reloads() -> None:
    load = _Counter()
    value = RefreshedValue(load)
    assert load.calls == 0
    assert value.get() == 1
    assert value.get() == 1
    value.refresh()
    assert value.get() == 2
    assert load.calls == 2


def test_ttl_reloads_once_expired(refresh_clock) -> None:
    load = _Counter()
    value = RefreshedValue(load, ttl_seconds=60.0)
    assert value.get() == 1
    refresh_clock.now += 59.0
    assert value.get() == 1
    refresh_clock.now += 1.0
    assert value.get() == 2


def test_zero_ttl_reloads_every_get() -> None:
    load = _Counter()
    value = RefreshedValue(load, ttl_seconds=0)
    assert [value.get(), value.get()] == [1, 2]


def test_failed_load_is_retried_on_next_get() -> None:
    outcomes = iter([OSError("down"), "ok"])

    def load() -> str:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    value = RefreshedValue(load, ttl_seconds=60.0)
    with pytest.raises(OSError, match="down"):
        value.get()
    assert value.get() == "ok"


def test_concurrent_expired_gets_load_once() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow_load() -> int:
        calls.append(1)
        started.set()
        release.wait(5)
        return len(calls)

    value = RefreshedValue(slow_load, ttl_seconds=60.0)
    results: list[int] = []
    threads = [threading.Thread(target=lambda: results.append(value.get())) for _ in range(4)]
    threads[0].start()
    assert started.wait(5)
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.1)  # let the other gets block on the in-flight load
    release.set()
    for thread in threads:
        thread.join(5)
    assert calls == [1]
    assert results == [1, 1, 1, 1]


def test_initial_is_served_until_a_refresh_succeeds() -> None:
    outcomes = iter([RuntimeError("down"), "fresh"])

    def load() -> str:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    value = RefreshedValue(load, initial="seed")
    assert value.get() == "seed"
    with pytest.raises(RuntimeError, match="down"):
        value.refresh()
    assert value.get() == "seed"
    value.refresh()
    assert value.get() == "fresh"

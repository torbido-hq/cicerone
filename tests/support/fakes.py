"""Shared test doubles."""

from __future__ import annotations


class FakeLock:
    """LockBackend that records ``acquire`` / ``owned`` / ``release`` calls in ``events``."""

    def __init__(self, *, acquires: bool = True, owned: bool = True) -> None:
        self.acquires = acquires
        self.is_owned = owned
        self.events: list[str] = []

    def acquire(self) -> bool:
        self.events.append("acquire")
        return self.acquires

    def release(self) -> None:
        self.events.append("release")

    def owned(self) -> bool:
        self.events.append("owned")
        return self.is_owned

    def is_locked(self) -> bool:
        return True

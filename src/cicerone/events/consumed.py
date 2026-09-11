"""In-process overlay of item ids seen in incremental events."""

from __future__ import annotations

import threading
from collections import defaultdict


class ConsumedOverlay:
    """User → item ids from events applied since process start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_user: dict[str, set[str]] = defaultdict(set)

    def add(self, user_id: str, item_id: str) -> None:
        with self._lock:
            self._by_user[str(user_id)].add(str(item_id))

    def add_many(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        with self._lock:
            for user_id, item_id in pairs:
                self._by_user[str(user_id)].add(str(item_id))

    def item_ids(self, user_id: str) -> set[str]:
        with self._lock:
            return set(self._by_user.get(str(user_id), ()))

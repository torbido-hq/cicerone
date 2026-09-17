"""Process-local overlay of item ids seen in incremental events."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

from cicerone.config.constants import (
    DEFAULT_CONSUMED_OVERLAY_MAX_USERS,
    DEFAULT_SERVE_CONSUMED_LOOKBACK,
)


class ConsumedOverlay:
    """Recent incremental item ids for this process. Cross-replica hide uses [input]."""

    def __init__(
        self,
        *,
        max_items_per_user: int = DEFAULT_SERVE_CONSUMED_LOOKBACK,
        max_users: int = DEFAULT_CONSUMED_OVERLAY_MAX_USERS,
    ) -> None:
        if max_items_per_user < 1:
            raise ValueError("max_items_per_user must be >= 1")
        if max_users < 1:
            raise ValueError("max_users must be >= 1")
        self._lock = threading.RLock()
        self._max_items_per_user = max_items_per_user
        self._max_users = max_users
        self._by_user: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()

    @contextmanager
    def mutation(self) -> Iterator[None]:
        with self._lock:
            yield

    def add(self, user_id: str, item_id: str) -> None:
        with self._lock:
            self._remember(str(user_id), str(item_id))

    def add_many(self, pairs: list[tuple[str, str]]) -> None:
        self.replace_pairs([], pairs)

    def replace_pairs(
        self,
        discard: list[tuple[str, str]],
        add: list[tuple[str, str]],
    ) -> None:
        if not discard and not add:
            return
        with self._lock:
            for user_id, item_id in discard:
                self._forget(str(user_id), str(item_id))
            for user_id, item_id in add:
                self._remember(str(user_id), str(item_id))

    def item_ids(self, user_id: str) -> set[str]:
        with self._lock:
            items = self._by_user.get(str(user_id))
            return set(items) if items is not None else set()

    def discard(self, user_id: str, item_id: str | None = None) -> None:
        with self._lock:
            self._forget(str(user_id), None if item_id is None else str(item_id))

    def _forget(self, user_id: str, item_id: str | None) -> None:
        if item_id is None:
            self._by_user.pop(user_id, None)
            return
        items = self._by_user.get(user_id)
        if items is None:
            return
        items.pop(item_id, None)
        if not items:
            self._by_user.pop(user_id, None)

    def _remember(self, user_id: str, item_id: str) -> None:
        items = self._by_user.get(user_id)
        if items is None:
            items = OrderedDict()
            self._by_user[user_id] = items
        else:
            self._by_user.move_to_end(user_id)
        if item_id in items:
            items.move_to_end(item_id)
        else:
            items[item_id] = None
            while len(items) > self._max_items_per_user:
                items.popitem(last=False)
        while len(self._by_user) > self._max_users:
            self._by_user.popitem(last=False)

"""Shared helpers and item-filter mixin for recommendation readers."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pandas as pd

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io import recommendation_schema as _rec
from cicerone.values import item_true_mask

USER_COLUMN = _rec.USER_COLUMN
ITEM_COLUMN = _rec.ITEM_COLUMN
RANK_COLUMN = _rec.RANK_COLUMN
SCORE_COLUMN = _rec.SCORE_COLUMN
SOURCE_COLUMN = _rec.SOURCE_COLUMN
VARIANT_COLUMN = _rec.VARIANT_COLUMN
RECOMMENDATION_COLUMNS = _rec.RECOMMENDATION_COLUMNS
ITEMS_SNAPSHOT_FILENAME = "items_snapshot.parquet"

# Cold-start without __cold_start__: popular/latest only (never warm "blended"),
# prefer popular → latest → min user_id. Missing-table: ProgrammingError / OperationalError.
_FALLBACK_SOURCES = frozenset({POPULAR_SOURCE, LATEST_SOURCE})
_FALLBACK_SOURCE_PRIORITY = {POPULAR_SOURCE: 0, LATEST_SOURCE: 1}


def normalize_items_snapshot(
    items: pd.DataFrame | None,
    *,
    category_column: str | None = None,
    availability_filters: Sequence[str] = (),
) -> pd.DataFrame | None:
    """Cast filter columns once so serve requests can reuse the frame as-is."""
    if items is None or items.empty:
        return items
    out = items.copy()
    if ITEM_COLUMN not in out.columns:
        return out
    out[ITEM_COLUMN] = out[ITEM_COLUMN].astype(str)
    if category_column and category_column in out.columns:
        out[category_column] = out[category_column].astype(str)
    for column in availability_filters:
        if column in out.columns:
            out[column] = item_true_mask(out[column])
    return out


def _best_fallback_user_id(priorities: dict[str, int]) -> str | None:
    """Lowest source priority wins; ties break on lexicographically smallest user id."""
    if not priorities:
        return None
    return min(priorities, key=lambda user_id: (priorities[user_id], user_id))


def _pick_fallback_user(candidates: pd.DataFrame) -> str | None:
    """Stable fallback user: prefer popular_fallback, then latest, then min user_id."""
    if candidates.empty or USER_COLUMN not in candidates.columns:
        return None
    frame = candidates[[USER_COLUMN]].copy()
    frame["_user"] = frame[USER_COLUMN].astype(str)
    if SOURCE_COLUMN in candidates.columns:
        frame["_src_pri"] = candidates[SOURCE_COLUMN].map(_FALLBACK_SOURCE_PRIORITY).fillna(99)
    else:
        frame["_src_pri"] = 99
    priorities = frame.groupby("_user", sort=False)["_src_pri"].min().astype(int).to_dict()
    return _best_fallback_user_id(priorities)


def select_cold_start_fallback(
    recommendations: pd.DataFrame,
    k: int,
    *,
    sentinel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pick ``__cold_start__`` rows, else one popular/latest user's top-K.

    Fallback user selection is deterministic across dataset and DB backends:
    prefer ``popular_fallback``, then ``latest``, then lexicographically smallest
    ``user_id``.

    ``sentinel`` may be a pre-fetched ``__cold_start__`` slice (e.g. from SQL
    or the per-user index); when omitted, it is derived from ``recommendations``.
    """
    empty = recommendations.iloc[0:0]
    if k < 1:
        return empty

    if sentinel is None:
        if recommendations.empty or USER_COLUMN not in recommendations.columns:
            return empty
        sentinel = (
            recommendations[recommendations[USER_COLUMN].astype(str) == COLD_START_USER_ID]
            .sort_values(RANK_COLUMN, kind="mergesort")
            .head(k)
            .reset_index(drop=True)
        )
    elif not sentinel.empty:
        sentinel = sentinel.sort_values(RANK_COLUMN, kind="mergesort").head(k).reset_index(drop=True)

    if not sentinel.empty:
        return sentinel

    if recommendations.empty or SOURCE_COLUMN not in recommendations.columns:
        return empty
    candidates = recommendations[recommendations[SOURCE_COLUMN].isin(_FALLBACK_SOURCES)]
    if candidates.empty:
        return empty
    sample_user = _pick_fallback_user(candidates)
    if sample_user is None:
        return empty
    rows = candidates[candidates[USER_COLUMN].astype(str) == sample_user].sort_values(
        RANK_COLUMN, kind="mergesort"
    )
    return rows.head(k).reset_index(drop=True)


def _index_recommendations_by_user(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalize user ids once and build an O(1) lookup for serve requests."""
    if frame.empty or USER_COLUMN not in frame.columns:
        return {}
    indexed = frame.copy()
    indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
    if RANK_COLUMN in indexed.columns:
        indexed = indexed.sort_values(RANK_COLUMN, kind="mergesort")
    return {
        user_id: group.reset_index(drop=True) for user_id, group in indexed.groupby(USER_COLUMN, sort=False)
    }


def _resolve_fallback_user_id(by_user: dict[str, pd.DataFrame]) -> str | None:
    """Pick popular/latest fallback user from the per-user index (no full-frame scan)."""
    priorities: dict[str, int] = {}
    for user_id, rows in by_user.items():
        if user_id == COLD_START_USER_ID or rows.empty or SOURCE_COLUMN not in rows.columns:
            continue
        sources = set(rows[SOURCE_COLUMN].astype(str))
        if not sources & _FALLBACK_SOURCES:
            continue
        priorities[user_id] = min(_FALLBACK_SOURCE_PRIORITY.get(src, 99) for src in sources)
    return _best_fallback_user_id(priorities)


class _ItemFilterMixin:
    """Shared items-snapshot filter configuration for recommendation readers.

    Call ``_init_item_filter_state()`` from subclass ``__init__`` before
    ``refresh`` / mixin methods. Methods also lazy-init if that was skipped.
    """

    _items: pd.DataFrame | None
    _items_version: int
    _category_column: str | None
    _availability_filters: list[str]
    _lock: threading.RLock

    def _init_item_filter_state(self) -> None:
        self._items = None
        self._items_version = 0
        self._category_column = None
        self._availability_filters = []
        self._lock = threading.RLock()

    def _ensure_item_filter_state(self) -> None:
        if getattr(self, "_lock", None) is None:
            self._init_item_filter_state()

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        self._ensure_item_filter_state()
        with self._lock:
            self._category_column = category_column
            self._availability_filters = list(availability_filters)
            self._items = normalize_items_snapshot(
                self._items,
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            self._items_version += 1

    def items_version(self) -> int:
        self._ensure_item_filter_state()
        with self._lock:
            return self._items_version

    def get_items(self) -> pd.DataFrame | None:
        self._ensure_item_filter_state()
        with self._lock:
            return self._items

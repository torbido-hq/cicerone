"""Serve-time category / availability filtering over the items snapshot."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import pandas as pd

from cicerone.io.base import RecommendationReader
from cicerone.io.recommendation_reader import ITEM_COLUMN, normalize_items_snapshot
from cicerone.values import item_true_mask

_FILTER_CACHE_REBUILDS = 8


def configure_reader_item_filters(
    reader: RecommendationReader,
    *,
    category_column: str | None,
    availability_filters: Sequence[str],
) -> None:
    reader.configure_item_filters(
        category_column=category_column,
        availability_filters=availability_filters,
    )


def available_item_ids(items: pd.DataFrame, availability_filters: Sequence[str]) -> frozenset[str] | None:
    if not availability_filters or ITEM_COLUMN not in items.columns:
        return None
    mask = pd.Series(True, index=items.index)
    for column in availability_filters:
        if column not in items.columns:
            continue
        mask &= item_true_mask(items[column])
    return frozenset(items.loc[mask, ITEM_COLUMN].astype(str).tolist())


class ItemsFilterCache:
    """Reuse one normalized items snapshot between refreshes."""

    def __init__(
        self,
        reader: RecommendationReader,
        *,
        category_column: str,
        availability_filters: Sequence[str],
    ) -> None:
        self._reader = reader
        self._category_column = category_column
        self._availability_filters = list(availability_filters)
        self._snapshot: (
            tuple[object, pd.DataFrame | None, frozenset[str] | None, dict[str, frozenset[str]]] | None
        ) = None
        self._lock = threading.Lock()

    def get(self) -> tuple[pd.DataFrame | None, frozenset[str] | None, dict[str, frozenset[str]]]:
        for _ in range(_FILTER_CACHE_REBUILDS):
            version = self._reader.items_version()
            cached = self._snapshot
            if cached is not None and cached[0] == version:
                return cached[1], cached[2], cached[3]
            built = self._rebuild(version)
            with self._lock:
                latest = self._reader.items_version()
                cached = self._snapshot
                if cached is not None and cached[0] == latest:
                    return cached[1], cached[2], cached[3]
                if latest == version:
                    self._snapshot = built
                    return built[1], built[2], built[3]
        version = self._reader.items_version()
        built = self._rebuild(version)
        with self._lock:
            latest = self._reader.items_version()
            cached = self._snapshot
            if cached is not None and cached[0] == latest:
                return cached[1], cached[2], cached[3]
            if latest == version:
                self._snapshot = built
            return built[1], built[2], built[3]

    def _rebuild(
        self, version: object
    ) -> tuple[object, pd.DataFrame | None, frozenset[str] | None, dict[str, frozenset[str]]]:
        items = normalize_items_snapshot(
            self._reader.get_items(),
            category_column=self._category_column,
            availability_filters=self._availability_filters,
        )
        available = (
            available_item_ids(items, self._availability_filters)
            if items is not None and not items.empty
            else None
        )
        ids_by_category: dict[str, frozenset[str]] = {}
        if (
            items is not None
            and not items.empty
            and self._category_column in items.columns
            and ITEM_COLUMN in items.columns
        ):
            item_ids = items[ITEM_COLUMN].astype(str)
            for cat, idx in items.groupby(self._category_column, sort=False).groups.items():
                ids_by_category[str(cat)] = frozenset(item_ids.loc[idx].tolist())
        return (version, items, available, ids_by_category)


def filter_recommendations(
    recs: pd.DataFrame,
    *,
    items: pd.DataFrame | None,
    available_ids: frozenset[str] | None,
    category: str | None,
    category_column: str,
    exclude_unavailable: bool,
    ids_by_category: dict[str, frozenset[str]] | None = None,
    on_missing_category_column: Callable[[], None] | None = None,
) -> pd.DataFrame:
    if recs.empty:
        return recs
    out = recs
    if items is None or items.empty:
        return out.reset_index(drop=True)

    item_ids = out[ITEM_COLUMN].astype(str)

    if category is not None:
        if category_column not in items.columns:
            if on_missing_category_column is not None:
                on_missing_category_column()
            return out.iloc[0:0].reset_index(drop=True)
        if ids_by_category is not None:
            allowed: set[str] | frozenset[str] = ids_by_category.get(str(category), frozenset())
        else:
            allowed = set(items.loc[items[category_column] == str(category), ITEM_COLUMN].astype(str))
        out = out[item_ids.isin(allowed)]
        item_ids = out[ITEM_COLUMN].astype(str)

    if exclude_unavailable and available_ids is not None:
        out = out[item_ids.isin(available_ids)]

    return out.reset_index(drop=True)

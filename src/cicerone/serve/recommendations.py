"""Recommendation lookup for serve: assignment, variant and cold-start fallbacks, filtering, consumed hide."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from cicerone.config import Settings
from cicerone.experiment.assignment import (
    assignment_needs_snapshot,
    resolve_assignment,
    snapshot_variant_names,
)
from cicerone.io.base import RecommendationReader
from cicerone.io.recommendation_schema import VARIANT_COLUMN, has_variant_column, pick_fallback_variant
from cicerone.serve.consumed import drop_consumed, merge_fill
from cicerone.serve.item_filters import ItemsFilterCache, filter_recommendations

logger = logging.getLogger(__name__)

AssignmentOverlay = tuple[str | None, tuple[str, str] | None]
ItemsSnapshot = tuple[pd.DataFrame | None, frozenset[str] | None, dict[str, frozenset[str]]]


@dataclass(frozen=True)
class ServedRecommendations:
    recs: pd.DataFrame
    used_fallback: bool
    experiment_id: str | None
    variant: str | None


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        reader: RecommendationReader,
        *,
        items_cache: ItemsFilterCache,
        availability_filters: Sequence[str],
        assignment_overlay: Callable[[], AssignmentOverlay],
    ) -> None:
        self._settings = settings
        self._reader = reader
        self._items_cache = items_cache
        self._availability_filters = list(availability_filters)
        self._assignment_overlay = assignment_overlay
        self._category_column = settings.serve.category_column
        self._missing_category_warned = False

    def get(
        self,
        user_id: str,
        top_k: int,
        *,
        category: str | None,
        exclude_unavailable: bool,
        consumed_ids: set[str] | None = None,
        items_snapshot: ItemsSnapshot | None = None,
        recs: pd.DataFrame | None = None,
        fallback_recs: pd.DataFrame | None = None,
        assignment: tuple[str | None, str | None] | None = None,
    ) -> ServedRecommendations | None:
        """Ranked top-``top_k`` rows, or ``None`` when nothing (not even cold-start) matches.

        Batch callers pass prefetched ``recs`` / ``fallback_recs``, the ``assignment`` and one
        ``items_snapshot`` so every user in a request sees the same data.
        """
        items, available_ids, ids_by_category = items_snapshot or self._items_cache.get()
        consumed_ids = consumed_ids or set()
        can_filter = bool(
            (
                items is not None
                and not items.empty
                and (category is not None or (exclude_unavailable and self._availability_filters))
            )
            or consumed_ids
        )
        fetch_k = max(top_k * 5, top_k) if can_filter else top_k
        experiment_id, variant = assignment if assignment is not None else self._assign(user_id)
        recs, variant = self._fetch_with_variant_fallback(
            lambda v: self._reader.get_recommendations(user_id, fetch_k, variant=v), variant, first=recs
        )
        used_fallback = False
        if recs.empty:
            used_fallback = True
            recs, variant = self._fetch_with_variant_fallback(
                lambda v: self._reader.get_cold_start_fallback(fetch_k, variant=v),
                variant,
                first=fallback_recs,
            )
        if recs.empty:
            return None
        if not has_variant_column(recs):
            experiment_id, variant = None, None

        after_availability = filter_recommendations(
            recs,
            items=items,
            available_ids=available_ids,
            category=None,
            category_column=self._category_column,
            exclude_unavailable=exclude_unavailable,
            ids_by_category=ids_by_category,
            on_missing_category_column=self.warn_missing_category_column,
        )
        after_hide = drop_consumed(after_availability, consumed_ids)
        dropped_by_hide_or_availability = len(after_hide) < len(recs)
        filtered = filter_recommendations(
            after_hide,
            items=items,
            available_ids=available_ids,
            category=category,
            category_column=self._category_column,
            exclude_unavailable=False,
            ids_by_category=ids_by_category,
            on_missing_category_column=self.warn_missing_category_column,
        )
        if self._settings.serve.fallback_fill and len(filtered) < top_k and dropped_by_hide_or_availability:
            filler = (
                fallback_recs
                if fallback_recs is not None
                else self._reader.get_cold_start_fallback(fetch_k, variant=variant)
            )
            filler = filter_recommendations(
                filler,
                items=items,
                available_ids=available_ids,
                category=category,
                category_column=self._category_column,
                exclude_unavailable=exclude_unavailable,
                ids_by_category=ids_by_category,
                on_missing_category_column=self.warn_missing_category_column,
            )
            filtered = merge_fill(filtered, filler, k=top_k, exclude=consumed_ids)
        filtered = filtered.head(top_k).reset_index(drop=True)
        if not filtered.empty:
            filtered = filtered.copy()
            filtered["rank"] = range(1, len(filtered) + 1)
        return ServedRecommendations(
            recs=filtered,
            used_fallback=used_fallback,
            experiment_id=experiment_id,
            variant=variant,
        )

    def _assign(self, user_id: str) -> tuple[str | None, str | None]:
        promoted, active_pair = self._assignment_overlay()
        snapshot = (
            snapshot_variant_names(self._reader)
            if assignment_needs_snapshot(self._settings, promoted_variant=promoted, active_pair=active_pair)
            else None
        )
        return resolve_assignment(
            self._settings,
            user_id,
            promoted_variant=promoted,
            active_pair=active_pair,
            snapshot_names=snapshot,
        )

    @staticmethod
    def _fetch_with_variant_fallback(
        fetch: Callable[[str | None], pd.DataFrame],
        variant: str | None,
        *,
        first: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        recs = first if first is not None else fetch(variant)
        if not recs.empty or variant is None:
            return recs, variant
        leftover = fetch(None)
        if not leftover.empty and has_variant_column(leftover):
            picked = pick_fallback_variant(leftover[VARIANT_COLUMN].tolist())
            if picked:
                variant = picked
        return leftover, variant

    def warn_missing_category_column(self) -> None:
        if self._missing_category_warned:
            return
        self._missing_category_warned = True
        logger.warning(
            "Serve category filter requested but items have no column %r — returning empty",
            self._category_column,
        )

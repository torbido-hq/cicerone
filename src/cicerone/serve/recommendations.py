"""Recommendation lookup for serve: assignment, variant and cold-start fallbacks, filtering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from cicerone.config import Settings
from cicerone.experiment.assignment import AssignmentOverlay, assign_user, snapshot_variant_names
from cicerone.experiment.store import ExperimentStore
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import RecommendationReader
from cicerone.io.recommendation_schema import VARIANT_COLUMN, has_variant_column, pick_fallback_variant
from cicerone.serve.item_filters import ItemsFilterCache, build_items_cache, filter_recommendations

logger = logging.getLogger(__name__)


def read_assignment_overlay(settings: Settings, store: ExperimentStore | None) -> AssignmentOverlay:
    if store is None or not settings.experiment.enabled:
        return None, None
    return store.assignment_overlay(settings.experiment.id)


def build_recommendation_service(
    settings: Settings,
    reader: RecommendationReader,
    feature_config: FeatureConfig | None,
    assignment_overlay: Callable[[], AssignmentOverlay],
) -> RecommendationService:
    return RecommendationService(
        settings,
        reader,
        items_cache=build_items_cache(settings, reader, feature_config),
        assignment_overlay=assignment_overlay,
    )


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
        assignment_overlay: Callable[[], AssignmentOverlay],
    ) -> None:
        self._settings = settings
        self._reader = reader
        self._items_cache = items_cache
        self._assignment_overlay = assignment_overlay
        self._missing_category_warned = False

    @property
    def items_cache(self) -> ItemsFilterCache:
        return self._items_cache

    def get(
        self,
        user_id: str,
        top_k: int,
        *,
        category: str | None,
        exclude_unavailable: bool,
    ) -> ServedRecommendations | None:
        """Ranked top-``top_k`` rows, or ``None`` when nothing (not even cold-start) matches."""
        items, available_ids, ids_by_category = self._items_cache.get()
        can_filter = bool(
            items is not None
            and not items.empty
            and (category is not None or (exclude_unavailable and self._items_cache.availability_filters))
        )
        fetch_k = top_k * 5 if can_filter else top_k
        experiment_id, variant = assign_user(
            self._settings, user_id, self._assignment_overlay(), lambda: snapshot_variant_names(self._reader)
        )
        recs, variant = self._fetch_with_variant_fallback(
            lambda v: self._reader.get_recommendations(user_id, fetch_k, variant=v), variant
        )
        used_fallback = False
        if recs.empty:
            used_fallback = True
            recs, variant = self._fetch_with_variant_fallback(
                lambda v: self._reader.get_cold_start_fallback(fetch_k, variant=v), variant
            )
        if recs.empty:
            return None
        if not has_variant_column(recs):
            experiment_id, variant = None, None

        filtered = filter_recommendations(
            recs,
            items=items,
            available_ids=available_ids,
            category=category,
            category_column=self._items_cache.category_column,
            exclude_unavailable=exclude_unavailable,
            ids_by_category=ids_by_category,
            on_missing_category_column=self._warn_missing_category_column,
        )
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

    @staticmethod
    def _fetch_with_variant_fallback(
        fetch: Callable[[str | None], pd.DataFrame], variant: str | None
    ) -> tuple[pd.DataFrame, str | None]:
        recs = fetch(variant)
        if not recs.empty or variant is None:
            return recs, variant
        leftover = fetch(None)
        if not leftover.empty and has_variant_column(leftover):
            picked = pick_fallback_variant(leftover[VARIANT_COLUMN].tolist())
            if picked:
                variant = picked
        return leftover, variant

    def _warn_missing_category_column(self) -> None:
        if self._missing_category_warned:
            return
        self._missing_category_warned = True
        logger.warning(
            "Serve category filter requested but items have no column %r — returning empty",
            self._items_cache.category_column,
        )

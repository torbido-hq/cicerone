"""Per-strategy recommend (threaded jobs, cache hits)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cicerone.blending import rank_latest_items, resolve_latest_date_column
from cicerone.dataset import BuiltDataset
from cicerone.model.constants import SOURCE_COLUMN, WEIGHT_COLUMN
from cicerone.model.recommend_cache import (
    RecommendCache,
    _dataset_fingerprint,
    _items_fingerprint,
    _recommend_cache_key,
)
from cicerone.model.recommend_cohort import _CohortPlan
from cicerone.model.strategies import STRATEGIES, RecommenderModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StrategyFrames:
    frames: list[pd.DataFrame]
    personalized_frames: list[pd.DataFrame]
    popular_frames: list[pd.DataFrame]
    latest_by_cohort: dict[object, list[tuple[str, int, float]]]
    date_column: str | None
    latest_available: bool


@dataclass(frozen=True)
class _StrategyRecommendJob:
    slot: int
    name: str
    source_label: str
    personalized: bool
    recommend_users: list
    allowed_items: list[Any] | None
    cache_key: tuple[Hashable, ...]


def _recommend_per_strategy(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    recommend_models: list[str],
    cohort_plan: _CohortPlan,
    *,
    blending_enabled: bool,
    blending_latest_date_columns: tuple[str, ...],
    top_k: int,
    recommend_cache: RecommendCache | None = None,
    max_workers: int = 1,
) -> _StrategyFrames:
    frames: list[pd.DataFrame] = []
    personalized_frames: list[pd.DataFrame] = []
    popular_frames: list[pd.DataFrame] = []
    latest_by_cohort: dict[object, list[tuple[str, int, float]]] = {}

    date_column = (
        resolve_latest_date_column(built.items, blending_latest_date_columns) if blending_enabled else None
    )
    latest_available = bool(blending_enabled and date_column is not None and built.items is not None)
    if blending_enabled and not latest_available:
        logger.info(
            "Blending: no usable date column among %s on items — disabling 'latest' "
            "and redistributing its weight onto popular",
            list(blending_latest_date_columns),
        )

    dataset = built.dataset
    slots: list[pd.DataFrame | None] = []
    slot_meta: list[tuple[bool, bool]] = []
    jobs: list[_StrategyRecommendJob] = []
    for name in recommend_models:
        strategy = STRATEGIES[name]
        if strategy.personalized and not cohort_plan.has_any_warm_user:
            continue
        if name not in models:
            raise ValueError(f"Fitted model for strategy {name!r} is missing; available: {sorted(models)}")

        for cohort_key_value, cohort_users in cohort_plan.cohorts:
            allowed_items = cohort_plan.allowed_by_cohort[cohort_key_value]
            if not allowed_items:
                continue

            if strategy.personalized:
                cohort_warm = [u for u in cohort_users if str(u) in cohort_plan.known_users]
                if strategy.requires_interactions:
                    cohort_warm = [u for u in cohort_warm if str(u) in cohort_plan.interacting_users]
                if not cohort_warm:
                    continue
                recommend_users = cohort_warm
            else:
                recommend_users = cohort_users

            cache_key = _recommend_cache_key(
                "strategy",
                name,
                cohort_key_value,
                cohort_plan.recommend_k,
                _items_fingerprint(allowed_items),
                _items_fingerprint(recommend_users),
                bool(strategy.personalized),
                _dataset_fingerprint(dataset),
            )
            cached = recommend_cache.get(cache_key) if recommend_cache is not None else None
            if cached is not None:
                slots.append(cached.copy())
            else:
                jobs.append(
                    _StrategyRecommendJob(
                        slot=len(slots),
                        name=name,
                        source_label=strategy.source_label,
                        personalized=strategy.personalized,
                        recommend_users=recommend_users,
                        allowed_items=allowed_items,
                        cache_key=cache_key,
                    )
                )
                slots.append(None)
            slot_meta.append((strategy.personalized, name == "popular"))

    if jobs:
        model_locks = {name: threading.Lock() for name in {job.name for job in jobs}}

        def _run_job(job: _StrategyRecommendJob) -> tuple[int, pd.DataFrame]:
            with model_locks[job.name]:
                recs = (
                    models[job.name]
                    .recommend(
                        users=job.recommend_users,
                        dataset=dataset,
                        k=cohort_plan.recommend_k,
                        filter_viewed=job.personalized,
                        items_to_recommend=job.allowed_items,
                    )
                    .copy()
                )
            recs[SOURCE_COLUMN] = job.source_label
            recs[WEIGHT_COLUMN] = 1.0
            return job.slot, recs

        workers = max(1, int(max_workers))
        if workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                filled = list(pool.map(_run_job, jobs))
        else:
            filled = [_run_job(job) for job in jobs]
        for job, (slot, recs) in zip(jobs, filled, strict=True):
            slots[slot] = recs
            if recommend_cache is not None:
                recommend_cache[job.cache_key] = recs.copy()

    for recs, (personalized, is_popular) in zip(slots, slot_meta, strict=True):
        if recs is None:
            raise RuntimeError("strategy recommend slot was not filled")
        frames.append(recs)
        if blending_enabled:
            if personalized:
                personalized_frames.append(recs)
            elif is_popular:
                popular_frames.append(recs)

    if blending_enabled and latest_available and date_column is not None and built.items is not None:
        latest_k = cohort_plan.recommend_k if cohort_plan.has_boosts else top_k
        for cohort_key, _cohort_users in cohort_plan.cohorts:
            allowed_items = cohort_plan.allowed_by_cohort[cohort_key]
            if not allowed_items:
                continue
            latest_key = _recommend_cache_key(
                "latest",
                date_column,
                cohort_key,
                latest_k,
                _items_fingerprint(allowed_items),
                _dataset_fingerprint(dataset),
            )
            cached_latest = recommend_cache.get(latest_key) if recommend_cache is not None else None
            if cached_latest is not None:
                latest_by_cohort[cohort_key] = list(cached_latest)
            else:
                ranked = rank_latest_items(built.items, date_column, allowed_items, latest_k)
                latest_by_cohort[cohort_key] = ranked
                if recommend_cache is not None:
                    recommend_cache[latest_key] = list(ranked)

    return _StrategyFrames(
        frames=frames,
        personalized_frames=personalized_frames,
        popular_frames=popular_frames,
        latest_by_cohort=latest_by_cohort,
        date_column=date_column,
        latest_available=latest_available,
    )

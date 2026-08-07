"""Recommend + combine fitted strategies (cohorts, blending, boosts)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools import Columns

from cicerone.blending import (
    COLD_START_USER_ID,
    PERSONALIZED_SOURCE,
    POPULAR_SOURCE,
    append_cold_start_rows,
    blend_for_users,
    expand_latest_ranking,
    interaction_counts,
    rank_latest_items,
    resolve_latest_date_column,
)
from cicerone.config import (
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    EpochMetricsSettings,
    validate_model_weights,
    validate_rrf_k,
)
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, FeatureConfig
from cicerone.ids import interacting_external_user_ids
from cicerone.model.combine import _combine_by_priority, _combine_by_weighted_fusion
from cicerone.model.constants import RRF_K, SOURCE_COLUMN, WEIGHT_COLUMN
from cicerone.model.fit import ModelRunPlan, fit_strategies, plan_model_run
from cicerone.model.strategies import STRATEGIES, RecommenderModel
from cicerone.policy import (
    allowed_items_for_cohort,
    apply_boosts,
    group_users_by_cohort,
    has_user_scoped_eligibility,
    index_users_by_id,
    is_user_scoped,
    resolve_eligibility,
)

logger = logging.getLogger(__name__)


def _recommend_k(top_k: int, has_boosts: bool, overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR) -> int:
    if not has_boosts:
        return top_k
    factor = overfetch_factor if overfetch_factor >= 1 else DEFAULT_BOOST_OVERFETCH_FACTOR
    return max(top_k, top_k * factor)


@dataclass(frozen=True)
class _CohortPlan:
    cohorts: list[tuple[object, list[str]]]
    allowed_by_cohort: dict[object, list]
    eligibility: list
    users_frame: pd.DataFrame | None
    known_users: set
    interacting_users: set
    has_any_warm_user: bool
    unique_target_users: list[str]
    all_item_ids: Sequence
    has_boosts: bool
    recommend_k: int


@dataclass(frozen=True)
class _StrategyFrames:
    frames: list[pd.DataFrame]
    personalized_frames: list[pd.DataFrame]
    popular_frames: list[pd.DataFrame]
    latest_by_cohort: dict[object, list[tuple[str, int, float]]]
    date_column: str | None
    latest_available: bool


def _resolve_cohort_plan(
    built: BuiltDataset,
    config: FeatureConfig,
    target_users: list[str],
    recommend_models: list[str],
    top_k: int,
) -> _CohortPlan:
    dataset = built.dataset
    all_item_ids = dataset.item_id_map.external_ids
    eligibility = resolve_eligibility(config)
    users_frame = built.users if built.users is not None and not built.users.empty else None
    if has_user_scoped_eligibility(eligibility) and users_frame is None:
        logger.warning(
            "User-scoped eligibility rules are configured but no users frame is available — "
            "applying only item-global rules"
        )
        eligibility = [r for r in eligibility if not is_user_scoped(r)]
    use_cohorts = has_user_scoped_eligibility(eligibility) and users_frame is not None
    has_boosts = bool(config.boosts)
    recommend_k = _recommend_k(top_k, has_boosts, config.boost_overfetch_factor)

    known_users = set(dataset.user_id_map.external_ids)
    unique_target_users = list(dict.fromkeys(target_users))
    has_any_warm_user = bool(known_users.intersection(unique_target_users))
    needs_interacting_users = any(
        STRATEGIES[name].requires_interactions for name in recommend_models if name in STRATEGIES
    )
    interacting_users = interacting_external_user_ids(built) if needs_interacting_users else set()

    users_by_id = index_users_by_id(users_frame)
    if use_cohorts:
        cohorts = group_users_by_cohort(
            unique_target_users, users_frame, eligibility, users_by_id=users_by_id
        )
        allowed_by_cohort = {
            key: allowed_items_for_cohort(
                cohort_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
            for key, cohort_users in cohorts
        }
    else:
        allowed_by_cohort = {
            None: allowed_items_for_cohort(
                unique_target_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
        }
        cohorts = [(None, unique_target_users)]

    return _CohortPlan(
        cohorts=cohorts,
        allowed_by_cohort=allowed_by_cohort,
        eligibility=eligibility,
        users_frame=users_frame,
        known_users=known_users,
        interacting_users=interacting_users,
        has_any_warm_user=has_any_warm_user,
        unique_target_users=unique_target_users,
        all_item_ids=all_item_ids,
        has_boosts=has_boosts,
        recommend_k=recommend_k,
    )


def _recommend_per_strategy(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    recommend_models: list[str],
    cohort_plan: _CohortPlan,
    *,
    blending_enabled: bool,
    blending_latest_date_columns: tuple[str, ...],
    top_k: int,
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
                cohort_warm = [u for u in cohort_users if u in cohort_plan.known_users]
                if strategy.requires_interactions:
                    cohort_warm = [u for u in cohort_warm if u in cohort_plan.interacting_users]
                if not cohort_warm:
                    continue
                recommend_users = cohort_warm
            else:
                recommend_users = cohort_users

            recs = models[name].recommend(
                users=recommend_users,
                dataset=dataset,
                k=cohort_plan.recommend_k,
                filter_viewed=strategy.personalized,
                items_to_recommend=allowed_items,
            )
            recs[SOURCE_COLUMN] = strategy.source_label
            recs[WEIGHT_COLUMN] = 1.0
            frames.append(recs)
            if blending_enabled:
                if strategy.personalized:
                    personalized_frames.append(recs)
                elif name == "popular":
                    popular_frames.append(recs)

    if blending_enabled and latest_available and date_column is not None and built.items is not None:
        latest_k = cohort_plan.recommend_k if cohort_plan.has_boosts else top_k
        for cohort_key, _cohort_users in cohort_plan.cohorts:
            allowed_items = cohort_plan.allowed_by_cohort[cohort_key]
            if not allowed_items:
                continue
            latest_by_cohort[cohort_key] = rank_latest_items(
                built.items, date_column, allowed_items, latest_k
            )

    return _StrategyFrames(
        frames=frames,
        personalized_frames=personalized_frames,
        popular_frames=popular_frames,
        latest_by_cohort=latest_by_cohort,
        date_column=date_column,
        latest_available=latest_available,
    )


def _combine_strategy_frames(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    config: FeatureConfig,
    recommend_models: list[str],
    cohort_plan: _CohortPlan,
    strategy_frames: _StrategyFrames,
    *,
    blending_enabled: bool,
    weights: dict[str, float] | None,
    rrf_k: float | None,
    top_k: int,
) -> pd.DataFrame:
    combine_k = cohort_plan.recommend_k if cohort_plan.has_boosts else top_k
    empty_recs = pd.DataFrame(
        columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]
    )

    if blending_enabled:
        blending = config.blending
        personalized = (
            pd.concat(strategy_frames.personalized_frames, ignore_index=True)
            if strategy_frames.personalized_frames
            else empty_recs
        )
        if not personalized.empty:
            personalized = personalized.copy()
            personalized[SOURCE_COLUMN] = PERSONALIZED_SOURCE

        popular = (
            pd.concat(strategy_frames.popular_frames, ignore_index=True)
            if strategy_frames.popular_frames
            else empty_recs
        )
        if not popular.empty:
            popular = popular.copy()
            popular[SOURCE_COLUMN] = POPULAR_SOURCE

        latest_parts: list[pd.DataFrame] = []
        for cohort_key, cohort_users in cohort_plan.cohorts:
            ranked = strategy_frames.latest_by_cohort.get(cohort_key)
            if ranked:
                latest_parts.append(expand_latest_ranking(ranked, cohort_users))
        latest_frame = pd.concat(latest_parts, ignore_index=True) if latest_parts else None

        counts = interaction_counts(built.interactions)
        combined = blend_for_users(
            personalized=personalized,
            popular=popular,
            latest=latest_frame,
            counts=counts,
            target_users=cohort_plan.unique_target_users,
            config=blending,
            top_k=combine_k,
            latest_available=strategy_frames.latest_available,
        )

        cold_popular = empty_recs.copy()
        cold_shared_latest: list[tuple[str, int, float]] | None = None
        if "popular" in models:
            global_rules = [rule for rule in cohort_plan.eligibility if not is_user_scoped(rule)]
            global_allowed = allowed_items_for_cohort(
                [],
                None,
                built.items,
                global_rules,
                cohort_plan.all_item_ids,
            )
            if global_allowed:
                cold_popular = models["popular"].recommend(
                    users=[COLD_START_USER_ID],
                    dataset=built.dataset,
                    k=combine_k,
                    filter_viewed=False,
                    items_to_recommend=global_allowed,
                )
                cold_popular[SOURCE_COLUMN] = POPULAR_SOURCE
                if (
                    strategy_frames.latest_available
                    and strategy_frames.date_column is not None
                    and built.items is not None
                ):
                    cold_shared_latest = rank_latest_items(
                        built.items,
                        strategy_frames.date_column,
                        global_allowed,
                        combine_k,
                    )

        combined = append_cold_start_rows(
            combined,
            popular=cold_popular,
            latest=None,
            config=blending,
            top_k=combine_k,
            latest_available=strategy_frames.latest_available,
            shared_latest=cold_shared_latest,
        )
        if WEIGHT_COLUMN in combined.columns:
            combined = combined.drop(columns=[WEIGHT_COLUMN])
        return combined

    if not strategy_frames.frames:
        return empty_recs
    if weights is not None:
        label_weights = {STRATEGIES[name].source_label: weights.get(name, 1.0) for name in recommend_models}
        stamped: list[pd.DataFrame] = []
        for frame in strategy_frames.frames:
            part = frame.copy()
            part[WEIGHT_COLUMN] = part[SOURCE_COLUMN].map(label_weights).fillna(1.0)
            stamped.append(part)
        source_label_order = [STRATEGIES[name].source_label for name in recommend_models]
        return _combine_by_weighted_fusion(
            stamped, combine_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    return _combine_by_priority(strategy_frames.frames, combine_k)


def recommend_with_models(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str],
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    run_plan: ModelRunPlan | None = None,
) -> pd.DataFrame:
    """Runs recommend + combine on already-fitted strategies (no fit).

    Phases: resolve cohorts → recommend per strategy → combine → boosts.
    Used by ``train_and_recommend`` and by ``artifact.recommend_from_artifact``.
    """
    blending = config.blending
    blending_enabled = blending.enabled
    if run_plan is None:
        run_plan = plan_model_run(
            enabled_models,
            blending_enabled=blending_enabled,
            content_fallback_enabled=None,
        )
    enabled_models = list(run_plan.enabled_models)
    recommend_models = list(run_plan.recommend_models)
    if blending_enabled and "latest" in enabled_models:
        logger.warning(
            "Blending is enabled: date-based 'latest' comes from items; "
            "strategy 'latest' (trending PopularModel) is skipped for this run"
        )
    if blending_enabled and weights is not None:
        logger.warning(
            "Blending is enabled: job.model_weights / AutoML weights are ignored "
            "(per-user blend curve controls source mix instead)"
        )

    if weights is not None and not blending_enabled:
        unknown_weights = [name for name in weights if name not in recommend_models]
        if unknown_weights:
            raise ValueError(
                f"model_weights key(s) {unknown_weights} are not in recommend models {recommend_models}"
            )
        validate_model_weights(weights)
    validate_rrf_k(rrf_k)

    cohort_plan = _resolve_cohort_plan(built, config, target_users, recommend_models, top_k)
    strategy_frames = _recommend_per_strategy(
        models,
        built,
        recommend_models,
        cohort_plan,
        blending_enabled=blending_enabled,
        blending_latest_date_columns=tuple(blending.latest_date_columns),
        top_k=top_k,
    )
    combined = _combine_strategy_frames(
        models,
        built,
        config,
        recommend_models,
        cohort_plan,
        strategy_frames,
        blending_enabled=blending_enabled,
        weights=weights,
        rrf_k=rrf_k,
        top_k=top_k,
    )

    if cohort_plan.has_boosts:
        combined = apply_boosts(combined, built.items, config.boosts, top_k=top_k)

    return combined.reset_index(drop=True)


def train_and_recommend(
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str] | None = None,
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
    epoch_metrics: EpochMetricsSettings | None = None,
    item_based_k_neighbors: int | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_fallback_enabled: bool | None = None,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    run_plan: ModelRunPlan | None = None,
) -> pd.DataFrame:
    """Fit enabled strategies, then recommend + combine. See ``fit_strategies``
    and ``recommend_with_models`` for the split used by model artifacts.

    Pass ``run_plan`` from ``plan_model_run`` when the caller already resolved
    models (e.g. job). Otherwise ``content_fallback_enabled`` is the config
    flag, or ``None`` to derive from ``enabled_models`` (AutoML lists).
    """
    if run_plan is None:
        run_plan = plan_model_run(
            enabled_models,
            blending_enabled=config.blending.enabled,
            content_fallback_enabled=content_fallback_enabled,
        )
    _, fitted = fit_strategies(
        built,
        target_users,
        enabled_models=list(run_plan.recommend_models),
        strategy_cache=strategy_cache,
        max_workers=max_workers,
        epoch_metrics=epoch_metrics,
        epoch_metrics_top_k=top_k,
        item_based_k_neighbors=item_based_k_neighbors,
        model_configs=model_configs,
        content_fallback_max_neighbors=content_fallback_max_neighbors,
        content_feature_columns=config.item_features,
    )
    return recommend_with_models(
        fitted,
        built,
        target_users,
        config,
        top_k=top_k,
        enabled_models=list(run_plan.enabled_models),
        weights=weights,
        rrf_k=rrf_k,
        run_plan=run_plan,
    )

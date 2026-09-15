"""Combine per-strategy frames (blend, weighted RRF, priority)."""

from __future__ import annotations

import pandas as pd
from rectools import Columns

from cicerone.blending import (
    COLD_START_USER_ID,
    PERSONALIZED_SOURCE,
    POPULAR_SOURCE,
    append_cold_start_rows,
    blend_for_users,
    interaction_counts,
    rank_latest_items,
)
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig
from cicerone.model.combine import combine_by_priority, combine_by_weighted_fusion
from cicerone.model.constants import RRF_K, SOURCE_COLUMN, WEIGHT_COLUMN
from cicerone.model.recommend_cohort import _CohortPlan
from cicerone.model.recommend_strategy import _StrategyFrames
from cicerone.model.strategies import STRATEGIES, RecommenderModel
from cicerone.policy import allowed_items_for_cohort, is_user_scoped


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

        latest_by_user: dict[str, list[tuple[str, int, float]]] = {}
        for cohort_key, cohort_users in cohort_plan.cohorts:
            ranked = strategy_frames.latest_by_cohort.get(cohort_key)
            if ranked:
                for user_id in cohort_users:
                    latest_by_user[str(user_id)] = ranked

        counts = interaction_counts(built.interactions)
        combined = blend_for_users(
            personalized=personalized,
            popular=popular,
            latest=None,
            counts=counts,
            target_users=cohort_plan.unique_target_users,
            config=blending,
            top_k=combine_k,
            latest_available=strategy_frames.latest_available,
            latest_by_user=latest_by_user or None,
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
        return combine_by_weighted_fusion(
            stamped, combine_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    return combine_by_priority(strategy_frames.frames, combine_k)

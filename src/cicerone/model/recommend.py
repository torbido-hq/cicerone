"""Recommend + combine fitted strategies (cohorts, blending, boosts)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cicerone.config import (
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    EpochMetricsSettings,
    validate_model_weights,
    validate_rrf_k,
)
from cicerone.config.settings import ExplainSettings
from cicerone.dataset import BuiltDataset
from cicerone.explain import attach_reasons
from cicerone.feature_config import FeatureConfig
from cicerone.model.fit import ModelRunPlan, fit_strategies, plan_model_run
from cicerone.model.recommend_cache import (  # noqa: F401
    RecommendCache,
    _dataset_fingerprint,
    _items_fingerprint,
    _recommend_cache_key,
)
from cicerone.model.recommend_cohort import (
    _resolve_cohort_plan,
    boost_overfetch_k,  # noqa: F401
)
from cicerone.model.recommend_combine import _combine_strategy_frames
from cicerone.model.recommend_strategy import _recommend_per_strategy
from cicerone.model.strategies import RecommenderModel
from cicerone.policy import apply_boosts

logger = logging.getLogger(__name__)


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
    recommend_cache: RecommendCache | None = None,
    max_workers: int = 1,
    explain: ExplainSettings | None = None,
) -> pd.DataFrame:
    """Recommend + combine on already-fitted strategies (no fit).

    ``recommend_cache`` keys include strategy, cohort, k, allowlist, user set,
    filter_viewed, and dataset identity so a shared dict is safe across
    differing recommend() inputs.
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
        recommend_cache=recommend_cache,
        max_workers=max_workers,
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

    explain_settings = explain if explain is not None else ExplainSettings()
    if cohort_plan.has_boosts:
        combined = apply_boosts(
            combined,
            built.items,
            config.boosts,
            top_k=top_k,
            record_hits=explain_settings.enabled,
        )

    return attach_reasons(
        combined.reset_index(drop=True),
        items=built.items,
        interactions=built.interactions,
        feature_columns=config.item_features,
        settings=explain_settings,
    )


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
    recommend_cache: RecommendCache | None = None,
    explain: ExplainSettings | None = None,
) -> pd.DataFrame:
    """Fit enabled strategies, then recommend + combine."""
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
        recommend_cache=recommend_cache,
        max_workers=max_workers,
        explain=explain,
    )

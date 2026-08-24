"""Fit enabled strategies (optionally in parallel) and resolve run model lists."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from typing import Any

from rectools.dataset import Dataset

from cicerone.config import DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS, EpochMetricsSettings
from cicerone.content_fallback import build_content_fallback_model
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureColumn
from cicerone.model.constants import (
    DEFAULT_MODELS,
    LIGHTFM_NUM_THREADS_PARALLEL,
    LIGHTFM_NUM_THREADS_SEQUENTIAL,
)
from cicerone.model.epoch_metrics import (
    fit_with_epoch_metrics,
    interactions_for_epoch_metrics,
)
from cicerone.model.strategies import (
    STRATEGIES,
    RecommenderModel,
    as_recommender_model,
    build_strategy_model,
)
from cicerone.model_config import SEQUENTIAL_STRATEGY, default_model_configs, resolve_model_configs

logger = logging.getLogger(__name__)

_FIT_POOL_LIGHTFM_THREADS = LIGHTFM_NUM_THREADS_SEQUENTIAL


def _init_fit_worker(lightfm_threads: int) -> None:
    """ProcessPool initializer: keep LightFM single-threaded inside workers."""
    global _FIT_POOL_LIGHTFM_THREADS
    _FIT_POOL_LIGHTFM_THREADS = lightfm_threads


def _resolve_model_configs_for_fit(
    model_configs: dict[str, dict[str, Any]] | None,
    item_based_k_neighbors: int | None,
) -> dict[str, dict[str, Any]]:
    """Apply optional k_neighbors override onto a copy of model configs."""
    if model_configs is None and item_based_k_neighbors is None:
        return default_model_configs()
    if model_configs is None:
        return resolve_model_configs(
            legacy_k_neighbors=item_based_k_neighbors,
            legacy_k_neighbors_explicit=item_based_k_neighbors is not None,
        )
    configs = {name: deepcopy(cfg) for name, cfg in model_configs.items()}
    if item_based_k_neighbors is not None:
        item_cfg = configs.setdefault("item_based", deepcopy(default_model_configs()["item_based"]))
        model_section = item_cfg.setdefault("model", {})
        model_section["K"] = int(item_based_k_neighbors)
        configs["item_based"] = item_cfg
    return configs


def _fit_strategy(
    name: str,
    dataset: Dataset,
    epoch_interactions: object | None,
    epoch_metrics: EpochMetricsSettings | None,
    epoch_metrics_top_k: int,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_feature_columns: list[FeatureColumn] | None = None,
    content_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_items: object | None = None,
    content_interactions: object | None = None,
) -> tuple[str, RecommenderModel]:
    """Fit one strategy (picklable for ProcessPoolExecutor workers)."""
    strategy = STRATEGIES[name]
    if name == "content_fallback":
        model = as_recommender_model(
            build_content_fallback_model(
                feature_columns=content_feature_columns or [],
                max_neighbors=content_max_neighbors,
                items=content_items,  # type: ignore[arg-type]
                interactions=content_interactions,  # type: ignore[arg-type]
            )
        )
    elif strategy.factory is not None:
        model = strategy.factory()
    else:
        model = build_strategy_model(
            name,
            model_configs=model_configs,
            lightfm_num_threads=_FIT_POOL_LIGHTFM_THREADS,
        )
    if name in {"collaborative", SEQUENTIAL_STRATEGY} and epoch_metrics is not None:
        if epoch_interactions is None:
            raise ValueError("epoch_interactions is required when epoch metric logging is enabled")
        fit_with_epoch_metrics(
            model,
            dataset,
            epoch_interactions,  # type: ignore[arg-type]
            settings=epoch_metrics,
            top_k=epoch_metrics_top_k,
            label="Collaborative" if name == "collaborative" else "Sequential",
        )
    else:
        model.fit(dataset)
    return name, model


def _resolve_enabled_models(enabled_models: list[str] | None) -> list[str]:
    resolved = enabled_models if enabled_models is not None else DEFAULT_MODELS
    if not resolved:
        raise ValueError(
            "enabled_models is empty; provide at least one model name, or omit enabled_models/pass None "
            "to use the default"
        )
    unknown_models = [name for name in resolved if name not in STRATEGIES]
    if unknown_models:
        raise ValueError(f"Unknown model(s) {unknown_models}; available: {sorted(STRATEGIES)}")
    return resolved


def content_fallback_enabled_from_models(models: list[str] | tuple[str, ...] | None) -> bool:
    """True when ``content_fallback`` is present in a stored/candidate model list."""
    if not models:
        return False
    return "content_fallback" in models


@dataclass(frozen=True)
class ModelRunPlan:
    """Resolved enabled + recommend model lists for one run."""

    enabled_models: tuple[str, ...]
    recommend_models: tuple[str, ...]

    @property
    def content_fallback_active(self) -> bool:
        return "content_fallback" in self.recommend_models


def resolve_recommend_models(
    enabled_models: list[str],
    blending_enabled: bool,
    content_fallback_enabled: bool = False,
) -> list[str]:
    """Adjust enabled models for content_fallback insertion/drop and blending."""
    models = list(enabled_models)
    if not content_fallback_enabled:
        if "content_fallback" in models:
            logger.info(
                "content_fallback is listed in models but job.content_fallback.enabled is false — skipping"
            )
            models = [name for name in models if name != "content_fallback"]
    elif "content_fallback" not in models:
        insert_at = len(models)
        for index, name in enumerate(models):
            if name in STRATEGIES and not STRATEGIES[name].personalized:
                insert_at = index
                break
        models.insert(insert_at, "content_fallback")

    if not blending_enabled:
        return models
    if "latest" in models:
        models = [name for name in models if name != "latest"]
    if "popular" not in models:
        models.append("popular")
    return models


def plan_model_run(
    enabled_models: list[str] | None,
    *,
    blending_enabled: bool,
    content_fallback_enabled: bool | None = None,
) -> ModelRunPlan:
    """Build a ``ModelRunPlan`` (enabled + recommend lists)."""
    resolved = _resolve_enabled_models(enabled_models)
    if content_fallback_enabled is None:
        content_fallback_enabled = content_fallback_enabled_from_models(resolved)
    recommend = resolve_recommend_models(
        resolved, blending_enabled, content_fallback_enabled=content_fallback_enabled
    )
    return ModelRunPlan(enabled_models=tuple(resolved), recommend_models=tuple(recommend))


def resolve_run_models(
    enabled_models: list[str] | None,
    *,
    blending_enabled: bool,
    content_fallback_enabled: bool = False,
) -> tuple[list[str], list[str]]:
    """Like ``plan_model_run``, returning plain lists for existing callers."""
    plan = plan_model_run(
        enabled_models,
        blending_enabled=blending_enabled,
        content_fallback_enabled=content_fallback_enabled,
    )
    return list(plan.enabled_models), list(plan.recommend_models)


def fit_strategies(
    built: BuiltDataset,
    target_users: list[str],
    enabled_models: list[str] | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
    epoch_metrics: EpochMetricsSettings | None = None,
    epoch_metrics_top_k: int = 10,
    item_based_k_neighbors: int | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_feature_columns: list[FeatureColumn] | None = None,
) -> tuple[list[str], dict[str, RecommenderModel]]:
    """Fit (or cache-hit) enabled strategies; parallel when ``max_workers > 1``."""
    dataset = built.dataset
    enabled_models = _resolve_enabled_models(enabled_models)
    resolved_configs = _resolve_model_configs_for_fit(model_configs, item_based_k_neighbors)

    known_users = set(dataset.user_id_map.external_ids)
    warm_users = [u for u in target_users if u in known_users]
    cold_users = [u for u in target_users if u not in known_users]
    if cold_users:
        if any(not STRATEGIES[name].personalized for name in enabled_models):
            logger.info(
                "%d/%d users have no usable signal yet; falling back to non-personalized strategies for them",
                len(cold_users),
                len(target_users),
            )
        else:
            logger.info(
                "%d/%d users have no usable signal yet and no non-personalized strategy is "
                "enabled; they will receive no recommendations",
                len(cold_users),
                len(target_users),
            )

    models: dict[str, RecommenderModel] = {}
    if strategy_cache is not None:
        for name in enabled_models:
            if name in strategy_cache:
                models[name] = strategy_cache[name]

    to_fit = list(
        dict.fromkeys(
            name
            for name in enabled_models
            if name not in models and not (STRATEGIES[name].personalized and not warm_users)
        )
    )
    # Pre-slice interactions in the parent to shrink ProcessPool pickles.
    epoch_interactions = (
        interactions_for_epoch_metrics(dataset, built.interactions, epoch_metrics.max_users)
        if epoch_metrics is not None and ("collaborative" in to_fit or SEQUENTIAL_STRATEGY in to_fit)
        else None
    )
    content_cols = list(content_feature_columns or [])
    content_items = built.items if "content_fallback" in to_fit else None
    content_interactions = built.interactions if "content_fallback" in to_fit else None
    if to_fit:
        if max_workers > 1 and len(to_fit) > 1:
            with ProcessPoolExecutor(
                max_workers=min(max_workers, len(to_fit)),
                initializer=_init_fit_worker,
                initargs=(LIGHTFM_NUM_THREADS_PARALLEL,),
            ) as executor:
                for name, model in executor.map(
                    _fit_strategy,
                    to_fit,
                    repeat(dataset),
                    repeat(epoch_interactions),
                    repeat(epoch_metrics),
                    repeat(epoch_metrics_top_k),
                    repeat(resolved_configs),
                    repeat(content_cols),
                    repeat(content_fallback_max_neighbors),
                    repeat(content_items),
                    repeat(content_interactions),
                ):
                    logger.info("Fitted '%s' on %d interactions", name, len(built.interactions))
                    models[name] = model
        else:
            for name in to_fit:
                logger.info("Fitting '%s' on %d interactions", name, len(built.interactions))
                _, model = _fit_strategy(
                    name,
                    dataset,
                    epoch_interactions,
                    epoch_metrics,
                    epoch_metrics_top_k,
                    resolved_configs,
                    content_cols,
                    content_fallback_max_neighbors,
                    content_items,
                    content_interactions,
                )
                models[name] = model
        if strategy_cache is not None:
            for name in to_fit:
                strategy_cache[name] = models[name]

    return enabled_models, models

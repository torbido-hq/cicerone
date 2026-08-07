"""Train one or more recommendation strategies and combine their outputs.

Public API is re-exported here so ``from cicerone.model import …`` keeps working
after the former monolithic ``model.py`` was split into this package.
"""

from __future__ import annotations

from cicerone.model.combine import _combine_by_priority, _combine_by_weighted_fusion
from cicerone.model.constants import (
    COLLABORATIVE_EPOCHS,
    DEFAULT_MODELS,
    LATEST_WINDOW_DAYS,
    RANDOM_STATE,
    RRF_K,
    SOURCE_COLUMN,
    WEIGHT_COLUMN,
)
from cicerone.model.epoch_metrics import (
    _epoch_metric_total_epochs,
    _fit_lightfm_with_epoch_metrics,
    _sample_epoch_metric_users,
    _should_log_epoch,
    _warn_on_epoch_metric_trajectory,
)
from cicerone.model.fit import (
    ModelRunPlan,
    content_fallback_enabled_from_models,
    fit_strategies,
    plan_model_run,
    resolve_recommend_models,
    resolve_run_models,
)
from cicerone.model.recommend import (
    _recommend_k,
    recommend_with_models,
    train_and_recommend,
)
from cicerone.model.strategies import (
    STRATEGIES,
    RecommenderModel,
    Strategy,
    _as_recommender_model,
    _validate_strategy_names,
    build_strategy_model,
)

__all__ = [
    "COLLABORATIVE_EPOCHS",
    "DEFAULT_MODELS",
    "LATEST_WINDOW_DAYS",
    "ModelRunPlan",
    "RANDOM_STATE",
    "RRF_K",
    "SOURCE_COLUMN",
    "STRATEGIES",
    "Strategy",
    "WEIGHT_COLUMN",
    "RecommenderModel",
    "build_strategy_model",
    "content_fallback_enabled_from_models",
    "fit_strategies",
    "plan_model_run",
    "recommend_with_models",
    "resolve_recommend_models",
    "resolve_run_models",
    "train_and_recommend",
    # Test / internal helpers kept on the package root for stable imports.
    "_as_recommender_model",
    "_combine_by_priority",
    "_combine_by_weighted_fusion",
    "_epoch_metric_total_epochs",
    "_fit_lightfm_with_epoch_metrics",
    "_recommend_k",
    "_sample_epoch_metric_users",
    "_should_log_epoch",
    "_validate_strategy_names",
    "_warn_on_epoch_metric_trajectory",
]

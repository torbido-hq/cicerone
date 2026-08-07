"""Public model API (``from cicerone.model import …``)."""

from __future__ import annotations

from cicerone.model.constants import (
    COLLABORATIVE_EPOCHS,
    DEFAULT_MODELS,
    LATEST_WINDOW_DAYS,
    RANDOM_STATE,
    RRF_K,
    SOURCE_COLUMN,
    WEIGHT_COLUMN,
)
from cicerone.model.fit import (
    ModelRunPlan,
    content_fallback_enabled_from_models,
    fit_strategies,
    plan_model_run,
    resolve_recommend_models,
    resolve_run_models,
)
from cicerone.model.recommend import recommend_with_models, train_and_recommend
from cicerone.model.strategies import (
    STRATEGIES,
    RecommenderModel,
    Strategy,
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
]

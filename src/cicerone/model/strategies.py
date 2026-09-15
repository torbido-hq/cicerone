"""Strategy registry and RecTools model construction."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
from rectools.dataset import Dataset
from rectools.models import model_from_config

from cicerone.config import STRATEGY_NAMES, ConfigError
from cicerone.content_fallback import CONTENT_FALLBACK_SOURCE, ContentFallbackModel
from cicerone.model.constants import LIGHTFM_NUM_THREADS_SEQUENTIAL
from cicerone.model_config import (
    LIGHTFM_WRAPPER_CLS,
    RECTOOLS_STRATEGY_NAMES,
    SEQUENTIAL_EXTRA_HINT,
    SEQUENTIAL_STRATEGY,
    default_model_configs,
    rectools_model_config,
    sequential_extra_available,
)


class RecommenderModel(Protocol):
    def fit(self, dataset: Dataset) -> object: ...

    def recommend(
        self,
        *,
        users: list[str],
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: list | None = None,
    ) -> pd.DataFrame: ...


_RECOMMEND_PARAMS = {"users", "dataset", "k", "filter_viewed", "items_to_recommend"}


def as_recommender_model(model: object) -> RecommenderModel:
    """Fail fast if `model` does not implement RecommenderModel."""
    fit = getattr(model, "fit", None)
    recommend = getattr(model, "recommend", None)
    if not callable(fit) or not callable(recommend):
        raise TypeError(
            f"{type(model).__name__} does not implement the RecommenderModel protocol "
            "(missing a callable fit() and/or recommend())"
        )
    recommend_params = set(inspect.signature(recommend).parameters)
    missing_params = _RECOMMEND_PARAMS - recommend_params
    if missing_params:
        raise TypeError(
            f"{type(model).__name__}.recommend() is missing expected parameter(s) {sorted(missing_params)}; "
            "the RecommenderModel protocol may have drifted from the installed rectools/implicit version"
        )
    return model  # type: ignore[return-value]


@dataclass(frozen=True)
class Strategy:
    personalized: bool
    source_label: str
    # Item-KNN / sequential / content_fallback need history; LightFM hybrid can score feature-only users.
    requires_interactions: bool = False
    # Optional factory (tests / content_fallback); skips model_from_config.
    factory: Callable[[], RecommenderModel] | None = None


def build_strategy_model(
    name: str,
    *,
    model_configs: dict[str, dict[str, Any]] | None = None,
    lightfm_num_threads: int | None = None,
) -> RecommenderModel:
    """Build one RecTools strategy via ``model_from_config``."""
    if name not in RECTOOLS_STRATEGY_NAMES:
        raise ValueError(
            f"Cannot build {name!r} via RecTools config; available: {list(RECTOOLS_STRATEGY_NAMES)}"
        )
    configs = model_configs if model_configs is not None else default_model_configs()
    if name not in configs:
        raise ValueError(f"No model config for strategy {name!r}; have {sorted(configs)}")
    cfg = deepcopy(configs[name])
    if name == "collaborative" and cfg.get("cls", LIGHTFM_WRAPPER_CLS) == LIGHTFM_WRAPPER_CLS:
        threads = lightfm_num_threads if lightfm_num_threads is not None else LIGHTFM_NUM_THREADS_SEQUENTIAL
        cfg["num_threads"] = threads
    if name == SEQUENTIAL_STRATEGY and not sequential_extra_available():
        raise ConfigError(f"strategy {name!r} requires torch; {SEQUENTIAL_EXTRA_HINT}")
    return as_recommender_model(model_from_config(rectools_model_config(cfg)))


def _build_content_fallback() -> RecommenderModel:
    """Placeholder; real fit builds ContentFallbackModel with items/features."""
    return as_recommender_model(ContentFallbackModel())


STRATEGIES: dict[str, Strategy] = {
    "collaborative": Strategy(personalized=True, source_label="personalized"),
    "item_based": Strategy(
        personalized=True,
        source_label="item_based",
        requires_interactions=True,
    ),
    "sequential": Strategy(
        personalized=True,
        source_label="sequential",
        requires_interactions=True,
    ),
    "ease": Strategy(
        personalized=True,
        source_label="ease",
        requires_interactions=True,
    ),
    "als": Strategy(
        personalized=True,
        source_label="als",
        requires_interactions=True,
    ),
    "content_fallback": Strategy(
        factory=_build_content_fallback,
        personalized=True,
        source_label=CONTENT_FALLBACK_SOURCE,
        requires_interactions=True,
    ),
    "popular": Strategy(personalized=False, source_label="popular_fallback"),
    "popular_in_category": Strategy(personalized=False, source_label="popular_in_category"),
    "latest": Strategy(personalized=False, source_label="latest"),
    "random": Strategy(personalized=False, source_label="random"),
}


def validate_strategy_names(strategies: dict[str, Strategy], strategy_names: tuple[str, ...]) -> None:
    """Raise if STRATEGIES keys drift from ``STRATEGY_NAMES``."""
    if set(strategies) != set(strategy_names):
        raise RuntimeError(
            f"cicerone.model.STRATEGIES keys {sorted(strategies)} must match "
            f"cicerone.config.STRATEGY_NAMES {sorted(strategy_names)} — update both together"
        )


validate_strategy_names(STRATEGIES, STRATEGY_NAMES)

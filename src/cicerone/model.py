"""Trains one or more recommendation strategies (see STRATEGIES) and combines
their outputs into top-K recommendations per user, with a non-personalized
fallback for cold-start users who have too little (or no) personal signal.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Protocol

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from lightfm import LightFM
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel, LightFMWrapperModel, PopularModel

from cicerone.config import STRATEGY_NAMES, validate_model_weights, validate_rrf_k
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, FeatureConfig
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

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "popular"]
LATEST_WINDOW_DAYS = 14
# Reciprocal rank fusion constant (Cormack et al., 2009); default for rrf_k.
RRF_K = 60
SOURCE_COLUMN = "source"
WEIGHT_COLUMN = "_weight"  # internal-only; dropped before returning to callers


def _recommend_k(top_k: int, has_boosts: bool, overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR) -> int:
    if not has_boosts:
        return top_k
    factor = overfetch_factor if overfetch_factor >= 1 else DEFAULT_BOOST_OVERFETCH_FACTOR
    return max(top_k, top_k * factor)


class RecommenderModel(Protocol):
    def fit(self, dataset: Dataset) -> object: ...

    def recommend(
        self,
        *,
        users: list,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: list,
    ) -> pd.DataFrame: ...


_RECOMMEND_PARAMS = {"users", "dataset", "k", "filter_viewed", "items_to_recommend"}


def _as_recommender_model(model: object) -> RecommenderModel:
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
    factory: Callable[[], RecommenderModel]
    personalized: bool
    source_label: str


def _build_collaborative() -> RecommenderModel:
    return _as_recommender_model(
        LightFMWrapperModel(
            LightFM(
                no_components=64,
                loss="warp",
                learning_rate=0.05,
                item_alpha=1e-6,
                user_alpha=1e-6,
                random_state=RANDOM_STATE,
            ),
            epochs=30,
            num_threads=4,
        )
    )


def _build_item_based() -> RecommenderModel:
    return _as_recommender_model(ImplicitItemKNNWrapperModel(TFIDFRecommender(K=20)))


def _build_popular() -> RecommenderModel:
    return _as_recommender_model(PopularModel())


def _build_latest() -> RecommenderModel:
    return _as_recommender_model(
        PopularModel(popularity="n_interactions", period=pd.Timedelta(days=LATEST_WINDOW_DAYS))
    )


STRATEGIES: dict[str, Strategy] = {
    "collaborative": Strategy(_build_collaborative, personalized=True, source_label="personalized"),
    "item_based": Strategy(_build_item_based, personalized=True, source_label="item_based"),
    "popular": Strategy(_build_popular, personalized=False, source_label="popular_fallback"),
    "latest": Strategy(_build_latest, personalized=False, source_label="latest"),
}


def _validate_strategy_names(strategies: dict[str, Strategy], strategy_names: tuple[str, ...]) -> None:
    """Raises if STRATEGIES' keys and cicerone.config.STRATEGY_NAMES drift apart."""
    if set(strategies) != set(strategy_names):
        raise RuntimeError(
            f"cicerone.model.STRATEGIES keys {sorted(strategies)} must match "
            f"cicerone.config.STRATEGY_NAMES {sorted(strategy_names)} — update both together"
        )


_validate_strategy_names(STRATEGIES, STRATEGY_NAMES)


def _combine_by_priority(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    """Fill top-K from strategy outputs in list order: earlier strategies keep
    duplicate (user, item) pairs and fill slots before later ones.
    """
    tagged = []
    for priority, frame in enumerate(frames):
        part = frame.copy()
        part["_priority"] = priority
        tagged.append(part)
    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, "_priority", Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    return combined.drop(columns=[WEIGHT_COLUMN, "_priority"])


def _combine_by_weighted_fusion(
    frames: list[pd.DataFrame], top_k: int, rrf_k: float, source_label_order: list[str]
) -> pd.DataFrame:
    """Weighted reciprocal rank fusion: each strategy's contribution to an
    item's fused score is `weight / (rrf_k + rank)`, summed across every
    strategy that recommended that (user, item) pair. Combined source labels
    are joined in `source_label_order` rather than alphabetically.
    """
    combined = pd.concat(frames, ignore_index=True)
    combined[Columns.Score] = combined[WEIGHT_COLUMN] / (rrf_k + combined[Columns.Rank])

    def _join_labels_in_order(labels: pd.Series) -> str:
        present = set(labels)
        return "+".join(label for label in source_label_order if label in present)

    fused = combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, _join_labels_in_order),
        }
    )
    fused = fused.sort_values([Columns.User, Columns.Score], ascending=[True, False])
    fused[Columns.Rank] = fused.groupby(Columns.User).cumcount() + 1
    fused = fused.groupby(Columns.User, as_index=False).head(top_k)
    return fused[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]


def _fit_strategy(name: str, dataset: Dataset) -> tuple[str, RecommenderModel]:
    """Fit one strategy (picklable for ProcessPoolExecutor workers)."""
    model = STRATEGIES[name].factory()
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


def fit_strategies(
    built: BuiltDataset,
    target_users: list[str],
    enabled_models: list[str] | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
) -> tuple[list[str], dict[str, RecommenderModel]]:
    """Fit (or cache-hit) enabled strategies. ``max_workers > 1`` fits in parallel."""
    dataset = built.dataset
    enabled_models = _resolve_enabled_models(enabled_models)

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
    if to_fit:
        if max_workers > 1 and len(to_fit) > 1:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(to_fit))) as executor:
                for name, model in executor.map(_fit_strategy, to_fit, repeat(dataset)):
                    logger.info("Fitted '%s' on %d interactions", name, len(built.interactions))
                    models[name] = model
        else:
            for name in to_fit:
                logger.info("Fitting '%s' on %d interactions", name, len(built.interactions))
                _, model = _fit_strategy(name, dataset)
                models[name] = model
        if strategy_cache is not None:
            for name in to_fit:
                strategy_cache[name] = models[name]

    return enabled_models, models


def recommend_with_models(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str],
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
) -> pd.DataFrame:
    """Runs recommend + combine on already-fitted strategies (no fit).

    Used by ``train_and_recommend`` and by ``artifact.recommend_from_artifact``
    so a loaded model artifact can produce recommendations without re-training.
    """
    enabled_models = _resolve_enabled_models(enabled_models)
    if weights is not None:
        unknown_weights = [name for name in weights if name not in enabled_models]
        if unknown_weights:
            raise ValueError(
                f"model_weights key(s) {unknown_weights} are not in enabled_models {enabled_models}"
            )
        validate_model_weights(weights)
    validate_rrf_k(rrf_k)

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

    frames = []
    for name in enabled_models:
        strategy = STRATEGIES[name]
        if strategy.personalized and not has_any_warm_user:
            continue
        if name not in models:
            raise ValueError(f"Fitted model for strategy {name!r} is missing; available: {sorted(models)}")

        for cohort_key_value, cohort_users in cohorts:
            allowed_items = allowed_by_cohort[cohort_key_value]
            if not allowed_items:
                continue

            if strategy.personalized:
                cohort_warm = [u for u in cohort_users if u in known_users]
                if not cohort_warm:
                    continue
                recommend_users = cohort_warm
            else:
                recommend_users = cohort_users

            recs = models[name].recommend(
                users=recommend_users,
                dataset=dataset,
                k=recommend_k,
                filter_viewed=strategy.personalized,
                items_to_recommend=allowed_items,
            )
            recs[SOURCE_COLUMN] = strategy.source_label
            recs[WEIGHT_COLUMN] = weights.get(name, 1.0) if weights is not None else 1.0
            frames.append(recs)

    if not frames:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    combine_k = recommend_k if has_boosts else top_k
    if weights is not None:
        source_label_order = [STRATEGIES[name].source_label for name in enabled_models]
        combined = _combine_by_weighted_fusion(
            frames, combine_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    else:
        combined = _combine_by_priority(frames, combine_k)

    if has_boosts:
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
) -> pd.DataFrame:
    """Fit enabled strategies, then recommend + combine. See ``fit_strategies``
    and ``recommend_with_models`` for the split used by model artifacts.
    """
    resolved_models, fitted = fit_strategies(
        built,
        target_users,
        enabled_models=enabled_models,
        strategy_cache=strategy_cache,
        max_workers=max_workers,
    )
    return recommend_with_models(
        fitted,
        built,
        target_users,
        config,
        top_k=top_k,
        enabled_models=resolved_models,
        weights=weights,
        rrf_k=rrf_k,
    )

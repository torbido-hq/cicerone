"""Trains one or more recommendation strategies (see STRATEGIES) and combines
their outputs into top-K recommendations per user, with a non-personalized
fallback for cold-start users who have too little (or no) personal signal.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from lightfm import LightFM
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel, LightFMWrapperModel, PopularModel

from cicerone.config import STRATEGY_NAMES, validate_model_weights, validate_rrf_k
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "popular"]
LATEST_WINDOW_DAYS = 14
# Reciprocal rank fusion constant (Cormack et al., 2009); default for rrf_k.
RRF_K = 60
SOURCE_COLUMN = "source"
# Internal-only; dropped from the output before it's returned to callers.
WEIGHT_COLUMN = "_weight"


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
    """Verifies (at strategy-construction time, not first use) that `model`
    implements the RecommenderModel protocol expected by train_and_recommend.
    A rectools/implicit upgrade that renames or drops one of `.fit`/.recommend`'s
    parameters would otherwise only surface as a confusing TypeError deep
    inside a fold loop the first time the strategy is actually used; this
    fails fast with a clear message naming the offending model/strategy instead.
    """
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


def _recommendable_item_ids(
    items: pd.DataFrame | None, filter_columns: list[str], all_item_ids: pd.Index
) -> list:
    if items is None or not filter_columns:
        return list(all_item_ids)
    mask = pd.Series(True, index=items.index)
    for column in filter_columns:
        if column not in items.columns:
            logger.warning("Configured item_availability_filters column '%s' not found — skipping", column)
            continue
        mask &= items[column].fillna(False)
    allowed = set(items.loc[mask, "item_id"])
    return [i for i in all_item_ids if i in allowed] or list(all_item_ids)


def _combine_by_priority(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    """Concatenates strategy outputs in list order; earlier strategies win
    ties for the same (user, item) pair.
    """
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    return combined.drop(columns=[WEIGHT_COLUMN])


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


def train_and_recommend(
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str] | None = None,
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
) -> pd.DataFrame:
    """`strategy_cache`, if given, caches fitted models by strategy name so
    callers evaluating multiple candidates against the same built dataset
    (see cicerone.automl) can avoid re-fitting a shared strategy.
    """
    dataset = built.dataset
    enabled_models = enabled_models if enabled_models is not None else DEFAULT_MODELS
    if not enabled_models:
        raise ValueError(
            "enabled_models is empty; provide at least one model name, or omit enabled_models/pass None "
            "to use the default"
        )
    unknown_models = [name for name in enabled_models if name not in STRATEGIES]
    if unknown_models:
        raise ValueError(f"Unknown model(s) {unknown_models}; available: {sorted(STRATEGIES)}")
    if weights is not None:
        unknown_weights = [name for name in weights if name not in enabled_models]
        if unknown_weights:
            raise ValueError(
                f"model_weights key(s) {unknown_weights} are not in enabled_models {enabled_models}"
            )
        validate_model_weights(weights)
    validate_rrf_k(rrf_k)

    all_item_ids = dataset.item_id_map.external_ids
    allowed_items = _recommendable_item_ids(built.items, config.item_availability_filters, all_item_ids)

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
    unique_target_users = list(dict.fromkeys(target_users))

    frames = []
    for name in enabled_models:
        strategy = STRATEGIES[name]
        if strategy.personalized and not warm_users:
            continue
        if strategy_cache is not None and name in strategy_cache:
            model = strategy_cache[name]
        else:
            model = strategy.factory()
            logger.info("Fitting '%s' on %d interactions", name, len(built.interactions))
            model.fit(dataset)
            if strategy_cache is not None:
                strategy_cache[name] = model
        recs = model.recommend(
            users=warm_users if strategy.personalized else unique_target_users,
            dataset=dataset,
            k=top_k,
            filter_viewed=strategy.personalized,
            items_to_recommend=allowed_items,
        )
        recs[SOURCE_COLUMN] = strategy.source_label
        recs[WEIGHT_COLUMN] = weights.get(name, 1.0) if weights is not None else 1.0
        frames.append(recs)

    if not frames:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    if weights is not None:
        source_label_order = [STRATEGIES[name].source_label for name in enabled_models]
        combined = _combine_by_weighted_fusion(
            frames, top_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    else:
        combined = _combine_by_priority(frames, top_k)
    return combined.reset_index(drop=True)

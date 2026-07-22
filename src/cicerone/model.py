"""Trains one or more recommendation strategies (see STRATEGIES) and combines
their outputs into top-K recommendations per user, with a non-personalized
fallback for cold-start users who have too little (or no) personal signal.
"""

from __future__ import annotations

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

from cicerone.config import STRATEGY_NAMES
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "popular"]
LATEST_WINDOW_DAYS = 14
# Reciprocal rank fusion constant (Cormack et al., 2009) — dampens the
# influence of very low ranks without needing per-strategy score normalization.
# Default when Settings.rrf_k / train_and_recommend(rrf_k=...) is not set.
RRF_K = 60
# Not a rectools-defined column (see rectools.Columns) — our own "which
# strategy/strategies produced this row" tag, kept as a module constant so
# it's not repeated as a string literal throughout this file.
SOURCE_COLUMN = "source"
# Internal-only column: per-strategy weight used by weighted fusion, dropped
# from the final output before it's returned to callers.
WEIGHT_COLUMN = "_weight"


def validate_model_weights(weights: dict[str, float] | None, *, context: str = "model_weights") -> None:
    """Raises ValueError if any weight is negative. Shared by train_and_recommend
    and cicerone.automl's candidate parsing so both fail on the same invalid
    configurations with the same error shape (`context` only changes the
    message prefix so each caller's error reads naturally).
    """
    if weights is None:
        return
    negative_weights = {name: weight for name, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ValueError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    """Raises ValueError if rrf_k is set but not positive. Shared by
    train_and_recommend and cicerone.automl's candidate parsing (see
    validate_model_weights).
    """
    if rrf_k is not None and rrf_k <= 0:
        raise ValueError(f"{context} must be positive, got {rrf_k}")


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


@dataclass(frozen=True)
class Strategy:
    factory: Callable[[], RecommenderModel]
    # Personalized strategies only run for warm users (filter_viewed=True);
    # non-personalized ones run for every target user and backfill the rest.
    personalized: bool
    source_label: str


def _build_collaborative() -> LightFMWrapperModel:
    return LightFMWrapperModel(
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


def _build_item_based() -> ImplicitItemKNNWrapperModel:
    return ImplicitItemKNNWrapperModel(TFIDFRecommender(K=20))


def _build_popular() -> PopularModel:
    return PopularModel()


def _build_latest() -> PopularModel:
    return PopularModel(popularity="n_interactions", period=pd.Timedelta(days=LATEST_WINDOW_DAYS))


STRATEGIES: dict[str, Strategy] = {
    "collaborative": Strategy(_build_collaborative, personalized=True, source_label="personalized"),
    "item_based": Strategy(_build_item_based, personalized=True, source_label="item_based"),
    "popular": Strategy(_build_popular, personalized=False, source_label="popular_fallback"),
    "latest": Strategy(_build_latest, personalized=False, source_label="latest"),
}


def _validate_strategy_names(strategies: dict[str, Strategy], strategy_names: tuple[str, ...]) -> None:
    """Fails fast (at import time, for the module-level call below) if
    STRATEGIES' keys and cicerone.config.STRATEGY_NAMES -- the canonical list
    Settings.models is validated against at config-load time -- ever drift
    apart, e.g. a strategy added/renamed in one place but not the other.
    """
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
    ties for the same (user, item) pair (e.g. a personalized hit takes
    precedence over a popularity/latest backfill for that pair).
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
    strategy that recommended that (user, item) pair. Rank-based (rather
    than raw-score-based) so heterogeneous strategies — LightFM scores,
    ItemKNN similarities, popularity counts — combine without normalization.
    Per-strategy weights are read from the "_weight" column each frame was
    tagged with in train_and_recommend. Combined source labels for a given
    (user, item) pair are joined in `source_label_order` (i.e. the
    configured enabled_models priority order) rather than alphabetically,
    so e.g. "popular_fallback+latest" reads consistently with how the
    strategies were configured/prioritized, not by coincidence of spelling.
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
    strategy_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """`strategy_cache`, if given, is read from and written to (keyed by
    strategy name) so a caller evaluating multiple candidates against the
    *same* built dataset/target_users/top_k -- e.g. cicerone.automl
    backtesting several candidates per fold -- can avoid re-fitting a
    strategy that's shared by more than one candidate. Unused by the
    single-config job.py call path (defaults to None: no caching, one fit
    per call, exactly the prior behavior).
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
    # `weights is not None` (rather than truthiness) so an explicitly configured
    # but empty `[job.model_weights]` still opts into fusion mode, just with
    # every enabled strategy defaulting to weight 1.0.
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
        logger.info(
            "%d/%d users have no usable signal yet; falling back to non-personalized strategies for them",
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
            recs = strategy_cache[name].copy()
        else:
            model = strategy.factory()
            logger.info("Fitting '%s' on %d interactions", name, len(built.interactions))
            model.fit(dataset)
            recs = model.recommend(
                users=warm_users if strategy.personalized else unique_target_users,
                dataset=dataset,
                k=top_k,
                filter_viewed=strategy.personalized,
                items_to_recommend=allowed_items,
            )
            recs[SOURCE_COLUMN] = strategy.source_label
            if strategy_cache is not None:
                strategy_cache[name] = recs.copy()
        recs = recs.copy()
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

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

from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "popular"]
LATEST_WINDOW_DAYS = 14


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


def train_and_recommend(
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str] | None = None,
) -> pd.DataFrame:
    dataset = built.dataset
    enabled_models = enabled_models if enabled_models is not None else DEFAULT_MODELS
    unknown_models = [name for name in enabled_models if name not in STRATEGIES]
    if unknown_models:
        raise ValueError(f"Unknown model(s) {unknown_models}; available: {sorted(STRATEGIES)}")

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
        recs["source"] = strategy.source_label
        frames.append(recs)

    if not frames:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    return combined.reset_index(drop=True)

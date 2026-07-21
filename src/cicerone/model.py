"""Trains the LightFM model (via rectools) and produces top-K recommendations
for every known user, with a popularity fallback for cold-start users who
have too little (or no) personal signal.
"""

from __future__ import annotations

import logging

import pandas as pd
from lightfm import LightFM
from rectools import Columns
from rectools.models import LightFMWrapperModel, PopularModel

from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _recommendable_item_ids(items: pd.DataFrame | None, filter_columns: list[str], all_item_ids: pd.Index) -> list:
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
    built: BuiltDataset, target_users: list[str], config: FeatureConfig, top_k: int
) -> pd.DataFrame:
    dataset = built.dataset

    model = LightFMWrapperModel(
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
    logger.info("Fitting LightFM on %d interactions", len(built.interactions))
    model.fit(dataset)

    all_item_ids = dataset.item_id_map.external_ids
    allowed_items = _recommendable_item_ids(built.items, config.item_availability_filters, all_item_ids)

    known_users = set(dataset.user_id_map.external_ids)
    warm_users = [u for u in target_users if u in known_users]
    cold_users = [u for u in target_users if u not in known_users]
    if cold_users:
        logger.info("%d/%d users have no usable signal yet; falling back to popularity for them",
                    len(cold_users), len(target_users))

    personalized = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score])
    if warm_users:
        personalized = model.recommend(
            users=warm_users,
            dataset=dataset,
            k=top_k,
            filter_viewed=True,
            items_to_recommend=allowed_items,
        )
        personalized["source"] = "personalized"

    popular_model = PopularModel()
    popular_model.fit(dataset)
    popular = popular_model.recommend(
        users=list(dict.fromkeys(target_users)),
        dataset=dataset,
        k=top_k,
        filter_viewed=False,
        items_to_recommend=allowed_items,
    )
    popular["source"] = "popular_fallback"

    combined = pd.concat([personalized, popular], ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    return combined.reset_index(drop=True)

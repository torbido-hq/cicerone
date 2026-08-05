"""Build a rectools Dataset from events/users/items (see cicerone.io).

Input contract:

  events (required)
    user_id, item_id, event_type, quantity (optional), occurred_at

  users (optional — features + per-user eligibility)
    user_id + columns from user_features / [[eligibility]]

  items (optional — features, availability, eligibility, boosts)
    item_id + columns from item_features / item_availability_filters /
    [[eligibility]] / [[boost]]

Only events is required; missing users/items degrade gracefully.
Weights and feature columns come from config/features.toml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from cicerone.feature_config import FeatureColumn, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltDataset:
    dataset: Dataset
    interactions: pd.DataFrame
    items: pd.DataFrame | None
    users: pd.DataFrame | None = None


def _time_decay_multiplier(occurred_at: pd.Series, half_life_days: float) -> pd.Series:
    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - occurred_at).dt.total_seconds() / 86_400
    age_days = age_days.clip(lower=0)
    return 0.5 ** (age_days / half_life_days)


def build_interactions(events: pd.DataFrame, config: FeatureConfig, half_life_days: float) -> pd.DataFrame:
    """Weighted/aggregated interactions without building a rectools Dataset."""
    df = events.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    df["quantity"] = df.get("quantity", 1)
    df["quantity"] = df["quantity"].fillna(1).clip(lower=1)

    base = df["event_type"].map(config.event_weights)
    unknown = df["event_type"][base.isna()].unique()
    if len(unknown):
        logger.warning("Dropping event_type values missing from event_weights config: %s", unknown)
    df = df.assign(base_weight=base).dropna(subset=["base_weight"])

    df["_qty_multiplier"] = np.where(
        df["event_type"].isin(config.quantity_scaled_events), np.log1p(df["quantity"]), 1.0
    )

    for event_type, cap in config.event_caps.items():
        mask = df["event_type"] == event_type
        if not mask.any():
            continue
        rank = df[mask].groupby(["user_id", "item_id"]).cumcount()
        drop_idx = df[mask].index[rank >= cap]
        df = df.drop(index=drop_idx)

    decay = _time_decay_multiplier(df["occurred_at"], half_life_days)
    df["weight"] = df["base_weight"] * df["_qty_multiplier"] * decay

    aggregated = df.groupby(["user_id", "item_id"], as_index=False).agg(
        weight=("weight", "sum"), datetime=("occurred_at", "max")
    )
    # Floor negative review sums: rectools/LightFM expect non-negative weights.
    aggregated["weight"] = aggregated["weight"].clip(lower=1e-3)

    aggregated = aggregated.rename(columns={"user_id": Columns.User, "item_id": Columns.Item})
    aggregated[Columns.Weight] = aggregated.pop("weight")
    aggregated[Columns.Datetime] = aggregated.pop("datetime")
    return aggregated


def _explode_features(
    df: pd.DataFrame, id_column: str, rectools_id_column: str, columns: list[FeatureColumn]
) -> pd.DataFrame:
    frames = []
    for feature in columns:
        if feature.column not in df.columns:
            logger.warning("Configured feature column '%s' not found — skipping", feature.column)
            continue
        part = df[[id_column, feature.column]].rename(
            columns={id_column: rectools_id_column, feature.column: "value"}
        )
        if feature.type == "list":
            part = part.explode("value")
        part = part.dropna(subset=["value"])
        part["feature"] = feature.column
        frames.append(part[[rectools_id_column, "feature", "value"]])
    if not frames:
        return pd.DataFrame(columns=[rectools_id_column, "feature", "value"])
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    events: pd.DataFrame,
    users: pd.DataFrame | None,
    items: pd.DataFrame | None,
    config: FeatureConfig,
    half_life_days: float,
) -> BuiltDataset:
    interactions = build_interactions(events, config, half_life_days)

    user_features_df = (
        _explode_features(users, "user_id", Columns.User, config.user_features) if users is not None else None
    )
    item_features_df = (
        _explode_features(items, "item_id", Columns.Item, config.item_features) if items is not None else None
    )

    has_user_features = user_features_df is not None and not user_features_df.empty
    has_item_features = item_features_df is not None and not item_features_df.empty

    cat_user_features = (
        list(user_features_df["feature"].unique())
        if user_features_df is not None and has_user_features
        else None
    )
    cat_item_features = (
        list(item_features_df["feature"].unique())
        if item_features_df is not None and has_item_features
        else None
    )

    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=user_features_df if has_user_features else None,
        cat_user_features=cat_user_features,
        item_features_df=item_features_df if has_item_features else None,
        cat_item_features=cat_item_features,
    )
    return BuiltDataset(dataset=dataset, interactions=interactions, items=items, users=users)

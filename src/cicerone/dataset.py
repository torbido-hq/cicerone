"""Builds a rectools Dataset from the raw events/users/items provided by the
configured input source (see cicerone.io).

Input contract:

  events   (required)
    user_id       str   stable user identifier
    item_id       str   stable item/product identifier
    event_type    str   must have an entry in config/features.toml -> event_weights
    quantity      int   optional, defaults to 1 (used by quantity_scaled_events)
    occurred_at   datetime64  UTC timestamp of the event

  users   (optional — enables warm/cold user features)
    user_id  str
    + one column per entry in config/features.toml -> user_features

  items   (optional — enables warm/cold item features + fallback ranking)
    item_id  str
    + one column per entry in config/features.toml -> item_features
    + boolean columns listed in item_availability_filters

Only events is required; users/items features are best-effort and missing
inputs degrade gracefully to an interactions-only model. Which event types
carry weight, and which user/item columns feed the model, are NOT hardcoded
here — see cicerone.feature_config / config/features.toml.
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


def _time_decay_multiplier(occurred_at: pd.Series, half_life_days: float) -> pd.Series:
    now = pd.Timestamp.utcnow()
    age_days = (now - occurred_at).dt.total_seconds() / 86_400
    age_days = age_days.clip(lower=0)
    return 0.5 ** (age_days / half_life_days)


def build_interactions(events: pd.DataFrame, config: FeatureConfig, half_life_days: float) -> pd.DataFrame:
    """Builds the weighted/aggregated interactions DataFrame alone, without
    constructing user/item feature matrices or a rectools Dataset. Exposed
    (not `_`-prefixed) for callers that only need interactions -- e.g.
    cicerone.automl scoring a held-out fold's ground truth, where building a
    full Dataset (feature exploding included) would be wasted work.
    """
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

    # Apply per-(user, item, event_type) caps before decay, so noisy
    # high-frequency signals (e.g. views) can't drown out rarer ones.
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
    # Negative reviews can push a pair below zero; rectools/LightFM expects
    # non-negative implicit weights, so floor at a small positive epsilon
    # instead of dropping the row (an epsilon still ranks it near-last).
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

    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=user_features_df if has_user_features else None,
        # rectools' cat_*_features must list the actual feature *names* (the
        # values in the "feature" column) to treat as categorical — every
        # feature we build is categorical/list-exploded, so pass them all.
        cat_user_features=list(user_features_df["feature"].unique()) if has_user_features else None,
        item_features_df=item_features_df if has_item_features else None,
        cat_item_features=list(item_features_df["feature"].unique()) if has_item_features else None,
    )
    return BuiltDataset(dataset=dataset, interactions=interactions, items=items)

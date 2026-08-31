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

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from cicerone.feature_config import FeatureColumn, FeatureConfig
from cicerone.weighting import event_row_weights

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltDataset:
    dataset: Dataset
    interactions: pd.DataFrame
    items: pd.DataFrame | None
    users: pd.DataFrame | None = None


@dataclass(frozen=True)
class _NormalizedFeatures:
    """Optional exploded feature frame + categorical names for Dataset.construct.

    Both fields are None, or both are set (never mixed).
    """

    frame: pd.DataFrame | None
    categorical: list[str] | None


def _normalize_feature_df(df: pd.DataFrame | None) -> _NormalizedFeatures:
    """Collapse empty/missing frames to None; otherwise read categorical names.

    Expects ``None`` or an ``_explode_features`` result (``feature`` column present).
    """
    if df is None or df.empty:
        return _NormalizedFeatures(frame=None, categorical=None)
    if "feature" not in df.columns:
        raise ValueError(
            "_normalize_feature_df expected an _explode_features frame with a 'feature' column, "
            f"got columns {list(df.columns)}"
        )
    return _NormalizedFeatures(frame=df, categorical=list(df["feature"].unique()))


def _time_decay_multiplier(occurred_at: pd.Series, half_life_days: float) -> pd.Series:
    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - occurred_at).dt.total_seconds() / 86_400
    age_days = age_days.clip(lower=0)
    return 0.5 ** (age_days / half_life_days)


def build_interactions(events: pd.DataFrame, config: FeatureConfig, half_life_days: float) -> pd.DataFrame:
    """Weighted/aggregated interactions without building a rectools Dataset."""
    df = events.copy()
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    if "quantity" not in df.columns:
        df["quantity"] = 1.0
    else:
        original_qty = df["quantity"]
        parsed_qty = pd.to_numeric(original_qty, errors="coerce")
        parsed_qty = parsed_qty.mask(original_qty.isna(), 1)
        df["quantity"] = parsed_qty.clip(lower=1)

    row_weight = event_row_weights(
        df["event_type"],
        df["quantity"],
        event_weights=config.event_weights,
        quantity_scaled_events=config.quantity_scaled_events,
    )
    unknown_type = ~df["event_type"].isin(set(config.event_weights))
    unknown = df["event_type"][unknown_type].unique()
    if len(unknown):
        logger.warning("Dropping event_type values missing from event_weights config: %s", unknown)
    bad_qty = df["event_type"][row_weight.isna() & ~unknown_type].unique()
    if len(bad_qty):
        logger.warning(
            "Dropping quantity-scaled events with non-numeric quantity: %s",
            bad_qty,
        )
    df = df.assign(row_weight=row_weight).dropna(subset=["row_weight"])

    if config.event_caps:
        capped_types = set(config.event_caps)
        mask = df["event_type"].isin(capped_types)
        if mask.any():
            # One sort for all capped event types; keep most recent ``cap`` per
            # (user, item, event_type).
            capped = df.loc[mask].sort_values("occurred_at", ascending=False, kind="mergesort")
            rank = capped.groupby(["user_id", "item_id", "event_type"], sort=False).cumcount()
            limits = capped["event_type"].map(config.event_caps)
            df = df.drop(index=capped.index[rank >= limits])

    decay = _time_decay_multiplier(df["occurred_at"], half_life_days)
    df["weight"] = df["row_weight"] * decay

    aggregated = df.groupby(["user_id", "item_id"], as_index=False).agg(
        weight=("weight", "sum"), datetime=("occurred_at", "max")
    )
    # Drop non-positive aggregates (rectools/LightFM require weight > 0).
    before = len(aggregated)
    aggregated = aggregated.loc[aggregated["weight"] > 0].reset_index(drop=True)
    dropped = before - len(aggregated)
    if dropped:
        logger.info(
            "Dropped %d (user, item) pairs with non-positive aggregated weight",
            dropped,
        )

    aggregated = aggregated.rename(columns={"user_id": Columns.User, "item_id": Columns.Item})
    aggregated[Columns.Weight] = aggregated.pop("weight")
    aggregated[Columns.Datetime] = aggregated.pop("datetime")
    return aggregated


def _explode_features(
    df: pd.DataFrame, id_column: str, rectools_id_column: str, columns: list[FeatureColumn]
) -> pd.DataFrame:
    """Long-form features with columns ``{id, feature, value}`` (possibly empty)."""
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

    user_features = _normalize_feature_df(
        _explode_features(users, "user_id", Columns.User, config.user_features) if users is not None else None
    )
    item_features = _normalize_feature_df(
        _explode_features(items, "item_id", Columns.Item, config.item_features) if items is not None else None
    )

    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=user_features.frame,
        cat_user_features=user_features.categorical,
        item_features_df=item_features.frame,
        cat_item_features=item_features.categorical,
    )
    return BuiltDataset(dataset=dataset, interactions=interactions, items=items, users=users)

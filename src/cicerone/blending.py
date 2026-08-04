"""Per-user weighted blending of personalized / popular / latest sources.

Replaces the binary warm/cold fallback with a gradual curve: the personalized
weight grows with the user's interaction count; the remainder is split between
popular and latest (item publication date). See ``[blending]`` in
``config/features.toml``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import pandas as pd
from rectools import Columns

from cicerone.feature_config import BlendingConfig

logger = logging.getLogger(__name__)

SOURCE_COLUMN = "source"
BLENDED_SOURCE = "blended"
PERSONALIZED_SOURCE = "personalized"
POPULAR_SOURCE = "popular_fallback"
LATEST_SOURCE = "latest"
COLD_START_USER_ID = "__cold_start__"

# Sources whose contribution counts toward a "blended" label when >1 participate.
BLENDABLE_SOURCES = (PERSONALIZED_SOURCE, POPULAR_SOURCE, LATEST_SOURCE)

_WEIGHT_EPS = 1e-9


def personalized_weight(n_interactions: int, config: BlendingConfig) -> float:
    """Map interaction count → weight in [0, 1] for the personalized source."""
    n = max(0, int(n_interactions))
    if config.curve == "linear":
        saturate = config.saturate_at if config.saturate_at > 0 else 1.0
        return min(1.0, n / saturate)
    # sigmoid
    return 1.0 / (1.0 + math.exp(-config.steepness * (n - config.midpoint)))


def source_weights(
    n_interactions: int,
    config: BlendingConfig,
    *,
    latest_available: bool,
) -> dict[str, float]:
    """Return non-negative weights for personalized / popular / latest.

    Weights sum to 1. When ``latest`` is unavailable its share is absorbed by
    ``popular``.
    """
    p = personalized_weight(n_interactions, config)
    remainder = max(0.0, 1.0 - p)
    if latest_available:
        popular = remainder * config.popular_share
        latest = remainder * (1.0 - config.popular_share)
    else:
        popular = remainder
        latest = 0.0
    return {
        PERSONALIZED_SOURCE: p,
        POPULAR_SOURCE: popular,
        LATEST_SOURCE: latest,
    }


def resolve_latest_date_column(
    items: pd.DataFrame | None,
    candidates: Sequence[str],
) -> str | None:
    """First usable datetime column from ``candidates`` present in ``items``."""
    if items is None or items.empty or not candidates:
        return None
    for column in candidates:
        if column not in items.columns:
            continue
        parsed = pd.to_datetime(items[column], errors="coerce", utc=True)
        if parsed.notna().any():
            return column
    return None


def interaction_counts(interactions: pd.DataFrame) -> dict[str, int]:
    """Count rows per user in a rectools interactions frame."""
    if interactions.empty or Columns.User not in interactions.columns:
        return {}
    counts = interactions.groupby(Columns.User).size()
    return {str(user): int(count) for user, count in counts.items()}


def build_latest_ranking(
    items: pd.DataFrame,
    date_column: str,
    allowed_item_ids: Sequence,
    top_k: int,
    target_users: Sequence[str],
) -> pd.DataFrame:
    """Non-personalized top-K by newest ``date_column``, expanded to every user.

    Availability / eligibility must already be encoded in ``allowed_item_ids``
    (filter before blend, not after).
    """
    if not target_users or top_k < 1 or not allowed_item_ids:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    allowed = {str(i) for i in allowed_item_ids}
    frame = items.loc[items["item_id"].astype(str).isin(allowed)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    frame["_date"] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=["_date"])
    if frame.empty:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    frame = frame.sort_values(["_date", "item_id"], ascending=[False, True]).head(top_k)
    frame = frame.reset_index(drop=True)
    ranks = list(range(1, len(frame) + 1))
    # Newer → higher score so boosts/fusion stay coherent with other sources.
    scores = [float(len(frame) - i + 1) for i in ranks]
    item_ids = frame["item_id"].astype(str).tolist()

    rows = [
        {
            Columns.User: user_id,
            Columns.Item: item_id,
            Columns.Rank: rank,
            Columns.Score: score,
            SOURCE_COLUMN: LATEST_SOURCE,
        }
        for user_id in target_users
        for item_id, rank, score in zip(item_ids, ranks, scores, strict=True)
    ]
    return pd.DataFrame(rows)


def _normalize_source_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])
    out = frame.copy()
    out[Columns.User] = out[Columns.User].astype(str)
    out[Columns.Item] = out[Columns.Item].astype(str)
    out[SOURCE_COLUMN] = source_label
    return out[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]


def _rrf_contrib(rank: float, weight: float, rrf_k: float) -> float:
    return weight / (rrf_k + rank)


def blend_for_users(
    *,
    personalized: pd.DataFrame,
    popular: pd.DataFrame,
    latest: pd.DataFrame | None,
    counts: Mapping[str, int],
    target_users: Sequence[str],
    config: BlendingConfig,
    top_k: int,
    latest_available: bool,
) -> pd.DataFrame:
    """Weighted reciprocal-rank fusion with per-user source weights."""
    frames = {
        PERSONALIZED_SOURCE: _normalize_source_frame(personalized, PERSONALIZED_SOURCE),
        POPULAR_SOURCE: _normalize_source_frame(popular, POPULAR_SOURCE),
        LATEST_SOURCE: (
            _normalize_source_frame(latest, LATEST_SOURCE)
            if latest is not None and latest_available
            else pd.DataFrame(
                columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]
            )
        ),
    }

    by_user_item: dict[tuple[str, str], dict[str, float | set[str]]] = {}
    for user_id in dict.fromkeys(str(u) for u in target_users):
        weights = source_weights(
            counts.get(user_id, 0),
            config,
            latest_available=latest_available,
        )
        for source_label, frame in frames.items():
            weight = weights.get(source_label, 0.0)
            if weight <= _WEIGHT_EPS:
                continue
            user_rows = frame[frame[Columns.User] == user_id]
            for row in user_rows.itertuples(index=False):
                key = (user_id, str(getattr(row, Columns.Item)))
                entry = by_user_item.setdefault(key, {"score": 0.0, "sources": set()})
                entry["score"] = float(entry["score"]) + _rrf_contrib(
                    float(getattr(row, Columns.Rank)), weight, config.rrf_k
                )
                sources = entry["sources"]
                assert isinstance(sources, set)
                sources.add(source_label)

    if not by_user_item:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])

    rows = []
    for (user_id, item_id), entry in by_user_item.items():
        sources = entry["sources"]
        assert isinstance(sources, set)
        label = BLENDED_SOURCE if len(sources) > 1 else next(iter(sources))
        rows.append(
            {
                Columns.User: user_id,
                Columns.Item: item_id,
                Columns.Score: float(entry["score"]),
                SOURCE_COLUMN: label,
            }
        )

    combined = pd.DataFrame(rows)
    combined = combined.sort_values(
        [Columns.User, Columns.Score, Columns.Item], ascending=[True, False, True]
    )
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    return combined[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]].reset_index(
        drop=True
    )


def append_cold_start_rows(
    recommendations: pd.DataFrame,
    *,
    popular: pd.DataFrame,
    latest: pd.DataFrame | None,
    config: BlendingConfig,
    top_k: int,
    latest_available: bool,
) -> pd.DataFrame:
    """Ensure a shared ``__cold_start__`` lookup row set exists for serve fallback."""
    cold = blend_for_users(
        personalized=pd.DataFrame(
            columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]
        ),
        popular=_retag_user(popular, COLD_START_USER_ID),
        latest=_retag_user(latest, COLD_START_USER_ID) if latest is not None else None,
        counts={COLD_START_USER_ID: 0},
        target_users=[COLD_START_USER_ID],
        config=config,
        top_k=top_k,
        latest_available=latest_available,
    )
    if cold.empty:
        return recommendations
    without = recommendations[recommendations[Columns.User].astype(str) != COLD_START_USER_ID]
    return pd.concat([without, cold], ignore_index=True)


def _retag_user(frame: pd.DataFrame | None, user_id: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])
    # Use one representative user's ranking (same non-personalized list for all).
    sample_user = frame[Columns.User].astype(str).iloc[0]
    rows = frame[frame[Columns.User].astype(str) == sample_user].copy()
    rows[Columns.User] = user_id
    return rows

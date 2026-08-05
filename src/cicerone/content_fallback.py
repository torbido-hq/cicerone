"""Content-based cold-item fallback: recommend zero-interaction items by feature similarity.

Uses one-hot encodings of configured item_features (categorical / list) and
cosine similarity against each user's interaction history. No free-text / TF-IDF
path — features.toml item_features are pure categoricals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cicerone.feature_config import FeatureColumn
from cicerone.ids import interactions_item_column, interactions_user_column, items_id_column

logger = logging.getLogger(__name__)

CONTENT_FALLBACK_SOURCE = "content_fallback"
# Cap history length so cold×history cosine stays bounded for heavy users.
_MAX_HISTORY_ITEMS = 50
# Soft cap on a single dense cosine block; larger cold sets are scored in batches.
_DENSE_SIM_BATCH_PRODUCT = 50_000


def _is_missing(value: object) -> bool:
    """True for None / NaN / pd.NA / NaT; False for containers and other values."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    # pd.isna on array-likes returns an array; treat the value as present.
    if isinstance(result, (np.ndarray, pd.Series, list)):
        return False
    return bool(result)


def _feature_dict(
    row: pd.Series,
    feature_columns: Sequence[FeatureColumn | tuple[str, str]],
) -> dict[str, float]:
    """Map one item row to {feature=value: 1.0} tokens for DictVectorizer."""
    tokens: dict[str, float] = {}
    for spec in feature_columns:
        if isinstance(spec, FeatureColumn):
            column, ftype = spec.column, spec.type
        else:
            column, ftype = spec
        if column not in row.index:
            continue
        value = row[column]
        if ftype == "list":
            if _is_missing(value):
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for entry in values:
                if _is_missing(entry):
                    continue
                tokens[f"{column}={entry}"] = 1.0
        else:
            if _is_missing(value):
                continue
            tokens[f"{column}={value}"] = 1.0
    return tokens


def _max_cosine_scores(cold_matrix, hist_matrix) -> np.ndarray:
    """Per-cold-item max cosine vs history (exact).

    Batches the cold axis when a single dense cold×history block would be large,
    so catalogs with many zero-interaction items do not allocate one huge matrix.
    History is already capped at ``_MAX_HISTORY_ITEMS`` by the caller.
    """
    n_cold = int(cold_matrix.shape[0])
    n_hist = int(hist_matrix.shape[0])
    if n_cold == 0 or n_hist == 0:
        return np.zeros(n_cold, dtype=float)

    batch_size = max(1, _DENSE_SIM_BATCH_PRODUCT // max(n_hist, 1))
    if n_cold <= batch_size:
        sim = cosine_similarity(cold_matrix, hist_matrix, dense_output=True)
        return sim.max(axis=1)

    scores = np.empty(n_cold, dtype=float)
    for start in range(0, n_cold, batch_size):
        end = min(start + batch_size, n_cold)
        block = cosine_similarity(cold_matrix[start:end], hist_matrix, dense_output=True)
        scores[start:end] = block.max(axis=1)
    return scores


class ContentFallbackModel:
    """RecommenderModel-compatible strategy for brand-new (zero-event) items."""

    def __init__(
        self,
        feature_columns: Sequence[FeatureColumn | tuple[str, str]] | None = None,
        max_neighbors: int = 50,
        items: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
    ) -> None:
        self.feature_columns: list[FeatureColumn | tuple[str, str]] = list(feature_columns or [])
        self.max_neighbors = max_neighbors
        self.items = items
        self.interactions = interactions
        self._item_ids: list = []
        self._item_index: dict = {}
        self._matrix = None
        self._cold_ids: list = []
        self._cold_indices: np.ndarray = np.array([], dtype=int)
        self._user_history: dict = {}
        self._vectorizer: DictVectorizer | None = None

    def _reset_item_state(self) -> None:
        self._item_ids = []
        self._item_index = {}
        self._matrix = None
        self._cold_ids = []
        self._cold_indices = np.array([], dtype=int)

    def _release_fit_frames(self) -> None:
        # Drop source frames once matrices / history are built (artifact size + RAM).
        self.items = None
        self.interactions = None

    def fit(self, dataset: Dataset) -> ContentFallbackModel:
        del dataset  # history/cold set come from interactions + items frames
        self._user_history = defaultdict(list)
        if self.interactions is not None and not self.interactions.empty:
            user_col = interactions_user_column(self.interactions)
            item_col = interactions_item_column(self.interactions)
            for user_id, item_id in zip(
                self.interactions[user_col].tolist(),
                self.interactions[item_col].tolist(),
                strict=True,
            ):
                self._user_history[user_id].append(item_id)

        if self.items is None or self.items.empty or not self.feature_columns:
            logger.info("Content fallback: no items/features — strategy will emit no rows")
            self._reset_item_state()
            self._release_fit_frames()
            return self

        id_col = items_id_column(self.items)
        interacted = set()
        if self.interactions is not None and not self.interactions.empty:
            item_col = interactions_item_column(self.interactions)
            interacted = set(self.interactions[item_col].tolist())

        dicts = []
        item_ids = []
        for _, row in self.items.iterrows():
            item_id = row[id_col]
            tokens = _feature_dict(row, self.feature_columns)
            if not tokens:
                continue
            dicts.append(tokens)
            item_ids.append(item_id)

        if not dicts:
            self._reset_item_state()
            self._release_fit_frames()
            return self

        self._vectorizer = DictVectorizer(sparse=True)
        self._matrix = self._vectorizer.fit_transform(dicts)
        self._item_ids = item_ids
        self._item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        cold_indices = [i for i, item_id in enumerate(item_ids) if item_id not in interacted]
        self._cold_indices = np.asarray(cold_indices, dtype=int)
        self._cold_ids = [item_ids[i] for i in cold_indices]
        logger.info(
            "Content fallback: %d catalog items with features, %d cold (zero-interaction)",
            len(item_ids),
            len(self._cold_ids),
        )
        self._release_fit_frames()
        return self

    def recommend(
        self,
        *,
        users: list,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: list | None = None,
    ) -> pd.DataFrame:
        del dataset
        empty = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Score, Columns.Rank])
        if self._matrix is None or len(self._cold_ids) == 0 or k < 1:
            return empty

        allow = None if items_to_recommend is None else set(items_to_recommend)
        cold_mask = []
        cold_ids_filtered = []
        for idx, item_id in zip(self._cold_indices.tolist(), self._cold_ids, strict=True):
            if allow is not None and item_id not in allow:
                continue
            cold_mask.append(idx)
            cold_ids_filtered.append(item_id)
        if not cold_mask:
            return empty

        cold_matrix = self._matrix[cold_mask]
        item_index = self._item_index
        take = min(k, self.max_neighbors, len(cold_ids_filtered))

        rows: list[dict] = []
        for user in users:
            history = self._user_history.get(user, [])
            if not history:
                continue
            # Most recent interactions first when history was appended in event order.
            recent_history = history[-_MAX_HISTORY_ITEMS:]
            hist_indices = [item_index[i] for i in recent_history if i in item_index]
            if not hist_indices:
                continue
            hist_matrix = self._matrix[hist_indices]
            scores = _max_cosine_scores(cold_matrix, hist_matrix)
            seen = set(history) if filter_viewed else set()
            candidates = [
                (cold_ids_filtered[i], float(scores[i]))
                for i in range(len(cold_ids_filtered))
                if cold_ids_filtered[i] not in seen and scores[i] > 0
            ]
            if not candidates:
                continue
            if take >= len(candidates):
                ranked = sorted(candidates, key=lambda pair: pair[1], reverse=True)
            else:
                score_arr = np.fromiter((s for _, s in candidates), dtype=float, count=len(candidates))
                top_idx = np.argpartition(score_arr, -take)[-take:]
                ranked = sorted(
                    (candidates[i] for i in top_idx),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            for rank, (item_id, score) in enumerate(ranked, start=1):
                rows.append(
                    {
                        Columns.User: user,
                        Columns.Item: item_id,
                        Columns.Score: score,
                        Columns.Rank: rank,
                    }
                )

        if not rows:
            return empty
        return pd.DataFrame(rows)


def build_content_fallback_model(
    feature_columns: Sequence[FeatureColumn],
    max_neighbors: int,
    items: pd.DataFrame | None,
    interactions: pd.DataFrame | None,
) -> ContentFallbackModel:
    """Factory used by model._fit_strategy (ProcessPool-picklable FeatureColumns)."""
    return ContentFallbackModel(
        feature_columns=list(feature_columns),
        max_neighbors=max_neighbors,
        items=items,
        interactions=interactions,
    )

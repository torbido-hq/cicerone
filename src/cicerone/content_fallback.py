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

logger = logging.getLogger(__name__)

CONTENT_FALLBACK_SOURCE = "content_fallback"

_USER_ID_COLUMNS = (Columns.User, "user_id")
_ITEM_ID_COLUMNS = (Columns.Item, "item_id")


def _require_id_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, frame_name: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(
        f"{frame_name} is missing a required id column; expected one of {list(candidates)}, "
        f"got columns {list(frame.columns)}"
    )


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
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for entry in values:
                if entry is None or (isinstance(entry, float) and np.isnan(entry)):
                    continue
                tokens[f"{column}={entry}"] = 1.0
        else:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            tokens[f"{column}={value}"] = 1.0
    return tokens


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
        self._matrix = None
        self._cold_ids: list = []
        self._cold_indices: np.ndarray = np.array([], dtype=int)
        self._user_history: dict = {}
        self._vectorizer: DictVectorizer | None = None

    def fit(self, dataset: Dataset) -> ContentFallbackModel:
        del dataset  # history/cold set come from interactions + items frames
        self._user_history = defaultdict(list)
        if self.interactions is not None and not self.interactions.empty:
            user_col = _require_id_column(self.interactions, _USER_ID_COLUMNS, frame_name="interactions")
            item_col = _require_id_column(self.interactions, _ITEM_ID_COLUMNS, frame_name="interactions")
            for user_id, item_id in zip(
                self.interactions[user_col].tolist(),
                self.interactions[item_col].tolist(),
                strict=True,
            ):
                self._user_history[user_id].append(item_id)

        if self.items is None or self.items.empty or not self.feature_columns:
            logger.info("Content fallback: no items/features — strategy will emit no rows")
            self._item_ids = []
            self._matrix = None
            self._cold_ids = []
            self._cold_indices = np.array([], dtype=int)
            return self

        id_col = _require_id_column(self.items, _ITEM_ID_COLUMNS, frame_name="items")
        interacted = set()
        if self.interactions is not None and not self.interactions.empty:
            item_col = _require_id_column(self.interactions, _ITEM_ID_COLUMNS, frame_name="interactions")
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
            self._item_ids = []
            self._matrix = None
            self._cold_ids = []
            self._cold_indices = np.array([], dtype=int)
            return self

        self._vectorizer = DictVectorizer(sparse=True)
        self._matrix = self._vectorizer.fit_transform(dicts)
        self._item_ids = item_ids
        cold_indices = [i for i, item_id in enumerate(item_ids) if item_id not in interacted]
        self._cold_indices = np.asarray(cold_indices, dtype=int)
        self._cold_ids = [item_ids[i] for i in cold_indices]
        logger.info(
            "Content fallback: %d catalog items with features, %d cold (zero-interaction)",
            len(item_ids),
            len(self._cold_ids),
        )
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
        item_index = {item_id: i for i, item_id in enumerate(self._item_ids)}
        take = min(k, self.max_neighbors, len(cold_ids_filtered))

        rows: list[dict] = []
        for user in users:
            history = self._user_history.get(user, [])
            if not history:
                continue
            hist_indices = [item_index[i] for i in history if i in item_index]
            if not hist_indices:
                continue
            hist_matrix = self._matrix[hist_indices]
            # cold × history cosine; score = max similarity to any history item
            sim = cosine_similarity(cold_matrix, hist_matrix, dense_output=True)
            scores = sim.max(axis=1)
            seen = set(history) if filter_viewed else set()
            ranked = sorted(
                (
                    (cold_ids_filtered[i], float(scores[i]))
                    for i in range(len(cold_ids_filtered))
                    if cold_ids_filtered[i] not in seen and scores[i] > 0
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )[:take]
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

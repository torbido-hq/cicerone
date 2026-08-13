"""Cheap incremental update: refresh popular/latest slices; write-through."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.store import empty_recommendations_frame, load_recommendations_frame
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import OutputSink
from cicerone.io.recommendation_reader import (
    ITEM_COLUMN,
    RANK_COLUMN,
    RECOMMENDATION_COLUMNS,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
)

logger = logging.getLogger(__name__)

INCREMENTAL_SOURCE = "incremental"
_PRESERVE_SOURCES = frozenset(
    {
        "personalized",
        "item_based",
        "content_fallback",
        "blended",
    }
)


class IncrementalUpdater:
    """Merge micro-batch events into existing top-K rows and write via OutputSink.

    Does not import LightFM / RecTools. Personalized rows are preserved; popular
    and latest slices for affected users (and ``__cold_start__``) are refreshed
    from the flushed batch. Full collaborative refits wait for ``job.run()``.
    """

    def __init__(
        self,
        *,
        sink: OutputSink,
        output_settings: Any,
        feature_config: FeatureConfig | None,
        top_k: int,
        busy_check: Callable[[], bool] | None = None,
        on_success: Callable[[], None] | None = None,
    ):
        self._sink = sink
        self._output_settings = output_settings
        self._feature_config = feature_config
        self._top_k = top_k
        self._busy_check = busy_check
        self._on_success = on_success
        self._last_success_at: datetime | None = None
        self._events_applied = 0

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    @property
    def events_applied(self) -> int:
        return self._events_applied

    def apply(self, events: Sequence[NormalizedEvent]) -> int:
        if not events:
            return 0
        if self._busy_check is not None and self._busy_check():
            logger.info("Skipping incremental update: full retrain in progress")
            return 0

        batch = events_to_dataframe(events)
        existing = load_recommendations_frame(self._output_settings)
        if existing.empty:
            existing = empty_recommendations_frame()

        popular_ranking = self._popular_ranking(batch)
        latest_ranking = self._latest_ranking(batch)
        affected_users = sorted(set(batch[USER_COLUMN].astype(str)))

        frames = [existing[~existing[USER_COLUMN].astype(str).isin([*affected_users, COLD_START_USER_ID])]]
        for user_id in affected_users:
            prior = existing[existing[USER_COLUMN].astype(str) == user_id]
            frames.append(self._merge_user_rows(user_id, prior, popular_ranking, latest_ranking, batch))
        frames.append(self._cold_start_rows(popular_ranking, latest_ranking))
        frames = [frame for frame in frames if frame is not None and not frame.empty]

        merged = pd.concat(frames, ignore_index=True) if frames else empty_recommendations_frame()
        merged = merged[list(RECOMMENDATION_COLUMNS)]
        self._sink.write_recommendations(merged)

        now = datetime.now(UTC)
        manifest = {
            "triggered_by": "incremental",
            "status": "success",
            "error": None,
            "generated_at": now.isoformat(),
            "n_events": len(events),
            "incremental_events_applied": len(events),
            "last_incremental_at": now.isoformat(),
            "n_users_with_recommendations": int(merged[USER_COLUMN].nunique()) if not merged.empty else 0,
            "top_k": self._top_k,
            "partial_outputs": True,
        }
        self._sink.write_manifest(manifest)
        self._last_success_at = now
        self._events_applied += len(events)
        if self._on_success is not None:
            self._on_success()
        logger.info(
            "Incremental update wrote recommendations for %d user(s) from %d event(s)",
            len(affected_users),
            len(events),
        )
        return len(events)

    def _event_weight(self, event_type: str, quantity: int) -> float:
        if self._feature_config is None:
            return float(quantity)
        weights = self._feature_config.event_weights
        if event_type not in weights:
            return 0.0
        weight = float(weights[event_type])
        if event_type in self._feature_config.quantity_scaled_events:
            weight *= math.log1p(max(quantity, 0))
        return weight

    def _popular_ranking(self, batch: pd.DataFrame) -> pd.DataFrame:
        scores: dict[str, float] = {}
        for row in batch.itertuples(index=False):
            weight = self._event_weight(str(row.event_type), int(row.quantity))
            if weight == 0.0:
                continue
            item_id = str(row.item_id)
            scores[item_id] = scores.get(item_id, 0.0) + weight
        if not scores:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self._top_k]
        return pd.DataFrame(
            [
                {ITEM_COLUMN: item_id, SCORE_COLUMN: score, SOURCE_COLUMN: POPULAR_SOURCE}
                for item_id, score in ranked
            ]
        )

    def _latest_ranking(self, batch: pd.DataFrame) -> pd.DataFrame:
        if batch.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = batch.copy()
        frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
        frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], utc=True)
        latest = (
            frame.sort_values("occurred_at", ascending=False)
            .drop_duplicates(subset=[ITEM_COLUMN], keep="first")
            .head(self._top_k)
        )
        rows = []
        for rank, row in enumerate(latest.itertuples(index=False), start=1):
            rows.append(
                {
                    ITEM_COLUMN: str(row.item_id),
                    SCORE_COLUMN: float(self._top_k - rank + 1),
                    SOURCE_COLUMN: LATEST_SOURCE,
                }
            )
        return (
            pd.DataFrame(rows) if rows else pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        )

    def _merge_user_rows(
        self,
        user_id: str,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        batch: pd.DataFrame,
    ) -> pd.DataFrame:
        preserved = (
            prior[prior[SOURCE_COLUMN].astype(str).isin(_PRESERVE_SOURCES)].copy()
            if not prior.empty
            else prior
        )
        user_batch = batch[batch[USER_COLUMN].astype(str) == user_id]
        boost_items = (
            user_batch.sort_values("occurred_at", ascending=False)[ITEM_COLUMN]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        boost = pd.DataFrame(
            [
                {
                    ITEM_COLUMN: item_id,
                    SCORE_COLUMN: float(len(boost_items) - index),
                    SOURCE_COLUMN: INCREMENTAL_SOURCE,
                }
                for index, item_id in enumerate(boost_items[: self._top_k])
            ]
        )

        parts = [frame for frame in (preserved, boost, popular, latest) if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        combined[USER_COLUMN] = user_id
        combined[ITEM_COLUMN] = combined[ITEM_COLUMN].astype(str)
        # Earlier frames win on (user, item); fill top-K.
        combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[[USER_COLUMN, ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN]].reset_index(
            drop=True
        )

    def _cold_start_rows(self, popular: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
        parts = [frame for frame in (popular, latest) if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[USER_COLUMN] = COLD_START_USER_ID
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[[USER_COLUMN, ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN]].reset_index(
            drop=True
        )

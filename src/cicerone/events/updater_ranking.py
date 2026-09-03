"""Popular and latest ranking from an incremental event batch."""

from __future__ import annotations

import pandas as pd

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE
from cicerone.feature_config import FeatureConfig
from cicerone.io.recommendation_reader import ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN
from cicerone.io.recommendation_schema import REASONS_COLUMN
from cicerone.reasons import dump_source_reasons
from cicerone.weighting import event_row_weights


class UpdaterRanking:
    _feature_config: FeatureConfig | None
    _top_k: int
    _explain_enabled: bool

    def _row_signal_weights(self, batch: pd.DataFrame) -> pd.Series:
        if self._feature_config is None:
            return pd.to_numeric(batch["quantity"], errors="coerce").fillna(0.0)
        return event_row_weights(
            batch["event_type"].astype(str),
            batch["quantity"],
            event_weights=self._feature_config.event_weights,
            quantity_scaled_events=self._feature_config.quantity_scaled_events,
        ).fillna(0.0)

    def _aligned_weights(self, batch: pd.DataFrame, weights: pd.Series | None) -> pd.Series:
        if weights is None:
            return self._row_signal_weights(batch)
        return weights.reindex(batch.index)

    def _signal_rows(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        if batch.empty:
            return batch
        aligned = self._aligned_weights(batch, weights)
        keep = aligned > 0
        scored = batch.loc[keep]
        if scored.empty:
            return scored
        return scored.assign(_signal_weight=aligned.loc[keep])

    def _popular_ranking(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        empty = pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        scored = self._signal_rows(batch, weights)
        if scored.empty:
            return empty
        summed = scored.groupby(scored["item_id"].astype(str), sort=False)["_signal_weight"].sum()
        summed = summed.loc[summed > 0]
        if summed.empty:
            return empty
        ranked = summed.reset_index()
        ranked.columns = [ITEM_COLUMN, SCORE_COLUMN]
        ranked = ranked.sort_values(
            [SCORE_COLUMN, ITEM_COLUMN], ascending=[False, True], kind="mergesort"
        ).head(self._top_k)
        ranked[SOURCE_COLUMN] = POPULAR_SOURCE
        if self._explain_enabled:
            ranked[REASONS_COLUMN] = [
                dump_source_reasons(POPULAR_SOURCE, rank=rank) for rank in range(1, len(ranked) + 1)
            ]
        return ranked.reset_index(drop=True)

    def _latest_ranking(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        if batch.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = self._signal_rows(batch, weights)
        if frame.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = frame.copy()
        frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
        latest = (
            frame.sort_values("occurred_at", ascending=False, kind="mergesort")
            .drop_duplicates(subset=[ITEM_COLUMN], keep="first")
            .head(self._top_k)
        )
        rows = []
        for rank, row in enumerate(latest.itertuples(index=False), start=1):
            item = {
                ITEM_COLUMN: str(row.item_id),
                SCORE_COLUMN: float(self._top_k - rank + 1),
                SOURCE_COLUMN: LATEST_SOURCE,
            }
            if self._explain_enabled:
                item[REASONS_COLUMN] = dump_source_reasons(LATEST_SOURCE, rank=rank)
            rows.append(item)
        return (
            pd.DataFrame(rows) if rows else pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        )

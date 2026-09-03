"""Merge incremental ranking with preserved personalized and experiment rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.events.store import empty_recommendations_frame
from cicerone.io.recommendation_reader import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
)
from cicerone.io.recommendation_schema import (
    FALLBACK_VARIANT,
    REASONS_COLUMN,
    VARIANT_COLUMN,
    collapse_mixed_variants,
    filter_variant_rows,
    pick_fallback_variant,
    recommendation_output_columns,
)
from cicerone.reasons import dump_source_reasons

INCREMENTAL_SOURCE = "incremental"
_PRESERVE_LABELS = frozenset(
    {
        "personalized",
        "item_based",
        "sequential",
        "content_fallback",
        "blended",
    }
)
# Reserve slots so recent interactions can enter top-K even when preserved rows fill it.
_BOOST_SLOT_FRACTION = 0.3


def _source_parts(source: str) -> set[str]:
    return {part for part in source.split("+") if part}


def _is_preserved_source(source: str) -> bool:
    return bool(_source_parts(source) & _PRESERVE_LABELS)


def _overlaps_source_parts(source: str, parts: set[str]) -> bool:
    return bool(_source_parts(source) & parts)


class UpdaterMerge:
    _variant_names: tuple[str, ...]
    _assign_variant: Callable[[str], str | None] | None
    _top_k: int
    _explain_enabled: bool

    if TYPE_CHECKING:

        def _signal_rows(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame: ...

    def _merge_user_rows(
        self,
        user_id: str,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        batch: pd.DataFrame,
        weights: pd.Series | None = None,
        online_rows: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        variants = self._variants_for()
        if not variants:
            prior = collapse_mixed_variants(prior)
            merged = self._merge_one_list(
                user_id, prior, popular, latest, batch, weights, online_rows=online_rows
            )
            return self._stamp_collapsed_variant(merged, prior)
        has_variant = VARIANT_COLUMN in prior.columns and not prior.empty
        primary = FALLBACK_VARIANT if FALLBACK_VARIANT in variants else variants[0]
        assigned = self._assigned_variant(user_id) or primary
        empty_batch = batch.iloc[0:0]
        parts = []
        for variant in variants:
            prior_slice = (
                filter_variant_rows(prior, variant)
                if has_variant
                else (prior if variant == primary else empty_recommendations_frame())
            )
            inject = variant == assigned
            merged = self._merge_one_list(
                user_id,
                prior_slice,
                popular if inject else popular.iloc[0:0],
                latest if inject else latest.iloc[0:0],
                batch if inject else empty_batch,
                weights if inject else None,
                online_rows=online_rows if inject else None,
            )
            if merged.empty:
                continue
            merged = merged.copy()
            merged[VARIANT_COLUMN] = variant
            parts.append(merged)
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _variants_for(self) -> tuple[str, ...]:
        return self._variant_names

    def _assigned_variant(self, user_id: str) -> str | None:
        if self._assign_variant is not None:
            name = self._assign_variant(user_id)
            return str(name) if name else None
        variants = self._variant_names
        if not variants:
            return None
        return FALLBACK_VARIANT if FALLBACK_VARIANT in variants else variants[0]

    def _stamp_collapsed_variant(self, merged: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
        if merged.empty or VARIANT_COLUMN not in prior.columns or prior.empty:
            return merged
        pick = pick_fallback_variant(prior[VARIANT_COLUMN].tolist())
        if pick is None:
            return merged
        stamped = merged.copy()
        stamped[VARIANT_COLUMN] = pick
        return stamped

    def _merge_one_list(
        self,
        user_id: str,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        batch: pd.DataFrame,
        weights: pd.Series | None = None,
        online_rows: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not prior.empty and SOURCE_COLUMN in prior.columns:
            mask = prior[SOURCE_COLUMN].astype(str).map(_is_preserved_source)
            preserved = prior.loc[mask].copy()
        else:
            preserved = prior.iloc[0:0] if not prior.empty else prior
        online_part, kept = self._split_online_preserved(preserved, online_rows)

        user_batch = self._signal_rows(batch[batch[USER_COLUMN].astype(str) == user_id], weights)
        has_signal = not user_batch.empty
        preserved_ids: set[str] = set()
        if not kept.empty:
            preserved_ids.update(kept[ITEM_COLUMN].astype(str))
        if not online_part.empty:
            preserved_ids.update(online_part[ITEM_COLUMN].astype(str))
        boost_items = (
            user_batch.sort_values("occurred_at", ascending=False, kind="mergesort")[ITEM_COLUMN]
            .astype(str)
            .drop_duplicates()
            .tolist()
            if has_signal
            else []
        )
        boost_items = [item_id for item_id in boost_items if item_id not in preserved_ids]
        boost_slots = max(1, int(self._top_k * _BOOST_SLOT_FRACTION)) if boost_items else 0
        boost = pd.DataFrame(
            [
                {
                    ITEM_COLUMN: item_id,
                    SCORE_COLUMN: float(len(boost_items) - index),
                    SOURCE_COLUMN: INCREMENTAL_SOURCE,
                    **(
                        {REASONS_COLUMN: dump_source_reasons(INCREMENTAL_SOURCE, rank=index + 1)}
                        if self._explain_enabled
                        else {}
                    ),
                }
                for index, item_id in enumerate(boost_items[:boost_slots])
            ]
        )
        preserve_cap = max(0, self._top_k - boost_slots)
        if not kept.empty:
            if RANK_COLUMN in kept.columns:
                kept = kept.sort_values(RANK_COLUMN, kind="mergesort")
            kept = kept.head(preserve_cap)
        preserved_parts = [frame for frame in (online_part, kept) if not frame.empty]
        preserved = (
            pd.concat(preserved_parts, ignore_index=True)
            if preserved_parts
            else empty_recommendations_frame()
        )

        # Batch-global popular/latest only for users with signal or preserved rows
        # (unknown/zero-weight events must not rewrite popular-only users).
        use_global = has_signal or not preserved.empty
        parts = [boost, preserved]
        if use_global:
            parts.extend((popular, latest))
        parts = [frame for frame in parts if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        combined[USER_COLUMN] = user_id
        combined[ITEM_COLUMN] = combined[ITEM_COLUMN].astype(str)
        combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[recommendation_output_columns(combined)].reset_index(drop=True)

    def _split_online_preserved(
        self, preserved: pd.DataFrame, online_rows: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        empty = preserved.iloc[0:0] if not preserved.empty else preserved
        if online_rows is None or online_rows.empty:
            return empty, preserved
        online = online_rows.copy()
        if SOURCE_COLUMN in online.columns:
            online = online.loc[online[SOURCE_COLUMN].astype(str).map(_is_preserved_source)]
        if online.empty:
            return empty, preserved
        online_parts: set[str] = set()
        if SOURCE_COLUMN in online.columns:
            for label in online[SOURCE_COLUMN].astype(str):
                online_parts.update(_source_parts(label))
        if preserved.empty:
            kept = preserved
        elif SOURCE_COLUMN in preserved.columns:
            drop = (
                preserved[SOURCE_COLUMN]
                .astype(str)
                .map(lambda source: _overlaps_source_parts(source, online_parts))
            )
            kept = preserved.loc[~drop]
        else:
            kept = empty
        return online, kept

    def _cold_start_rows(
        self,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
    ) -> pd.DataFrame:
        variants = self._variants_for()
        if not variants:
            prior = collapse_mixed_variants(prior)
            return self._stamp_collapsed_variant(self._cold_start_one_list(prior, popular, latest), prior)
        has_variant = VARIANT_COLUMN in prior.columns and not prior.empty
        primary = FALLBACK_VARIANT if FALLBACK_VARIANT in variants else variants[0]
        parts = []
        for variant in variants:
            prior_slice = (
                prior[prior[VARIANT_COLUMN].astype(str) == variant]
                if has_variant
                else (prior if variant == primary else empty_recommendations_frame())
            )
            merged = self._cold_start_one_list(prior_slice, popular, latest)
            if merged.empty:
                continue
            merged = merged.copy()
            merged[VARIANT_COLUMN] = variant
            parts.append(merged)
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _cold_start_one_list(
        self,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
    ) -> pd.DataFrame:
        # Prefers batch popular/latest, then keeps prior cold-start fill.
        parts = [frame for frame in (popular, latest, prior) if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        if ITEM_COLUMN in combined.columns:
            combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[USER_COLUMN] = COLD_START_USER_ID
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[recommendation_output_columns(combined)].reset_index(drop=True)

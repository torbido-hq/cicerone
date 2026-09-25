"""Production replay of served recommendation lists."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools.metrics import (
    MAP,
    MRR,
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    HitRate,
    IntraListDiversity,
    MeanInvUserFreq,
    Precision,
    Recall,
    Serendipity,
    calc_metrics,
)
from rectools.metrics.distances import PairwiseHammingDistanceCalculator

from cicerone.blending import COLD_START_USER_ID
from cicerone.evaluation.metrics import OCCURRED_AT, _frame, _ratio
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServedEvalReport:
    n_users: int
    n_users_with_events: int
    metrics: dict[str, float]
    by_source: dict[str, dict[str, float]]
    generated_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_users_with_events": self.n_users_with_events,
            "metrics": dict(self.metrics),
            "by_source": {key: dict(value) for key, value in sorted(self.by_source.items())},
            "generated_at": self.generated_at,
        }


def replay_ks(configured: Sequence[int], *, top_k: int) -> tuple[int, ...]:
    if configured:
        return tuple(sorted({int(value) for value in configured if 1 <= int(value) <= top_k})) or (top_k,)
    values = {top_k, *[k for k in (5, 10) if k <= top_k]}
    return tuple(sorted(values))


def filter_recs_to_assigned(
    recommendations: pd.DataFrame,
    assigned: Mapping[str, str] | None,
) -> pd.DataFrame:
    if recommendations.empty or not assigned or VARIANT_COLUMN not in recommendations.columns:
        return recommendations
    expected = recommendations[USER_COLUMN].astype(str).map(assigned)
    return recommendations.loc[expected.isna() | (recommendations[VARIANT_COLUMN].astype(str) == expected)]


def filter_events_to_recommended(
    events: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    assigned: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if events.empty or recommendations.empty:
        return events.iloc[0:0].copy() if not events.empty else events
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    recs = filter_recs_to_assigned(recs, assigned)
    keys = recs.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    frame = events.copy()
    frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    return frame.merge(keys, on=[USER_COLUMN, ITEM_COLUMN], how="inner")


def _hit_rate(reco: pd.DataFrame, relevant: pd.DataFrame, *, k: int) -> float:
    if reco.empty or relevant.empty:
        return 0.0
    truth = relevant.groupby(USER_COLUMN)[ITEM_COLUMN].agg(set)
    hits = 0
    scored = 0
    top = reco[reco[RANK_COLUMN] <= k] if RANK_COLUMN in reco.columns else reco
    for user_id, group in top.groupby(USER_COLUMN):
        items = truth.get(str(user_id))
        if items is None:
            continue
        scored += 1
        if set(group[ITEM_COLUMN].astype(str)) & set(items):
            hits += 1
    return _ratio(hits, scored)


def _ranking_metric_defs(k: int) -> dict[str, object]:
    return {
        f"HitRate@{k}": HitRate(k=k),
        f"MAP@{k}": MAP(k=k),
        f"NDCG@{k}": NDCG(k=k),
        f"Recall@{k}": Recall(k=k),
        f"MRR@{k}": MRR(k=k),
        f"Precision@{k}": Precision(k=k),
    }


def _catalog_metric_defs(k: int, *, with_prev: bool) -> dict[str, object]:
    metrics: dict[str, object] = {f"CatalogCoverage@{k}": CatalogCoverage(k=k, normalize=True)}
    if with_prev:
        metrics[f"MeanInvUserFreq@{k}"] = MeanInvUserFreq(k=k)
        metrics[f"AvgRecPopularity@{k}"] = AvgRecPopularity(k=k)
    return metrics


def _discrete_feature(value: object) -> str | None:
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    return str(value)


def _list_tokens(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        tokens = [_discrete_feature(item) for item in value]
        return [token for token in tokens if token]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _dummies_for_column(series: pd.Series, *, column: str, kind: str) -> pd.DataFrame:
    if kind == "list":
        exploded = series.map(_list_tokens).explode()
        exploded = exploded[exploded.notna() & exploded.astype(str).ne("")]
        if exploded.empty:
            return pd.DataFrame(index=series.index)
        encoded = pd.get_dummies(exploded.astype(str), prefix=column)
        return encoded.groupby(level=0).max().reindex(series.index, fill_value=0)
    mapped = series.map(_discrete_feature)
    if mapped.notna().sum() == 0:
        return pd.DataFrame(index=series.index)
    encoded = pd.get_dummies(mapped.astype("string"), prefix=column, dummy_na=False)
    return encoded.reindex(series.index, fill_value=0)


def _item_features(
    catalog: pd.DataFrame | Sequence[object] | None,
    item_features: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame | None:
    if not isinstance(catalog, pd.DataFrame) or catalog.empty or ITEM_COLUMN not in catalog.columns:
        return None
    specs = [(name, kind) for name, kind in (item_features or ()) if name in catalog.columns]
    if not specs:
        return None
    columns = [name for name, _kind in specs]
    frame = catalog.loc[:, [ITEM_COLUMN, *columns]].copy()
    frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    frame = frame.drop_duplicates(subset=[ITEM_COLUMN], keep="last")
    indexed = frame.set_index(ITEM_COLUMN)
    parts = [_dummies_for_column(indexed[name], column=name, kind=kind) for name, kind in specs]
    encoded = pd.concat(parts, axis=1)
    if encoded.empty or encoded.shape[1] == 0:
        return None
    return encoded.astype(float)


def _reco_with_known_features(reco: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    known = set(features.index.astype(str))
    frame = reco.copy()
    frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    missing = ~frame[ITEM_COLUMN].isin(known)
    bad_users = set(frame.loc[missing, USER_COLUMN].astype(str))
    if not bad_users:
        return frame
    return frame.loc[~frame[USER_COLUMN].astype(str).isin(bad_users)]


def _ild_metrics(
    recs: pd.DataFrame,
    ks: Sequence[int],
    *,
    catalog: pd.DataFrame | Sequence[object] | None,
    item_features: Sequence[tuple[str, str]] | None,
) -> dict[str, float]:
    if recs.empty or RANK_COLUMN not in recs.columns:
        return {}
    features = _item_features(catalog, item_features)
    if features is None:
        return {}
    try:
        calculator = PairwiseHammingDistanceCalculator(features)
    except (ValueError, TypeError, KeyError):
        logger.exception("RecTools PairwiseHammingDistanceCalculator failed")
        return {}
    metrics: dict[str, float] = {}
    for k in ks:
        reco_k = recs[recs[RANK_COLUMN] <= k]
        reco_k = _reco_with_known_features(reco_k, features)
        if reco_k.empty:
            continue
        try:
            computed = calc_metrics(
                {f"IntraListDiversity@{k}": IntraListDiversity(k=k, distance_calculator=calculator)},
                reco=reco_k,
            )
            metrics.update({key: float(value) for key, value in computed.items()})
        except (ValueError, TypeError, KeyError):
            logger.exception("RecTools IntraListDiversity failed for k=%s", k)
    return metrics


def _prev_interactions(events: pd.DataFrame, generated_at: str | None) -> pd.DataFrame:
    if events.empty or generated_at is None or OCCURRED_AT not in events.columns:
        return events.iloc[0:0]
    start = pd.to_datetime(generated_at, utc=True, errors="coerce")
    if pd.isna(start):
        return events.iloc[0:0]
    frame = events.loc[events[OCCURRED_AT] <= start, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["weight"] = 1.0
    return frame


def _served_catalog(
    catalog: pd.DataFrame | Sequence[object] | None,
    recs: pd.DataFrame,
    events: pd.DataFrame,
) -> list[str]:
    if isinstance(catalog, pd.DataFrame) and not catalog.empty and ITEM_COLUMN in catalog.columns:
        return list(dict.fromkeys(catalog[ITEM_COLUMN].astype(str)))
    if catalog is not None and not isinstance(catalog, pd.DataFrame):
        return list(dict.fromkeys(str(item) for item in catalog))
    seen: list[str] = []
    for frame in (events, recs):
        if frame is not None and not frame.empty and ITEM_COLUMN in frame.columns:
            seen.extend(frame[ITEM_COLUMN].astype(str))
    return list(dict.fromkeys(seen))


def evaluate_served(
    recommendations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    generated_at: str | None,
    ks: Sequence[int],
    event_types: Sequence[str],
    history: pd.DataFrame | None = None,
    catalog: pd.DataFrame | Sequence[object] | None = None,
    assigned: Mapping[str, str] | None = None,
    item_features: Sequence[tuple[str, str]] | None = None,
) -> ServedEvalReport | None:
    if recommendations is None or recommendations.empty:
        return None
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    recs = filter_recs_to_assigned(recs, assigned)
    if recs.empty:
        return None
    all_events = _frame(events)
    if all_events.empty:
        return ServedEvalReport(
            n_users=int(recs[USER_COLUMN].nunique()),
            n_users_with_events=0,
            metrics=_ild_metrics(recs, ks, catalog=catalog, item_features=item_features),
            by_source={},
            generated_at=generated_at,
        )
    window_events = all_events
    if generated_at and OCCURRED_AT in window_events.columns:
        start = pd.to_datetime(generated_at, utc=True, errors="coerce")
        if pd.notna(start):
            window_events = window_events[window_events[OCCURRED_AT] > start]
    if event_types and "event_type" in window_events.columns:
        window_events = window_events[window_events["event_type"].astype(str).isin(set(event_types))]
    if history is not None and not history.empty and OCCURRED_AT in window_events.columns:
        live = recs.copy()
        if generated_at:
            live["generated_at"] = generated_at
            combined = pd.concat([history, live], ignore_index=True)
        else:
            combined = history
        hist_recs = _recs_from_history(combined, window_events)
        if not hist_recs.empty:
            hist_recs = hist_recs[hist_recs[USER_COLUMN] != COLD_START_USER_ID]
            hist_recs = filter_recs_to_assigned(hist_recs, assigned)
        if not hist_recs.empty:
            recs = hist_recs
    relevant = window_events.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    n_users = int(recs[USER_COLUMN].nunique())
    n_with_events = int(relevant[USER_COLUMN].nunique()) if not relevant.empty else 0
    prev = _prev_interactions(all_events, generated_at)
    catalog_ids = _served_catalog(catalog, recs, all_events)
    metrics: dict[str, float] = {}
    for k in ks:
        reco_k = recs[recs[RANK_COLUMN] <= k] if RANK_COLUMN in recs.columns else recs
        if not relevant.empty and not recs.empty:
            interactions = relevant.copy()
            interactions["weight"] = 1.0
            extra: dict[str, object] = {}
            extra.update(_catalog_metric_defs(k, with_prev=not prev.empty))
            try:
                computed = calc_metrics(
                    {**_ranking_metric_defs(k), **extra},
                    reco=reco_k,
                    interactions=interactions,
                    catalog=catalog_ids,
                    prev_interactions=prev if not prev.empty else None,
                )
                metrics.update({key: float(value) for key, value in computed.items()})
            except Exception:
                logger.exception("RecTools calc_metrics failed for k=%s", k)
                metrics[f"HitRate@{k}"] = _hit_rate(recs, relevant, k=k)
            if not prev.empty and catalog_ids:
                try:
                    surprise = calc_metrics(
                        {f"Serendipity@{k}": Serendipity(k=k)},
                        reco=reco_k,
                        interactions=interactions,
                        catalog=catalog_ids,
                        prev_interactions=prev,
                    )
                    metrics.update({key: float(value) for key, value in surprise.items()})
                except (ValueError, TypeError, KeyError):
                    logger.exception("RecTools Serendipity failed for k=%s", k)
        else:
            metrics[f"HitRate@{k}"] = _hit_rate(recs, relevant, k=k)
    metrics.update(_ild_metrics(recs, ks, catalog=catalog, item_features=item_features))
    by_source: dict[str, dict[str, float]] = {}
    if SOURCE_COLUMN in recs.columns:
        for source, group in recs.groupby(SOURCE_COLUMN, dropna=True):
            source_metrics: dict[str, float] = {}
            for k in ks:
                if not relevant.empty and not group.empty:
                    reco_k = group[group[RANK_COLUMN] <= k] if RANK_COLUMN in group.columns else group
                    interactions = relevant.copy()
                    interactions["weight"] = 1.0
                    try:
                        computed = calc_metrics(
                            {f"HitRate@{k}": HitRate(k=k)},
                            reco=reco_k,
                            interactions=interactions,
                        )
                        source_metrics.update({key: float(value) for key, value in computed.items()})
                    except Exception:
                        logger.exception("RecTools HitRate failed for source=%s k=%s", source, k)
                        source_metrics[f"HitRate@{k}"] = _hit_rate(group, relevant, k=k)
                else:
                    source_metrics[f"HitRate@{k}"] = _hit_rate(group, relevant, k=k)
            by_source[str(source)] = source_metrics
    return ServedEvalReport(
        n_users=n_users,
        n_users_with_events=n_with_events,
        metrics=metrics,
        by_source=by_source,
        generated_at=generated_at,
    )


def _recs_from_history(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use each user's newest snapshot with generated_at <= that user's first event."""
    hist = history.copy()
    hist[USER_COLUMN] = hist[USER_COLUMN].astype(str)
    hist[ITEM_COLUMN] = hist[ITEM_COLUMN].astype(str)
    hist["generated_at"] = pd.to_datetime(hist["generated_at"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["generated_at"])
    hist = hist.drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN, "generated_at"], keep="last")
    if hist.empty or events.empty or OCCURRED_AT not in events.columns:
        return hist.iloc[0:0]
    first = events.groupby(USER_COLUMN, sort=False)[OCCURRED_AT].min().reset_index()
    first[USER_COLUMN] = first[USER_COLUMN].astype(str)
    snaps = (
        hist.loc[:, [USER_COLUMN, "generated_at"]]
        .drop_duplicates()
        .sort_values("generated_at")
        .rename(columns={"generated_at": "snap_at"})
    )
    first = first.sort_values(OCCURRED_AT).rename(columns={OCCURRED_AT: "event_at"})
    chosen = pd.merge_asof(
        first,
        snaps,
        by=USER_COLUMN,
        left_on="event_at",
        right_on="snap_at",
        direction="backward",
    )
    chosen = chosen.dropna(subset=["snap_at"])
    if chosen.empty:
        return hist.iloc[0:0]
    picked = chosen.loc[:, [USER_COLUMN, "snap_at"]].rename(columns={"snap_at": "generated_at"})
    return hist.merge(picked, on=[USER_COLUMN, "generated_at"], how="inner")

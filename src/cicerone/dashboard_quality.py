"""Dashboard Quality page: CTR/CVR and optional production replay."""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import pandas as pd

from cicerone.config import Settings
from cicerone.evaluation import (
    conversion_event_types,
    conversion_events_for_settings,
    evaluate_tracking,
    generated_ats_from_track,
    load_metric_events,
)
from cicerone.evaluation.context import prefer_history
from cicerone.io.base import ManifestReader
from cicerone.track.store import TrackStore
from cicerone.track.store_common import DASHBOARD_TRACK_FLOOR_HOURS, lookback_since

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_CATALOG_PREFIXES = ("CatalogCoverage", "MeanInvUserFreq", "AvgRecPopularity")
_DIVERSITY_PREFIXES = ("IntraListDiversity", "Serendipity")


def quality_context(settings: Settings, reader: ManifestReader | None = None) -> dict[str, Any]:
    store = TrackStore(settings.output)
    report: dict[str, Any] | None = None
    error: str | None = None
    try:
        report = store.read_eval()
    except Exception:
        logger.exception("Failed to read track eval report")
        error = "Could not load quality metrics."
    track_eval = None
    served_eval = None
    if isinstance(report, dict):
        raw_track = report.get("track_eval")
        raw_served = report.get("served_eval")
        track_eval = raw_track if isinstance(raw_track, dict) else None
        served_eval = raw_served if isinstance(raw_served, dict) else None
    used_live_track = False
    if track_eval is None and settings.track.enabled:
        track_eval = _live_track_eval(settings, store)
        used_live_track = track_eval is not None
        if used_live_track:
            error = None
    empty_track = (not settings.track.enabled) or _no_impressions(track_eval)
    track_as_of = None
    track_live = used_live_track
    if not used_live_track:
        if isinstance(report, dict):
            raw_as_of = report.get("generated_at")
            if isinstance(raw_as_of, str) and raw_as_of:
                track_as_of = raw_as_of
        if track_as_of is None and isinstance(served_eval, dict):
            raw_served_at = served_eval.get("generated_at")
            if isinstance(raw_served_at, str) and raw_served_at:
                track_as_of = raw_served_at
    ranking_metrics, catalog_metrics, diversity_metrics = _split_replay_metrics(served_eval)
    recent_runs: list[dict[str, Any]] = []
    history_single = False
    if reader is not None:
        try:
            history = reader.read_recent(settings.dashboard.history_limit)
        except Exception:
            logger.exception("Failed to read manifests for Quality history")
            history = []
        recent_runs = _quality_history(history)
        history_single = len(recent_runs) == 1 and settings.output.kind == "dataset"
    current_stamp = _eval_stamp(report.get("generated_at") if isinstance(report, dict) else None)
    if current_stamp is None:
        current_stamp = _eval_stamp(served_eval.get("generated_at") if served_eval else None)
    return {
        "track_enabled": settings.track.enabled,
        "eval_enabled": settings.eval.enabled,
        "log_impressions": settings.serve.log_impressions,
        "track_eval": track_eval,
        "served_eval": served_eval,
        "ranking_metrics": ranking_metrics,
        "catalog_metrics": catalog_metrics,
        "diversity_metrics": diversity_metrics,
        "track_as_of": track_as_of,
        "track_live": track_live,
        "replay_metric_names": _source_metric_names(served_eval),
        "quality_deltas": _quality_deltas(
            track_eval,
            served_eval,
            recent_runs,
            skip_first_track=not used_live_track,
            skip_first_replay=True,
            current_stamp=current_stamp,
        ),
        "quality_history": recent_runs,
        "quality_history_single": history_single,
        "rank_curve_inverted": _rank_curve_inverted(track_eval, settings.track.min_impressions),
        "error": error,
        "empty_track": empty_track,
    }


def _source_metric_names(served_eval: dict[str, Any] | None) -> list[str]:
    if not served_eval:
        return []
    names: list[str] = []
    by_source = served_eval.get("by_source")
    if not isinstance(by_source, dict):
        return names
    for raw in by_source.values():
        if not isinstance(raw, dict):
            continue
        for name in raw:
            key = str(name)
            if key not in names:
                names.append(key)
    return names


def _replay_metric_names(served_eval: dict[str, Any] | None) -> list[str]:
    return _source_metric_names(served_eval)


def _is_catalog_metric(name: str) -> bool:
    return name.startswith(_CATALOG_PREFIXES)


def _is_diversity_metric(name: str) -> bool:
    return name.startswith(_DIVERSITY_PREFIXES)


def _split_replay_metrics(
    served_eval: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    ranking: dict[str, float] = {}
    catalog: dict[str, float] = {}
    diversity: dict[str, float] = {}
    if not served_eval:
        return ranking, catalog, diversity
    metrics = served_eval.get("metrics")
    if not isinstance(metrics, dict):
        return ranking, catalog, diversity
    for name, raw in metrics.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        key = str(name)
        if _is_catalog_metric(key):
            catalog[key] = value
        elif _is_diversity_metric(key):
            diversity[key] = value
        else:
            ranking[key] = value
    return ranking, catalog, diversity


def _parse_eval_blob(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _overall_float(track_eval: dict[str, Any] | None, key: str) -> float | None:
    if not track_eval:
        return None
    overall = track_eval.get("overall")
    if not isinstance(overall, dict):
        return None
    try:
        return float(overall[key])
    except (KeyError, TypeError, ValueError):
        return None


def _metric_at_largest_k(served_eval: dict[str, Any] | None, prefix: str) -> tuple[str | None, float | None]:
    if not served_eval:
        return None, None
    metrics = served_eval.get("metrics")
    if not isinstance(metrics, dict):
        return None, None
    best_name: str | None = None
    best_k = -1
    best_value: float | None = None
    marker = f"{prefix}@"
    for name, raw in metrics.items():
        key = str(name)
        if not key.startswith(marker):
            continue
        try:
            k = int(key.rsplit("@", 1)[1])
            value = float(raw)
        except (IndexError, TypeError, ValueError):
            continue
        if k > best_k:
            best_k = k
            best_name = key
            best_value = value
    return best_name, best_value


def _quality_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in history:
        if run.get("triggered_by") == "incremental":
            continue
        status = run.get("status")
        if status not in (None, "success"):
            continue
        track_eval = _parse_eval_blob(run.get("track_eval"))
        served_eval = _parse_eval_blob(run.get("served_eval"))
        if track_eval is None and served_eval is None:
            continue
        ndcg_name, ndcg = _metric_at_largest_k(served_eval, "NDCG")
        coverage_name, coverage = _metric_at_largest_k(served_eval, "CatalogCoverage")
        rows.append(
            {
                "generated_at": run.get("generated_at"),
                "track_eval": track_eval,
                "served_eval": served_eval,
                "ctr": _overall_float(track_eval, "ctr"),
                "cvr_click": _overall_float(track_eval, "cvr_click"),
                "ndcg": ndcg,
                "ndcg_name": ndcg_name,
                "coverage": coverage,
                "coverage_name": coverage_name,
            }
        )
    return rows


def _format_delta_rate(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta * 100:.2f} pp"


def _format_delta_metric(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def _subtract(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _eval_stamp(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _same_served_eval(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_stamp = _eval_stamp(left.get("generated_at"))
    right_stamp = _eval_stamp(right.get("generated_at"))
    return bool(left_stamp and right_stamp and left_stamp == right_stamp)


def _previous_quality_row(
    history: list[dict[str, Any]],
    *,
    skip_current: bool,
    current_stamp: str | None = None,
    current_served: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not skip_current:
        return history[0] if history else None
    served_stamp = _eval_stamp(current_served.get("generated_at") if current_served else None)
    if current_stamp is None and served_stamp is None:
        return history[1] if len(history) > 1 else None
    for row in history:
        row_stamp = _eval_stamp(row.get("generated_at"))
        if current_stamp and row_stamp and row_stamp > current_stamp:
            continue
        if _same_served_eval(row.get("served_eval"), current_served):
            continue
        return row
    return None


def _empty_quality_deltas() -> dict[str, str | None]:
    return {
        "ctr": "—",
        "cvr_click": "—",
        "ndcg": "—",
        "ndcg_name": None,
        "coverage": "—",
        "coverage_name": None,
        "miuf": "—",
        "miuf_name": None,
        "popularity": "—",
        "popularity_name": None,
        "ild": "—",
        "ild_name": None,
        "serendipity": "—",
        "serendipity_name": None,
    }


def _named_cutoff_delta(
    current_name: str | None,
    current_value: float | None,
    previous_name: str | None,
    previous_value: float | None,
) -> str:
    if current_name and previous_name and current_name != previous_name:
        previous_value = None
    return _format_delta_metric(_subtract(current_value, previous_value))


def _quality_deltas(
    track_eval: dict[str, Any] | None,
    served_eval: dict[str, Any] | None,
    history: list[dict[str, Any]],
    *,
    skip_first_track: bool,
    skip_first_replay: bool,
    current_stamp: str | None = None,
) -> dict[str, str | None]:
    out = _empty_quality_deltas()
    ndcg_name, ndcg = _metric_at_largest_k(served_eval, "NDCG")
    coverage_name, coverage = _metric_at_largest_k(served_eval, "CatalogCoverage")
    miuf_name, miuf = _metric_at_largest_k(served_eval, "MeanInvUserFreq")
    popularity_name, popularity = _metric_at_largest_k(served_eval, "AvgRecPopularity")
    ild_name, ild = _metric_at_largest_k(served_eval, "IntraListDiversity")
    serendipity_name, serendipity = _metric_at_largest_k(served_eval, "Serendipity")
    out["ndcg_name"] = ndcg_name
    out["coverage_name"] = coverage_name
    out["miuf_name"] = miuf_name
    out["popularity_name"] = popularity_name
    out["ild_name"] = ild_name
    out["serendipity_name"] = serendipity_name
    track_previous = _previous_quality_row(
        history,
        skip_current=skip_first_track,
        current_stamp=current_stamp,
        current_served=served_eval,
    )
    if track_previous is not None:
        out["ctr"] = _format_delta_rate(
            _subtract(_overall_float(track_eval, "ctr"), track_previous.get("ctr"))
        )
        out["cvr_click"] = _format_delta_rate(
            _subtract(_overall_float(track_eval, "cvr_click"), track_previous.get("cvr_click"))
        )
    replay_previous = _previous_quality_row(
        history,
        skip_current=skip_first_replay,
        current_stamp=current_stamp,
        current_served=served_eval,
    )
    if replay_previous is None:
        return out
    prev_served = replay_previous.get("served_eval")
    prev_ndcg_name, prev_ndcg = _metric_at_largest_k(prev_served, "NDCG")
    prev_coverage_name, prev_coverage = _metric_at_largest_k(prev_served, "CatalogCoverage")
    prev_miuf_name, prev_miuf = _metric_at_largest_k(prev_served, "MeanInvUserFreq")
    prev_popularity_name, prev_popularity = _metric_at_largest_k(prev_served, "AvgRecPopularity")
    prev_ild_name, prev_ild = _metric_at_largest_k(prev_served, "IntraListDiversity")
    prev_serendipity_name, prev_serendipity = _metric_at_largest_k(prev_served, "Serendipity")
    out["ndcg"] = _named_cutoff_delta(ndcg_name, ndcg, prev_ndcg_name, prev_ndcg)
    out["coverage"] = _named_cutoff_delta(coverage_name, coverage, prev_coverage_name, prev_coverage)
    out["miuf"] = _named_cutoff_delta(miuf_name, miuf, prev_miuf_name, prev_miuf)
    out["popularity"] = _named_cutoff_delta(
        popularity_name, popularity, prev_popularity_name, prev_popularity
    )
    out["ild"] = _named_cutoff_delta(ild_name, ild, prev_ild_name, prev_ild)
    out["serendipity"] = _named_cutoff_delta(
        serendipity_name, serendipity, prev_serendipity_name, prev_serendipity
    )
    return out


def _rank_curve_inverted(track_eval: dict[str, Any] | None, min_impressions: int) -> bool:
    if not track_eval:
        return False
    by_rank = track_eval.get("by_rank")
    if not isinstance(by_rank, dict):
        return False
    floor = min_impressions if min_impressions > 0 else 100
    points: list[tuple[int, float]] = []
    for rank, raw in by_rank.items():
        if not isinstance(raw, dict):
            continue
        try:
            impressions = int(raw.get("n_impressions") or 0)
            if impressions < floor:
                continue
            points.append((int(rank), float(raw["ctr"])))
        except (TypeError, ValueError, KeyError):
            continue
    points.sort()
    if len(points) < 2:
        return False
    return any(left[1] < right[1] for left, right in zip(points, points[1:], strict=False))


def _future_or(future: Future[_T], label: str, default: _T) -> _T:
    try:
        return future.result()
    except Exception:
        logger.exception("Failed to %s", label)
        return default


def _no_impressions(track_eval: dict[str, Any] | None) -> bool:
    if not track_eval:
        return True
    overall = track_eval.get("overall")
    if not isinstance(overall, dict):
        return True
    return int(overall.get("n_impressions") or 0) <= 0


def _live_track_eval(settings: Settings, store: TrackStore) -> dict[str, Any] | None:
    try:
        since = lookback_since(
            window_hours=settings.track.attribution_window_hours,
            floor_hours=DASHBOARD_TRACK_FLOOR_HOURS,
        )
        rows = store.read_rows(since=since)
    except Exception:
        logger.exception("Failed to read track rows for Quality")
        return None
    if not rows:
        return None
    conversions = pd.DataFrame()
    recs = None
    try:
        from cicerone.events.store import load_recommendations_frame

        wanted = generated_ats_from_track(rows)

        def _load_conversions() -> pd.DataFrame:
            types = conversion_event_types(
                settings.track.conversion_event_types,
                primary_metric=settings.experiment.primary_metric,
            )
            return conversion_events_for_settings(
                load_metric_events(settings, event_types=types, since=since),
                settings,
            )

        def _load_recs() -> pd.DataFrame | None:
            frame = load_recommendations_frame(settings.output)
            if frame is not None and frame.empty:
                return None
            return frame

        def _load_history() -> pd.DataFrame | None:
            if not wanted:
                return None
            return store.read_history(generated_ats=wanted)

        with ThreadPoolExecutor(max_workers=3) as pool:
            conv_f = pool.submit(_load_conversions)
            recs_f = pool.submit(_load_recs)
            hist_f = pool.submit(_load_history)
            conversions = _future_or(conv_f, "load conversions for live Quality metrics", pd.DataFrame())
            current = _future_or(recs_f, "load recommendations for live Quality metrics", None)
            history = _future_or(hist_f, "read recommendation history for Quality", None)
            recs = prefer_history(history, current)
    except Exception:
        logger.exception("Failed to load conversions for live Quality metrics")
    try:
        return evaluate_tracking(
            track_rows=rows,
            conversions=conversions,
            recommendations=recs,
            window_hours=settings.track.attribution_window_hours,
        ).as_dict()
    except Exception:
        logger.exception("Failed to compute live track eval")
        return None

"""Dashboard Quality page: CTR/CVR and optional production replay."""

from __future__ import annotations

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
from cicerone.track.store import TrackStore

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def quality_context(settings: Settings) -> dict[str, Any]:
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
    return {
        "track_enabled": settings.track.enabled,
        "eval_enabled": settings.eval.enabled,
        "log_impressions": settings.serve.log_impressions,
        "track_eval": track_eval,
        "served_eval": served_eval,
        "track_as_of": track_as_of,
        "track_live": track_live,
        "replay_metric_names": _replay_metric_names(served_eval),
        "error": error,
        "empty_track": empty_track,
    }


def _replay_metric_names(served_eval: dict[str, Any] | None) -> list[str]:
    if not served_eval:
        return []
    names: list[str] = []
    metrics = served_eval.get("metrics")
    if isinstance(metrics, dict):
        names.extend(str(name) for name in metrics)
    by_source = served_eval.get("by_source")
    if isinstance(by_source, dict):
        for raw in by_source.values():
            if not isinstance(raw, dict):
                continue
            for name in raw:
                key = str(name)
                if key not in names:
                    names.append(key)
    return names


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
        rows = store.read_rows()
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
                load_metric_events(settings, event_types=types),
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

"""Previous-run scoring, input load, and track persistence for a job run."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings, Settings
from cicerone.config.constants import TRACK_KIND_IMPRESSION
from cicerone.evaluation import (
    conversion_event_types,
    conversion_events_for_settings,
    evaluate_served,
    evaluate_tracking,
    generated_ats_from_track,
    replay_ks,
)
from cicerone.evaluation.context import concat_history, stamp_recommendations
from cicerone.events.store import load_recommendations_frame
from cicerone.experiment.assignment import resolve_assignment
from cicerone.experiment.store import ExperimentStore
from cicerone.io.base import InputSource
from cicerone.io.factory import build_manifest_reader
from cicerone.io.recommendation_schema import USER_COLUMN, VARIANT_COLUMN, pick_fallback_variant
from cicerone.locks import LockLostError, WriterLockBusyError, held_writer_lock
from cicerone.track.store import TrackStore
from cicerone.track.store_common import _utc_stamp

logger = logging.getLogger(__name__)


def try_load(label: str, fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        logger.exception("Failed to %s", label)
        return default


def try_load_pair(
    left_label: str,
    left: Callable[[], Any],
    left_default: Any,
    right_label: str,
    right: Callable[[], Any],
    right_default: Any,
    *,
    parallel: bool,
) -> tuple[Any, Any]:
    if not parallel:
        return try_load(left_label, left, left_default), try_load(right_label, right, right_default)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_f = pool.submit(try_load, left_label, left, left_default)
        right_f = pool.submit(try_load, right_label, right, right_default)
        return left_f.result(), right_f.result()


def replay_assignments(
    settings: Settings,
    recs: pd.DataFrame,
    track_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str] | None:
    if recs.empty or VARIANT_COLUMN not in recs.columns:
        return None
    names = {str(value) for value in recs[VARIANT_COLUMN].dropna().astype(str) if str(value)}
    if len(names) <= 1:
        return None
    assigned: dict[str, str] = {}
    timed: list[tuple[tuple[int, str], str, str]] = []
    for row in track_rows:
        if str(row.get("kind") or "") != TRACK_KIND_IMPRESSION:
            continue
        user_id = str(row.get("user_id") or "")
        variant = str(row.get("variant") or "")
        if not user_id or variant not in names:
            continue
        stamp = _utc_stamp(row.get("occurred_at"))
        if stamp is None:
            continue
        timed.append(((int(stamp.value), str(row.get("event_id") or "")), user_id, variant))
    for _key, user_id, variant in sorted(timed, key=lambda item: item[0]):
        assigned.setdefault(user_id, variant)
    if settings.experiment.enabled:
        promoted, pair = ExperimentStore(settings.output).assignment_overlay(settings.experiment.id)
        for raw_user in recs[USER_COLUMN].astype(str).unique():
            user_id = str(raw_user)
            if user_id in assigned or user_id == COLD_START_USER_ID:
                continue
            _experiment_id, assigned_variant = resolve_assignment(
                settings, user_id, promoted_variant=promoted, active_pair=pair
            )
            if assigned_variant:
                assigned[user_id] = assigned_variant
    else:
        pick = pick_fallback_variant(list(names))
        if pick:
            for user_id in recs[USER_COLUMN].astype(str).unique():
                assigned.setdefault(str(user_id), pick)
    return assigned or None


def persist_track_outputs(
    store: TrackStore,
    *,
    kind: str,
    eval_report: Mapping[str, Any],
    recommendations: pd.DataFrame | None,
    generated_at: str,
    fence_check: Callable[[], bool] | None = None,
) -> None:
    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("write track eval", lambda: store.write_eval(eval_report)),
    ]
    if recommendations is not None:
        tasks.append(
            (
                "append recommendation history",
                lambda: store.append_history(recommendations, generated_at=generated_at),
            )
        )
    lock = getattr(store, "_writer_lock", None)

    def _run_serial() -> None:
        for label, fn in tasks:
            try_load(label, fn, None)

    if lock is not None:
        try:
            with held_writer_lock(
                lock,
                fence_check=fence_check,
                fence_lost="retrain lock lost before write",
                fence_kind="retrain",
            ):
                _run_serial()
        except (WriterLockBusyError, LockLostError):
            logger.exception("Failed to persist track outputs")
        return
    if kind == "db":
        _run_serial()
        return
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for label, fn in tasks:
            pool.submit(try_load, label, fn, None)


def score_previous_run(
    settings: Settings,
    events: pd.DataFrame,
    last_manifest: dict[str, Any] | None,
    items: pd.DataFrame | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not settings.track.enabled and not settings.eval.enabled:
        return None, None
    previous_generated_at = None
    if last_manifest:
        previous_generated_at = last_manifest.get("generated_at")
        previous_generated_at = str(previous_generated_at) if previous_generated_at else None
    store = TrackStore(settings.output)

    def _load_recs() -> pd.DataFrame | None:
        recs = load_recommendations_frame(settings.output)
        if recs is not None and recs.empty:
            return None
        return recs

    def _load_track() -> list[dict[str, Any]]:
        if not settings.track.enabled:
            return []
        return store.read_rows()

    previous_recs, track_rows = try_load_pair(
        "load previous recommendations for eval",
        _load_recs,
        None,
        "read track rows",
        _load_track,
        [],
        parallel=settings.output.kind != "db",
    )
    wanted = generated_ats_from_track(track_rows, previous_generated_at)
    history = None
    if wanted:
        try:
            history = store.read_history(generated_ats=wanted)
            if history is not None and history.empty:
                history = None
        except Exception:
            logger.exception("Failed to read recommendation history")
            history = None
    recs_for_track = concat_history(history, stamp_recommendations(previous_recs, previous_generated_at))
    assigned: dict[str, str] | None = None
    replay_failed = False
    if settings.eval.enabled and previous_recs is not None and previous_generated_at:
        try:
            assigned = replay_assignments(settings, previous_recs, track_rows)
        except Exception:
            logger.exception("Failed to compute served eval")
            replay_failed = True

    def _compute_track() -> dict[str, Any] | None:
        if not settings.track.enabled:
            return None
        try:
            conversions = conversion_events_for_settings(events, settings)
            return evaluate_tracking(
                track_rows=track_rows,
                conversions=conversions,
                recommendations=recs_for_track,
                window_hours=settings.track.attribution_window_hours,
            ).as_dict()
        except Exception:
            logger.exception("Failed to compute track eval")
            return None

    def _compute_served() -> dict[str, Any] | None:
        if replay_failed or not settings.eval.enabled or previous_recs is None or not previous_generated_at:
            return None
        try:
            types = settings.eval.event_types or conversion_event_types(
                settings.track.conversion_event_types,
                primary_metric=settings.experiment.primary_metric,
            )
            report = evaluate_served(
                previous_recs,
                events,
                generated_at=previous_generated_at,
                ks=replay_ks(settings.eval.ks, top_k=settings.top_k),
                event_types=types,
                history=history,
                catalog=items,
                assigned=assigned,
            )
            return report.as_dict() if report is not None else None
        except Exception:
            logger.exception("Failed to compute served eval")
            return None

    run_both = bool(
        settings.track.enabled
        and settings.eval.enabled
        and previous_recs is not None
        and previous_generated_at
    )
    if run_both:
        with ThreadPoolExecutor(max_workers=2) as pool:
            track_f = pool.submit(_compute_track)
            served_f = pool.submit(_compute_served)
            return track_f.result(), served_f.result()
    return _compute_track(), _compute_served()


def read_input(
    source: InputSource, output: IOSettings
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        events_future = executor.submit(source.read_events)
        users_future = executor.submit(source.read_users)
        items_future = executor.submit(source.read_items)
        manifest_future = executor.submit(
            try_load,
            "read last manifest",
            lambda: build_manifest_reader(output).read_latest(),
            None,
        )
        return (
            events_future.result(),
            users_future.result(),
            items_future.result(),
            manifest_future.result(),
        )

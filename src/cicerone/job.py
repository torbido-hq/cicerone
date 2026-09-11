"""Single recommendation job run: input → dataset → (AutoML) → train → write."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, NamedTuple

import pandas as pd

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, build_artifact, dumps_artifact
from cicerone.automl import evaluate_candidates, select_best_candidate
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings, Settings, load_settings
from cicerone.config.constants import (
    ALLOCATION_THOMPSON,
    DEFAULT_LOG_FORMAT,
    TRACK_KIND_IMPRESSION,
)
from cicerone.dataset import build_dataset
from cicerone.evaluation import (
    conversion_event_types,
    conversion_events_for_settings,
    evaluate_served,
    evaluate_tracking,
    generated_ats_from_track,
    replay_ks,
)
from cicerone.evaluation.context import concat_history, stamp_recommendations
from cicerone.events.store import load_items_catalog_size, load_recommendations_frame
from cicerone.experiment import (
    ResolvedRecipe,
    apply_recipe,
    recipes_manifest_json,
    resolve_recipes,
    union_models,
)
from cicerone.experiment.assignment import resolve_assignment
from cicerone.experiment.guardrails import evaluate_guardrails
from cicerone.experiment.store import ExperimentStore, merge_experiment_state
from cicerone.experiment.thompson import (
    allocate_thompson,
    select_active_recipes,
    track_rows_since,
    window_trials_from_slices,
)
from cicerone.feature_config import load_feature_config
from cicerone.io.base import InputSource
from cicerone.io.factory import build_input_source, build_manifest_reader, build_output_sink
from cicerone.io.recommendation_schema import (
    USER_COLUMN,
    VARIANT_COLUMN,
    filter_variant_rows,
    pick_fallback_variant,
)
from cicerone.locks import LockLostError
from cicerone.model import (
    DEFAULT_MODELS,
    RRF_K,
    RecommenderModel,
    fit_strategies,
    plan_model_run,
    recommend_with_models,
    train_and_recommend,
)
from cicerone.model.recommend import RecommendCache
from cicerone.publish import build_publisher
from cicerone.track.store import TrackStore
from cicerone.track.store_common import _utc_stamp

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 500


class ThompsonSelection(NamedTuple):
    recipes: tuple[ResolvedRecipe, ...]
    state: dict[str, Any] | None = None


def _replay_assignments(
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


def _target_user_ids(events: pd.DataFrame, users: pd.DataFrame | None) -> list[str]:
    columns = [events[USER_COLUMN]]
    if users is not None:
        columns.append(users[USER_COLUMN])
    ids: set[str] = set()
    for column in columns:
        for user_id in column.dropna():
            ids.add(str(user_id))
    return sorted(ids)


_MANIFEST_DEFAULTS: dict[str, Any] = {
    "triggered_by": None,
    "lock_backend": None,
    "status": "failed",
    "error": None,
    "n_events": None,
    "n_target_users": None,
    "n_users_with_recommendations": None,
    "n_items": None,
    "top_k": None,
    "models": "",
    "model_weights": "",
    "rrf_k": None,
    "artifact_written": False,
    "artifact_schema_version": None,
    "partial_outputs": False,
    "automl_enabled": False,
    "automl_metrics": "",
    "experiment_id": "",
    "experiment_variants": "",
    "track_eval": "",
    "served_eval": "",
}


def _try_load(label: str, fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        logger.exception("Failed to %s", label)
        return default


def _persist_track_outputs(
    store: TrackStore,
    *,
    kind: str,
    eval_report: Mapping[str, Any],
    recommendations: pd.DataFrame | None,
    generated_at: str,
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
    if kind == "db":
        for label, fn in tasks:
            _try_load(label, fn, None)
        return
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for label, fn in tasks:
            pool.submit(_try_load, label, fn, None)


def _score_previous_run(
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

    with ThreadPoolExecutor(max_workers=2) as pool:
        recs_f = pool.submit(_try_load, "load previous recommendations for eval", _load_recs, None)
        track_f = pool.submit(_try_load, "read track rows", _load_track, [])
        previous_recs = recs_f.result()
        track_rows = track_f.result()
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
        if not settings.eval.enabled or previous_recs is None or not previous_generated_at:
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
                assigned=_replay_assignments(settings, previous_recs, track_rows),
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


def _read_input(
    source: InputSource, output: IOSettings
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        events_future = executor.submit(source.read_events)
        users_future = executor.submit(source.read_users)
        items_future = executor.submit(source.read_items)
        manifest_future = executor.submit(
            _try_load,
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


def _ensure_fence(fence_check: Callable[[], bool] | None) -> None:
    if fence_check is not None and not fence_check():
        raise LockLostError("retrain lock lost before write")


def _select_thompson_recipes(
    settings: Settings,
    recipes: tuple[ResolvedRecipe, ...],
    events: pd.DataFrame,
) -> ThompsonSelection:
    if len(recipes) < 2:
        return ThompsonSelection(recipes)
    experiment = settings.experiment
    store = ExperimentStore(settings.output)
    failed = object()
    with ThreadPoolExecutor(max_workers=2) as pool:
        state_f = pool.submit(_try_load, "read experiment state", store.read_state, failed)
        track_f = pool.submit(
            _try_load,
            "read track rows",
            lambda: TrackStore(settings.output).read_rows(experiment_id=experiment.id),
            failed,
        )
        raw_state = state_f.result()
        raw_track = track_f.result()
    if raw_state is failed or not (raw_state is None or isinstance(raw_state, dict)):
        return ThompsonSelection(recipes)
    previous: dict[str, Any] | None = raw_state
    if raw_track is failed or not isinstance(raw_track, list):
        return ThompsonSelection(recipes)
    track_rows: list[dict[str, Any]] = raw_track
    if previous and str(previous.get("experiment_id") or "") != experiment.id:
        previous = None
    promoted = str(previous["promoted_variant"]) if previous and previous.get("promoted_variant") else None
    has_pair = bool(previous and previous.get("champion") and previous.get("challenger"))
    if not track_rows and not has_pair:
        logger.warning("Thompson allocation fail closed: empty track")
        return ThompsonSelection(recipes)
    window_started = ""
    if previous is not None and has_pair:
        window_started = str(previous.get("window_started_at") or "")
    window_rows = track_rows_since(track_rows, window_started or None)
    names = [recipe.name for recipe in recipes]
    try:
        conversions = conversion_events_for_settings(events, settings)
        with ThreadPoolExecutor(max_workers=2) as pool:
            recs_f = pool.submit(
                _try_load,
                "load recommendations for Thompson guardrails",
                lambda: load_recommendations_frame(settings.output),
                None,
            )
            catalog_f = pool.submit(
                _try_load,
                "load catalog size for Thompson guardrails",
                lambda: load_items_catalog_size(settings.output),
                None,
            )
            recs = recs_f.result()
            catalog_size = catalog_f.result()
        report = evaluate_tracking(
            track_rows=window_rows,
            conversions=conversions,
            recommendations=recs,
            window_hours=settings.track.attribution_window_hours,
        )
        window_trials = window_trials_from_slices(
            report.by_variant, attribution=experiment.attribution, names=names
        )
        if not has_pair and not any(item.impressions for item in window_trials.values()):
            logger.warning("Thompson allocation fail closed: no variant signal")
            return ThompsonSelection(recipes)
        guardrails_ok = False
        if recs is not None and not recs.empty and VARIANT_COLUMN in recs.columns:
            pair_names: tuple[str, ...] = tuple(names)
            if previous is not None and has_pair:
                pair_names = (str(previous.get("champion")), str(previous.get("challenger")))
            guardrails_ok = all(
                evaluate_guardrails(
                    filter_variant_rows(recs, name),
                    variant=name,
                    catalog_size=catalog_size,
                ).ok
                for name in pair_names
                if name
            )
        allocation = allocate_thompson(
            names=names,
            previous=previous,
            window_trials=window_trials,
            min_impressions=settings.track.min_impressions,
            rotate_min_prob=experiment.rotate_min_prob,
            promoted_variant=promoted,
            guardrails_ok=guardrails_ok,
        )
        selected = select_active_recipes(
            recipes,
            champion=allocation.champion,
            challenger=allocation.challenger,
            explore_traffic=experiment.explore_traffic,
        )
        pending = merge_experiment_state(
            previous,
            experiment_id=experiment.id,
            promoted_variant=promoted,
            promoted_at=(str(previous["promoted_at"]) if previous and previous.get("promoted_at") else None),
            **allocation.as_state(),
        )
        logger.info(
            "Thompson allocation: champion=%s challenger=%s rotated=%s",
            allocation.champion,
            allocation.challenger,
            allocation.rotated,
        )
        return ThompsonSelection(selected, pending)
    except Exception:
        logger.exception("Thompson allocation fail closed")
        return ThompsonSelection(recipes)


def _recommendation_user_count(recommendations: pd.DataFrame) -> int:
    if recommendations.empty or USER_COLUMN not in recommendations.columns:
        return 0
    user_ids = recommendations[USER_COLUMN].astype(str)
    return int(user_ids[user_ids != COLD_START_USER_ID].nunique())


def run(triggered_by: str = "manual", *, fence_check: Callable[[], bool] | None = None) -> None:
    settings = load_settings()
    feature_config = load_feature_config(settings.feature_config_path)
    sink = build_output_sink(settings.output)
    publisher = None

    manifest = dict(_MANIFEST_DEFAULTS)
    manifest["triggered_by"] = triggered_by
    manifest["lock_backend"] = settings.trigger.lock_backend
    manifest["top_k"] = settings.top_k
    manifest["automl_enabled"] = settings.automl.enabled
    track_eval_payload: dict[str, Any] | None = None
    served_eval_payload: dict[str, Any] | None = None
    recommendations: pd.DataFrame | None = None
    eval_generated_at: str | None = None
    pending_thompson: dict[str, Any] | None = None

    try:
        publisher = build_publisher(settings)
        source = build_input_source(settings.input)
        events, users, items, last_manifest = _read_input(source, settings.output)

        logger.info(
            "Loaded %d events, %s users, %s items",
            len(events),
            len(users) if users is not None else "n/a",
            len(items) if items is not None else "n/a",
        )

        if last_manifest and last_manifest.get("generated_at"):
            eval_generated_at = str(last_manifest["generated_at"])
        track_eval_payload, served_eval_payload = _score_previous_run(settings, events, last_manifest, items)

        built = build_dataset(events, users, items, feature_config, half_life_days=settings.half_life_days)

        target_users = _target_user_ids(events, users)

        automl_result = None
        enabled_models, weights, rrf_k = settings.models, settings.model_weights, settings.rrf_k
        if settings.automl.enabled:
            candidate_results = evaluate_candidates(
                events,
                users,
                items,
                feature_config,
                top_k=settings.top_k,
                half_life_days=settings.half_life_days,
                candidates=settings.automl.candidates,
                n_splits=settings.automl.n_splits,
                test_days=settings.automl.test_days,
                max_workers=settings.max_workers,
                model_configs=settings.model_configs,
                sequential_min_median_interactions=settings.sequential_min_median_interactions,
                debias=settings.automl.debias,
                content_fallback_enabled=settings.content_fallback_enabled,
            )
            automl_result = select_best_candidate(
                candidate_results, primary_metric=settings.automl.primary_metric
            )
            enabled_models = automl_result.candidate.models
            weights = automl_result.candidate.weights
            rrf_k = automl_result.candidate.rrf_k
            logger.info(
                "AutoML selected '%s' (metrics=%s, over %d fold(s))",
                automl_result.candidate.label,
                automl_result.metrics,
                automl_result.n_folds,
            )

        fitted: dict[str, RecommenderModel] = {}
        if settings.experiment.enabled and last_manifest is None:
            try:
                last_manifest = build_manifest_reader(settings.output).read_latest()
            except Exception:
                logger.exception("Failed to read last manifest for experiment recipes")

        recipes: tuple[ResolvedRecipe, ...] = ()
        if settings.experiment.enabled:
            recipes = resolve_recipes(
                settings,
                feature_config,
                automl_models=(
                    list(enabled_models) if automl_result is not None and enabled_models is not None else None
                ),
                automl_weights=weights if automl_result is not None else None,
                automl_rrf_k=rrf_k if automl_result is not None else None,
                last_manifest=last_manifest,
            )
            logger.info(
                "Experiment %s: %d variant(s) %s",
                settings.experiment.id,
                len(recipes),
                ",".join(recipe.name for recipe in recipes),
            )
            if settings.experiment.allocation == ALLOCATION_THOMPSON:
                selected = _select_thompson_recipes(settings, recipes, events)
                recipes = selected.recipes
                pending_thompson = selected.state
                logger.info(
                    "Experiment %s after allocation: %d variant(s) %s",
                    settings.experiment.id,
                    len(recipes),
                    ",".join(recipe.name for recipe in recipes),
                )

        recommend_cache: RecommendCache = {}
        if recipes:
            union = union_models(recipes)
            recommend_names: list[str] = []
            for recipe in recipes:
                recipe_plan = plan_model_run(
                    list(recipe.models),
                    blending_enabled=recipe.blending.enabled,
                    content_fallback_enabled=settings.content_fallback_enabled,
                )
                for name in recipe_plan.recommend_models:
                    if name not in recommend_names:
                        recommend_names.append(name)
            fit_plan = plan_model_run(
                recommend_names,
                blending_enabled=False,
                content_fallback_enabled=settings.content_fallback_enabled,
            )
            _, fitted = fit_strategies(
                built,
                target_users,
                enabled_models=list(fit_plan.recommend_models),
                strategy_cache=fitted if settings.save_model_artifact else None,
                max_workers=settings.max_workers,
                epoch_metrics=settings.epoch_metrics,
                epoch_metrics_top_k=settings.top_k,
                item_based_k_neighbors=settings.item_based_k_neighbors,
                model_configs=settings.model_configs,
                content_fallback_max_neighbors=settings.content_fallback_max_neighbors,
                content_feature_columns=feature_config.item_features,
            )
            frames: list[pd.DataFrame] = []
            for recipe in recipes:
                recipe_config = apply_recipe(feature_config, recipe)
                recipe_plan = plan_model_run(
                    list(recipe.models),
                    blending_enabled=recipe.blending.enabled,
                    content_fallback_enabled=settings.content_fallback_enabled,
                )
                variant_recs = recommend_with_models(
                    fitted,
                    built,
                    target_users,
                    recipe_config,
                    top_k=settings.top_k,
                    enabled_models=list(recipe_plan.enabled_models),
                    weights=recipe.weights,
                    rrf_k=recipe.rrf_k,
                    run_plan=recipe_plan,
                    recommend_cache=recommend_cache,
                    max_workers=settings.max_workers,
                    explain=settings.explain,
                )
                variant_recs = variant_recs.copy()
                variant_recs[VARIANT_COLUMN] = recipe.name
                frames.append(variant_recs)
            recommendations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            run_plan = fit_plan
            enabled_models = union
            weights = recipes[0].weights
            rrf_k = recipes[0].rrf_k
            manifest["experiment_id"] = settings.experiment.id
            manifest["experiment_variants"] = recipes_manifest_json(recipes)
        else:
            run_plan = plan_model_run(
                enabled_models or DEFAULT_MODELS,
                blending_enabled=feature_config.blending.enabled,
                content_fallback_enabled=settings.content_fallback_enabled,
            )
            recommendations = train_and_recommend(
                built,
                target_users,
                feature_config,
                top_k=settings.top_k,
                enabled_models=list(run_plan.enabled_models),
                weights=weights,
                rrf_k=rrf_k,
                strategy_cache=fitted if settings.save_model_artifact else None,
                max_workers=settings.max_workers,
                epoch_metrics=settings.epoch_metrics,
                item_based_k_neighbors=settings.item_based_k_neighbors,
                model_configs=settings.model_configs,
                content_fallback_max_neighbors=settings.content_fallback_max_neighbors,
                run_plan=run_plan,
                explain=settings.explain,
            )

        run_models = list(run_plan.recommend_models)
        model_weights_str = (
            ",".join(f"{name}={weights.get(name, 1.0)}" for name in run_models) if weights is not None else ""
        )

        artifact_bytes: bytes | None = None
        if settings.save_model_artifact:
            artifact_models = [name for name in run_models if name in fitted]
            artifact_weights = (
                {name: weights.get(name, 1.0) for name in artifact_models} if weights is not None else None
            )
            artifact_bytes = dumps_artifact(
                build_artifact(
                    fitted=fitted,
                    built=built,
                    feature_config=feature_config,
                    models=artifact_models,
                    model_weights=artifact_weights,
                    rrf_k=rrf_k if rrf_k is not None else RRF_K,
                )
            )

        # Artifact → snapshot → recommendations; success only after all writes.
        outputs_written = False
        _ensure_fence(fence_check)
        try:
            if artifact_bytes is not None:
                _ensure_fence(fence_check)
                sink.write_model_artifact(artifact_bytes)
                manifest["artifact_written"] = True
                manifest["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

            if items is not None and not items.empty:
                _ensure_fence(fence_check)
                sink.write_items_snapshot(items)

            _ensure_fence(fence_check)
            sink.write_recommendations(recommendations)
            outputs_written = True
            if pending_thompson is not None:
                ExperimentStore(settings.output).write_state(pending_thompson)
            if publisher is not None:
                publisher.publish(recommendations)
        except Exception:
            if outputs_written or manifest.get("artifact_written"):
                manifest["partial_outputs"] = True
            raise

        try:
            _ensure_fence(fence_check)
        except LockLostError:
            if outputs_written or manifest.get("artifact_written"):
                manifest["partial_outputs"] = True
            raise

        manifest.update(
            {
                "status": "success",
                "n_events": int(len(events)),
                "n_target_users": len(target_users),
                "n_users_with_recommendations": _recommendation_user_count(recommendations),
                "n_items": int(built.dataset.item_id_map.external_ids.shape[0]),
                "models": ",".join(run_models),
                "model_weights": model_weights_str,
                "rrf_k": rrf_k if rrf_k is not None else RRF_K,
                "automl_metrics": (
                    ",".join(
                        f"{name}={automl_result.metrics[name]:.4f}" for name in sorted(automl_result.metrics)
                    )
                    if automl_result is not None
                    else ""
                ),
                "track_eval": json.dumps(track_eval_payload) if track_eval_payload else "",
                "served_eval": json.dumps(served_eval_payload) if served_eval_payload else "",
            }
        )
    except Exception as exc:
        error_message = str(exc)
        if len(error_message) > _MAX_ERROR_LENGTH:
            error_message = error_message[:_MAX_ERROR_LENGTH] + "... (truncated)"
        manifest["error"] = error_message
        raise
    finally:
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                logger.exception("Failed to close recommendation publisher")
        manifest["generated_at"] = datetime.now(UTC).isoformat()
        try:
            sink.write_manifest(manifest)
        except Exception:
            logger.exception("Failed to write manifest; original job error (if any) is preserved")
            if manifest.get("status") == "success":
                raise
        logger.info("Job finished: %s", json.dumps(manifest))
        if manifest.get("status") == "success" and (settings.track.enabled or settings.eval.enabled):
            _persist_track_outputs(
                TrackStore(settings.output),
                kind=settings.output.kind,
                eval_report={
                    "generated_at": eval_generated_at,
                    "track_eval": track_eval_payload,
                    "served_eval": served_eval_payload,
                },
                recommendations=recommendations,
                generated_at=str(manifest["generated_at"]),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

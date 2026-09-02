"""Single recommendation job run: input → dataset → (AutoML) → train → write."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, build_artifact, dumps_artifact
from cicerone.automl import evaluate_candidates, select_best_candidate
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import Settings, load_settings
from cicerone.config.constants import DEFAULT_LOG_FORMAT
from cicerone.dataset import build_dataset
from cicerone.evaluation import (
    conversion_event_types,
    evaluate_served,
    evaluate_tracking,
    replay_ks,
)
from cicerone.events.store import load_recommendations_frame
from cicerone.experiment import (
    ResolvedRecipe,
    apply_recipe,
    recipes_manifest_json,
    resolve_recipes,
    union_models,
)
from cicerone.feature_config import load_feature_config
from cicerone.io.base import InputSource
from cicerone.io.factory import build_input_source, build_manifest_reader, build_output_sink
from cicerone.io.recommendation_schema import USER_COLUMN, VARIANT_COLUMN
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

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 500


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
    previous_recs = None
    try:
        previous_recs = load_recommendations_frame(settings.output)
        if previous_recs is not None and previous_recs.empty:
            previous_recs = None
    except Exception:
        logger.exception("Failed to load previous recommendations for eval")
        previous_recs = None
    store = TrackStore(settings.output)
    track_rows: list[dict[str, Any]] = []
    if settings.track.enabled:
        try:
            track_rows = store.read_rows()
        except Exception:
            logger.exception("Failed to read track rows")
            track_rows = []
    wanted = {str(row.get("generated_at") or "") for row in track_rows}
    wanted.discard("")
    if previous_generated_at:
        wanted.add(previous_generated_at)
    history = None
    if settings.track.enabled and wanted:
        try:
            history = store.read_history(generated_ats=wanted)
            if history is not None and history.empty:
                history = None
        except Exception:
            logger.exception("Failed to read recommendation history")
            history = None
    recs_for_track = previous_recs
    if recs_for_track is not None and previous_generated_at:
        recs_for_track = recs_for_track.copy()
        recs_for_track["generated_at"] = previous_generated_at
    if history is not None:
        recs_for_track = (
            pd.concat([history, recs_for_track], ignore_index=True) if recs_for_track is not None else history
        )
    track_payload = None
    served_payload = None
    if settings.track.enabled:
        try:
            types = conversion_event_types(
                settings.track.conversion_event_types,
                primary_metric=settings.experiment.primary_metric,
            )
            conversions = events
            if not conversions.empty and "event_type" in conversions.columns:
                conversions = conversions[conversions["event_type"].astype(str).isin(set(types))]
            track_payload = evaluate_tracking(
                track_rows=track_rows,
                conversions=conversions,
                recommendations=recs_for_track,
                window_hours=settings.track.attribution_window_hours,
            ).as_dict()
        except Exception:
            logger.exception("Failed to compute track eval")
    if settings.eval.enabled and previous_recs is not None and previous_generated_at:
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
            )
            served_payload = report.as_dict() if report is not None else None
        except Exception:
            logger.exception("Failed to compute served eval")
    return track_payload, served_payload


def _read_input(source: InputSource) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        events_future = executor.submit(source.read_events)
        users_future = executor.submit(source.read_users)
        items_future = executor.submit(source.read_items)
        return events_future.result(), users_future.result(), items_future.result()


def _ensure_fence(fence_check: Callable[[], bool] | None) -> None:
    if fence_check is not None and not fence_check():
        raise LockLostError("retrain lock lost before write")


def _recommendation_user_count(recommendations: pd.DataFrame) -> int:
    if recommendations.empty or USER_COLUMN not in recommendations.columns:
        return 0
    user_ids = recommendations[USER_COLUMN].astype(str)
    return int(user_ids[user_ids != COLD_START_USER_ID].nunique())


def run(triggered_by: str = "manual", *, fence_check: Callable[[], bool] | None = None) -> None:
    settings = load_settings()
    feature_config = load_feature_config(settings.feature_config_path)
    sink = build_output_sink(settings.output)
    publisher = build_publisher(settings)

    manifest = dict(_MANIFEST_DEFAULTS)
    manifest["triggered_by"] = triggered_by
    manifest["lock_backend"] = settings.trigger.lock_backend
    manifest["top_k"] = settings.top_k
    manifest["automl_enabled"] = settings.automl.enabled
    track_eval_payload: dict[str, Any] | None = None
    served_eval_payload: dict[str, Any] | None = None
    recommendations: pd.DataFrame | None = None

    try:
        source = build_input_source(settings.input)
        events, users, items = _read_input(source)

        logger.info(
            "Loaded %d events, %s users, %s items",
            len(events),
            len(users) if users is not None else "n/a",
            len(items) if items is not None else "n/a",
        )

        last_manifest = None
        try:
            last_manifest = build_manifest_reader(settings.output).read_latest()
        except Exception:
            logger.exception("Failed to read last manifest")
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
            if manifest.get("status") != "success":
                pass  # keep original job failure
            else:
                raise
        logger.info("Job finished: %s", json.dumps(manifest))
        if manifest.get("status") == "success" and (settings.track.enabled or settings.eval.enabled):
            store = TrackStore(settings.output)
            try:
                store.write_eval({"track_eval": track_eval_payload, "served_eval": served_eval_payload})
            except Exception:
                logger.exception("Failed to write track eval")
            if recommendations is not None:
                try:
                    store.append_history(recommendations, generated_at=str(manifest["generated_at"]))
                except Exception:
                    logger.exception("Failed to append recommendation history")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

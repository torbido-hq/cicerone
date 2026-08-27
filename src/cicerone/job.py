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
from cicerone.config import load_settings
from cicerone.dataset import build_dataset
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
from cicerone.io.recommendation_schema import VARIANT_COLUMN
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 500

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
}


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
    if recommendations.empty or "user_id" not in recommendations.columns:
        return 0
    user_ids = recommendations["user_id"].astype(str)
    return int(user_ids[user_ids != COLD_START_USER_ID].nunique())


def run(triggered_by: str = "manual", *, fence_check: Callable[[], bool] | None = None) -> None:
    settings = load_settings()
    feature_config = load_feature_config(settings.feature_config_path)
    sink = build_output_sink(settings.output)

    manifest = dict(_MANIFEST_DEFAULTS)
    manifest["triggered_by"] = triggered_by
    manifest["lock_backend"] = settings.trigger.lock_backend
    manifest["top_k"] = settings.top_k
    manifest["automl_enabled"] = settings.automl.enabled

    try:
        source = build_input_source(settings.input)
        events, users, items = _read_input(source)

        logger.info(
            "Loaded %d events, %s users, %s items",
            len(events),
            len(users) if users is not None else "n/a",
            len(items) if items is not None else "n/a",
        )

        built = build_dataset(events, users, items, feature_config, half_life_days=settings.half_life_days)

        known_users = set(users["user_id"]) if users is not None else set()
        target_users = sorted(set(events["user_id"]) | known_users)

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
        last_manifest = None
        if settings.experiment.enabled:
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
            }
        )
    except Exception as exc:
        error_message = str(exc)
        if len(error_message) > _MAX_ERROR_LENGTH:
            error_message = error_message[:_MAX_ERROR_LENGTH] + "... (truncated)"
        manifest["error"] = error_message
        raise
    finally:
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


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

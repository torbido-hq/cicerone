"""Single recommendation job run: input → dataset → (AutoML) → train → write."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any, NamedTuple

import pandas as pd

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, build_artifact, dumps_artifact
from cicerone.automl import evaluate_candidates, select_best_candidate
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import Settings, load_settings
from cicerone.config.constants import ALLOCATION_THOMPSON, DEFAULT_LOG_FORMAT
from cicerone.dataset import build_dataset
from cicerone.evaluation import conversion_events_for_settings, evaluate_tracking
from cicerone.events.store import load_items_catalog_size, load_recommendations_frame
from cicerone.experiment import (
    ResolvedRecipe,
    apply_recipe,
    recipes_manifest_json,
    resolve_recipes,
    union_models,
)
from cicerone.experiment.guardrails import evaluate_guardrails
from cicerone.experiment.store import ExperimentStore, merge_experiment_state
from cicerone.experiment.thompson import (
    allocate_thompson,
    select_active_recipes,
    track_rows_since,
    window_trials_from_slices,
)
from cicerone.feature_config import load_feature_config
from cicerone.io.factory import build_input_source, build_manifest_reader, build_output_sink
from cicerone.io.recommendation_schema import USER_COLUMN, VARIANT_COLUMN, filter_variant_rows
from cicerone.job_eval import OPTIONAL_EVAL_ERRORS as _OPTIONAL_EVAL_ERRORS
from cicerone.job_eval import PUBLISH_ERRORS as _PUBLISH_ERRORS
from cicerone.job_eval import SINK_WRITE_ERRORS as _SINK_WRITE_ERRORS
from cicerone.job_eval import log_caught as _log_caught
from cicerone.job_eval import persist_track_outputs as _persist_track_outputs
from cicerone.job_eval import read_input as _read_input
from cicerone.job_eval import replay_assignments as _replay_assignments  # noqa: F401
from cicerone.job_eval import score_previous_run as _score_previous_run
from cicerone.job_eval import try_load as _try_load
from cicerone.job_eval import try_load_pair as _try_load_pair
from cicerone.job_output import (
    ensure_fence,
    ensure_publication_fence,
    skip_stale_job_manifest,
    truncate_job_error,
    write_job_manifest,
)
from cicerone.locks import (
    LockBackend,
    LockLostError,
    WriterLockBusyError,
    acquire_blocking,
    build_dataset_writer_lock,
    build_lock_backend,
    build_output_writer_lock,
    has_distributed_lock,
)
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
from cicerone.publish.sidecar import sidecar_generation_current
from cicerone.track.store import TrackStore

logger = logging.getLogger(__name__)


class ThompsonSelection(NamedTuple):
    recipes: tuple[ResolvedRecipe, ...]
    state: dict[str, Any] | None = None


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


def _refresh_pending_thompson(store: ExperimentStore, pending: dict[str, Any]) -> dict[str, Any]:
    latest = store.read_state()
    if not latest or str(latest.get("experiment_id") or "") != str(pending.get("experiment_id") or ""):
        return pending
    return merge_experiment_state(
        pending,
        experiment_id=str(pending.get("experiment_id") or ""),
        promoted_variant=(str(latest["promoted_variant"]) if latest.get("promoted_variant") else None),
        promoted_at=(str(latest["promoted_at"]) if latest.get("promoted_at") else None),
    )


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
    raw_state = _try_load("read experiment state", store.read_state, failed)
    raw_track = _try_load(
        "read track rows",
        lambda: TrackStore(settings.output).read_rows(experiment_id=experiment.id),
        failed,
    )
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
        recs, catalog_size = _try_load_pair(
            "load recommendations for Thompson guardrails",
            lambda: load_recommendations_frame(settings.output),
            None,
            "load catalog size for Thompson guardrails",
            lambda: load_items_catalog_size(settings.output),
            None,
            parallel=settings.output.kind != "db",
        )
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
    except (LockLostError, WriterLockBusyError):
        raise
    except _OPTIONAL_EVAL_ERRORS as exc:
        _log_caught("Thompson allocation fail closed", exc, log=logger)
        return ThompsonSelection(recipes)


def _maybe_acquire_direct_retrain_lock(
    settings: Settings, fence_check: Callable[[], bool] | None
) -> LockBackend | None:
    if fence_check is not None or not has_distributed_lock(settings):
        return None
    lock = build_lock_backend(settings)
    if not acquire_blocking(lock):
        raise WriterLockBusyError("retrain lock busy")
    return lock


def _recommendation_user_count(recommendations: pd.DataFrame) -> int:
    if recommendations.empty or USER_COLUMN not in recommendations.columns:
        return 0
    user_ids = recommendations[USER_COLUMN].astype(str)
    return int(user_ids[user_ids != COLD_START_USER_ID].nunique())


def run(triggered_by: str = "manual", *, fence_check: Callable[[], bool] | None = None) -> None:
    settings = load_settings()
    retrain_lock = _maybe_acquire_direct_retrain_lock(settings, fence_check)
    if retrain_lock is not None:
        fence_check = retrain_lock.owned
    try:
        _run_job(settings, triggered_by=triggered_by, fence_check=fence_check)
    finally:
        if retrain_lock is not None:
            retrain_lock.release()


def _run_job(settings: Settings, triggered_by: str, fence_check: Callable[[], bool] | None) -> None:
    started_at = datetime.now(UTC).isoformat()
    feature_config = load_feature_config(settings.feature_config_path)
    publication_lock = build_output_writer_lock(settings)
    writer_lock = build_dataset_writer_lock(settings)
    sink = build_output_sink(
        settings.output,
        writer_lock=publication_lock,
        fence_check=fence_check,
        fence_lost="retrain lock lost before write",
        fence_kind="retrain",
    )
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
    manifest_written = False
    replace_success_manifest = False

    try:
        publisher = build_publisher(settings, connect=False)
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
            last_manifest = _try_load(
                "read last manifest for experiment recipes",
                lambda: build_manifest_reader(settings.output).read_latest(),
                None,
            )

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
                ),
                hmac_key=settings.output.artifact_hmac_key,
            )

        # Artifact → snapshot → recommendations; success only after all writes.
        outputs_written = False
        recs_write = getattr(sink, "recommendations_write", None)
        ensure_fence(fence_check)
        try:
            with recs_write() if callable(recs_write) else nullcontext():
                try:
                    if artifact_bytes is not None:
                        ensure_publication_fence(sink, fence_check)
                        sink.write_model_artifact(artifact_bytes)
                        manifest["artifact_written"] = True
                        manifest["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

                    if items is not None and not items.empty:
                        ensure_publication_fence(sink, fence_check)
                        sink.write_items_snapshot(items)

                    ensure_publication_fence(sink, fence_check)
                    sink.write_recommendations(recommendations)
                    outputs_written = True
                    if pending_thompson is not None:
                        ensure_publication_fence(sink, fence_check)
                        store = ExperimentStore(
                            settings.output,
                            writer_lock=publication_lock,
                            fence_check=fence_check,
                            fence_lost="retrain lock lost before write",
                            fence_kind="retrain",
                        )
                        store.write_state(_refresh_pending_thompson(store, pending_thompson))
                    ensure_publication_fence(sink, fence_check)
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
                                    f"{name}={automl_result.metrics[name]:.4f}"
                                    for name in sorted(automl_result.metrics)
                                )
                                if automl_result is not None
                                else ""
                            ),
                            "track_eval": json.dumps(track_eval_payload) if track_eval_payload else "",
                            "served_eval": json.dumps(served_eval_payload) if served_eval_payload else "",
                        }
                    )
                    manifest["generated_at"] = datetime.now(UTC).isoformat()
                    ensure_publication_fence(sink, fence_check)
                    if write_job_manifest(sink, manifest):
                        manifest_written = True
                except _SINK_WRITE_ERRORS as exc:
                    if outputs_written or manifest.get("artifact_written"):
                        manifest["partial_outputs"] = True
                    if (
                        not manifest_written
                        and manifest.get("status") != "success"
                        and not skip_stale_job_manifest(
                            fence_check=fence_check,
                            exc=exc,
                        )
                    ):
                        manifest["error"] = truncate_job_error(exc)
                        manifest["generated_at"] = datetime.now(UTC).isoformat()
                        try:
                            if write_job_manifest(sink, manifest, skip_if_newer_than=started_at):
                                manifest_written = True
                        except _SINK_WRITE_ERRORS as manifest_exc:
                            _log_caught(
                                "Failed to write manifest; original job error (if any) is preserved",
                                manifest_exc,
                                log=logger,
                            )
                    raise
            if publisher is not None and manifest.get("status") == "success":
                try:
                    ensure_fence(fence_check)
                    publisher.connect()
                    ensure_fence(fence_check)
                    if sidecar_generation_current(settings.output, str(manifest.get("generated_at") or "")):
                        publisher.publish(recommendations)
                    else:
                        logger.info("Skipping publish: recommendations were superseded")
                except LockLostError:
                    raise
                except _PUBLISH_ERRORS as exc:
                    _log_caught("Publish failed after successful write", exc, log=logger)
        except _SINK_WRITE_ERRORS:
            if outputs_written or manifest.get("artifact_written"):
                manifest["partial_outputs"] = True
            raise
    except Exception as exc:
        manifest["error"] = truncate_job_error(exc)
        if manifest.get("status") == "success":
            manifest["status"] = "failed"
            manifest_written = False
            replace_success_manifest = True
        raise
    finally:
        persist_exc: BaseException | None = None
        close_exc: BaseException | None = None
        if publisher is not None:
            try:
                publisher.close()
            except _PUBLISH_ERRORS as exc:
                _log_caught("Failed to close recommendation publisher", exc, log=logger)
            except Exception as exc:
                close_exc = exc
        if manifest.get("status") == "success" and (settings.track.enabled or settings.eval.enabled):
            try:
                _persist_track_outputs(
                    TrackStore(
                        settings.output,
                        writer_lock=writer_lock,
                        fence_check=fence_check,
                        fence_lost="retrain lock lost before write",
                        fence_kind="retrain",
                    ),
                    kind=settings.output.kind,
                    eval_report={
                        "generated_at": eval_generated_at,
                        "track_eval": track_eval_payload,
                        "served_eval": served_eval_payload,
                    },
                    recommendations=recommendations,
                    generated_at=str(manifest["generated_at"]),
                    fence_check=fence_check,
                )
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["error"] = truncate_job_error(exc)
                manifest_written = False
                persist_exc = exc
                replace_success_manifest = True
        if (
            close_exc is not None
            and manifest.get("status") == "success"
            and not isinstance(close_exc, LockLostError)
        ):
            manifest["status"] = "failed"
            manifest["error"] = truncate_job_error(close_exc)
            manifest_written = False
            replace_success_manifest = True
        leftover_exc = persist_exc or sys.exc_info()[1]
        skip_if_newer_than = None if replace_success_manifest else started_at
        if not manifest_written and not skip_stale_job_manifest(
            fence_check=fence_check,
            exc=leftover_exc,
        ):
            manifest["generated_at"] = datetime.now(UTC).isoformat()
            try:
                holder = getattr(sink, "recommendations_write", None)
                if callable(holder):
                    with holder():
                        write_job_manifest(sink, manifest, skip_if_newer_than=skip_if_newer_than)
                else:
                    write_job_manifest(sink, manifest, skip_if_newer_than=skip_if_newer_than)
            except _SINK_WRITE_ERRORS as exc:
                _log_caught(
                    "Failed to write manifest; original job error (if any) is preserved",
                    exc,
                    log=logger,
                )
                if manifest.get("status") == "success":
                    raise
        logger.info("Job finished: %s", json.dumps(manifest))
        if persist_exc is not None:
            raise persist_exc
        if close_exc is not None:
            raise close_exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    try:
        run()
    except Exception as exc:
        _log_caught("Recommendation job failed", exc, log=logger)
        sys.exit(1)

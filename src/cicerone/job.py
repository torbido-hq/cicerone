"""Single recommendation job run: input → dataset → (AutoML) → train → write."""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, build_artifact, dumps_artifact
from cicerone.automl import evaluate_candidates, select_best_candidate
from cicerone.config import load_settings
from cicerone.dataset import build_dataset
from cicerone.feature_config import load_feature_config
from cicerone.io.base import InputSource
from cicerone.io.factory import build_input_source, build_output_sink
from cicerone.model import (
    DEFAULT_MODELS,
    RRF_K,
    RecommenderModel,
    plan_model_run,
    train_and_recommend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 500

_MANIFEST_DEFAULTS: dict[str, Any] = {
    "triggered_by": None,
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
}


def _read_input(source: InputSource) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        events_future = executor.submit(source.read_events)
        users_future = executor.submit(source.read_users)
        items_future = executor.submit(source.read_items)
        return events_future.result(), users_future.result(), items_future.result()


def run(triggered_by: str = "manual") -> None:
    settings = load_settings()
    feature_config = load_feature_config(settings.feature_config_path)
    sink = build_output_sink(settings.output)

    manifest = dict(_MANIFEST_DEFAULTS)
    manifest["triggered_by"] = triggered_by
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
            content_fallback_max_neighbors=settings.content_fallback_max_neighbors,
            run_plan=run_plan,
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

        # Persist in order: artifact → items snapshot → recommendations last
        # (serve reads recommendations). Mark success only after all writes
        # succeed so a failed manifest never pairs with orphaned recs without
        # ``partial_outputs``.
        outputs_written = False
        try:
            if artifact_bytes is not None:
                sink.write_model_artifact(artifact_bytes)
                manifest["artifact_written"] = True
                manifest["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

            if items is not None and not items.empty:
                sink.write_items_snapshot(items)

            sink.write_recommendations(recommendations)
            outputs_written = True
        except Exception:
            if outputs_written or manifest.get("artifact_written"):
                manifest["partial_outputs"] = True
            raise

        manifest.update(
            {
                "status": "success",
                "n_events": int(len(events)),
                "n_target_users": len(target_users),
                "n_users_with_recommendations": int(recommendations["user_id"].nunique()),
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
                # Avoid masking the original failure when manifest write also fails.
                pass
            else:
                raise
        logger.info("Job finished: %s", json.dumps(manifest))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

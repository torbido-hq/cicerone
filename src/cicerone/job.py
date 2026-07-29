"""Entry point for a single run of the recommendation job:
configured input (dataset or db) -> build dataset -> train LightFM ->
recommend -> configured output (dataset or db).
"""

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
from cicerone.model import DEFAULT_MODELS, RRF_K, RecommenderModel, train_and_recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Cap on manifest["error"]'s length, since it's str(exc) from an arbitrary
# exception and gets persisted/displayed on the dashboard as-is.
_MAX_ERROR_LENGTH = 500


# Every run writes exactly one manifest with this fixed key set, including
# on failure, so a "db" output's manifest table never gets an INSERT with a
# different column set from one run to the next.
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
    "automl_enabled": False,
    "automl_metrics": "",
}


def _read_input(source: InputSource) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Reads events/users/items concurrently — the three reads are
    independent I/O calls (S3/local file or DB query).
    """
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
    manifest["automl_enabled"] = settings.automl_enabled

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
        if settings.automl_enabled:
            candidate_results = evaluate_candidates(
                events,
                users,
                items,
                feature_config,
                top_k=settings.top_k,
                half_life_days=settings.half_life_days,
                candidates=settings.automl_candidates,
                n_splits=settings.automl_n_splits,
                test_days=settings.automl_test_days,
            )
            automl_result = select_best_candidate(
                candidate_results, primary_metric=settings.automl_primary_metric
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
        recommendations = train_and_recommend(
            built,
            target_users,
            feature_config,
            top_k=settings.top_k,
            enabled_models=enabled_models,
            weights=weights,
            rrf_k=rrf_k,
            strategy_cache=fitted if settings.save_model_artifact else None,
        )

        sink.write_recommendations(recommendations)

        resolved_models = enabled_models or DEFAULT_MODELS
        # `weights is not None` (not truthiness) so an empty/partial
        # model_weights table still reports the effective weight
        # (defaulting to 1.0) for every enabled model.
        model_weights_str = (
            ",".join(f"{name}={weights.get(name, 1.0)}" for name in resolved_models)
            if weights is not None
            else ""
        )

        if settings.save_model_artifact:
            artifact_models = [name for name in resolved_models if name in fitted]
            artifact_weights = (
                {name: weight for name, weight in weights.items() if name in artifact_models}
                if weights is not None
                else None
            )
            artifact = build_artifact(
                fitted=fitted,
                built=built,
                feature_config=feature_config,
                # Only strategies that were actually fitted (personalized
                # strategies are skipped when there are no warm users).
                models=artifact_models,
                model_weights=artifact_weights,
                rrf_k=rrf_k if rrf_k is not None else RRF_K,
            )
            sink.write_model_artifact(dumps_artifact(artifact))
            manifest["artifact_written"] = True
            manifest["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

        manifest.update(
            {
                "status": "success",
                "n_events": int(len(events)),
                "n_target_users": len(target_users),
                "n_users_with_recommendations": int(recommendations["user_id"].nunique()),
                "n_items": int(built.dataset.item_id_map.external_ids.shape[0]),
                "models": ",".join(resolved_models),
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
        sink.write_manifest(manifest)
        logger.info("Job finished: %s", json.dumps(manifest))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

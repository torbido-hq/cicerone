"""Entry point for a single run of the recommendation job:
configured input (dataset or db) -> build dataset -> train LightFM ->
recommend -> configured output (dataset or db).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from cicerone.automl import evaluate_candidates, select_best_candidate
from cicerone.config import load_settings
from cicerone.dataset import build_dataset
from cicerone.feature_config import load_feature_config
from cicerone.io.factory import build_input_source, build_output_sink
from cicerone.model import DEFAULT_MODELS, RRF_K, train_and_recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    settings = load_settings()
    feature_config = load_feature_config(settings.feature_config_path)

    source = build_input_source(settings.input)
    sink = build_output_sink(settings.output)

    events = source.read_events()
    users = source.read_users()
    items = source.read_items()

    logger.info(
        "Loaded %d events, %s users, %s items",
        len(events),
        len(users) if users is not None else "n/a",
        len(items) if items is not None else "n/a",
    )

    built = build_dataset(events, users, items, feature_config, half_life_days=settings.half_life_days)

    target_users = sorted(set(events["user_id"]) | (set(users["user_id"]) if users is not None else set()))

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

    recommendations = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=settings.top_k,
        enabled_models=enabled_models,
        weights=weights,
        rrf_k=rrf_k,
    )

    sink.write_recommendations(recommendations)

    resolved_models = enabled_models or DEFAULT_MODELS
    # `weights is not None` (rather than truthiness) so fusion mode with an
    # empty/partial `[job.model_weights]` still reports the *effective*
    # weight (defaulting to 1.0) for every enabled model, instead of hiding
    # implicit defaults behind an empty string in the manifest.
    if weights is not None:
        effective_weights = {name: weights.get(name, 1.0) for name in resolved_models}
        model_weights_str = ",".join(f"{name}={weight}" for name, weight in effective_weights.items())
    else:
        model_weights_str = ""

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_events": int(len(events)),
        "n_target_users": len(target_users),
        "n_users_with_recommendations": int(recommendations["user_id"].nunique()),
        "n_items": int(built.dataset.item_id_map.external_ids.shape[0]),
        "top_k": settings.top_k,
        "models": ",".join(resolved_models),
        "model_weights": model_weights_str,
        "rrf_k": rrf_k if rrf_k is not None else RRF_K,
        "automl_enabled": settings.automl_enabled,
        "automl_metrics": (
            ",".join(f"{name}={automl_result.metrics[name]:.4f}" for name in sorted(automl_result.metrics))
            if automl_result is not None
            else ""
        ),
    }
    sink.write_manifest(manifest)
    logger.info("Job finished: %s", json.dumps(manifest))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

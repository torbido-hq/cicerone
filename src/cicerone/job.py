"""Entry point for a single run of the recommendation job:
configured input (dataset or db) -> build dataset -> train LightFM ->
recommend -> configured output (dataset or db).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

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
    recommendations = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=settings.top_k,
        enabled_models=settings.models,
        weights=settings.model_weights,
        rrf_k=settings.rrf_k,
    )

    sink.write_recommendations(recommendations)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_events": int(len(events)),
        "n_target_users": len(target_users),
        "n_users_with_recommendations": int(recommendations["user_id"].nunique()),
        "n_items": int(built.dataset.item_id_map.external_ids.shape[0]),
        "top_k": settings.top_k,
        "models": ",".join(settings.models or DEFAULT_MODELS),
        "model_weights": (
            ",".join(f"{name}={weight}" for name, weight in settings.model_weights.items())
            if settings.model_weights
            else ""
        ),
        "rrf_k": settings.rrf_k if settings.rrf_k is not None else RRF_K,
    }
    sink.write_manifest(manifest)
    logger.info("Job finished: %s", json.dumps(manifest))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        sys.exit(1)

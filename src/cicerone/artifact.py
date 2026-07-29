"""Versioned, portable fitted-model artifacts.

The batch job can optionally persist the fitted strategies (+ the rectools
Dataset and FeatureConfig needed to recommend from them) as a single
serialized file. That artifact is loadable without re-running
``job.run`` / ``fit``, so a future thin inference layer would not need to
redesign the training side.

Serve mode does **not** load this artifact — it still reads precomputed
recommendation rows only (no ML deps in the request path).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rectools.dataset import Dataset

from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig
from cicerone.model import RecommenderModel, recommend_with_models

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FILENAME = "model.artifact"


@dataclass
class ModelArtifact:
    """Portable bundle of fitted strategies and everything needed to recommend."""

    schema_version: int
    created_at: str
    models: list[str]
    model_weights: dict[str, float] | None
    rrf_k: float | None
    fitted: dict[str, RecommenderModel]
    dataset: Dataset
    items: pd.DataFrame | None
    feature_config: FeatureConfig


def build_artifact(
    *,
    fitted: dict[str, RecommenderModel],
    built: BuiltDataset,
    feature_config: FeatureConfig,
    models: list[str],
    model_weights: dict[str, float] | None,
    rrf_k: float | None,
) -> ModelArtifact:
    return ModelArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        models=list(models),
        model_weights=dict(model_weights) if model_weights is not None else None,
        rrf_k=rrf_k,
        fitted=dict(fitted),
        dataset=built.dataset,
        items=built.items.copy() if built.items is not None else None,
        feature_config=feature_config,
    )


def dumps_artifact(artifact: ModelArtifact) -> bytes:
    return pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)


def loads_artifact(payload: bytes) -> ModelArtifact:
    artifact = pickle.loads(payload)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(
            f"Artifact payload did not unpickle to ModelArtifact, got {type(artifact).__name__}"
        )
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema_version {artifact.schema_version}; "
            f"this build expects {ARTIFACT_SCHEMA_VERSION}"
        )
    return artifact


def save_artifact(path: Path | str, artifact: ModelArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps_artifact(artifact))
    logger.info("Wrote model artifact to %s (schema_version=%d)", path, artifact.schema_version)


def load_artifact(path: Path | str) -> ModelArtifact:
    path = Path(path)
    logger.info("Loading model artifact from %s", path)
    return loads_artifact(path.read_bytes())


def recommend_from_artifact(
    artifact: ModelArtifact,
    target_users: list[str],
    top_k: int,
) -> pd.DataFrame:
    """Produce recommendations from a loaded artifact without re-fitting."""
    # interactions are unused at recommend time; pass an empty frame.
    built = BuiltDataset(dataset=artifact.dataset, interactions=pd.DataFrame(), items=artifact.items)
    return recommend_with_models(
        artifact.fitted,
        built,
        target_users,
        artifact.feature_config,
        top_k=top_k,
        enabled_models=artifact.models,
        weights=artifact.model_weights,
        rrf_k=artifact.rrf_k,
    )

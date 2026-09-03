"""Versioned fitted-model artifacts (RecTools save/load + pickle envelope).

Trust boundary: load only trusted batch-job artifacts — never untrusted
uploads (pickle RCE). Not used on the serve HTTP request path; the events
worker may load it when `[events.online]` is enabled.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rectools.dataset import Dataset
from rectools.models import load_model
from rectools.models.base import ModelBase

from cicerone.config.settings import ExplainSettings
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import FeatureConfig
from cicerone.model import (
    RecommenderModel,
    recommend_with_models,
)

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 3
ARTIFACT_FILENAME = "model.artifact"
_META_NAME = "meta.json"
_BUNDLE_NAME = "bundle.pkl"
_MODELS_DIR = "models/"
_RECTOOLS_SUFFIX = ".rectools"
_PICKLE_SUFFIX = ".pkl"


@dataclass(frozen=True)
class ModelArtifact:
    schema_version: int
    created_at: datetime
    models: tuple[str, ...]
    model_weights: dict[str, float] | None
    rrf_k: float | None
    fitted: dict[str, RecommenderModel]
    dataset: Dataset
    items: pd.DataFrame | None
    feature_config: FeatureConfig
    users: pd.DataFrame | None = None


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
        created_at=datetime.now(UTC),
        models=tuple(models),
        model_weights=dict(model_weights) if model_weights is not None else None,
        rrf_k=rrf_k,
        fitted=dict(fitted),
        dataset=built.dataset,
        items=built.items.copy() if built.items is not None else None,
        feature_config=feature_config,
        users=built.users.copy() if built.users is not None else None,
    )


def _is_rectools_model(model: object) -> bool:
    return isinstance(model, ModelBase)


def _dump_rectools_model(model: ModelBase) -> bytes:
    buffer = io.BytesIO()
    model.save(buffer)
    return buffer.getvalue()


def _load_rectools_model(payload: bytes) -> RecommenderModel:
    return load_model(io.BytesIO(payload))  # type: ignore[return-value]


def dumps_artifact(artifact: ModelArtifact) -> bytes:
    """Serialize artifact (zip: meta + pickle envelope + model blobs)."""
    meta = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at.isoformat(),
        "models": list(artifact.models),
        "model_weights": artifact.model_weights,
        "rrf_k": artifact.rrf_k,
        "model_formats": {
            name: ("rectools" if _is_rectools_model(model) else "pickle")
            for name, model in artifact.fitted.items()
        },
    }
    bundle = {
        "dataset": artifact.dataset,
        "items": artifact.items,
        "users": artifact.users,
        "feature_config": artifact.feature_config,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_META_NAME, json.dumps(meta, sort_keys=True))
        zf.writestr(_BUNDLE_NAME, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
        for name, model in artifact.fitted.items():
            if _is_rectools_model(model):
                zf.writestr(
                    f"{_MODELS_DIR}{name}{_RECTOOLS_SUFFIX}",
                    _dump_rectools_model(model),  # type: ignore[arg-type]
                )
            else:
                zf.writestr(
                    f"{_MODELS_DIR}{name}{_PICKLE_SUFFIX}",
                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
                )
    return buffer.getvalue()


def _zip_member_allowed(name: str) -> bool:
    if name in {_META_NAME, _BUNDLE_NAME}:
        return True
    if not name.startswith(_MODELS_DIR):
        return False
    rest = name[len(_MODELS_DIR) :]
    if not rest or "/" in rest or rest in {".", ".."}:
        return False
    return rest.endswith(_RECTOOLS_SUFFIX) or rest.endswith(_PICKLE_SUFFIX)


def _assert_safe_artifact_zip(zf: zipfile.ZipFile) -> set[str]:
    names = zf.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Artifact zip has duplicate members")
    for name in names:
        parts = name.split("/")
        if ".." in parts or name.startswith("/") or "\\" in name or not _zip_member_allowed(name):
            raise ValueError(f"Artifact zip has unexpected member {name!r}")
    unique = set(names)
    if _META_NAME not in unique or _BUNDLE_NAME not in unique:
        raise ValueError("Artifact zip is missing meta.json or bundle.pkl")
    return unique


def loads_artifact(payload: bytes) -> ModelArtifact:
    """Deserialize a trusted v3 ModelArtifact zip. Legacy bare pickle is refused."""
    try:
        buffer = io.BytesIO(payload)
        with zipfile.ZipFile(buffer, mode="r") as zf:
            names = _assert_safe_artifact_zip(zf)
            meta: dict[str, Any] = json.loads(zf.read(_META_NAME).decode("utf-8"))
            schema_version = int(meta["schema_version"])
            if schema_version != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported artifact schema_version {schema_version}; "
                    f"this build expects {ARTIFACT_SCHEMA_VERSION}"
                )
            models = list(meta["models"])
            model_files = {name for name in names if name.startswith(_MODELS_DIR)}
            allowed_stems = set(models)
            for path in model_files:
                stem = path[len(_MODELS_DIR) :]
                strategy = stem
                for suffix in (_RECTOOLS_SUFFIX, _PICKLE_SUFFIX):
                    if stem.endswith(suffix):
                        strategy = stem[: -len(suffix)]
                        break
                if strategy not in allowed_stems:
                    raise ValueError(f"Artifact zip has unexpected member {path!r}")
            bundle = pickle.loads(zf.read(_BUNDLE_NAME))
            fitted: dict[str, RecommenderModel] = {}
            model_formats = meta.get("model_formats") or {}
            for name in models:
                fmt = model_formats.get(name)
                rectools_path = f"{_MODELS_DIR}{name}{_RECTOOLS_SUFFIX}"
                pickle_path = f"{_MODELS_DIR}{name}{_PICKLE_SUFFIX}"
                if fmt == "rectools" or (fmt is None and rectools_path in names):
                    fitted[name] = _load_rectools_model(zf.read(rectools_path))
                elif fmt == "pickle" or pickle_path in names:
                    fitted[name] = pickle.loads(zf.read(pickle_path))
                else:
                    raise ValueError(f"Artifact is missing serialized model for strategy {name!r}")
            return ModelArtifact(
                schema_version=schema_version,
                created_at=datetime.fromisoformat(str(meta["created_at"])),
                models=tuple(models),
                model_weights=(
                    dict(meta["model_weights"]) if meta.get("model_weights") is not None else None
                ),
                rrf_k=meta.get("rrf_k"),
                fitted=fitted,
                dataset=bundle["dataset"],
                items=bundle["items"],
                feature_config=bundle["feature_config"],
                users=bundle.get("users"),
            )
    except zipfile.BadZipFile as exc:
        raise TypeError("Artifact payload is not a v3 zip") from exc


def save_artifact(path: Path | str, artifact: ModelArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps_artifact(artifact))
    logger.info("Wrote model artifact to %s (schema_version=%d)", path, artifact.schema_version)


def load_artifact(path: Path | str) -> ModelArtifact:
    path = Path(path)
    logger.info("Loading model artifact from %s", path)
    return loads_artifact(path.read_bytes())


def save_rectools_model(path: Path | str, model: ModelBase) -> None:
    """``model.save`` → bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_rectools_model(path: Path | str) -> RecommenderModel:
    """``load_model`` from bytes."""
    return load_model(Path(path))  # type: ignore[return-value]


def _interactions_from_dataset(dataset: Dataset) -> pd.DataFrame:
    """External-id interaction table already stored inside the RecTools Dataset."""
    frame = dataset.get_raw_interactions()
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame


def recommend_from_artifact(
    artifact: ModelArtifact,
    target_users: list[str],
    top_k: int,
    *,
    explain: ExplainSettings | None = None,
) -> pd.DataFrame:
    interactions = _interactions_from_dataset(artifact.dataset)
    built = BuiltDataset(
        dataset=artifact.dataset,
        interactions=interactions,
        items=artifact.items,
        users=artifact.users,
    )
    if artifact.feature_config.blending.enabled and interactions.empty:
        logger.warning(
            "recommend_from_artifact with blending.enabled: stored Dataset has no "
            "interactions, so the blend curve sees n_interactions=0 for every user"
        )
    weights = dict(artifact.model_weights) if artifact.model_weights is not None else None
    return recommend_with_models(
        dict(artifact.fitted),
        built,
        target_users,
        artifact.feature_config,
        top_k=top_k,
        enabled_models=list(artifact.models),
        weights=weights,
        rrf_k=artifact.rrf_k,
        explain=explain,
    )

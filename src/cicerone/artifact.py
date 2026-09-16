"""Versioned fitted-model artifacts (RecTools save/load + pickle envelope).

Trust boundary: load only trusted batch-job artifacts — never untrusted
uploads (pickle RCE). Not used on the serve HTTP request path; the events
worker may load it when `[events.online]` is enabled.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import pickle
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rectools.dataset import Dataset
from rectools.models import load_model
from rectools.models.base import ModelBase

from cicerone.config.constants import ARTIFACT_HMAC_KEY_MIN_BYTES, DEFAULT_MAX_ARTIFACT_BYTES
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
_HMAC_NAME = "hmac.sha256"
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


def _hmac_key_bytes(hmac_key: str | None) -> bytes | None:
    if hmac_key is None:
        return None
    raw = hmac_key.encode("utf-8") if isinstance(hmac_key, str) else hmac_key
    if len(raw) < ARTIFACT_HMAC_KEY_MIN_BYTES:
        raise ValueError(f"artifact HMAC key must be at least {ARTIFACT_HMAC_KEY_MIN_BYTES} bytes")
    return raw


_HMAC_READ_CHUNK = 1024 * 1024
_HMAC_HEX_BYTES = hashlib.sha256().digest_size * 2


def _hmac_update_member(digest: hmac.HMAC, name: str, size: int, chunks: Iterable[bytes]) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(size.to_bytes(8, "big"))
    remaining = size
    for chunk in chunks:
        digest.update(chunk)
        remaining -= len(chunk)
        if remaining < 0:
            raise ValueError("Artifact HMAC member size does not match")
    if remaining != 0:
        raise ValueError("Artifact HMAC member size does not match")


def _iter_zip_chunks(handle: Any) -> Iterable[bytes]:
    while True:
        chunk = handle.read(_HMAC_READ_CHUNK)
        if not chunk:
            return
        yield chunk


def dumps_artifact(artifact: ModelArtifact, *, hmac_key: str | None = None) -> bytes:
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
    key = _hmac_key_bytes(hmac_key)
    names: list[str] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

        def _write(name: str, payload: bytes) -> None:
            zf.writestr(name, payload)
            names.append(name)

        _write(_META_NAME, json.dumps(meta, sort_keys=True).encode("utf-8"))
        _write(_BUNDLE_NAME, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
        for name, model in artifact.fitted.items():
            if _is_rectools_model(model):
                _write(f"{_MODELS_DIR}{name}{_RECTOOLS_SUFFIX}", _dump_rectools_model(model))  # type: ignore[arg-type]
            else:
                _write(
                    f"{_MODELS_DIR}{name}{_PICKLE_SUFFIX}",
                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
                )
        if key is not None:
            digest = hmac.new(key, digestmod=hashlib.sha256)
            for name in sorted(names):
                info = zf.getinfo(name)
                with zf.open(name) as handle:
                    _hmac_update_member(digest, name, info.file_size, _iter_zip_chunks(handle))
            zf.writestr(_HMAC_NAME, digest.hexdigest().encode("ascii"))
    return buffer.getvalue()


def _zip_member_allowed(name: str) -> bool:
    if name in {_META_NAME, _BUNDLE_NAME, _HMAC_NAME}:
        return True
    if not name.startswith(_MODELS_DIR):
        return False
    rest = name[len(_MODELS_DIR) :]
    if not rest or "/" in rest or rest in {".", ".."}:
        return False
    return rest.endswith(_RECTOOLS_SUFFIX) or rest.endswith(_PICKLE_SUFFIX)


def _model_zip_path(name: str, fmt: str) -> str:
    if fmt == "rectools":
        return f"{_MODELS_DIR}{name}{_RECTOOLS_SUFFIX}"
    if fmt == "pickle":
        return f"{_MODELS_DIR}{name}{_PICKLE_SUFFIX}"
    raise ValueError(f"Artifact has unknown model format {fmt!r} for strategy {name!r}")


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


def _assert_declared_model_members(
    names: set[str], models: list[str], model_formats: object
) -> dict[str, str]:
    if not isinstance(model_formats, dict):
        raise ValueError("Artifact is missing model_formats")
    actual = {name for name in names if name.startswith(_MODELS_DIR)}
    expected: dict[str, str] = {}
    for name in models:
        fmt = model_formats.get(name)
        if not isinstance(fmt, str):
            raise ValueError(f"Artifact is missing model format for strategy {name!r}")
        expected[name] = _model_zip_path(name, fmt)
    expected_paths = set(expected.values())
    extra = actual - expected_paths
    if extra:
        raise ValueError(f"Artifact zip has unexpected member {sorted(extra)[0]!r}")
    for name, path in expected.items():
        if path not in actual:
            raise ValueError(f"Artifact is missing serialized model for strategy {name!r}")
    return expected


def _assert_artifact_size(zf: zipfile.ZipFile, max_bytes: int) -> None:
    uncompressed = 0
    for info in zf.infolist():
        if info.file_size > max_bytes:
            raise ValueError(f"Artifact member {info.filename!r} exceeds {max_bytes} bytes")
        uncompressed += info.file_size
        if uncompressed > max_bytes:
            raise ValueError(f"Artifact uncompressed size exceeds {max_bytes} bytes")


def _assert_hmac_member_size(zf: zipfile.ZipFile, names: set[str]) -> None:
    if _HMAC_NAME not in names:
        return
    if zf.getinfo(_HMAC_NAME).file_size != _HMAC_HEX_BYTES:
        raise ValueError("Artifact HMAC member has invalid size")


def _verify_artifact_hmac(zf: zipfile.ZipFile, names: set[str], key: bytes) -> None:
    if _HMAC_NAME not in names:
        raise ValueError("Artifact is missing HMAC member")
    _assert_hmac_member_size(zf, names)
    try:
        expected = zf.read(_HMAC_NAME).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Artifact HMAC member is not valid ASCII") from exc
    digest = hmac.new(key, digestmod=hashlib.sha256)
    for name in sorted(name for name in names if name != _HMAC_NAME):
        info = zf.getinfo(name)
        with zf.open(name) as handle:
            _hmac_update_member(digest, name, info.file_size, _iter_zip_chunks(handle))
    if not hmac.compare_digest(digest.hexdigest(), expected):
        raise ValueError("Artifact HMAC does not match")


def loads_artifact(
    payload: bytes,
    *,
    hmac_key: str | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> ModelArtifact:
    """Deserialize a trusted v3 ModelArtifact zip. Legacy bare pickle is refused."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    if len(payload) > max_bytes:
        raise ValueError(f"Artifact payload exceeds {max_bytes} bytes")
    try:
        buffer = io.BytesIO(payload)
        with zipfile.ZipFile(buffer, mode="r") as zf:
            _assert_artifact_size(zf, max_bytes)
            names = _assert_safe_artifact_zip(zf)
            _assert_hmac_member_size(zf, names)
            key = _hmac_key_bytes(hmac_key)
            if key is not None:
                _verify_artifact_hmac(zf, names, key)
            meta: dict[str, Any] = json.loads(zf.read(_META_NAME).decode("utf-8"))
            schema_version = int(meta["schema_version"])
            if schema_version != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported artifact schema_version {schema_version}; "
                    f"this build expects {ARTIFACT_SCHEMA_VERSION}"
                )
            models = list(meta["models"])
            expected_paths = _assert_declared_model_members(names, models, meta.get("model_formats"))
            bundle = pickle.loads(zf.read(_BUNDLE_NAME))
            fitted: dict[str, RecommenderModel] = {}
            for name in models:
                path = expected_paths[name]
                blob = zf.read(path)
                if path.endswith(_RECTOOLS_SUFFIX):
                    fitted[name] = _load_rectools_model(blob)
                else:
                    fitted[name] = pickle.loads(blob)
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


def save_artifact(path: Path | str, artifact: ModelArtifact, *, hmac_key: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps_artifact(artifact, hmac_key=hmac_key))
    logger.info("Wrote model artifact to %s (schema_version=%d)", path, artifact.schema_version)


def load_artifact(
    path: Path | str,
    *,
    hmac_key: str | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> ModelArtifact:
    path = Path(path)
    logger.info("Loading model artifact from %s", path)
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    return loads_artifact(payload, hmac_key=hmac_key, max_bytes=max_bytes)


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

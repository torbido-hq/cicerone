from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from cicerone.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    build_artifact,
    dumps_artifact,
    load_artifact,
    loads_artifact,
    recommend_from_artifact,
    save_artifact,
)
from cicerone.dataset import build_dataset
from cicerone.model import fit_strategies, train_and_recommend


def test_artifact_round_trip_recommendations_match(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    target_users = ["u1", "u2", "u3", "u4"]
    enabled = ["collaborative", "popular"]
    top_k = 3

    fitted: dict = {}
    original = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=top_k,
        enabled_models=enabled,
        strategy_cache=fitted,
    )

    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=enabled,
        model_weights=None,
        rrf_k=None,
    )
    path = tmp_path / "model.artifact"
    save_artifact(path, artifact)

    loaded = load_artifact(path)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert list(loaded.models) == enabled
    assert isinstance(loaded.created_at, datetime)
    assert loaded.created_at == artifact.created_at

    reloaded = recommend_from_artifact(loaded, target_users, top_k=top_k)
    pd.testing.assert_frame_equal(
        original.sort_values(["user_id", "rank"]).reset_index(drop=True),
        reloaded.sort_values(["user_id", "rank"]).reset_index(drop=True),
    )


def test_artifact_round_trip_with_weighted_fusion(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    target_users = ["u1", "u2"]
    enabled = ["collaborative", "popular"]
    weights = {"collaborative": 1.0, "popular": 0.5}
    top_k = 2

    fitted: dict = {}
    original = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=top_k,
        enabled_models=enabled,
        weights=weights,
        rrf_k=40,
        strategy_cache=fitted,
    )
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=enabled,
        model_weights=weights,
        rrf_k=40,
    )
    save_artifact(tmp_path / "model.artifact", artifact)
    reloaded = recommend_from_artifact(load_artifact(tmp_path / "model.artifact"), target_users, top_k=top_k)
    pd.testing.assert_frame_equal(
        original.sort_values(["user_id", "rank"]).reset_index(drop=True),
        reloaded.sort_values(["user_id", "rank"]).reset_index(drop=True),
    )


def test_loads_artifact_rejects_wrong_schema_version(
    feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    bad = replace(artifact, schema_version=ARTIFACT_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="Unsupported artifact schema_version"):
        loads_artifact(dumps_artifact(bad))


def test_loads_artifact_rejects_non_artifact_payload():
    with pytest.raises(TypeError, match="v3 zip"):
        loads_artifact(__import__("pickle").dumps({"not": "an artifact"}))


def test_loads_artifact_rejects_legacy_pickle_model_artifact(
    feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    with pytest.raises(TypeError, match="v3 zip"):
        loads_artifact(__import__("pickle").dumps(artifact))


def test_loads_artifact_rejects_unexpected_zip_member(
    feature_config, sample_events, sample_users, sample_items
):
    import io
    import zipfile

    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    buffer = io.BytesIO(dumps_artifact(artifact))
    out = io.BytesIO()
    with zipfile.ZipFile(buffer, "r") as src, zipfile.ZipFile(out, "w") as dest:
        for info in src.infolist():
            dest.writestr(info, src.read(info.filename))
        dest.writestr("evil.txt", b"nope")
    with pytest.raises(ValueError, match="unexpected member"):
        loads_artifact(out.getvalue())


def test_loads_artifact_rejects_path_traversal_member():
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dest:
        dest.writestr("meta.json", "{}")
        dest.writestr("bundle.pkl", b"x")
        dest.writestr("../evil.pkl", b"x")
    with pytest.raises(ValueError, match="unexpected member"):
        loads_artifact(out.getvalue())


def test_loads_artifact_rejects_both_formats_for_declared_model(
    feature_config, sample_events, sample_users, sample_items
):
    import io
    import zipfile

    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    buffer = io.BytesIO(dumps_artifact(artifact))
    out = io.BytesIO()
    with zipfile.ZipFile(buffer, "r") as src, zipfile.ZipFile(out, "w") as dest:
        names = set(src.namelist())
        for info in src.infolist():
            dest.writestr(info, src.read(info.filename))
        extra = "models/popular.pkl" if "models/popular.rectools" in names else "models/popular.rectools"
        dest.writestr(extra, b"extra")
    with pytest.raises(ValueError, match="unexpected member"):
        loads_artifact(out.getvalue())


def test_artifact_hmac_round_trip_and_reject_wrong_key(
    feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    key = "0123456789abcdef"
    payload = dumps_artifact(artifact, hmac_key=key)
    loaded = loads_artifact(payload, hmac_key=key)
    assert list(loaded.models) == ["popular"]
    with pytest.raises(ValueError, match="HMAC"):
        loads_artifact(payload, hmac_key="fedcba9876543210")
    with pytest.raises(ValueError, match="missing HMAC"):
        loads_artifact(dumps_artifact(artifact), hmac_key=key)


def test_dumps_artifact_rejects_short_hmac_key(feature_config, sample_events, sample_users, sample_items):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    with pytest.raises(ValueError, match="at least 16"):
        dumps_artifact(artifact, hmac_key="short")


def test_loads_artifact_rejects_nonpositive_max_bytes():
    with pytest.raises(ValueError, match="max_bytes"):
        loads_artifact(b"x", max_bytes=0)


def test_loads_artifact_rejects_oversize_payload():
    with pytest.raises(ValueError, match="exceeds"):
        loads_artifact(b"PK\x03\x04" + b"x" * 20, max_bytes=8)


def test_load_artifact_rejects_oversize_file(tmp_path):
    path = tmp_path / "model.artifact"
    path.write_bytes(b"PK\x03\x04" + b"x" * 20)
    with pytest.raises(ValueError, match="exceeds"):
        load_artifact(path, max_bytes=8)


def test_load_artifact_rejects_nonpositive_max_bytes(tmp_path):
    path = tmp_path / "model.artifact"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="max_bytes"):
        load_artifact(path, max_bytes=0)


def test_loads_artifact_rejects_oversize_uncompressed_member():
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as dest:
        dest.writestr("meta.json", "{}")
        dest.writestr("bundle.pkl", b"x" * 50_000)
    payload = out.getvalue()
    assert len(payload) < 500
    with pytest.raises(ValueError, match="exceeds"):
        loads_artifact(payload, max_bytes=500)


def test_fit_strategies_populates_cache(feature_config, sample_events, sample_users, sample_items):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    cache: dict = {}
    enabled, fitted = fit_strategies(
        built, ["u1", "u2"], enabled_models=["popular", "latest"], strategy_cache=cache
    )
    assert enabled == ["popular", "latest"]
    assert set(fitted) == {"popular", "latest"}
    assert set(cache) == {"popular", "latest"}

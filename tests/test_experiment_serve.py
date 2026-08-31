from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient
from test_serve import _FakeManifest, _FakeReader, _feature_config, _settings

from cicerone.config import IOSettings
from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.experiment.assignment import assign_variant
from cicerone.serve import create_app


def _experiment_settings(**overrides) -> ExperimentSettings:
    defaults: dict = dict(
        enabled=True,
        id="rrf-vs-blend",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    defaults.update(overrides)
    return ExperimentSettings(**defaults)


def _variant_recs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "control-item",
                "rank": 1,
                "score": 0.9,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "u1",
                "item_id": "treatment-item",
                "rank": 1,
                "score": 0.8,
                "source": "personalized",
                "variant": "treatment",
            },
            {
                "user_id": "__cold_start__",
                "item_id": "cold-control",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "control",
            },
            {
                "user_id": "__cold_start__",
                "item_id": "cold-treatment",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "treatment",
            },
        ]
    )


def test_recommendations_omit_experiment_fields_when_disabled():
    app = create_app(
        _settings(),
        _FakeReader(_variant_recs()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["experiment_id"] is None
    assert body["variant"] is None
    assert {row["item_id"] for row in body["items"]} == {"control-item"}


def test_filter_variant_rows_collapses_mixed_when_unspecified():
    from cicerone.io.recommendation_schema import filter_variant_rows

    rows = filter_variant_rows(_variant_recs(), None)
    assert set(rows[rows["user_id"] == "u1"]["item_id"].astype(str)) == {"control-item"}


def test_variant_helpers_empty_and_no_control():
    from cicerone.io.recommendation_schema import (
        RECOMMENDATION_COLUMNS,
        collapse_mixed_variants,
        filter_variant_rows,
        pick_fallback_variant,
        recommendation_output_columns,
    )

    assert pick_fallback_variant([]) is None
    assert pick_fallback_variant(["", "zeta", "alpha"]) == "alpha"
    assert pick_fallback_variant([float("nan"), pd.NA, None, "treatment"]) == "treatment"
    mixed_nan = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["keep", "drop"],
            "rank": [1, 1],
            "score": [0.9, 0.8],
            "source": ["personalized", "personalized"],
            "variant": ["treatment", float("nan")],
        }
    )
    collapsed = collapse_mixed_variants(mixed_nan)
    assert list(collapsed["item_id"].astype(str)) == ["keep"]
    assert recommendation_output_columns(object()) == list(RECOMMENDATION_COLUMNS)
    empty = pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source", "variant"])
    assert collapse_mixed_variants(empty).empty
    assert filter_variant_rows(empty, "control").empty
    assert filter_variant_rows(pd.DataFrame({"user_id": ["u1"]}), None).equals(
        pd.DataFrame({"user_id": ["u1"]})
    )
    nan_rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["keep", "drop"],
            "rank": [1, 1],
            "score": [0.9, 0.8],
            "source": ["personalized", "personalized"],
            "variant": ["control", float("nan")],
        }
    )
    assert list(filter_variant_rows(nan_rows, "control")["item_id"].astype(str)) == ["keep"]
    assert filter_variant_rows(nan_rows, "nan").empty


def test_recommendations_filter_assigned_variant(tmp_path):
    assigned = assign_variant("rrf-vs-blend", "u1", (("control", 0.5), ("treatment", 0.5)))
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(experiment=_experiment_settings(), output=output),
        _FakeReader(_variant_recs()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["experiment_id"] == "rrf-vs-blend"
    assert body["variant"] == assigned
    assert [row["item_id"] for row in body["items"]] == [f"{assigned}-item"]


def test_recommendations_omit_experiment_fields_without_variant_column(tmp_path):
    recs = _variant_recs().drop(columns=["variant"])
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(experiment=_experiment_settings(log_exposures=True), output=output),
        _FakeReader(recs),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["experiment_id"] is None
    assert body["variant"] is None
    assert {row["item_id"] for row in body["items"]} == {"control-item", "treatment-item"}
    assert not (tmp_path / "exposures.jsonl").exists()


def test_recommendations_filter_automl_challenger_without_variants(tmp_path):
    assigned = assign_variant("auto", "u1", (("control", 0.5), ("treatment", 0.5)))
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(
            experiment=_experiment_settings(
                id="auto",
                automl_challenger=True,
                variants=(),
            ),
            output=output,
        ),
        _FakeReader(_variant_recs()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["experiment_id"] == "auto"
    assert body["variant"] == assigned
    assert [row["item_id"] for row in body["items"]] == [f"{assigned}-item"]


def test_recommendations_log_exposures_when_enabled(tmp_path):
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(experiment=_experiment_settings(log_exposures=True), output=output),
        _FakeReader(_variant_recs()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    lines = (tmp_path / "exposures.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["user_id"] == "u1"
    assert row["experiment_id"] == "rrf-vs-blend"
    assert row["variant"] in {"control", "treatment"}


def test_recommendations_exposure_generated_at_matches_response(tmp_path):
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    app = create_app(
        _settings(experiment=_experiment_settings(log_exposures=True), output=output),
        _FakeReader(_variant_recs()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    row = json.loads((tmp_path / "exposures.jsonl").read_text().strip().splitlines()[0])
    assert row["generated_at"] == body["generated_at"]
    assert body["generated_at"] == "2026-08-04T12:00:00+00:00"

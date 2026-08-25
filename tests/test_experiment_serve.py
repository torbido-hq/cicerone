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
    assert {row["item_id"] for row in body["items"]} == {"control-item", "treatment-item"}


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

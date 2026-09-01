from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from conftest import make_settings
from sqlalchemy import create_engine

from cicerone.config import IOSettings
from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.dashboard_experiments import clear_promotion, experiment_context, promote_winner
from cicerone.experiment.evaluate import exposure_row
from cicerone.experiment.store import ExperimentStore, experiment_state
from cicerone.io.recommendation_schema import VARIANT_COLUMN

REPO_FEATURES = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _settings(tmp_path, **experiment_overrides):
    out = tmp_path / "out"
    inp = tmp_path / "in"
    out.mkdir()
    inp.mkdir()
    params = {
        "enabled": True,
        "id": "exp-1",
        "primary_metric": "purchase",
        "log_exposures": True,
        "variants": (
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
        **experiment_overrides,
    }
    experiment = ExperimentSettings(**params)
    return make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(inp)}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        experiment=experiment,
    )


def _write_frames(settings, *, events, recs, exposures=None) -> None:
    inp = Path(settings.input.options["path"])
    out = Path(settings.output.options["path"])
    pd.DataFrame(events).to_parquet(inp / "events.parquet", index=False)
    pd.DataFrame(recs).to_parquet(out / "recommendations.parquet", index=False)
    if exposures:
        ExperimentStore(settings.output).append_exposures(exposures)


def test_experiment_context_disabled(tmp_path):
    settings = make_settings(feature_config_path=str(REPO_FEATURES))
    context = experiment_context(settings)
    assert context["enabled"] is False
    assert context["report"] is None


def test_promote_winner_when_undecided(tmp_path):
    settings = _settings(tmp_path)
    _write_frames(
        settings,
        events=[{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1}],
        recs=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "treatment",
            },
        ],
    )
    assert promote_winner(settings, "ghost") == "Unknown variant 'ghost'"
    error = promote_winner(settings, "treatment")
    assert error is not None
    assert "not ready" in error


def test_promote_winner_when_treatment_wins(tmp_path):
    settings = _settings(tmp_path)
    events = []
    recs = []
    exposures = []
    for i in range(40):
        events.append(
            {
                "user_id": f"c{i}",
                "item_id": f"i{i % 10}",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": "2026-01-02T00:00:00Z",
            }
        )
        events.append(
            {
                "user_id": f"t{i}",
                "item_id": f"i{i % 10}",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-02T00:00:00Z",
            }
        )
        recs.append(
            {
                "user_id": f"c{i}",
                "item_id": f"i{i % 10}",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        )
        recs.append(
            {
                "user_id": f"t{i}",
                "item_id": f"i{(i + 3) % 10}",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "treatment",
            }
        )
        exposures.append(
            exposure_row(
                user_id=f"c{i}",
                experiment_id="exp-1",
                variant="control",
                generated_at=None,
                exposed_at=pd.Timestamp("2026-01-01T00:00:00Z"),
            )
        )
        exposures.append(
            exposure_row(
                user_id=f"t{i}",
                experiment_id="exp-1",
                variant="treatment",
                generated_at=None,
                exposed_at=pd.Timestamp("2026-01-01T00:00:00Z"),
            )
        )
    _write_frames(settings, events=events, recs=recs, exposures=exposures)
    context = experiment_context(settings)
    report = context["report"]
    assert report is not None
    assert report.can_promote is True
    assert report.winner == "treatment"
    assert promote_winner(settings, "control") == "Winner is 'treatment', not 'control'"
    assert promote_winner(settings, "treatment") is None
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] == "treatment"
    assert experiment_context(settings)["promoted_variant"] == "treatment"
    assert clear_promotion(settings) is None
    cleared = ExperimentStore(settings.output).read_state()
    assert cleared is not None
    assert cleared["promoted_variant"] is None


def test_experiment_context_coverage_uses_items_snapshot(tmp_path):
    settings = _settings(tmp_path, log_exposures=False)
    events = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 3}",
            "event_type": "purchase",
            "quantity": 1,
        }
        for i in range(12)
    ]
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 3}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "treatment",
        }
        for i in range(12)
    ]
    _write_frames(settings, events=events, recs=recs)
    pd.DataFrame({"item_id": [f"cat{i}" for i in range(100)]}).to_parquet(
        Path(settings.output.options["path"]) / "items_snapshot.parquet",
        index=False,
    )
    report = experiment_context(settings)["report"]
    assert report is not None
    assert "guardrails" in report.promote_blocked_by
    assert any("coverage" in item.failures for item in report.guardrails)


def test_clear_promotion_requires_enabled_experiment():
    settings = make_settings(feature_config_path=str(REPO_FEATURES))
    assert clear_promotion(settings) == "No experiment is enabled"


def test_experiment_context_missing_feature_config(tmp_path):
    settings = _settings(tmp_path)
    settings = make_settings(
        feature_config_path=str(tmp_path / "missing.toml"),
        input=settings.input,
        output=settings.output,
        experiment=settings.experiment,
    )
    context = experiment_context(settings)
    assert context["error"] == "No experiment variants to evaluate."
    assert promote_winner(settings, "control") == "Experiment report is not available"


def test_experiment_context_invalid_feature_config(tmp_path):
    settings = _settings(tmp_path)
    bad = tmp_path / "bad.toml"
    bad.write_text("not = toml [[[")
    settings = make_settings(
        feature_config_path=str(bad),
        input=settings.input,
        output=settings.output,
        experiment=settings.experiment,
    )
    context = experiment_context(settings)
    assert context["error"] == "No experiment variants to evaluate."


def test_experiment_context_missing_events_file(tmp_path):
    settings = _settings(tmp_path, log_exposures=False)
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ]
    ).to_parquet(Path(settings.output.options["path"]) / "recommendations.parquet", index=False)
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_db_events(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'in.db'}"
    engine = create_engine(url)
    pd.DataFrame([{"user_id": "u1", "event_type": "purchase", "quantity": 1}]).to_sql(
        "events", engine, index=False
    )
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=IOSettings(kind="db", options={"database_url": url}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            primary_metric="purchase",
            log_exposures=False,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
    )
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_db_events_missing_table(tmp_path):
    db_path = tmp_path / "empty.db"
    url = f"sqlite+pysqlite:///{db_path}"
    create_engine(url).connect().close()
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=IOSettings(kind="db", options={"database_url": url}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            log_exposures=False,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
    )
    context = experiment_context(settings)
    assert context["report"] is not None or context["error"]


def test_experiment_context_tolerates_load_failures(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _write_frames(
        settings,
        events=[{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1}],
        recs=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ],
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _boom)
    monkeypatch.setattr("cicerone.dashboard_experiments._load_metric_events", _boom)
    monkeypatch.setattr("cicerone.dashboard_experiments.load_recommendation_guardrail_rows", _boom)
    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_exposures", _boom)
    monkeypatch.setattr("cicerone.dashboard_experiments.load_items_catalog_size", _boom)
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_reuses_cached_promote_state_on_read_failure(tmp_path, monkeypatch):
    settings = _settings(tmp_path, id="exp-cache")
    _write_frames(
        settings,
        events=[{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1}],
        recs=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ],
    )
    ExperimentStore(settings.output).write_state(experiment_state("exp-cache", promoted_variant="treatment"))
    assert experiment_context(settings)["promoted_variant"] == "treatment"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _boom)
    assert experiment_context(settings)["promoted_variant"] == "treatment"


def test_experiment_context_promoted_variant_when_state_experiment_id_is_int(tmp_path, monkeypatch):
    settings = _settings(tmp_path, id="7")
    _write_frames(
        settings,
        events=[{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1}],
        recs=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ],
    )

    def _state(_self):
        return {
            "experiment_id": 7,
            "promoted_variant": "treatment",
            "promoted_at": "2026-09-02T00:00:00Z",
        }

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _state)
    assert experiment_context(settings)["promoted_variant"] == "treatment"


def test_experiment_context_recipes_from_manifest(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)
    _write_frames(
        settings,
        events=[{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1}],
        recs=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "control",
            }
        ],
    )
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {
                "experiment_variants": json.dumps(
                    [
                        {"name": "control", "traffic": 0.5, "models": ["popular"]},
                        {"name": "treatment", "traffic": 0.5, "models": ["collaborative"]},
                    ]
                )
            }

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    context = experiment_context(settings)
    assert context["error"] is None
    assert [recipe.name for recipe in context["recipes"]] == ["control", "treatment"]


def test_experiment_context_manifest_recipes_invalid_json(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {"experiment_variants": "{not-json"}

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    context = experiment_context(settings)
    assert context["error"] == "No experiment variants to evaluate."


def test_experiment_context_manifest_read_and_resolve_errors(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)

    class _BoomReader:
        def read_latest(self):
            raise RuntimeError("manifest")

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _BoomReader())
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.resolve_recipes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("recipes")),
    )
    context = experiment_context(settings)
    assert context["error"] == "No experiment variants to evaluate."


def test_experiment_context_events_query_falls_back(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=IOSettings(
            kind="db",
            options={"database_url": "sqlite+pysqlite://", "events_query": "SELECT 1"},
        ),
        output=settings.output,
        experiment=settings.experiment,
    )

    class _Source:
        def read_events(self):
            return pd.DataFrame([{"user_id": "u1", "event_type": "purchase", "quantity": 1}])

    monkeypatch.setattr("cicerone.dashboard_experiments.build_input_source", lambda _inp: _Source())
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_events_s3_missing(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)

    class _S3Missing(Exception):
        pass

    def _raise(*_args, **_kwargs):
        raise _S3Missing("missing")

    monkeypatch.setattr("cicerone.dashboard_experiments.read_parquet", _raise)
    monkeypatch.setattr("cicerone.dashboard_experiments.is_s3_not_found", lambda _exc: True)
    context = experiment_context(settings)
    assert context["report"] is not None

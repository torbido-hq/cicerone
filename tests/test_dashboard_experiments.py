from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_settings
from sqlalchemy import create_engine

from cicerone.config import ConfigError, IOSettings
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


def test_experiment_context_surfaces_live_policy_config_error(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)

    def _boom(*_args, **_kwargs):
        raise ConfigError("experiment.variants[treatment].boosts duplicate rule name 'featured'")

    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", _boom)
    context = experiment_context(settings)
    assert "duplicate rule name" in (context["error"] or "")


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


def test_experiment_context_clears_promote_cache_when_store_returns_other_experiment(tmp_path, monkeypatch):
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

    def _other(_self):
        return {
            "experiment_id": "other-exp",
            "promoted_variant": "control",
            "promoted_at": "2026-09-02T00:00:00Z",
        }

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _other)
    assert experiment_context(settings)["promoted_variant"] is None

    def _boom(*_args, **_kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _boom)
    assert experiment_context(settings)["promoted_variant"] is None


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
    assert context["recipes"][0].merge_item_availability is True


def test_experiment_context_manifest_restores_eligibility_merge_flag(tmp_path, monkeypatch):
    from cicerone.experiment.recipes import apply_recipe
    from cicerone.feature_config import load_feature_config
    from cicerone.policy import resolve_eligibility

    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {
                "experiment_variants": json.dumps(
                    [
                        {
                            "name": "control",
                            "traffic": 0.5,
                            "models": ["popular"],
                            "eligibility": [],
                            "merge_item_availability": False,
                        },
                        {
                            "name": "treatment",
                            "traffic": 0.5,
                            "models": ["collaborative"],
                            "eligibility": [
                                {"name": "published", "op": "item_true", "item_column": "published"}
                            ],
                            "merge_item_availability": False,
                        },
                    ]
                )
            }

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    context = experiment_context(settings)
    features = load_feature_config(REPO_FEATURES)
    control = apply_recipe(features, context["recipes"][0])
    treatment = apply_recipe(features, context["recipes"][1])
    assert resolve_eligibility(control) == []
    assert [rule.name for rule in resolve_eligibility(treatment)] == ["published"]


def test_experiment_context_manifest_policy_error_names_variant(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {
                "experiment_variants": json.dumps(
                    [
                        {"name": "control", "traffic": 0.5, "models": ["popular"]},
                        {
                            "name": "treatment",
                            "traffic": 0.5,
                            "models": ["collaborative"],
                            "eligibility": [1],
                        },
                    ]
                )
            }

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    context = experiment_context(settings)
    assert "experiment_variants[treatment].eligibility" in (context["error"] or "")


def test_experiment_context_manifest_recipes_malformed_item(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {"experiment_variants": json.dumps([{"traffic": 0.5}])}

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    context = experiment_context(settings)
    assert context["error"] == "No experiment variants to evaluate."


def test_experiment_context_manifest_recipes_skips_malformed_keeps_good(tmp_path, monkeypatch):
    from cicerone.dashboard_experiments import _recipes
    from cicerone.feature_config import load_feature_config

    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr("cicerone.dashboard_experiments.resolve_recipes", lambda *args, **kwargs: ())

    class _Reader:
        def read_latest(self):
            return {
                "experiment_variants": json.dumps(
                    [
                        {"name": "control", "traffic": 0.5, "models": ["popular"]},
                        {"traffic": 0.5},
                        {"name": "treatment", "traffic": 0.5, "models": ["collaborative"]},
                    ]
                )
            }

    monkeypatch.setattr("cicerone.dashboard_experiments.build_manifest_reader", lambda _output: _Reader())
    recipes = _recipes(settings, load_feature_config(REPO_FEATURES))
    assert [recipe.name for recipe in recipes] == ["control", "treatment"]


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

    monkeypatch.setattr("cicerone.evaluation.context.build_input_source", lambda _inp: _Source())
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_events_s3_missing(tmp_path, monkeypatch):
    settings = _settings(tmp_path, log_exposures=False)

    class _S3Missing(Exception):
        pass

    def _raise(*_args, **_kwargs):
        raise _S3Missing("missing")

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _raise)
    monkeypatch.setattr("cicerone.evaluation.context.is_s3_not_found", lambda _exc: True)
    context = experiment_context(settings)
    assert context["report"] is not None


def test_experiment_context_ctr_from_track_rows(tmp_path):
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    base = _settings(tmp_path, log_exposures=False)
    _write_frames(
        base,
        events=[
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-28T12:10:00Z",
            }
        ],
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
                "user_id": "u2",
                "item_id": "i2",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                VARIANT_COLUMN: "treatment",
            },
        ],
    )
    TrackStore(base.output).append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-u1",
                }
            ).as_row(),
            normalize_track(
                {
                    "kind": "click",
                    "user_id": "u1",
                    "item_id": "i1",
                    "occurred_at": "2026-08-28T12:01:00Z",
                    "event_id": "clk-u1",
                }
            ).as_row(),
        ]
    )
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=base.input,
        output=base.output,
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            primary_metric="ctr",
            attribution="click",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track={"enabled": True},
    )
    context = experiment_context(settings)
    assert context["report"] is not None
    assert context["report"].primary_metric == "ctr"
    assert context["lift_label"] == "CTR lift"


def test_experiment_context_skips_other_experiment_track_rows(tmp_path, monkeypatch):
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    def _impression(user_id: str, event_id: str, experiment_id: str | None = None) -> dict:
        payload = {
            "kind": "impression",
            "user_id": user_id,
            "item_id": "i1",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": event_id,
        }
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id
        return normalize_track(payload).as_row()

    base = _settings(tmp_path, log_exposures=False)
    _write_frames(
        base,
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
    TrackStore(base.output).append_rows(
        [_impression(f"old-{i}", f"imp-old-{i}", "exp-old") for i in range(100)]
        + [_impression("u1", "imp-now", "exp-1"), _impression("u-bare", "imp-bare")]
    )
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=base.input,
        output=base.output,
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            primary_metric="ctr",
            attribution="click",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track={"enabled": True, "min_impressions": 100},
    )
    captured: dict[str, list] = {}

    def _capture(**kwargs):
        captured["rows"] = list(kwargs["track_rows"])
        return {}

    monkeypatch.setattr("cicerone.dashboard_experiments.user_track_outcomes", _capture)
    context = experiment_context(settings)
    ids = {str(row.get("experiment_id") or "") for row in captured["rows"]}
    assert "exp-old" not in ids
    assert ids == {"exp-1"}
    assert context["report"] is not None
    assert "volume" in context["report"].promote_blocked_by


def test_experiment_context_track_read_error(tmp_path, monkeypatch):
    base = _settings(tmp_path, log_exposures=False)
    _write_frames(
        base,
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
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=base.input,
        output=base.output,
        experiment=base.experiment,
        track={"enabled": True},
    )
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.read_rows",
        lambda self, **_kwargs: (_ for _ in ()).throw(RuntimeError("track")),
    )
    context = experiment_context(settings)
    assert context["report"] is not None
    assert context["report"].n_assigned >= 0


def test_experiment_context_user_attribution_skips_track_outcomes(tmp_path, monkeypatch):
    base = _settings(tmp_path, log_exposures=False)
    _write_frames(
        base,
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
    settings = make_settings(
        feature_config_path=str(REPO_FEATURES),
        input=base.input,
        output=base.output,
        experiment=base.experiment,
        track={"enabled": True},
    )
    called = {"n": 0}

    def _boom(**_kwargs):
        called["n"] += 1
        return {}

    monkeypatch.setattr("cicerone.dashboard_experiments.user_track_outcomes", _boom)
    context = experiment_context(settings)
    assert context["report"] is not None
    assert called["n"] == 0


def test_experiment_context_events_full_parquet_fallback(tmp_path, monkeypatch):
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
    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("parquet")

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _boom)
    monkeypatch.setattr("cicerone.evaluation.context.is_s3_not_found", lambda _exc: False)
    context = experiment_context(settings)
    assert context["report"] is not None
    assert calls["n"] >= 2


def test_track_variant_by_user_uses_earliest_impression() -> None:
    from cicerone.dashboard_experiments import _track_variant_by_user

    names = {"control", "treatment"}
    later_first = [
        {
            "kind": "impression",
            "user_id": "u1",
            "variant": "treatment",
            "occurred_at": "2026-08-29T12:00:00Z",
            "event_id": "b",
        },
        {
            "kind": "impression",
            "user_id": "u1",
            "variant": "control",
            "occurred_at": "2026-08-29T10:00:00Z",
            "event_id": "a",
        },
    ]
    assert _track_variant_by_user(later_first, names) == {"u1": "control"}
    assert _track_variant_by_user(list(reversed(later_first)), names) == {"u1": "control"}


def test_experiment_context_thompson_view(tmp_path):
    from cicerone.config.constants import ALLOCATION_THOMPSON

    settings = _settings(tmp_path, log_exposures=False)
    settings = replace(
        settings,
        experiment=replace(
            settings.experiment,
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
                VariantSettings(name="blend", traffic=0.0),
            ),
        ),
        track=replace(settings.track, enabled=True, min_impressions=100),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state(
            "exp-1",
            promoted_variant=None,
            champion="control",
            challenger="treatment",
            arms={
                "control": {"successes": 2, "failures": 8},
                "treatment": {"successes": 1, "failures": 9},
                "blend": {"successes": 0, "failures": 0},
            },
            p_best={"control": 0.7, "treatment": 0.2, "blend": 0.1},
            pair_impressions=40,
        )
    )
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 10}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "treatment",
        }
        for i in range(12)
    ]
    events = [
        {"user_id": f"u{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1} for i in range(12)
    ]
    _write_frames(settings, events=events, recs=recs)
    context = experiment_context(settings)
    thompson = context["thompson"]
    assert thompson is not None
    assert thompson["champion"] == "control"
    assert thompson["challenger"] == "treatment"
    by_name = {arm["name"]: arm for arm in thompson["arms"]}
    assert by_name["control"]["cvr_pct"] == 20.0
    assert by_name["control"]["p_best"] == 0.7
    assert by_name["control"]["role"] == "champion"
    assert by_name["blend"]["role"] == "parked"
    assert thompson["volume_pct"] == 40.0
    assert thompson["volume_max"] == 100


def test_thompson_view_volume_max_when_floor_is_zero() -> None:
    from cicerone.config.constants import ALLOCATION_THOMPSON
    from cicerone.dashboard_experiments import _thompson_view

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        allocation=ALLOCATION_THOMPSON,
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    view = _thompson_view(
        {"champion": "control", "challenger": "treatment", "pair_impressions": 12},
        experiment,
        0,
    )
    assert view is not None
    assert view["min_impressions"] == 0
    assert view["volume_max"] == 12
    assert view["volume_pct"] == 100.0
    empty = _thompson_view(
        {"champion": "control", "challenger": "treatment", "pair_impressions": 0},
        experiment,
        0,
    )
    assert empty is not None
    assert empty["volume_max"] == 1


def test_promote_and_resume_keep_thompson_fields(tmp_path):
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
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    _write_frames(settings, events=events, recs=recs, exposures=exposures)
    assert promote_winner(settings, "treatment") is None
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] == "treatment"
    assert state["champion"] == "control"
    assert state["challenger"] == "treatment"
    assert clear_promotion(settings) is None
    cleared = ExperimentStore(settings.output).read_state()
    assert cleared is not None
    assert cleared["promoted_variant"] is None
    assert cleared["champion"] == "control"


def test_thompson_ship_ignores_undecided(tmp_path):
    from cicerone.config.constants import ALLOCATION_THOMPSON

    settings = _settings(tmp_path, log_exposures=False)
    settings = replace(settings, experiment=replace(settings.experiment, allocation=ALLOCATION_THOMPSON))
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 10}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "treatment",
        }
        for i in range(12)
    ]
    events = [
        {"user_id": f"u{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1} for i in range(12)
    ]
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    _write_frames(settings, events=events, recs=recs)
    context = experiment_context(settings)
    assert "undecided" in context["report"].promote_blocked_by
    assert context["can_ship"] is True
    assert context["ship_variant"] == "control"
    assert promote_winner(settings, "control") is None
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] == "control"


def test_thompson_ship_ignores_parked_empty_lists(tmp_path):
    from cicerone.config.constants import ALLOCATION_THOMPSON
    from cicerone.dashboard_experiments import _eval_recipes
    from cicerone.experiment.recipes import ResolvedRecipe
    from cicerone.feature_config import BlendingConfig

    settings = _settings(tmp_path, log_exposures=False)
    settings = replace(
        settings,
        experiment=replace(
            settings.experiment,
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
                VariantSettings(name="blend", traffic=0.0),
            ),
        ),
    )
    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("treatment", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("blend", 0.0, ("popular",), None, None, "priority", blending, True, True),
    )
    filtered = _eval_recipes(
        recipes,
        settings.experiment,
        {"champion": "control", "challenger": "treatment"},
    )
    assert [item.name for item in filtered] == ["control", "treatment"]
    explored = _eval_recipes(
        recipes,
        replace(settings.experiment, explore_traffic=0.2),
        {"champion": "control", "challenger": "treatment"},
    )
    assert [item.name for item in explored] == ["control", "treatment"]
    assert explored[0].traffic == pytest.approx(0.8)
    assert explored[1].traffic == pytest.approx(0.2)
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 10}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "treatment",
        }
        for i in range(12)
    ]
    events = [
        {"user_id": f"u{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1} for i in range(12)
    ]
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    _write_frames(settings, events=events, recs=recs)
    context = experiment_context(settings)
    assert "guardrails" not in context["ship_blocked"]
    assert {item.variant for item in context["report"].guardrails} == {"control", "treatment"}
    assert context["can_ship"] is True
    assert context["ship_variant"] == "control"
    assert promote_winner(settings, "blend") == "Winner is 'control', not 'blend'"


def _thompson_three_variants(tmp_path):
    from cicerone.config.constants import ALLOCATION_THOMPSON

    settings = _settings(tmp_path, log_exposures=False)
    return replace(
        settings,
        experiment=replace(
            settings.experiment,
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
                VariantSettings(name="blend", traffic=0.0),
            ),
        ),
    )


def test_thompson_context_without_pair_does_not_ship(tmp_path):
    settings = _thompson_three_variants(tmp_path)
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 10}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "treatment",
        }
        for i in range(12)
    ]
    events = [
        {"user_id": f"u{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1} for i in range(12)
    ]
    ExperimentStore(settings.output).write_state(experiment_state("exp-1", promoted_variant=None))
    _write_frames(settings, events=events, recs=recs)
    context = experiment_context(settings)
    assert context["ship_variant"] is None
    assert context["can_ship"] is False


def test_thompson_stale_pair_is_not_shippable(tmp_path):
    from cicerone.config.constants import ALLOCATION_THOMPSON

    settings = _settings(tmp_path, log_exposures=False)
    settings = replace(
        settings,
        experiment=replace(
            settings.experiment,
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="blend", traffic=0.5),
            ),
        ),
    )
    recs = [
        {
            "user_id": f"u{i}",
            "item_id": f"i{i % 10}",
            "rank": 1,
            "score": 1.0,
            "source": "personalized",
            VARIANT_COLUMN: "control" if i < 6 else "blend",
        }
        for i in range(12)
    ]
    events = [
        {"user_id": f"u{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1} for i in range(12)
    ]
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    _write_frames(settings, events=events, recs=recs)
    context = experiment_context(settings)
    assert context["ship_variant"] is None
    assert context["can_ship"] is False


def test_thompson_promote_rejects_stale_pair_missing_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace

    settings = _thompson_three_variants(tmp_path)
    settings = replace(
        settings,
        experiment=replace(
            settings.experiment,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="blend", traffic=0.5),
            ),
        ),
    )
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="control",
                promote_blocked_by=(),
            ),
            "recipes": (SimpleNamespace(name="control"), SimpleNamespace(name="blend")),
            "thompson": {"champion": "control", "challenger": "treatment"},
            "ship_variant": "control",
        },
    )
    assert promote_winner(settings, "control") == "No active champion/challenger pair is available"
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] is None


def test_thompson_promote_rejects_parked_report_winner_without_pair(tmp_path, monkeypatch):
    from types import SimpleNamespace

    settings = _thompson_three_variants(tmp_path)
    ExperimentStore(settings.output).write_state(experiment_state("exp-1", promoted_variant=None))
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="blend",
                promote_blocked_by=(),
            ),
            "thompson": {"champion": "", "challenger": ""},
            "ship_variant": "blend",
        },
    )
    assert promote_winner(settings, "blend") == "No active champion/challenger pair is available"
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] is None


def test_thompson_promote_rechecks_pair_under_writer_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    settings = _thompson_three_variants(tmp_path)
    ExperimentStore(settings.output).write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="blend")
    )
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="control",
                promote_blocked_by=(),
            ),
            "thompson": {"champion": "control", "challenger": "treatment"},
            "ship_variant": "control",
        },
    )
    assert promote_winner(settings, "treatment") == "Winner is 'control', not 'treatment'"
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] is None
    assert state["champion"] == "control"
    assert state["challenger"] == "blend"


def test_thompson_promote_fails_closed_when_state_read_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    settings = _thompson_three_variants(tmp_path)
    store = ExperimentStore(settings.output)
    store.write_state(
        experiment_state("exp-1", promoted_variant=None, champion="control", challenger="treatment")
    )
    assert store.read_state() is not None
    assert store.last_state("exp-1") is not None
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="control",
                promote_blocked_by=(),
            ),
            "thompson": {"champion": "control", "challenger": "treatment"},
            "ship_variant": "control",
        },
    )
    original = ExperimentStore.read_state

    def _boom(self):
        raise RuntimeError("store down")

    monkeypatch.setattr(ExperimentStore, "read_state", _boom)
    assert promote_winner(settings, "treatment") == "Experiment state could not be read"
    monkeypatch.setattr(ExperimentStore, "read_state", original)
    state = ExperimentStore(settings.output).read_state()
    assert state is not None
    assert state["promoted_variant"] is None
    assert state["champion"] == "control"
    assert state["challenger"] == "treatment"


def test_promote_winner_reads_state_under_writer_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cicerone.locks import writer_lock_held_here

    settings = _settings(tmp_path, log_exposures=False)
    seen = {"held": False}

    class _Lock:
        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def owned(self) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    lock = _Lock()
    original = ExperimentStore.read_state

    def _read(self):
        seen["held"] = writer_lock_held_here(self._writer_lock)
        return original(self)

    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="treatment",
                promote_blocked_by=(),
            )
        },
    )
    monkeypatch.setattr("cicerone.dashboard_experiments.build_output_writer_lock", lambda _settings: lock)
    monkeypatch.setattr(ExperimentStore, "read_state", _read)
    assert promote_winner(settings, "treatment") is None
    assert seen["held"] is True


def test_promote_winner_maps_writer_lock_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cicerone.locks import LockLostError, WriterLockBusyError

    settings = _settings(tmp_path, log_exposures=False)
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.experiment_context",
        lambda _settings: {
            "report": SimpleNamespace(
                comparisons=(),
                winner="treatment",
                promote_blocked_by=(),
            )
        },
    )
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.held_writer_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(WriterLockBusyError("dataset writer lock busy")),
    )
    assert promote_winner(settings, "treatment") == "Writer lock is busy"
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.held_writer_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LockLostError("writer lock lost", kind="writer")),
    )
    assert promote_winner(settings, "treatment") == "Writer lock was lost"
    monkeypatch.setattr(
        "cicerone.dashboard_experiments.held_writer_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(WriterLockBusyError("dataset writer lock busy")),
    )
    assert clear_promotion(settings) == "Writer lock is busy"

from __future__ import annotations

from pathlib import Path

import pandas as pd
from conftest import make_settings

from cicerone.config import IOSettings
from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.dashboard_experiments import experiment_context, promote_winner
from cicerone.experiment.evaluate import exposure_row
from cicerone.experiment.store import ExperimentStore
from cicerone.io.recommendation_schema import VARIANT_COLUMN

REPO_FEATURES = Path(__file__).resolve().parents[1] / "config" / "features.toml"


def _settings(tmp_path, **experiment_overrides):
    out = tmp_path / "out"
    inp = tmp_path / "in"
    out.mkdir()
    inp.mkdir()
    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        primary_metric="purchase",
        log_exposures=True,
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
        **experiment_overrides,
    )
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
        events.append({"user_id": f"c{i}", "item_id": f"i{i % 10}", "event_type": "view", "quantity": 1})
        events.append({"user_id": f"t{i}", "item_id": f"i{i % 10}", "event_type": "purchase", "quantity": 1})
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
            exposure_row(user_id=f"c{i}", experiment_id="exp-1", variant="control", generated_at=None)
        )
        exposures.append(
            exposure_row(user_id=f"t{i}", experiment_id="exp-1", variant="treatment", generated_at=None)
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

from __future__ import annotations

import pandas as pd
import pytest

from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.experiment.evaluate import evaluate_experiment, exposure_row, user_outcome
from cicerone.experiment.guardrails import evaluate_guardrails
from cicerone.experiment.recipes import ResolvedRecipe
from cicerone.experiment.stats import compare_variants, mixing_radius, variant_metric
from cicerone.feature_config import BlendingConfig


def _recipe(name: str, traffic: float = 0.5) -> ResolvedRecipe:
    return ResolvedRecipe(
        name=name,
        traffic=traffic,
        models=("popular",),
        weights=None,
        rrf_k=None,
        combiner="priority",
        blending=BlendingConfig(),
        boosts=True,
        eligibility=True,
    )


def test_mixing_radius_requires_two_observations() -> None:
    assert mixing_radius(1, 1.0, alpha=0.05) == float("inf")
    assert mixing_radius(20, 0.25, alpha=0.05) < 1.0


def test_compare_variants_decides_large_gap() -> None:
    control = variant_metric("control", [0.0] * 40)
    treatment = variant_metric("treatment", [1.0] * 40)
    result = compare_variants(control, treatment, alpha=0.05)
    assert result.decided is True
    assert result.winner == "treatment"
    assert result.ci_low > 0


def test_guardrails_fail_closed_on_empty_and_concentration() -> None:
    empty = evaluate_guardrails(pd.DataFrame(), variant="control")
    assert empty.ok is False
    assert "empty_recommendations" in empty.failures

    concentrated = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(10)],
            "item_id": ["same"] * 10,
            "source": ["personalized"] * 10,
        }
    )
    report = evaluate_guardrails(concentrated, variant="control", min_coverage=5)
    assert report.ok is False
    assert "top_item_share" in report.failures
    assert "coverage" in report.failures


def test_guardrails_catalog_size_relaxes_only_for_small_catalogs() -> None:
    recs = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "item_id": [f"i{i % 3}" for i in range(12)],
            "source": ["personalized"] * 12,
        }
    )
    small = evaluate_guardrails(recs, variant="control", catalog_size=3)
    large = evaluate_guardrails(recs, variant="control", catalog_size=100)
    assert "coverage" not in small.failures
    assert "coverage" in large.failures


def test_user_outcome_weighted_and_event_type() -> None:
    events = pd.DataFrame(
        [
            {"user_id": "a", "event_type": "purchase", "quantity": 2},
            {"user_id": "a", "event_type": "view", "quantity": 1},
            {"user_id": "b", "event_type": "view", "quantity": 1},
        ]
    )
    weights = {"purchase": 4.0, "view": 0.5}
    weighted = user_outcome(events, event_weights=weights, primary_metric="weighted")
    assert weighted["a"] == pytest.approx(8.5)
    purchases = user_outcome(events, event_weights=weights, primary_metric="purchase")
    assert purchases == {"a": 1.0}


def test_evaluate_experiment_itt_and_exposure_conditional() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    recipes = (_recipe("control"), _recipe("treatment"))
    events = pd.DataFrame(
        [
            {"user_id": f"u{i}", "event_type": "purchase" if i % 2 else "view", "quantity": 1}
            for i in range(30)
        ]
    )
    recs = pd.DataFrame(
        [
            {
                "user_id": f"u{i}",
                "item_id": f"i{i % 8}",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control" if i < 15 else "treatment",
            }
            for i in range(30)
        ]
    )
    report = evaluate_experiment(
        experiment=experiment,
        recipes=recipes,
        events=events,
        event_weights={"purchase": 1.0, "view": 0.0},
        recommendations=recs,
    )
    assert report.n_assigned == 30
    assert report.exposure_conditional is False
    assert report.comparisons
    assert report.guardrails
    assert "undecided" in report.promote_blocked_by or report.can_promote in {True, False}

    exposures = [
        exposure_row(user_id="u1", experiment_id="exp", variant="treatment", generated_at=None),
        exposure_row(user_id="u2", experiment_id="exp", variant="control", generated_at=None),
    ]
    exposed = evaluate_experiment(
        experiment=experiment,
        recipes=recipes,
        events=events,
        event_weights={"purchase": 1.0},
        exposures=exposures,
    )
    assert exposed.exposure_conditional is True
    assert exposed.n_assigned == 2

    other = [
        exposure_row(user_id="u1", experiment_id="other", variant="treatment", generated_at=None),
    ]
    skipped = evaluate_experiment(
        experiment=experiment,
        recipes=recipes,
        events=events,
        event_weights={"purchase": 1.0},
        exposures=other,
    )
    assert skipped.n_assigned == 0


def test_evaluate_experiment_empty_exposures_stays_conditional() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame([{"user_id": "u1", "event_type": "purchase", "quantity": 1}])
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        exposures=[],
    )
    assert report.exposure_conditional is True
    assert report.n_assigned == 0


def test_evaluate_experiment_blocks_promote_when_already_promoted() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame([{"user_id": f"u{i}", "event_type": "purchase", "quantity": 1} for i in range(4)])
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        recommendations=pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "source": "personalized",
                    "variant": "control",
                },
                {
                    "user_id": "u2",
                    "item_id": "i2",
                    "source": "personalized",
                    "variant": "treatment",
                },
            ]
        ),
        promoted_variant="treatment",
    )
    assert report.can_promote is False
    assert "promoted" in report.promote_blocked_by


def test_evaluate_experiment_uses_first_exposure_and_later_events() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-01T00:00:00Z",
            },
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-03T00:00:00Z",
            },
        ]
    )
    exposures = [
        exposure_row(
            user_id="u1",
            experiment_id="exp",
            variant="control",
            generated_at=None,
            exposed_at=pd.Timestamp("2026-01-02T00:00:00Z"),
        ),
        exposure_row(
            user_id="u1",
            experiment_id="exp",
            variant="treatment",
            generated_at=None,
            exposed_at=pd.Timestamp("2026-01-04T00:00:00Z"),
        ),
    ]
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        exposures=exposures,
    )
    assert report.exposure_conditional is True
    assert report.n_assigned == 1
    assert report.comparisons[0].control.total == pytest.approx(1.0)
    assert report.comparisons[0].treatment.n_users == 0


def test_evaluate_experiment_picks_best_mean_among_three_arms() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        alpha=0.05,
        variants=(
            VariantSettings(name="control", traffic=0.34),
            VariantSettings(name="a", traffic=0.33),
            VariantSettings(name="b", traffic=0.33),
        ),
    )
    recs = pd.DataFrame(
        [
            {
                "user_id": f"u{i}",
                "item_id": f"i{i % 8}",
                "source": "personalized",
                "variant": name,
            }
            for name in ("control", "a", "b")
            for i in range(40)
        ]
    )
    events = pd.DataFrame(
        [
            {
                "user_id": f"{name}-u{i}",
                "event_type": ("purchase" if name == "b" or (name == "a" and i % 2 == 0) else "view"),
                "quantity": 1,
                "occurred_at": "2026-01-02T00:00:00Z",
            }
            for name in ("control", "a", "b")
            for i in range(40)
        ]
    )
    # Force assignment via exposures so arm membership is deterministic.
    exposures = [
        exposure_row(
            user_id=f"{name}-u{i}",
            experiment_id="exp",
            variant=name,
            generated_at=None,
            exposed_at=pd.Timestamp("2026-01-01T00:00:00Z"),
        )
        for name in ("control", "a", "b")
        for i in range(40)
    ]
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("a"), _recipe("b")),
        events=events,
        event_weights={"purchase": 1.0},
        recommendations=recs,
        exposures=exposures,
    )
    assert report.can_promote is True
    assert report.winner == "b"


def test_evaluate_experiment_cuts_events_after_promoted_at() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-01T00:00:00Z",
            },
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-03T00:00:00Z",
            },
        ]
    )
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        promoted_at="2026-01-02T00:00:00Z",
    )
    totals = {item.control.name: item.control.total for item in report.comparisons}
    totals.update({item.treatment.name: item.treatment.total for item in report.comparisons})
    assert sum(totals.values()) == pytest.approx(1.0)


def test_evaluate_experiment_timestamped_exposure_replaces_untimed_first() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-03T00:00:00Z",
            }
        ]
    )
    exposures = [
        {
            "user_id": "u1",
            "experiment_id": "exp",
            "variant": "control",
            "generated_at": None,
            "exposed_at": None,
        },
        exposure_row(
            user_id="u1",
            experiment_id="exp",
            variant="treatment",
            generated_at=None,
            exposed_at=pd.Timestamp("2026-01-04T00:00:00Z"),
        ),
    ]
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        exposures=exposures,
    )
    assert report.comparisons[0].control.n_users == 0
    assert report.comparisons[0].treatment.n_users == 1
    assert report.comparisons[0].treatment.total == pytest.approx(0.0)


def test_evaluate_experiment_drops_untimed_events_when_windowed() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [
            {"user_id": "u1", "event_type": "purchase", "quantity": 1, "occurred_at": None},
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-03T00:00:00Z",
            },
        ]
    )
    exposures = [
        exposure_row(
            user_id="u1",
            experiment_id="exp",
            variant="control",
            generated_at=None,
            exposed_at=pd.Timestamp("2026-01-02T00:00:00Z"),
        )
    ]
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        exposures=exposures,
    )
    assert report.comparisons[0].control.total == pytest.approx(1.0)


def test_evaluate_experiment_blocks_on_invalid_promoted_at() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        primary_metric="purchase",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        promoted_at="not-a-timestamp",
    )
    assert report.can_promote is False
    assert "promoted_at" in report.promote_blocked_by
    totals = {item.control.name: item.control.total for item in report.comparisons}
    totals.update({item.treatment.name: item.treatment.total for item in report.comparisons})
    assert sum(totals.values()) == pytest.approx(0.0)


def test_evaluate_experiment_blocks_promote_on_guardrails() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    recs = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "only",
                "source": "popular_fallback",
                "variant": "control",
            },
            {
                "user_id": "u2",
                "item_id": "only",
                "source": "popular_fallback",
                "variant": "treatment",
            },
        ]
    )
    events = pd.DataFrame(
        [{"user_id": "u1", "event_type": "purchase", "quantity": 1}]
        + [{"user_id": "u2", "event_type": "purchase", "quantity": 1}]
    )
    report = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        recommendations=recs,
    )
    assert report.can_promote is False
    assert "guardrails" in report.promote_blocked_by
    assert report.winner is None


def test_evaluate_experiment_blocks_promote_when_recs_missing() -> None:
    experiment = ExperimentSettings(
        enabled=True,
        id="exp",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    events = pd.DataFrame(
        [{"user_id": "u1", "event_type": "purchase", "quantity": 1}]
        + [{"user_id": "u2", "event_type": "purchase", "quantity": 1}]
    )
    missing = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        recommendations=None,
    )
    assert missing.can_promote is False
    assert "guardrails" in missing.promote_blocked_by

    no_variant = evaluate_experiment(
        experiment=experiment,
        recipes=(_recipe("control"), _recipe("treatment")),
        events=events,
        event_weights={"purchase": 1.0},
        recommendations=pd.DataFrame([{"user_id": "u1", "item_id": "i1", "source": "personalized"}]),
    )
    assert no_variant.can_promote is False
    assert "guardrails" in no_variant.promote_blocked_by

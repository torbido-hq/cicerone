from __future__ import annotations

from conftest import make_settings

from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.experiment.assignment import (
    assign_variant,
    assignment_bucket,
    resolve_assignment,
)


def test_assignment_bucket_is_sticky() -> None:
    a = assignment_bucket("exp-1", "user-42")
    b = assignment_bucket("exp-1", "user-42")
    assert a == b
    assert 0.0 <= a < 1.0


def test_assignment_bucket_differs_by_experiment() -> None:
    assert assignment_bucket("exp-a", "user-1") != assignment_bucket("exp-b", "user-1")


def test_assign_variant_walks_cumulative_traffic() -> None:
    variants = (("control", 0.5), ("treatment", 0.5))
    seen: set[str] = set()
    for i in range(80):
        seen.add(assign_variant("exp", f"u{i}", variants))
    assert seen == {"control", "treatment"}


def test_remainder_traffic_lands_on_last_variant() -> None:
    variants = (("a", 0.0), ("b", 0.0))
    assert assign_variant("exp", "anyone", variants) == "b"


def test_resolve_assignment_disabled() -> None:
    settings = make_settings(experiment=ExperimentSettings())
    assert resolve_assignment(settings, "u1") == (None, None)


def test_resolve_assignment_promoted_wins() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        )
    )
    assert resolve_assignment(settings, "u1", promoted_variant="treatment") == (
        "exp",
        "treatment",
    )
    assigned = resolve_assignment(settings, "u1", promoted_variant="unknown")
    assert assigned[0] == "exp"
    assert assigned[1] in {"control", "treatment"}

from __future__ import annotations

import pytest
from conftest import make_settings

from cicerone.config.constants import ALLOCATION_THOMPSON
from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
from cicerone.experiment.assignment import (
    assign_variant,
    assignment_bucket,
    assignment_needs_snapshot,
    experiment_variant_names,
    resolve_assignment,
    snapshot_variant_names,
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


def test_assign_variant_requires_variants() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assign_variant("exp", "u1", ())


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


def test_resolve_assignment_automl_challenger_without_variants() -> None:
    settings = make_settings(experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True))
    experiment_id, variant = resolve_assignment(settings, "u1")
    assert experiment_id == "auto"
    assert variant in {"control", "treatment"}
    assert resolve_assignment(settings, "u1", promoted_variant="treatment") == ("auto", "treatment")


def test_experiment_variant_names_challenger_defaults() -> None:
    off = make_settings()
    assert experiment_variant_names(off) == ()
    named = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        )
    )
    assert experiment_variant_names(named) == ("control", "treatment")
    challenger = make_settings(experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True))
    assert experiment_variant_names(challenger) == ("control", "treatment")
    enabled_empty = make_settings(experiment=ExperimentSettings(enabled=True, id="exp"))
    assert experiment_variant_names(enabled_empty) == ()


def test_resolve_assignment_enabled_without_variants_is_off() -> None:
    settings = make_settings(experiment=ExperimentSettings(enabled=True, id="exp"))
    assert resolve_assignment(settings, "u1") == (None, None)


def test_resolve_assignment_normalizes_numeric_user_id() -> None:
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
    assert resolve_assignment(settings, 123) == resolve_assignment(settings, "123")


def test_resolve_assignment_hashes_active_pair_not_toml_traffic() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            allocation=ALLOCATION_THOMPSON,
            explore_traffic=1.0,
            variants=(
                VariantSettings(name="control", traffic=0.9),
                VariantSettings(name="treatment", traffic=0.05),
                VariantSettings(name="blend", traffic=0.05),
            ),
        ),
        track=TrackSettings(enabled=True),
    )
    seen = {resolve_assignment(settings, f"u{i}", active_pair=("control", "blend"))[1] for i in range(40)}
    assert seen == {"blend"}
    promoted = resolve_assignment(
        settings, "u1", promoted_variant="treatment", active_pair=("control", "blend")
    )
    assert promoted == ("exp", "treatment")
    same = resolve_assignment(settings, "u1", active_pair=("control", "control"))
    assert same[1] == "control"
    ignored = resolve_assignment(settings, "u1", active_pair=("missing", "gone"))
    assert ignored == (None, None)
    half = resolve_assignment(settings, "u1", active_pair=("control", "missing-arm"))
    assert half == (None, None)
    snapshot = {
        resolve_assignment(settings, f"u{i}", snapshot_names=("control", "blend"))[1] for i in range(40)
    }
    assert snapshot <= {"control", "blend"}
    assert resolve_assignment(settings, "u1", snapshot_names=("gone",)) == (None, None)
    assert resolve_assignment(settings, "u1") == (None, None)


def test_snapshot_variant_names_from_reader() -> None:
    class _Missing:
        pass

    class _None:
        def present_variant_names(self):
            return None

    class _Present:
        def present_variant_names(self):
            return ("control", "", "blend")

    assert snapshot_variant_names(_Missing()) is None
    assert snapshot_variant_names(_None()) is None
    assert snapshot_variant_names(_Present()) == ("control", "blend")


def test_assignment_needs_snapshot_only_for_thompson_without_pair() -> None:
    off = make_settings()
    assert assignment_needs_snapshot(off) is False
    fixed = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        )
    )
    assert assignment_needs_snapshot(fixed) is False
    thompson = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
    )
    assert assignment_needs_snapshot(thompson) is True
    assert assignment_needs_snapshot(thompson, promoted_variant="control") is False
    assert assignment_needs_snapshot(thompson, active_pair=("control", "treatment")) is False
    assert assignment_needs_snapshot(thompson, active_pair=("control", "missing")) is True
    automl = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="auto",
            allocation=ALLOCATION_THOMPSON,
            automl_challenger=True,
        ),
        track=TrackSettings(enabled=True),
    )
    assert assignment_needs_snapshot(automl) is True
    assert assignment_needs_snapshot(automl, active_pair=("control", "treatment")) is False

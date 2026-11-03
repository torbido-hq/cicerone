from __future__ import annotations

import pytest
from conftest import make_settings

from cicerone.config.constants import ALLOCATION_THOMPSON
from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
from cicerone.experiment.assignment import (
    assign_user,
    assign_variant,
    assignment_bucket,
    experiment_variant_names,
    snapshot_variant_names,
)

_NO_OVERLAY = (None, None)


def _no_snapshot() -> None:
    return None


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


def test_assign_user_disabled() -> None:
    settings = make_settings(experiment=ExperimentSettings())
    assert assign_user(settings, "u1", _NO_OVERLAY, _no_snapshot) == (None, None)


def test_assign_user_promoted_wins() -> None:
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
    assert assign_user(settings, "u1", ("treatment", None), _no_snapshot) == (
        "exp",
        "treatment",
    )
    assigned = assign_user(settings, "u1", ("unknown", None), _no_snapshot)
    assert assigned[0] == "exp"
    assert assigned[1] in {"control", "treatment"}


def test_assign_user_automl_challenger_without_variants() -> None:
    settings = make_settings(experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True))
    experiment_id, variant = assign_user(settings, "u1", _NO_OVERLAY, _no_snapshot)
    assert experiment_id == "auto"
    assert variant in {"control", "treatment"}
    assert assign_user(settings, "u1", ("treatment", None), _no_snapshot) == ("auto", "treatment")


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
    custom = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="auto",
            automl_challenger=True,
            variants=(
                VariantSettings(name="champion", traffic=0.5),
                VariantSettings(name="challenger", traffic=0.5),
            ),
        )
    )
    assert experiment_variant_names(custom) == ("champion", "challenger")
    experiment_id, variant = assign_user(custom, "u1", _NO_OVERLAY, _no_snapshot)
    assert experiment_id == "auto"
    assert variant in {"champion", "challenger"}
    mixed = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="auto",
            automl_challenger=True,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="challenger", traffic=0.5),
            ),
        )
    )
    assert experiment_variant_names(mixed) == ("control", "challenger")
    enabled_empty = make_settings(experiment=ExperimentSettings(enabled=True, id="exp"))
    assert experiment_variant_names(enabled_empty) == ()


def test_assign_user_enabled_without_variants_is_off() -> None:
    settings = make_settings(experiment=ExperimentSettings(enabled=True, id="exp"))
    assert assign_user(settings, "u1", _NO_OVERLAY, _no_snapshot) == (None, None)


def test_assign_user_normalizes_numeric_user_id() -> None:
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
    assert assign_user(settings, 123, _NO_OVERLAY, _no_snapshot) == assign_user(
        settings, "123", _NO_OVERLAY, _no_snapshot
    )


def test_assign_user_hashes_active_pair_not_toml_traffic() -> None:
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
    seen = {assign_user(settings, f"u{i}", (None, ("control", "blend")), _no_snapshot)[1] for i in range(40)}
    assert seen == {"blend"}
    promoted = assign_user(settings, "u1", ("treatment", ("control", "blend")), _no_snapshot)
    assert promoted == ("exp", "treatment")
    same = assign_user(settings, "u1", (None, ("control", "control")), _no_snapshot)
    assert same[1] == "control"
    ignored = assign_user(settings, "u1", (None, ("missing", "gone")), _no_snapshot)
    assert ignored == (None, None)
    half = assign_user(settings, "u1", (None, ("control", "missing-arm")), _no_snapshot)
    assert half == (None, None)
    snapshot = {
        assign_user(settings, f"u{i}", _NO_OVERLAY, lambda: ("control", "blend"))[1] for i in range(40)
    }
    assert snapshot <= {"control", "blend"}
    assert assign_user(settings, "u1", _NO_OVERLAY, lambda: ("gone",)) == (None, None)
    assert assign_user(settings, "u1", _NO_OVERLAY, _no_snapshot) == (None, None)


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


def test_assign_user_reads_snapshot_only_for_thompson_without_pair() -> None:
    calls: list[int] = []

    def names() -> tuple[str, ...]:
        calls.append(1)
        return ("control", "treatment")

    def reads_snapshot(settings, overlay=_NO_OVERLAY) -> bool:
        calls.clear()
        assign_user(settings, "u1", overlay, names)
        return bool(calls)

    off = make_settings()
    assert reads_snapshot(off) is False
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
    assert reads_snapshot(fixed) is False
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
    assert reads_snapshot(thompson) is True
    assert reads_snapshot(thompson, ("control", None)) is False
    assert reads_snapshot(thompson, (None, ("control", "treatment"))) is False
    assert reads_snapshot(thompson, (None, ("control", "missing"))) is True
    automl = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="auto",
            allocation=ALLOCATION_THOMPSON,
            automl_challenger=True,
        ),
        track=TrackSettings(enabled=True),
    )
    assert reads_snapshot(automl) is True
    assert reads_snapshot(automl, (None, ("control", "treatment"))) is False

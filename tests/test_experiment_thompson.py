from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cicerone.config import ConfigError
from cicerone.evaluation.metrics import SliceMetrics
from cicerone.experiment.recipes import ResolvedRecipe
from cicerone.experiment.thompson import (
    ArmCounts,
    _fit_mab,
    allocate_thompson,
    p_best,
    parse_arm_counts,
    pick_champion_name,
    require_bandits_extra,
    sample_arm,
    select_active_recipes,
    track_rows_since,
    window_trials_from_slices,
)
from cicerone.feature_config import BlendingConfig

pytest.importorskip("mabwiser")


def _slice(*, impressions: int, conversions: int) -> SliceMetrics:
    n_imp = impressions
    n_conv = min(conversions, impressions)
    rate = n_conv / n_imp if n_imp else 0.0
    return SliceMetrics(n_imp, 0, n_conv, n_conv, 0.0, rate, rate, 1)


def test_require_bandits_extra_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: False)
    with pytest.raises(ConfigError, match="bandits"):
        require_bandits_extra()


def test_fit_mab_sets_beta_counts_without_expanding_trials() -> None:
    mab = _fit_mab(
        ["control", "blend"],
        {"control": ArmCounts(80, 2), "blend": ArmCounts(1, 80)},
        seed=0,
    )
    policy = mab._imp
    assert policy.arm_to_success_count["control"] == 81
    assert policy.arm_to_fail_count["control"] == 3
    assert policy.arm_to_success_count["blend"] == 2
    assert policy.arm_to_fail_count["blend"] == 81


def test_window_trials_from_slices_uses_click_conversions() -> None:
    trials = window_trials_from_slices(
        {
            "control": _slice(impressions=10, conversions=4),
            "parked": _slice(impressions=99, conversions=50),
        },
        attribution="click",
        names=["control", "treatment"],
    )
    assert trials["control"] == ArmCounts(4, 6)
    assert trials["treatment"] == ArmCounts(0, 0)
    assert "parked" not in trials


def test_track_rows_since_filters_occurred_at() -> None:
    rows = [
        {"occurred_at": "2026-09-01T00:00:00Z", "kind": "impression"},
        {"occurred_at": "2026-09-03T00:00:00Z", "kind": "impression"},
    ]
    kept = track_rows_since(rows, "2026-09-02T00:00:00Z")
    assert [row["occurred_at"] for row in kept] == ["2026-09-03T00:00:00Z"]


def test_parse_arm_counts_ignores_junk() -> None:
    assert parse_arm_counts(None) == {}
    parsed = parse_arm_counts({"control": {"successes": 2, "failures": 3}, "x": "nope"})
    assert parsed["control"] == ArmCounts(2, 3)
    assert "x" not in parsed


def test_allocate_thompson_does_not_rotate_under_volume() -> None:
    result = allocate_thompson(
        names=["control", "blend", "rrf"],
        previous={
            "champion": "control",
            "challenger": "blend",
            "arms": {
                "control": {"successes": 8, "failures": 2},
                "blend": {"successes": 1, "failures": 9},
                "rrf": {"successes": 0, "failures": 0},
            },
            "pair_impressions": 5,
        },
        window_trials={
            "control": ArmCounts(1, 0),
            "blend": ArmCounts(0, 1),
            "rrf": ArmCounts(50, 0),
        },
        min_impressions=1000,
        rotate_min_prob=0.5,
        seed=0,
        draws=50,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.champion == "control"
    assert result.challenger == "blend"
    assert result.rotated is False
    assert result.arms["rrf"] == ArmCounts(0, 0)
    assert result.pair_impressions == 7


def test_allocate_thompson_does_not_mix_parked_window_trials() -> None:
    result = allocate_thompson(
        names=["control", "a", "b"],
        previous={
            "champion": "control",
            "challenger": "a",
            "arms": {
                "control": {"successes": 40, "failures": 5},
                "a": {"successes": 1, "failures": 40},
                "b": {"successes": 2, "failures": 40},
            },
            "pair_impressions": 80,
        },
        window_trials={"control": ArmCounts(5, 0), "a": ArmCounts(0, 5), "b": ArmCounts(90, 0)},
        min_impressions=10,
        rotate_min_prob=0.5,
        seed=1,
        draws=80,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.arms["b"].successes == 2
    assert result.champion == "control"


def test_allocate_thompson_promote_locks_champion() -> None:
    result = allocate_thompson(
        names=["control", "treatment", "blend"],
        previous={"champion": "control", "challenger": "blend", "pair_impressions": 500},
        window_trials={"control": ArmCounts(10, 0), "blend": ArmCounts(0, 10)},
        min_impressions=1,
        rotate_min_prob=0.01,
        promoted_variant="treatment",
        seed=0,
        draws=20,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.champion == "treatment"
    assert result.rotated is False
    assert result.challenger == "blend"


def test_allocate_thompson_rotates_when_champion_is_best() -> None:
    result = allocate_thompson(
        names=["control", "a", "b"],
        previous={
            "champion": "control",
            "challenger": "a",
            "arms": {
                "control": {"successes": 80, "failures": 2},
                "a": {"successes": 1, "failures": 80},
                "b": {"successes": 3, "failures": 80},
            },
            "pair_impressions": 200,
        },
        window_trials={"control": ArmCounts(20, 0), "a": ArmCounts(0, 20)},
        min_impressions=10,
        rotate_min_prob=0.5,
        seed=0,
        draws=100,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.champion == "control"
    assert result.p_best["control"] >= 0.5
    if result.rotated:
        assert result.challenger != "a"
        assert result.pair_impressions == 0


def test_select_active_recipes_rewrites_traffic() -> None:
    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.3, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("blend", 0.3, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("rrf", 0.4, ("popular",), None, None, "rrf", blending, True, True),
    )
    selected = select_active_recipes(recipes, champion="control", challenger="rrf", explore_traffic=0.4)
    assert [item.name for item in selected] == ["control", "rrf"]
    assert selected[0].traffic == pytest.approx(0.6)
    assert selected[1].traffic == pytest.approx(0.4)


def test_pick_champion_and_helpers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pick_champion_name(())
    assert pick_champion_name(["blend", "rrf"]) == "blend"
    assert pick_champion_name(["blend", "control"], promoted_variant="blend") == "blend"
    assert p_best((), {}) == {}
    with pytest.raises(ValueError, match="at least one arm"):
        sample_arm((), {})
    assert sample_arm(["only"], {}) == "only"
    blending = BlendingConfig(enabled=False)
    recipes = (
        ResolvedRecipe("control", 0.5, ("popular",), None, None, "priority", blending, True, True),
        ResolvedRecipe("blend", 0.5, ("popular",), None, None, "priority", blending, True, True),
    )
    missing = select_active_recipes(recipes, champion="nope", challenger="blend", explore_traffic=0.5)
    assert missing is recipes
    solo = select_active_recipes(recipes, champion="control", challenger="control", explore_traffic=0.5)
    assert [item.name for item in solo] == ["control"]
    assert solo[0].traffic == pytest.approx(1.0)
    kept = track_rows_since([{"occurred_at": "2026-09-01T00:00:00Z"}], "not-a-time")
    assert len(kept) == 1
    with pytest.raises(ValueError, match="at least two"):
        allocate_thompson(names=["only"])


def test_allocate_thompson_first_run_seeds_all_arms() -> None:
    result = allocate_thompson(
        names=["blend", "rrf"],
        window_trials={"blend": ArmCounts(3, 1), "rrf": ArmCounts(1, 3)},
        min_impressions=100,
        seed=0,
        draws=20,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.champion == "blend"
    assert result.challenger in {"blend", "rrf"}
    assert result.challenger != result.champion
    assert result.arms["blend"] == ArmCounts(3, 1)
    assert result.pair_impressions == 0


def test_allocate_thompson_other_name_when_promoted_was_challenger() -> None:
    result = allocate_thompson(
        names=["control", "treatment"],
        previous={"champion": "control", "challenger": "treatment", "pair_impressions": 10},
        promoted_variant="treatment",
        min_impressions=1,
        seed=0,
        draws=10,
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )
    assert result.champion == "treatment"
    assert result.challenger == "control"

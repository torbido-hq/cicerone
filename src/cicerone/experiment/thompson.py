"""Job-time Thompson sampling (MABWiser) for champion/challenger recipes."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cicerone.config.constants import (
    ALLOCATION_THOMPSON,
    BANDITS_EXTRA_HINT,
    THOMPSON_P_BEST_DRAWS,
    ConfigError,
)
from cicerone.evaluation.metrics import OCCURRED_AT
from cicerone.experiment.recipes import CONTROL_NAME, ResolvedRecipe
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN, VARIANT_COLUMN
from cicerone.track.normalize import assign_missing_event_id

_EMPTY = np.array([], dtype=object)
_EMPTY_REWARDS = np.array([], dtype=float)
COUNTED_IMPRESSIONS = "counted_impressions"
COUNTED_CONVERSIONS = "counted_conversions"


def bandits_extra_available() -> bool:
    return importlib.util.find_spec("mabwiser") is not None


def require_bandits_extra() -> None:
    if not bandits_extra_available():
        raise ConfigError(
            f"experiment.allocation = {ALLOCATION_THOMPSON!r} requires MABWiser; {BANDITS_EXTRA_HINT}"
        )


@dataclass(frozen=True)
class ArmCounts:
    successes: int
    failures: int

    @property
    def impressions(self) -> int:
        return self.successes + self.failures


@dataclass(frozen=True)
class ThompsonAllocation:
    champion: str
    challenger: str
    arms: dict[str, ArmCounts]
    p_best: dict[str, float]
    pair_impressions: int
    window_started_at: str
    rotated: bool

    def as_state(self) -> dict[str, Any]:
        return {
            "allocation": ALLOCATION_THOMPSON,
            "champion": self.champion,
            "challenger": self.challenger,
            "arms": {
                name: {"successes": item.successes, "failures": item.failures}
                for name, item in self.arms.items()
            },
            "p_best": dict(self.p_best),
            "pair_impressions": self.pair_impressions,
            "window_started_at": self.window_started_at,
        }


def parse_arm_counts(raw: object) -> dict[str, ArmCounts]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, ArmCounts] = {}
    for key, value in raw.items():
        name = str(key)
        if not name or not isinstance(value, Mapping):
            continue
        successes = max(0, int(value.get("successes") or 0))
        failures = max(0, int(value.get("failures") or 0))
        out[name] = ArmCounts(successes=successes, failures=failures)
    return out


@dataclass(frozen=True)
class CountedEvents:
    """Event ids already in the arm counts, kept for the overlap lookback."""

    impressions: dict[str, tuple[str, bool]] = field(default_factory=dict)
    conversions: dict[str, str] = field(default_factory=dict)

    def as_state(self) -> dict[str, Any]:
        return {
            COUNTED_IMPRESSIONS: {key: [at, converted] for key, (at, converted) in self.impressions.items()},
            COUNTED_CONVERSIONS: dict(self.conversions),
        }


def parse_counted_events(state: Mapping[str, Any] | None) -> CountedEvents | None:
    if not state or not isinstance(state.get(COUNTED_IMPRESSIONS), Mapping):
        return None
    impressions: dict[str, tuple[str, bool]] = {}
    for key, value in state[COUNTED_IMPRESSIONS].items():
        if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
            impressions[str(key)] = (str(value[0]), bool(value[1]))
    raw = state.get(COUNTED_CONVERSIONS)
    conversions = {str(key): str(value) for key, value in raw.items()} if isinstance(raw, Mapping) else {}
    return CountedEvents(impressions, conversions)


def _column(frame: pd.DataFrame, name: str) -> list[Any]:
    return list(frame[name]) if name in frame.columns else [None] * len(frame)


def _conversion_keys(frame: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for event_id, user, item, when, kind in zip(
        _column(frame, "event_id"),
        _column(frame, USER_COLUMN),
        _column(frame, ITEM_COLUMN),
        _column(frame, OCCURRED_AT),
        _column(frame, "event_type"),
        strict=True,
    ):
        if event_id is not None and not pd.isna(event_id) and str(event_id).strip():
            keys.append(str(event_id))
        else:
            keys.append(json.dumps([str(user), str(item), pd.Timestamp(when).isoformat(), str(kind)]))
    return keys


def window_trials_from_outcomes(
    impressions: pd.DataFrame,
    attributed: pd.DataFrame,
    *,
    names: Sequence[str],
    counted: CountedEvents,
    keep_since: datetime,
) -> tuple[dict[str, ArmCounts], CountedEvents]:
    """New trials per arm, counting each impression and each conversion once.

    An impression already counted as a failure that now has an uncounted
    conversion turns into a success (+1 success, -1 failure).
    """
    seen = dict(counted.impressions)
    used = dict(counted.conversions)
    first: dict[str, tuple[str, str]] = {}
    if not attributed.empty:
        ordered = attributed.sort_values(OCCURRED_AT, kind="mergesort")
        for prior, key, when in zip(
            ordered["prior_event_id"], _conversion_keys(ordered), ordered[OCCURRED_AT], strict=True
        ):
            if not pd.isna(prior) and key not in used:
                first.setdefault(str(prior), (key, pd.Timestamp(when).isoformat()))
    wanted = set(names)
    successes = dict.fromkeys(wanted, 0)
    failures = dict.fromkeys(wanted, 0)
    for event_id, variant, when in zip(
        _column(impressions, "event_id"),
        _column(impressions, VARIANT_COLUMN),
        _column(impressions, OCCURRED_AT),
        strict=True,
    ):
        ident = str(event_id)
        hit = first.get(ident)
        before = seen.get(ident)
        if before is not None and (before[1] or hit is None):
            continue
        seen[ident] = (pd.Timestamp(when).isoformat(), hit is not None)
        if hit is not None:
            used[hit[0]] = hit[1]
        name = str(variant)
        if name not in wanted:
            continue
        if hit is not None:
            successes[name] += 1
        if before is not None:
            failures[name] -= 1
        elif hit is None:
            failures[name] += 1
    cutoff = pd.Timestamp(keep_since)

    def recent(at: str) -> bool:
        stamp = pd.to_datetime(at, utc=True, errors="coerce")
        return not pd.isna(stamp) and stamp >= cutoff

    kept = CountedEvents(
        {key: value for key, value in seen.items() if recent(value[0])},
        {key: at for key, at in used.items() if recent(at)},
    )
    trials = {name: ArmCounts(successes=successes[name], failures=failures[name]) for name in wanted}
    return trials, kept


def track_rows_since(
    rows: Sequence[Mapping[str, Any]],
    started_at: str | None,
    *,
    until: datetime,
    lookback: timedelta = timedelta(0),
) -> list[dict[str, Any]]:
    end = pd.Timestamp(until)
    stamp = pd.to_datetime(started_at, utc=True, errors="coerce") if started_at else None
    if stamp is not None and pd.isna(stamp):
        return []
    if stamp is not None:
        stamp -= lookback
    out: list[dict[str, Any]] = []
    for row in rows:
        when = pd.to_datetime(row.get("occurred_at"), utc=True, errors="coerce")
        if pd.isna(when) or when >= end:
            continue
        if stamp is None or when >= stamp:
            out.append(assign_missing_event_id(row))
    return out


def add_counts(base: ArmCounts, extra: ArmCounts) -> ArmCounts:
    return ArmCounts(
        successes=base.successes + extra.successes,
        failures=max(0, base.failures + extra.failures),
    )


def pick_champion_name(names: Sequence[str], promoted_variant: str | None = None) -> str:
    if not names:
        raise ValueError("pick_champion_name requires at least one name")
    if promoted_variant is not None and promoted_variant in names:
        return promoted_variant
    if CONTROL_NAME in names:
        return CONTROL_NAME
    return str(names[0])


def _fit_mab(
    arms: Sequence[str],
    counts: Mapping[str, ArmCounts],
    *,
    seed: int | None,
) -> Any:
    from mabwiser.mab import MAB, LearningPolicy

    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = int(seed)
    mab = MAB(list(arms), LearningPolicy.ThompsonSampling(), **kwargs)
    mab.fit(_EMPTY, _EMPTY_REWARDS)
    policy = mab._imp
    for name in arms:
        arm = counts.get(name, ArmCounts(0, 0))
        # Beta(1+s, 1+f) without expanding one trial per impression.
        policy.arm_to_success_count[name] = 1 + arm.successes
        policy.arm_to_fail_count[name] = 1 + arm.failures
    return mab


def p_best(
    arms: Sequence[str],
    counts: Mapping[str, ArmCounts],
    *,
    draws: int = THOMPSON_P_BEST_DRAWS,
    seed: int | None = None,
) -> dict[str, float]:
    require_bandits_extra()
    if not arms:
        return {}
    n = max(1, int(draws))
    mab = _fit_mab(arms, counts, seed=seed)
    tallies = Counter(str(mab.predict()) for _ in range(n))
    return {name: float(tallies.get(name, 0)) / float(n) for name in arms}


def sample_arm(
    arms: Sequence[str],
    counts: Mapping[str, ArmCounts],
    *,
    seed: int | None = None,
) -> str:
    require_bandits_extra()
    if not arms:
        raise ValueError("sample_arm requires at least one arm")
    if len(arms) == 1:
        return str(arms[0])
    mab = _fit_mab(arms, counts, seed=seed)
    return str(mab.predict())


def select_active_recipes(
    recipes: tuple[ResolvedRecipe, ...],
    *,
    champion: str,
    challenger: str,
    explore_traffic: float,
) -> tuple[ResolvedRecipe, ...]:
    by_name = {recipe.name: recipe for recipe in recipes}
    if champion not in by_name:
        return recipes
    share = float(explore_traffic)
    if challenger not in by_name or challenger == champion:
        return (replace_traffic(by_name[champion], 1.0),)
    champ_share = max(0.0, 1.0 - share)
    return (
        replace_traffic(by_name[champion], champ_share),
        replace_traffic(by_name[challenger], share),
    )


def replace_traffic(recipe: ResolvedRecipe, traffic: float) -> ResolvedRecipe:
    return replace(recipe, traffic=float(traffic))


def allocate_thompson(
    *,
    names: Sequence[str],
    previous: Mapping[str, Any] | None = None,
    window_trials: Mapping[str, ArmCounts] | None = None,
    min_impressions: int = 0,
    rotate_min_prob: float = 0.9,
    promoted_variant: str | None = None,
    guardrails_ok: bool = True,
    now: datetime | None = None,
    draws: int = THOMPSON_P_BEST_DRAWS,
    seed: int | None = None,
) -> ThompsonAllocation:
    require_bandits_extra()
    labels = [str(name) for name in names if str(name)]
    if len(labels) < 2:
        raise ValueError("allocate_thompson requires at least two variant names")
    stamp = (now or datetime.now(UTC)).isoformat()
    prev = dict(previous or {})
    stored = parse_arm_counts(prev.get("arms"))
    window = dict(window_trials or {})
    prev_champion = str(prev.get("champion") or "")
    prev_challenger = str(prev.get("challenger") or "")
    pair_ok = prev_champion in labels and prev_challenger in labels
    champion = pick_champion_name(labels, promoted_variant or (prev_champion if pair_ok else None))
    arms = {name: stored.get(name, ArmCounts(0, 0)) for name in labels}
    if pair_ok:
        window_impressions = 0
        for name in (prev_champion, prev_challenger):
            extra = window.get(name, ArmCounts(0, 0))
            arms[name] = add_counts(arms[name], extra)
            window_impressions += extra.impressions
        pair_impressions = int(prev.get("pair_impressions") or 0) + window_impressions
        if promoted_variant and prev_champion in labels and prev_champion != champion:
            challenger = prev_champion
        else:
            challenger = prev_challenger if prev_challenger != champion else _other_name(labels, champion)
        if {champion, challenger} != {prev_champion, prev_challenger}:
            pair_impressions = 0
    else:
        for name in labels:
            extra = window.get(name, ArmCounts(0, 0))
            arms[name] = add_counts(arms[name], extra)
        pair_impressions = 0
        challenger = sample_arm([name for name in labels if name != champion], arms, seed=seed)
    rotated = False
    can_rotate = (
        pair_ok
        and promoted_variant is None
        and pair_impressions >= max(0, int(min_impressions))
        and guardrails_ok
    )
    probs = p_best(labels, arms, draws=draws, seed=seed)
    if can_rotate and float(probs.get(champion, 0.0)) >= float(rotate_min_prob):
        nxt = sample_arm([name for name in labels if name != champion], arms, seed=seed)
        if nxt != challenger:
            challenger = nxt
            pair_impressions = 0
            rotated = True
    return ThompsonAllocation(
        champion=champion,
        challenger=challenger,
        arms=arms,
        p_best=probs,
        pair_impressions=pair_impressions,
        window_started_at=stamp,
        rotated=rotated,
    )


def _other_name(names: Sequence[str], champion: str) -> str:
    for name in names:
        if name != champion:
            return name
    return champion

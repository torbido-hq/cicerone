"""Sticky user → variant assignment (blake2s, replica-safe)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from cicerone.config.constants import ALLOCATION_THOMPSON
from cicerone.config.settings import Settings
from cicerone.experiment.recipes import automl_challenger_pair
from cicerone.io.base import VariantSnapshotReader

AssignmentOverlay = tuple[str | None, tuple[str, str] | None]

_DIGEST_BYTES = 8
_DIGEST_SPAN = float(1 << (8 * _DIGEST_BYTES))


def assignment_bucket(experiment_id: str, user_id: str) -> float:
    """Map ``(experiment_id, user_id)`` to a stable value in ``[0, 1)``."""
    payload = f"{experiment_id}\0{user_id}".encode()
    digest = hashlib.blake2s(payload, digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") / _DIGEST_SPAN


def assign_variant(
    experiment_id: str,
    user_id: str,
    variants: Sequence[tuple[str, float]],
    *,
    promoted_variant: str | None = None,
) -> str:
    """Pick a variant name. ``promoted_variant`` (if known) wins for every user."""
    if not variants:
        raise ValueError("assign_variant requires at least one variant")
    names = [name for name, _traffic in variants]
    if promoted_variant is not None and promoted_variant in names:
        return promoted_variant
    bucket = assignment_bucket(experiment_id, user_id)
    cumulative = 0.0
    last_index = len(variants) - 1
    for index, (name, traffic) in enumerate(variants):
        cumulative += float(traffic)
        if index == last_index or bucket < cumulative:
            return name
    return names[-1]


def active_pair_traffic(
    champion: str,
    challenger: str,
    *,
    explore_traffic: float,
) -> list[tuple[str, float]]:
    share = float(explore_traffic)
    if champion == challenger:
        return [(champion, 1.0)]
    return [(champion, max(0.0, 1.0 - share)), (challenger, share)]


def assign_user(
    settings: Settings,
    user_id: str,
    overlay: AssignmentOverlay,
    snapshot_names: Callable[[], Sequence[str] | None],
) -> tuple[str | None, str | None]:
    """Return ``(experiment_id, variant)``; ``snapshot_names`` runs only for Thompson without a live pair."""
    experiment = settings.experiment
    if not experiment.enabled:
        return None, None
    if experiment.automl_challenger:
        control, treatment = automl_challenger_pair(experiment.variants)
        variants = [(control.name, float(control.traffic)), (treatment.name, float(treatment.traffic))]
    else:
        variants = [(item.name, item.traffic) for item in experiment.variants]
    if not variants:
        return None, None
    promoted_variant, active_pair = overlay
    names = {name for name, _traffic in variants}
    if experiment.allocation == ALLOCATION_THOMPSON and promoted_variant is None:
        if active_pair is not None and active_pair[0] in names and active_pair[1] in names:
            variants = active_pair_traffic(
                active_pair[0], active_pair[1], explore_traffic=experiment.explore_traffic
            )
        else:
            snapshot = snapshot_names()
            if snapshot is None:
                return None, None
            present = {str(name) for name in snapshot if name}
            variants = [(name, traffic) for name, traffic in variants if name in present]
            if not variants:
                return None, None
    variant = assign_variant(
        experiment.id,
        str(user_id),
        variants,
        promoted_variant=promoted_variant,
    )
    return experiment.id, variant


def snapshot_variant_names(reader: object) -> tuple[str, ...] | None:
    if not isinstance(reader, VariantSnapshotReader):
        return None
    names = reader.present_variant_names()
    if names is None:
        return None
    return tuple(str(name) for name in names if name)


def experiment_variant_names(settings: Settings) -> tuple[str, ...]:
    """Names incremental apply and serve hash over (challenger defaults if unset)."""
    experiment = settings.experiment
    if not experiment.enabled:
        return ()
    if experiment.automl_challenger:
        control, treatment = automl_challenger_pair(experiment.variants)
        return (control.name, treatment.name)
    return tuple(variant.name for variant in experiment.variants)

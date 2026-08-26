"""Sticky user → variant assignment (blake2s, replica-safe)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cicerone.config.settings import Settings
from cicerone.experiment.recipes import CONTROL_NAME, TREATMENT_NAME

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


def resolve_assignment(
    settings: Settings,
    user_id: str,
    *,
    promoted_variant: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(experiment_id, variant)`` or ``(None, None)`` when experiments are off."""
    experiment = settings.experiment
    if not experiment.enabled:
        return None, None
    variants = [(item.name, item.traffic) for item in experiment.variants]
    if not variants and experiment.automl_challenger:
        variants = [(CONTROL_NAME, 0.5), (TREATMENT_NAME, 0.5)]
    if not variants:
        return None, None
    variant = assign_variant(
        experiment.id,
        user_id,
        variants,
        promoted_variant=promoted_variant,
    )
    return experiment.id, variant

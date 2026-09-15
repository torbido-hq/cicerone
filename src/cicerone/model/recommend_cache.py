"""Recommend memoization keys and fingerprints."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

RecommendCache = dict[tuple[Hashable, ...], Any]


def _cache_key_part(value: object) -> Hashable:
    try:
        hash(value)
    except TypeError:
        if isinstance(value, Mapping):
            items = [(_cache_key_part(k), _cache_key_part(v)) for k, v in value.items()]
            return tuple(sorted(items, key=repr))
        if isinstance(value, set):
            return tuple(sorted((_cache_key_part(v) for v in value), key=repr))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(_cache_key_part(v) for v in value)
        return str(value)
    return value


def _recommend_cache_key(*parts: object) -> tuple[Hashable, ...]:
    return tuple(_cache_key_part(p) for p in parts)


def _items_fingerprint(allowed_items: Sequence | None) -> Hashable:
    if not allowed_items:
        return None
    return frozenset(map(str, allowed_items))


def _dataset_fingerprint(dataset: object) -> Hashable:
    fingerprint = getattr(dataset, "fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if fingerprint is not None:
        return _cache_key_part(fingerprint)
    version = getattr(dataset, "version", None)
    if version is not None:
        return _cache_key_part(version)
    return id(dataset)

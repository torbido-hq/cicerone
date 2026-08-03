"""Builds the configured input source / output sink / readers from IOSettings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from cicerone.config import IOSettings
from cicerone.io.base import InputSource, ManifestReader, OutputSink, RecommendationReader
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink
from cicerone.io.manifest_reader import DatasetManifestReader, DbManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader, DbRecommendationReader

T = TypeVar("T")


def _build(kind: str, mapping: dict[str, Callable[[dict], T]], *, label: str, options: dict) -> T:
    factory = mapping.get(kind)
    if factory is None:
        raise ValueError(f"Unknown {label} kind: {kind!r}")
    return factory(options)


def build_input_source(settings: IOSettings) -> InputSource:
    return _build(
        settings.kind,
        {"dataset": DatasetInputSource, "db": DatabaseInputSource},
        label="input",
        options=settings.options,
    )


def build_output_sink(settings: IOSettings) -> OutputSink:
    return _build(
        settings.kind,
        {"dataset": DatasetOutputSink, "db": DatabaseOutputSink},
        label="output",
        options=settings.options,
    )


def build_recommendation_reader(settings: IOSettings) -> RecommendationReader:
    return _build(
        settings.kind,
        {"dataset": DatasetRecommendationReader, "db": DbRecommendationReader},
        label="output",
        options=settings.options,
    )


def build_manifest_reader(settings: IOSettings) -> ManifestReader:
    return _build(
        settings.kind,
        {"dataset": DatasetManifestReader, "db": DbManifestReader},
        label="output",
        options=settings.options,
    )

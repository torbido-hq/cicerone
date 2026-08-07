"""Builds the configured input source / output sink / readers from IOSettings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from cicerone.config import IOSettings
from cicerone.io.base import InputSource, ManifestReader, OutputSink, RecommendationReader
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink
from cicerone.io.manifest_reader import DatasetManifestReader, DbManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader, DbRecommendationReader

T = TypeVar("T")

_BackendFactory = Callable[[dict[str, Any]], T]


def _build_from_registry(
    settings: IOSettings,
    registry: dict[str, _BackendFactory[T]],
    *,
    role: str,
) -> T:
    match settings.kind:
        case kind if kind in registry:
            return registry[kind](settings.options)
        case _:
            raise ValueError(f"Unknown {role} kind: {settings.kind!r}")


_INPUT_SOURCES: dict[str, _BackendFactory[InputSource]] = {
    "dataset": DatasetInputSource,
    "db": DatabaseInputSource,
}
_OUTPUT_SINKS: dict[str, _BackendFactory[OutputSink]] = {
    "dataset": DatasetOutputSink,
    "db": DatabaseOutputSink,
}
_RECOMMENDATION_READERS: dict[str, _BackendFactory[RecommendationReader]] = {
    "dataset": DatasetRecommendationReader,
    "db": DbRecommendationReader,
}
_MANIFEST_READERS: dict[str, _BackendFactory[ManifestReader]] = {
    "dataset": DatasetManifestReader,
    "db": DbManifestReader,
}


def build_input_source(settings: IOSettings) -> InputSource:
    return _build_from_registry(settings, _INPUT_SOURCES, role="input")


def build_output_sink(settings: IOSettings) -> OutputSink:
    return _build_from_registry(settings, _OUTPUT_SINKS, role="output")


def build_recommendation_reader(settings: IOSettings) -> RecommendationReader:
    return _build_from_registry(settings, _RECOMMENDATION_READERS, role="recommendation")


def build_manifest_reader(settings: IOSettings) -> ManifestReader:
    return _build_from_registry(settings, _MANIFEST_READERS, role="manifest")

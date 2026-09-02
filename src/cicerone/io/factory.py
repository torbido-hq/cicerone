"""Builds the configured input source / output sink / readers from IOSettings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from cicerone.config.settings import IOSettings
from cicerone.io.base import InputSource, ManifestReader, OutputSink, RecommendationReader, UserHistoryReader

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


def _dataset_input(options: dict[str, Any]) -> InputSource:
    from cicerone.io.dataset_store import DatasetInputSource

    return DatasetInputSource(options)


def _db_input(options: dict[str, Any]) -> InputSource:
    from cicerone.io.db_store import DatabaseInputSource

    return DatabaseInputSource(options)


def _dataset_output(options: dict[str, Any]) -> OutputSink:
    from cicerone.io.dataset_store import DatasetOutputSink

    return DatasetOutputSink(options)


def _db_output(options: dict[str, Any]) -> OutputSink:
    from cicerone.io.db_store import DatabaseOutputSink

    return DatabaseOutputSink(options)


def _dataset_recommendations(options: dict[str, Any]) -> RecommendationReader:
    from cicerone.io.recommendation_reader import DatasetRecommendationReader

    return DatasetRecommendationReader(options)


def _db_recommendations(options: dict[str, Any]) -> RecommendationReader:
    from cicerone.io.recommendation_reader import DbRecommendationReader

    return DbRecommendationReader(options)


def _dataset_manifest(options: dict[str, Any]) -> ManifestReader:
    from cicerone.io.manifest_reader import DatasetManifestReader

    return DatasetManifestReader(options)


def _db_manifest(options: dict[str, Any]) -> ManifestReader:
    from cicerone.io.manifest_reader import DbManifestReader

    return DbManifestReader(options)


_INPUT_SOURCES: dict[str, _BackendFactory[InputSource]] = {
    "dataset": _dataset_input,
    "db": _db_input,
}
_OUTPUT_SINKS: dict[str, _BackendFactory[OutputSink]] = {
    "dataset": _dataset_output,
    "db": _db_output,
}
_RECOMMENDATION_READERS: dict[str, _BackendFactory[RecommendationReader]] = {
    "dataset": _dataset_recommendations,
    "db": _db_recommendations,
}
_MANIFEST_READERS: dict[str, _BackendFactory[ManifestReader]] = {
    "dataset": _dataset_manifest,
    "db": _db_manifest,
}


def build_input_source(settings: IOSettings) -> InputSource:
    return _build_from_registry(settings, _INPUT_SOURCES, role="input")


def build_user_history_reader(settings: IOSettings) -> UserHistoryReader:
    return build_input_source(settings)


def build_output_sink(settings: IOSettings) -> OutputSink:
    return _build_from_registry(settings, _OUTPUT_SINKS, role="output")


def build_recommendation_reader(settings: IOSettings) -> RecommendationReader:
    return _build_from_registry(settings, _RECOMMENDATION_READERS, role="recommendation")


def build_manifest_reader(settings: IOSettings) -> ManifestReader:
    return _build_from_registry(settings, _MANIFEST_READERS, role="manifest")

"""Builds the configured input source / output sink / readers from IOSettings."""

from __future__ import annotations

from cicerone.config import IOSettings
from cicerone.io.base import InputSource, ManifestReader, OutputSink, RecommendationReader
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink
from cicerone.io.manifest_reader import DatasetManifestReader, DbManifestReader
from cicerone.io.recommendation_reader import DatasetRecommendationReader, DbRecommendationReader


def build_input_source(settings: IOSettings) -> InputSource:
    if settings.kind == "dataset":
        return DatasetInputSource(settings.options)
    if settings.kind == "db":
        return DatabaseInputSource(settings.options)
    raise ValueError(f"Unknown input kind: {settings.kind!r}")


def build_output_sink(settings: IOSettings) -> OutputSink:
    if settings.kind == "dataset":
        return DatasetOutputSink(settings.options)
    if settings.kind == "db":
        return DatabaseOutputSink(settings.options)
    raise ValueError(f"Unknown output kind: {settings.kind!r}")


def build_recommendation_reader(settings: IOSettings) -> RecommendationReader:
    if settings.kind == "dataset":
        return DatasetRecommendationReader(settings.options)
    if settings.kind == "db":
        return DbRecommendationReader(settings.options)
    raise ValueError(f"Unknown output kind: {settings.kind!r}")


def build_manifest_reader(settings: IOSettings) -> ManifestReader:
    if settings.kind == "dataset":
        return DatasetManifestReader(settings.options)
    if settings.kind == "db":
        return DbManifestReader(settings.options)
    raise ValueError(f"Unknown output kind: {settings.kind!r}")

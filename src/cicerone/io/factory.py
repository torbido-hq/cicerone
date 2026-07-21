"""Builds the configured input source / output sink.

Input and output are chosen independently via INPUT_KIND / OUTPUT_KIND
("dataset" or "db") — see cicerone.config for the full env var contract.
"""

from __future__ import annotations

from cicerone.config import IOSettings
from cicerone.io.base import InputSource, OutputSink
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink


def build_input_source(settings: IOSettings) -> InputSource:
    if settings.kind == "dataset":
        return DatasetInputSource(settings.dataset)
    if settings.kind == "db":
        return DatabaseInputSource(settings.database)
    raise ValueError(f"Unknown input kind: {settings.kind!r}")


def build_output_sink(settings: IOSettings) -> OutputSink:
    if settings.kind == "dataset":
        return DatasetOutputSink(settings.dataset)
    if settings.kind == "db":
        return DatabaseOutputSink(settings.database)
    raise ValueError(f"Unknown output kind: {settings.kind!r}")

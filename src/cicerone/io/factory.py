"""Builds the configured input source / output sink.

Input and output are chosen independently via the "kind" of their
[input]/[output] section in cicerone.toml ("dataset" or "db") — see
cicerone.config for the full configuration contract. Adding a new backend
kind means adding a case here and a module under cicerone.io; nothing about
the configuration loader needs to change since options are a generic dict.
"""

from __future__ import annotations

from cicerone.config import IOSettings
from cicerone.io.base import InputSource, OutputSink
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink


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

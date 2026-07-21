from __future__ import annotations

import pytest

from cicerone.config import IOSettings
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink
from cicerone.io.db_store import DatabaseInputSource, DatabaseOutputSink
from cicerone.io.factory import build_input_source, build_output_sink


def test_build_input_source_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert isinstance(build_input_source(settings), DatasetInputSource)


def test_build_output_sink_dataset(tmp_path):
    settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert isinstance(build_output_sink(settings), DatasetOutputSink)


def test_build_input_source_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_input_source(settings), DatabaseInputSource)


def test_build_output_sink_db():
    settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    assert isinstance(build_output_sink(settings), DatabaseOutputSink)


def test_build_input_source_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown input kind"):
        build_input_source(settings)


def test_build_output_sink_unknown_kind_raises():
    settings = IOSettings(kind="carrier-pigeon", options={})
    with pytest.raises(ValueError, match="Unknown output kind"):
        build_output_sink(settings)

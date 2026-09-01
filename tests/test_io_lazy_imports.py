"""Import-graph tests: dataset I/O must not load SQLAlchemy or boto3."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _run(code: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(_REPO / "src")}
    subprocess.check_call([sys.executable, "-c", code], cwd=str(_REPO), env=env)


def test_importing_options_does_not_load_boto3():
    _run(
        """
import sys
from cicerone.io import options
assert "boto3" not in sys.modules
assert "botocore" not in sys.modules
assert options.storage_backend({"storage_backend": "local"}) == "local"
assert "boto3" not in sys.modules
"""
    )


def test_dataset_factory_does_not_load_sqlalchemy_or_boto3(tmp_path):
    path = tmp_path.as_posix()
    _run(
        f"""
import sys
from cicerone.config.settings import IOSettings
from cicerone.io.factory import build_input_source, build_manifest_reader, build_output_sink

settings = IOSettings(kind="dataset", options={{"storage_backend": "local", "path": {path!r}}})
build_input_source(settings)
build_output_sink(settings)
build_manifest_reader(settings)
assert "sqlalchemy" not in sys.modules
assert "cicerone.io.db_store" not in sys.modules
assert "boto3" not in sys.modules
assert "cicerone.io.recommendation_reader" not in sys.modules
"""
    )


def test_event_registry_import_does_not_load_s3_or_db():
    _run(
        """
import sys
from cicerone.events.registry import registered_event_source_kinds

assert "db" in registered_event_source_kinds()
assert "sqlalchemy" not in sys.modules
assert "boto3" not in sys.modules
assert "cicerone.events.db" not in sys.modules
assert "cicerone.events.s3" not in sys.modules
assert "cicerone.events.redis_streams" not in sys.modules
"""
    )


def test_requirements_txt_does_not_pin_s3fs():
    text = (_REPO / "requirements.txt").read_text()
    assert "s3fs" not in text
    assert "aiobotocore" not in text

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from cicerone.io.manifest_reader import DatasetManifestReader


def _write_manifest(path, manifest: dict) -> None:
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_dataset_reader_read_latest_returns_the_manifest(tmp_path):
    _write_manifest(tmp_path, {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"})

    reader = DatasetManifestReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.read_latest() == {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}


def test_dataset_reader_read_latest_missing_file_returns_none(tmp_path):
    reader = DatasetManifestReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.read_latest() is None


def test_dataset_reader_read_latest_raises_on_corrupt_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("not valid json")

    reader = DatasetManifestReader({"storage_backend": "local", "path": str(tmp_path)})

    with pytest.raises(json.JSONDecodeError):
        reader.read_latest()


def test_dataset_reader_read_recent_returns_only_the_latest_run(tmp_path):
    _write_manifest(tmp_path, {"status": "failed", "error": "boom"})

    reader = DatasetManifestReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.read_recent(20) == [{"status": "failed", "error": "boom"}]


def test_dataset_reader_read_recent_empty_when_no_run_yet(tmp_path):
    reader = DatasetManifestReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.read_recent(20) == []


@pytest.fixture
def s3_options():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield {
            "storage_backend": "s3",
            "access_key_id": "test",
            "secret_access_key": "test",
            "bucket": "test-bucket",
            "prefix": "recommendations/latest",
        }


def test_dataset_reader_s3_backend_returns_latest_manifest(s3_options):
    client = boto3.client("s3", region_name="us-east-1")
    manifest = {"status": "success", "n_events": 10}
    client.put_object(
        Bucket=s3_options["bucket"],
        Key="recommendations/latest/manifest.json",
        Body=json.dumps(manifest).encode("utf-8"),
    )

    reader = DatasetManifestReader(s3_options)

    assert reader.read_latest() == manifest


def test_dataset_reader_s3_backend_missing_object_returns_none(s3_options):
    reader = DatasetManifestReader(s3_options)

    assert reader.read_latest() is None


def test_dataset_reader_s3_backend_raises_on_hard_failure(s3_options):
    # Real backend errors must propagate; only not-found returns None.
    from botocore.exceptions import ClientError

    reader = DatasetManifestReader({**s3_options, "bucket": "no-such-bucket"})

    with pytest.raises(ClientError):
        reader.read_latest()


def test_dataset_manifest_reader_validates_storage_options():
    with pytest.raises(RuntimeError, match="path"):
        DatasetManifestReader({"storage_backend": "local"})
    with pytest.raises(RuntimeError, match="bucket"):
        DatasetManifestReader(
            {
                "storage_backend": "s3",
                "access_key_id": "test",
                "secret_access_key": "test",
            }
        )

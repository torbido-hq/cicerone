from __future__ import annotations

import json

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from cicerone.config import DatasetLocation
from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink


# --- local backend -----------------------------------------------------------

def test_local_backend_round_trip(tmp_path):
    location = DatasetLocation(backend="local", path=str(tmp_path))
    sink = DatasetOutputSink(location)

    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_manifest({"n_events": 3})

    source = DatasetInputSource(location)
    (tmp_path / "events.parquet").write_bytes((tmp_path / "recommendations.parquet").read_bytes())
    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]

    manifest_path = tmp_path / "manifest.json"
    assert json.loads(manifest_path.read_text()) == {"n_events": 3}


def test_local_backend_optional_inputs_missing_return_none(tmp_path):
    location = DatasetLocation(backend="local", path=str(tmp_path))
    source = DatasetInputSource(location)

    assert source.read_users() is None
    assert source.read_items() is None


# --- s3 backend (mocked) ------------------------------------------------------

@pytest.fixture
def s3_location():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield DatasetLocation(
            backend="s3",
            endpoint_url=None,
            access_key_id="test",
            secret_access_key="test",
            bucket="test-bucket",
            prefix="datasets/latest",
        )


def test_s3_backend_round_trip(s3_location):
    sink = DatasetOutputSink(s3_location)
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.5, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_manifest({"n_events": 1})

    # The output sink writes recommendations.parquet; point a source at the
    # same prefix and re-read it back as if it were events.parquet.
    source = DatasetInputSource(s3_location)
    client = boto3.client("s3", region_name="us-east-1")
    body = client.get_object(Bucket=s3_location.bucket, Key="datasets/latest/recommendations.parquet")["Body"].read()
    client.put_object(Bucket=s3_location.bucket, Key="datasets/latest/events.parquet", Body=body)

    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]


def test_s3_backend_optional_inputs_missing_return_none(s3_location):
    source = DatasetInputSource(s3_location)
    assert source.read_users() is None
    assert source.read_items() is None


def test_s3_backend_no_prefix_writes_flat_key(s3_location):
    flat_location = DatasetLocation(
        backend="s3",
        access_key_id="test",
        secret_access_key="test",
        bucket=s3_location.bucket,
        prefix="",
    )
    sink = DatasetOutputSink(flat_location)
    sink.write_manifest({"ok": True})

    client = boto3.client("s3", region_name="us-east-1")
    body = client.get_object(Bucket=flat_location.bucket, Key="manifest.json")["Body"].read()
    assert json.loads(body) == {"ok": True}

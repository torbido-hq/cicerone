from __future__ import annotations

import json

import boto3
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from cicerone.io.dataset_store import DatasetInputSource, DatasetOutputSink

# --- local backend -----------------------------------------------------------


def test_local_backend_round_trip(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)

    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_items_snapshot(pd.DataFrame([{"item_id": "i1", "category": "beer"}]))
    sink.write_manifest({"n_events": 3})
    sink.write_model_artifact(b"fake-artifact-bytes")

    source = DatasetInputSource(options)
    (tmp_path / "events.parquet").write_bytes((tmp_path / "recommendations.parquet").read_bytes())
    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]

    manifest_path = tmp_path / "manifest.json"
    assert json.loads(manifest_path.read_text()) == {"n_events": 3}
    assert (tmp_path / "model.artifact").read_bytes() == b"fake-artifact-bytes"
    items_snap = pd.read_parquet(tmp_path / "items_snapshot.parquet")
    assert list(items_snap["item_id"]) == ["i1"]


def test_local_backend_optional_inputs_missing_return_none(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    source = DatasetInputSource(options)

    assert source.read_users() is None
    assert source.read_items() is None


def test_local_backend_optional_inputs_propagate_corrupt_file_errors(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    (tmp_path / "users.parquet").write_bytes(b"not-a-parquet-file")
    source = DatasetInputSource(options)
    with pytest.raises(ValueError):
        source.read_users()


def test_unknown_storage_backend_raises():
    with pytest.raises(ValueError, match="Unknown storage_backend"):
        DatasetInputSource({"storage_backend": "ftp"})
    with pytest.raises(ValueError, match="Unknown storage_backend"):
        DatasetOutputSink({"storage_backend": "ftp"})


def test_local_backend_missing_path_raises():
    with pytest.raises(RuntimeError, match="path"):
        DatasetInputSource({"storage_backend": "local"})
    with pytest.raises(RuntimeError, match="path"):
        DatasetOutputSink({"storage_backend": "local"})


@pytest.mark.parametrize("missing_key", ["access_key_id", "secret_access_key", "bucket"])
def test_s3_backend_missing_required_option_raises(missing_key):
    options = {
        "storage_backend": "s3",
        "access_key_id": "test",
        "secret_access_key": "test",
        "bucket": "test-bucket",
    }
    del options[missing_key]

    with pytest.raises(RuntimeError, match=missing_key):
        DatasetInputSource(options)
    with pytest.raises(RuntimeError, match=missing_key):
        DatasetOutputSink(options)


# --- s3 backend (mocked) ------------------------------------------------------


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
            "prefix": "datasets/latest",
        }


def test_s3_backend_round_trip(s3_options):
    sink = DatasetOutputSink(s3_options)
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.5, "source": "personalized"}])
    sink.write_recommendations(df)
    sink.write_manifest({"n_events": 1})

    # Re-read sink output as if it were events.parquet.
    source = DatasetInputSource(s3_options)
    client = boto3.client("s3", region_name="us-east-1")
    bucket = s3_options["bucket"]
    body = client.get_object(Bucket=bucket, Key="datasets/latest/recommendations.parquet")["Body"].read()
    client.put_object(Bucket=bucket, Key="datasets/latest/events.parquet", Body=body)

    events = source.read_events()
    assert list(events["user_id"]) == ["u1"]


def test_s3_backend_optional_inputs_missing_return_none(s3_options):
    source = DatasetInputSource(s3_options)
    assert source.read_users() is None
    assert source.read_items() is None


def test_s3_backend_optional_users_not_found_logs_warning_and_returns_none(s3_options, mocker, caplog):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Object not found"}},
        "GetObject",
    )
    mocker.patch.object(source, "_read", side_effect=client_error)

    with caplog.at_level("WARNING", logger="cicerone.io.dataset_store"):
        result = source.read_users()

    assert result is None
    assert any("users" in record.getMessage().lower() for record in caplog.records)


def test_s3_backend_optional_users_client_error_is_propagated_for_non_not_found_codes(s3_options, mocker):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
        "GetObject",
    )
    mocker.patch.object(source, "_read", side_effect=client_error)

    with pytest.raises(ClientError):
        source.read_users()


def test_s3_backend_optional_items_not_found_logs_warning_and_returns_none(s3_options, mocker, caplog):
    source = DatasetInputSource(s3_options)
    client_error = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "GetObject",
    )
    mocker.patch.object(source, "_read", side_effect=client_error)

    with caplog.at_level("WARNING", logger="cicerone.io.dataset_store"):
        result = source.read_items()

    assert result is None
    assert any("items" in record.getMessage().lower() for record in caplog.records)


def test_s3_backend_no_prefix_writes_flat_key(s3_options):
    flat_options = {
        "storage_backend": "s3",
        "access_key_id": "test",
        "secret_access_key": "test",
        "bucket": s3_options["bucket"],
        "prefix": "",
    }
    sink = DatasetOutputSink(flat_options)
    sink.write_manifest({"ok": True})

    client = boto3.client("s3", region_name="us-east-1")
    body = client.get_object(Bucket=flat_options["bucket"], Key="manifest.json")["Body"].read()
    assert json.loads(body) == {"ok": True}

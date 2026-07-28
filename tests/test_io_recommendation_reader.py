from __future__ import annotations

import io

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from cicerone.io.recommendation_reader import DatasetRecommendationReader


def _write_recommendations(path, rows) -> None:
    pd.DataFrame(rows).to_parquet(path / "recommendations.parquet", index=False)


def test_dataset_reader_returns_top_k_sorted_by_rank(tmp_path):
    _write_recommendations(
        tmp_path,
        [
            {"user_id": "u1", "item_id": "i3", "rank": 3, "score": 0.1, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
            {"user_id": "u2", "item_id": "i1", "rank": 1, "score": 0.7, "source": "personalized"},
        ],
    )

    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})

    recs = reader.get_recommendations("u1", k=2)

    assert list(recs["item_id"]) == ["i1", "i2"]
    assert list(recs["rank"]) == [1, 2]


def test_dataset_reader_unknown_user_returns_empty(tmp_path):
    _write_recommendations(tmp_path, [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}])

    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.get_recommendations("nobody", k=10).empty


def test_dataset_reader_refresh_picks_up_new_data(tmp_path):
    _write_recommendations(tmp_path, [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}])
    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})
    assert reader.get_recommendations("u2", k=10).empty

    _write_recommendations(tmp_path, [{"user_id": "u2", "item_id": "i9", "rank": 1, "score": 0.4}])
    reader.refresh()

    recs = reader.get_recommendations("u2", k=10)
    assert list(recs["item_id"]) == ["i9"]


def test_dataset_reader_refresh_keeps_previous_cache_on_error(tmp_path):
    _write_recommendations(tmp_path, [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9}])
    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})

    (tmp_path / "recommendations.parquet").unlink()
    reader.refresh()

    recs = reader.get_recommendations("u1", k=10)
    assert list(recs["item_id"]) == ["i1"]


def test_dataset_reader_construction_tolerates_missing_file(tmp_path):
    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})

    assert reader.get_recommendations("u1", k=10).empty


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


def test_dataset_reader_s3_backend_returns_top_k(s3_options):
    client = boto3.client("s3", region_name="us-east-1")
    df = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
        ]
    )
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    client.put_object(
        Bucket=s3_options["bucket"],
        Key="recommendations/latest/recommendations.parquet",
        Body=buffer.getvalue(),
    )

    reader = DatasetRecommendationReader(s3_options)

    recs = reader.get_recommendations("u1", k=1)
    assert list(recs["item_id"]) == ["i1"]


def test_dataset_reader_s3_backend_no_prefix_uses_flat_key(s3_options):
    flat_options = {
        "storage_backend": "s3",
        "access_key_id": "test",
        "secret_access_key": "test",
        "bucket": s3_options["bucket"],
        "prefix": "",
    }
    client = boto3.client("s3", region_name="us-east-1")
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    client.put_object(Bucket=flat_options["bucket"], Key="recommendations.parquet", Body=buffer.getvalue())

    reader = DatasetRecommendationReader(flat_options)

    assert list(reader.get_recommendations("u1", k=10)["item_id"]) == ["i1"]

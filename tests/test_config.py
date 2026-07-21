from __future__ import annotations

import pytest

from cicerone.config import load_settings


def test_load_settings_dataset_s3_backend(monkeypatch):
    monkeypatch.setenv("INPUT_KIND", "dataset")
    monkeypatch.setenv("INPUT_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("INPUT_S3_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("INPUT_S3_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("INPUT_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("INPUT_S3_BUCKET", "bucket-in")
    monkeypatch.setenv("INPUT_S3_PREFIX", "datasets/latest")

    monkeypatch.setenv("OUTPUT_KIND", "dataset")
    monkeypatch.setenv("OUTPUT_STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUTPUT_LOCAL_PATH", "/tmp/out")

    settings = load_settings()

    assert settings.input.kind == "dataset"
    assert settings.input.dataset.backend == "s3"
    assert settings.input.dataset.bucket == "bucket-in"
    assert settings.input.dataset.prefix == "datasets/latest"
    assert settings.output.dataset.backend == "local"
    assert settings.output.dataset.path == "/tmp/out"
    assert settings.top_k == 10
    assert settings.half_life_days == 90.0
    assert settings.feature_config_path == "/app/config/features.yml"


def test_load_settings_db_backend_with_defaults(monkeypatch):
    monkeypatch.setenv("INPUT_KIND", "db")
    monkeypatch.setenv("INPUT_DATABASE_URL", "postgresql://u:p@host/db")

    monkeypatch.setenv("OUTPUT_KIND", "db")
    monkeypatch.setenv("OUTPUT_DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setenv("OUTPUT_RECOMMENDATIONS_TABLE", "custom_recos")

    settings = load_settings()

    assert settings.input.kind == "db"
    assert settings.input.database.url == "postgresql://u:p@host/db"
    assert settings.input.database.events_table == "events"
    assert settings.input.database.events_query is None
    assert settings.output.database.recommendations_table == "custom_recos"
    assert settings.output.database.manifest_table == "recommendation_runs"


def test_load_settings_db_backend_with_custom_query(monkeypatch):
    monkeypatch.setenv("INPUT_KIND", "db")
    monkeypatch.setenv("INPUT_DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setenv("INPUT_EVENTS_QUERY", "SELECT * FROM order_items")
    monkeypatch.setenv("OUTPUT_KIND", "dataset")
    monkeypatch.setenv("OUTPUT_STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUTPUT_LOCAL_PATH", "/tmp/out")

    settings = load_settings()

    assert settings.input.database.events_query == "SELECT * FROM order_items"


def test_load_settings_missing_required_env_raises(monkeypatch):
    monkeypatch.delenv("INPUT_KIND", raising=False)
    monkeypatch.delenv("INPUT_S3_BUCKET", raising=False)
    monkeypatch.setenv("OUTPUT_KIND", "dataset")
    monkeypatch.setenv("OUTPUT_STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUTPUT_LOCAL_PATH", "/tmp/out")

    with pytest.raises(RuntimeError, match="INPUT_S3_ENDPOINT_URL"):
        load_settings()


def test_load_settings_unknown_storage_backend_raises(monkeypatch):
    monkeypatch.setenv("INPUT_KIND", "dataset")
    monkeypatch.setenv("INPUT_STORAGE_BACKEND", "ftp")
    monkeypatch.setenv("OUTPUT_KIND", "dataset")
    monkeypatch.setenv("OUTPUT_STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUTPUT_LOCAL_PATH", "/tmp/out")

    with pytest.raises(RuntimeError, match="INPUT_STORAGE_BACKEND"):
        load_settings()


def test_load_settings_unknown_kind_raises(monkeypatch):
    monkeypatch.setenv("INPUT_KIND", "carrier-pigeon")

    with pytest.raises(RuntimeError, match="INPUT_KIND"):
        load_settings()

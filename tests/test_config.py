from __future__ import annotations

import pytest

from cicerone.config import load_settings


def _write_toml(tmp_path, content: str) -> str:
    path = tmp_path / "cicerone.toml"
    path.write_text(content)
    return str(path)


def test_load_settings_dataset_backends(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [job]
        top_k = 20
        half_life_days = 30
        cron_schedule = "0 4 * * *"
        feature_config_path = "/custom/features.toml"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "s3"
        endpoint_url = "https://example.r2.cloudflarestorage.com"
        access_key_id = "key"
        secret_access_key = "secret"
        bucket = "bucket-in"
        prefix = "datasets/latest"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.kind == "dataset"
    assert settings.input.options["bucket"] == "bucket-in"
    assert settings.input.options["prefix"] == "datasets/latest"
    assert settings.output.kind == "dataset"
    assert settings.output.options["path"] == "/tmp/out"
    assert settings.top_k == 20
    assert settings.half_life_days == 30.0
    assert settings.cron_schedule == "0 4 * * *"
    assert settings.feature_config_path == "/custom/features.toml"


def test_load_settings_defaults_when_job_section_missing(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.top_k == 10
    assert settings.half_life_days == 90.0
    assert settings.cron_schedule == "0 3 * * *"
    assert settings.feature_config_path == "/app/config/features.toml"


def test_load_settings_db_backend_with_defaults(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "db"
        [input.options]
        database_url = "postgresql+psycopg://u:p@host/db"

        [output]
        kind = "db"
        [output.options]
        database_url = "postgresql+psycopg://u:p@host/db"
        recommendations_table = "custom_recos"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.kind == "db"
    assert settings.input.options["database_url"] == "postgresql+psycopg://u:p@host/db"
    assert "events_table" not in settings.input.options  # backend applies its own default
    assert settings.output.options["recommendations_table"] == "custom_recos"


def test_load_settings_resolves_env_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_BUCKET", "resolved-bucket")
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "s3"
        bucket = "${MY_SECRET_BUCKET}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["bucket"] == "resolved-bucket"


def test_load_settings_resolves_partial_env_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_NAME", "staging")
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "s3"
        bucket = "bucket"
        prefix = "datasets/${ENV_NAME}/latest"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["prefix"] == "datasets/staging/latest"


def test_load_settings_missing_env_placeholder_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "s3"
        bucket = "${SOME_UNSET_VAR}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    with pytest.raises(RuntimeError, match="SOME_UNSET_VAR"):
        load_settings(config_path)


def test_load_settings_missing_env_placeholder_error_names_config_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "s3"
        bucket = "${SOME_UNSET_VAR}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    with pytest.raises(RuntimeError, match=r"input\.options\.bucket"):
        load_settings(config_path)


def test_load_settings_resolves_multiple_placeholders_in_one_string(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_NAME", "staging")
    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        prefix = "${ENV_NAME}/${BUCKET_NAME}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["prefix"] == "staging/my-bucket"


def test_load_settings_resolves_env_placeholders_in_nested_dicts(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTED_KEY", "resolved-key")
    monkeypatch.setenv("NESTED_SECRET", "resolved-secret")
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [input.options.auth]
        access_key = "${NESTED_KEY}"
        secret_key = "${NESTED_SECRET}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["auth"] == {"access_key": "resolved-key", "secret_key": "resolved-secret"}


def test_load_settings_escaped_placeholder_is_left_literal(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        pattern = "$${NOT_A_VAR}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["pattern"] == "${NOT_A_VAR}"


def test_load_settings_resolves_env_placeholders_in_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TAG", "resolved-tag")
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        tags = ["${MY_TAG}", "literal"]

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["tags"] == ["resolved-tag", "literal"]


def test_load_settings_non_string_option_values_pass_through(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        retries = 3
        strict = true

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.options["retries"] == 3
    assert settings.input.options["strict"] is True


def test_load_settings_missing_config_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Config file not found"):
        load_settings(str(tmp_path / "does-not-exist.toml"))


def test_load_settings_missing_kind_raises(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    with pytest.raises(RuntimeError, match=r"\[input\]\.kind"):
        load_settings(config_path)


def test_load_settings_missing_section_raises(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    with pytest.raises(RuntimeError, match=r"Missing required config section: \[input\]$"):
        load_settings(config_path)


def test_load_settings_normalizes_kind_case(tmp_path):
    config_path = _write_toml(
        tmp_path,
        """
        [input]
        kind = "Dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "DATASET"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    settings = load_settings(config_path)

    assert settings.input.kind == "dataset"
    assert settings.output.kind == "dataset"


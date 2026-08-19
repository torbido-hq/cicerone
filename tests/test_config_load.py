from __future__ import annotations

import re

import pytest
from support.toml_config import write_toml

from cicerone.config import (
    ConfigError,
    EpochMetricsSettings,
    load_settings,
    resolve_epoch_metrics,
    resolve_max_workers,
)


def test_load_settings_dataset_backends(tmp_path):
    config_path = write_toml(
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
    assert settings.models is None
    assert settings.model_weights is None
    assert settings.rrf_k is None
    assert settings.item_based_k_neighbors == 20
    assert settings.content_fallback_enabled is False
    assert settings.content_fallback_max_neighbors == 50
    assert settings.sequential_min_median_interactions == 5
    assert settings.automl_enabled is False
    assert settings.automl_n_splits == 2
    assert settings.automl_test_days == 14
    assert settings.automl_primary_metric == "MAP"
    assert settings.automl_candidates is None


def test_load_settings_with_explicit_models(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["collaborative", "item_based", "popular", "latest"]

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

    assert settings.models == ["collaborative", "item_based", "popular", "latest"]


def test_load_settings_item_based_and_content_fallback(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        [job.item_based]
        k_neighbors = 15
        [job.content_fallback]
        enabled = true
        max_neighbors = 25

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

    assert settings.item_based_k_neighbors == 15
    assert settings.content_fallback_enabled is True
    assert settings.content_fallback_max_neighbors == 25


def test_load_settings_sequential_min_median_interactions(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        [job.sequential]
        min_median_interactions = 8

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
    assert settings.sequential_min_median_interactions == 8
    assert settings.model_configs["sequential"]["cls"] == "SASRecModel"
    assert settings.model_configs["sequential"]["architecture"] == "sasrec"


def test_load_settings_rejects_invalid_sequential_min_median(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        [job.sequential]
        min_median_interactions = 0

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
    with pytest.raises(ConfigError, match="job.sequential.min_median_interactions"):
        load_settings(config_path)


def test_load_settings_rejects_invalid_item_based_k_neighbors(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        [job.item_based]
        k_neighbors = 0

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

    with pytest.raises(ConfigError, match="job.item_based.k_neighbors"):
        load_settings(config_path)


def test_load_settings_rejects_unknown_model(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["collaborative", "not_a_real_model"]

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

    with pytest.raises(ConfigError, match="not_a_real_model"):
        load_settings(config_path)


def test_load_settings_rejects_empty_models(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = []

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

    # Empty models list is a config-load error, not a late train failure.
    with pytest.raises(ConfigError, match="job.models is empty"):
        load_settings(config_path)


def test_load_settings_with_explicit_model_weights(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["collaborative", "popular"]
        rrf_k = 45

        [job.model_weights]
        collaborative = 1.0
        popular = 0.3

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

    assert settings.model_weights == {"collaborative": 1.0, "popular": 0.3}
    assert settings.rrf_k == 45.0


def test_load_settings_rejects_model_weights_not_in_models(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["popular"]

        [job.model_weights]
        collaborative = 1.0

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
    with pytest.raises(ConfigError, match="model_weights"):
        load_settings(config_path)


def test_load_settings_rejects_negative_model_weight(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["popular"]

        [job.model_weights]
        popular = -1.0

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

    # Shared validate_model_weights runs at config load, not only at train time.
    with pytest.raises(ConfigError, match="non-negative"):
        load_settings(config_path)


def test_load_settings_rejects_non_positive_rrf_k(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        models = ["popular"]
        rrf_k = 0

        [job.model_weights]
        popular = 1.0

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

    with pytest.raises(ConfigError, match="job.rrf_k must be positive"):
        load_settings(config_path)


def test_load_settings_with_explicit_automl(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]

        [job.automl]
        enabled = true
        n_splits = 3
        test_days = 7
        primary_metric = "NDCG"

        [[job.automl.candidates]]
        models = ["popular"]

        [[job.automl.candidates]]
        models = ["popular", "latest"]
        [job.automl.candidates.weights]
        popular = 1.0
        latest = 0.5

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

    assert settings.automl_enabled is True
    assert settings.automl_n_splits == 3
    assert settings.automl_test_days == 7
    assert settings.automl_primary_metric == "NDCG"
    assert settings.automl_candidates == [
        {"models": ["popular"]},
        {"models": ["popular", "latest"], "weights": {"popular": 1.0, "latest": 0.5}},
    ]


def test_load_settings_defaults_when_job_section_missing(tmp_path):
    config_path = write_toml(
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
    assert settings.models is None
    assert settings.model_weights is None
    assert settings.rrf_k is None
    assert settings.save_model_artifact is False
    assert settings.max_workers == 1
    assert settings.epoch_metrics is None
    assert settings.automl_enabled is False
    assert settings.automl_n_splits == 2
    assert settings.automl_test_days == 14
    assert settings.automl_primary_metric == "MAP"
    assert settings.automl_candidates is None


def test_load_settings_save_model_artifact(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        save_model_artifact = true

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
    assert settings.save_model_artifact is True


def test_load_settings_max_workers_and_rejects_non_positive(tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    good = write_toml(
        good_dir,
        """
        [job]
        max_workers = 4
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
    assert load_settings(good).max_workers == 4

    omitted_dir = tmp_path / "omitted"
    omitted_dir.mkdir()
    omitted = write_toml(
        omitted_dir,
        """
        [job]
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
    assert load_settings(omitted).max_workers == 1
    assert resolve_max_workers() == 1
    assert resolve_max_workers(None) == 1
    assert resolve_max_workers(3) == 3

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = write_toml(
        bad_dir,
        """
        [job]
        max_workers = 0
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
    with pytest.raises(ConfigError, match="max_workers"):
        load_settings(bad)


def test_load_settings_log_epoch_metrics(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        log_epoch_metrics = true
        epoch_metrics_every = 3
        epoch_metrics_max_users = 100
        epoch_metrics_regression_drop = 0.5
        epoch_metrics_plateau_eps = 0.02
        epoch_metrics_plateau_window = 4

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
    assert settings.epoch_metrics == EpochMetricsSettings(
        every=3,
        max_users=100,
        regression_drop=0.5,
        plateau_eps=0.02,
        plateau_window=4,
    )
    assert resolve_epoch_metrics(log_epoch_metrics=True, every=3).every == 3
    assert resolve_epoch_metrics(log_epoch_metrics=False, every=3) is None
    assert resolve_epoch_metrics(log_epoch_metrics=True, every=None).every == 5


def test_load_settings_ignores_epoch_metrics_every_when_logging_disabled(tmp_path):
    # Invalid epoch_metrics_* must not fail load when logging is off.
    config_path = write_toml(
        tmp_path,
        """
        [job]
        log_epoch_metrics = false
        epoch_metrics_every = 0
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
    assert settings.epoch_metrics is None


def test_load_settings_rejects_non_positive_epoch_metrics_every_when_enabled(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        log_epoch_metrics = true
        epoch_metrics_every = 0
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
    with pytest.raises(ConfigError, match="epoch_metrics_every"):
        load_settings(config_path)


def test_load_settings_rejects_non_positive_half_life_days(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        half_life_days = 0
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
    with pytest.raises(ConfigError, match="half_life_days"):
        load_settings(config_path)


def test_load_settings_db_backend_with_defaults(tmp_path):
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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
    config_path = write_toml(
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


def test_load_settings_falls_back_to_default_path_when_env_var_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CICERONE_CONFIG_PATH", "")
    default_path = tmp_path / "does-not-exist.toml"
    monkeypatch.setattr("cicerone.config.DEFAULT_CONFIG_PATH", str(default_path))

    with pytest.raises(RuntimeError, match=f"Config file not found: {re.escape(str(default_path))}"):
        load_settings()


def test_load_settings_missing_kind_raises(tmp_path):
    config_path = write_toml(
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

    with pytest.raises(ConfigError, match=r"\[input\]\.kind"):
        load_settings(config_path)


def test_load_settings_missing_section_raises(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )

    with pytest.raises(ConfigError, match=r"Missing required config section: \[input\]$"):
        load_settings(config_path)


def test_load_settings_normalizes_kind_case(tmp_path):
    config_path = write_toml(
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


def _base_io_toml() -> str:
    return """
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
        """


def test_load_settings_mode_defaults_to_batch(tmp_path):
    config_path = write_toml(tmp_path, _base_io_toml())

    settings = load_settings(config_path)

    assert settings.mode == "batch"
    assert settings.serve_auth_token is None
    assert settings.trigger_enabled is False


def test_load_settings_rejects_unknown_mode(tmp_path):
    config_path = write_toml(tmp_path, f'[job]\nmode = "not_a_mode"\n{_base_io_toml()}')

    with pytest.raises(ConfigError, match="job.mode must be one of"):
        load_settings(config_path)


def test_load_settings_serve_mode_requires_auth_token(tmp_path):
    config_path = write_toml(tmp_path, f'[job]\nmode = "serve"\n{_base_io_toml()}')

    with pytest.raises(ConfigError, match="serve.auth_token is required"):
        load_settings(config_path)


def test_load_settings_serve_mode_with_auth_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SERVE_TOKEN", "secret-token")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        mode = "serve"

        [serve]
        auth_token = "${{MY_SERVE_TOKEN}}"
        host = "127.0.0.1"
        port = 9000
        default_k = 5
        refresh_interval_seconds = 30
        {_base_io_toml()}
        """,
    )

    settings = load_settings(config_path)

    assert settings.mode == "serve"
    assert settings.serve_auth_token == "secret-token"
    assert settings.serve_host == "127.0.0.1"
    assert settings.serve_port == 9000
    assert settings.serve_default_k == 5
    assert settings.serve_refresh_interval_seconds == 30.0
    assert settings.serve_category_column == "category"
    assert settings.serve_metrics_enabled is True
    assert settings.serve_metrics_token is None


def test_load_settings_serve_metrics_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_METRICS_TOKEN", "metrics-secret")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        mode = "serve"

        [serve]
        auth_token = "secret"
        metrics_token = "${{MY_METRICS_TOKEN}}"
        {_base_io_toml()}
        """,
    )

    settings = load_settings(config_path)

    assert settings.serve_metrics_token == "metrics-secret"
    assert settings.serve_metrics_enabled is True


def test_load_settings_serve_metrics_disabled(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        mode = "serve"

        [serve]
        auth_token = "secret"
        metrics_enabled = false
        {_base_io_toml()}
        """,
    )

    settings = load_settings(config_path)

    assert settings.serve_metrics_enabled is False


def test_load_settings_serve_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SERVE_TOKEN", "secret-token")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        mode = "serve"

        [serve]
        auth_token = "${{MY_SERVE_TOKEN}}"
        {_base_io_toml()}
        """,
    )

    settings = load_settings(config_path)

    assert settings.serve_host == "0.0.0.0"
    assert settings.serve_port == 8000
    assert settings.serve_default_k == 10
    assert settings.serve_refresh_interval_seconds == 60.0
    assert settings.serve_category_column == "category"
    assert settings.serve_metrics_enabled is True
    assert settings.serve_metrics_token is None


def test_load_settings_trigger_enabled_requires_auth_token(tmp_path):
    config_path = write_toml(tmp_path, f"[job]\n[job.trigger]\nenabled = true\n{_base_io_toml()}")

    with pytest.raises(ConfigError, match="job.trigger.auth_token is required"):
        load_settings(config_path)


def test_load_settings_trigger_enabled_with_auth_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TRIGGER_TOKEN", "trigger-secret")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]

        [job.trigger]
        enabled = true
        auth_token = "${{MY_TRIGGER_TOKEN}}"
        host = "127.0.0.1"
        port = 9090
        debounce_seconds = 30
        poll_input_bucket = true
        poll_interval_seconds = 120
        {_base_io_toml()}
        """,
    )

    settings = load_settings(config_path)

    assert settings.trigger_enabled is True
    assert settings.trigger_auth_token == "trigger-secret"
    assert settings.trigger_host == "127.0.0.1"
    assert settings.trigger_port == 9090
    assert settings.trigger_debounce_seconds == 30.0
    assert settings.trigger_poll_input_bucket is True
    assert settings.trigger_poll_interval_seconds == 120.0
    assert settings.trigger_lock_backend == "in_process"
    assert settings.trigger_lock_key == "cicerone:scheduler:run_guard"
    assert settings.trigger_lock_ttl_seconds == 86400.0
    assert settings.trigger_postgres_url is None
    assert settings.trigger_redis_url is None


def test_load_settings_trigger_defaults_when_disabled(tmp_path):
    config_path = write_toml(tmp_path, _base_io_toml())

    settings = load_settings(config_path)

    assert settings.trigger_enabled is False
    assert settings.trigger_host == "0.0.0.0"
    assert settings.trigger_port == 8080
    assert settings.trigger_debounce_seconds == 60.0
    assert settings.trigger_poll_input_bucket is False
    assert settings.trigger_poll_interval_seconds == 300.0
    assert settings.trigger_lock_backend == "in_process"
    assert settings.trigger_lock_key == "cicerone:scheduler:run_guard"
    assert settings.trigger_lock_ttl_seconds == 86400.0


def test_load_settings_lock_backend_rejects_unknown(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "zookeeper"
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="job.trigger.lock_backend must be one of"):
        load_settings(config_path)


def test_load_settings_postgres_lock_requires_database_url(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "postgres"
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="needs a database URL"):
        load_settings(config_path)


def test_load_settings_postgres_lock_accepts_explicit_url_with_dataset_output(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCK_PG", "postgresql+psycopg://u:p@h/db")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "postgres"
        postgres_url = "${{LOCK_PG}}"
        {_base_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.trigger_lock_backend == "postgres"
    assert settings.trigger_postgres_url == "postgresql+psycopg://u:p@h/db"


def test_load_settings_postgres_lock_accepts_output_db(tmp_path):
    config_path = write_toml(
        tmp_path,
        """
        [job]
        [job.trigger]
        lock_backend = "postgres"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "db"
        [output.options]
        database_url = "postgresql+psycopg://u:p@h/db"
        """,
    )
    settings = load_settings(config_path)
    assert settings.trigger_lock_backend == "postgres"
    assert settings.output.kind == "db"


def test_load_settings_redis_lock_requires_redis_url(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "redis"
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="job.trigger.redis_url is required"):
        load_settings(config_path)


def test_load_settings_redis_lock_with_url(tmp_path, monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "redis"
        redis_url = "${{REDIS_URL}}"
        {_base_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.trigger_lock_backend == "redis"
    assert settings.trigger_redis_url == "redis://localhost:6379/0"


def test_load_settings_lock_key_and_ttl(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "redis"
        redis_url = "redis://localhost:6379/0"
        lock_key = "shop-a:scheduler"
        lock_ttl_seconds = 3600
        {_base_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.trigger_lock_key == "shop-a:scheduler"
    assert settings.trigger_lock_ttl_seconds == 3600.0


def test_load_settings_lock_ttl_seconds_must_be_positive(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "redis"
        redis_url = "redis://localhost:6379/0"
        lock_ttl_seconds = 0
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="job.trigger.lock_ttl_seconds must be > 0"):
        load_settings(config_path)


def test_load_settings_lock_ttl_seconds_rejects_negative(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_backend = "redis"
        redis_url = "redis://localhost:6379/0"
        lock_ttl_seconds = -1
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="job.trigger.lock_ttl_seconds must be > 0"):
        load_settings(config_path)


def test_load_settings_empty_lock_key_rejected(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.trigger]
        lock_key = "   "
        {_base_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="job.trigger.lock_key must be a non-empty string"):
        load_settings(config_path)

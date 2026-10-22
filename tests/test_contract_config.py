from __future__ import annotations

import logging
import tomllib
import warnings
from typing import Any

import pytest
from support.contract import load_contract
from support.system_spec import REPO_ROOT

from cicerone.config import load_settings
from cicerone.feature_config import load_feature_config

FAKE_ENV = {
    "INPUT_S3_ENDPOINT_URL": "https://s3.example.test",
    "INPUT_S3_ACCESS_KEY_ID": "input-access-key",
    "INPUT_S3_SECRET_ACCESS_KEY": "input-secret-key",
    "INPUT_S3_BUCKET": "input-bucket",
    "OUTPUT_S3_ENDPOINT_URL": "https://s3.example.test",
    "OUTPUT_S3_ACCESS_KEY_ID": "output-access-key",
    "OUTPUT_S3_SECRET_ACCESS_KEY": "output-secret-key",
    "OUTPUT_S3_BUCKET": "output-bucket",
    "SERVE_AUTH_TOKEN": "serve-token",
}
SHIPPED_CONFIGS = {
    "cicerone.toml": load_settings,
    "cicerone.serve.toml": load_settings,
    "cicerone.dashboard.toml": load_settings,
    "features.toml": load_feature_config,
}


def toml_paths(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for key, value in table.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            paths.update(toml_paths(value, f"{path}."))
        else:
            paths[path] = value
    return paths


@pytest.mark.parametrize("name", sorted(SHIPPED_CONFIGS))
def test_shipped_config_keys_load_without_warnings(monkeypatch, caplog, name):
    for env_name, value in FAKE_ENV.items():
        monkeypatch.setenv(env_name, value)
    path = REPO_ROOT / "config" / name
    assert toml_paths(tomllib.loads(path.read_text())) == load_contract("shipped_configs.json")[name]

    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SHIPPED_CONFIGS[name](str(path))
    assert [record.getMessage() for record in caplog.records] == []

from __future__ import annotations

import json
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
# Enum-like values swap for another accepted value; anything else gets a generic change.
ALTERNATE_VALUES = {
    "serve": "batch",
    "dataset": "db",
    "s3": "local",
    "local": "s3",
    "0 3 * * *": "0 4 * * *",
    "0.0.0.0": "127.0.0.1",
}
UNCHANGEABLE_KEYS = {
    ("cicerone.toml", "job.mode"): "serve also needs serve.auth_token",
    ("cicerone.serve.toml", "serve.metrics_enabled"): "true also needs serve.metrics_token",
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


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(map(toml_value, value)) + "]"
    return "{" + ", ".join(f"{key} = {toml_value(item)}" for key, item in value.items()) + "}"


def changed_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, str):
        return ALTERNATE_VALUES.get(value, f"{value}-changed")
    return value[:-1]


def shipped_paths(name: str) -> dict[str, Any]:
    return toml_paths(tomllib.loads((REPO_ROOT / "config" / name).read_text()))


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


@pytest.mark.parametrize(
    ("name", "key"),
    [
        (name, key)
        for name in sorted(SHIPPED_CONFIGS)
        for key in sorted(shipped_paths(name))
        if (name, key) not in UNCHANGEABLE_KEYS
    ],
)
def test_shipped_config_keys_change_the_loaded_config(monkeypatch, tmp_path, name, key):
    for env_name, value in FAKE_ENV.items():
        monkeypatch.setenv(env_name, value)
    paths = shipped_paths(name)

    def load(values: dict[str, Any]) -> Any:
        path = tmp_path / name
        path.write_text("".join(f"{path_key} = {toml_value(value)}\n" for path_key, value in values.items()))
        return SHIPPED_CONFIGS[name](str(path))

    baseline = load(paths)
    assert baseline == SHIPPED_CONFIGS[name](str(REPO_ROOT / "config" / name))
    assert load({**paths, key: changed_value(paths[key])}) != baseline

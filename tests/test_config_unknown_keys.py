from __future__ import annotations

import logging
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import make_settings
from support.toml_config import MINIMAL_IO_TOML, write_toml

from cicerone.config import load_settings
from cicerone.config.unknown_keys import warn_unknown_keys

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def _unknown_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "cicerone.config.unknown_keys"]


def test_unknown_keys_warn_and_defaults_still_apply(tmp_path, caplog):
    config_path = write_toml(
        tmp_path,
        MINIMAL_IO_TOML
        + """
[job]
top_K = 50

[[job.automl.candidates]]
models = ["popular"]
wieghts = { popular = 1.0 }

[job.trigger]
enabeld = true

[serve]
prot = 9000

[events.online]
fit_partial_epoch = 3

[experiment]
[[experiment.variants]]
name = "control"
traffic = 1.0
trafic = 0.5

[servee]
port = 1
""",
    )
    with caplog.at_level(logging.WARNING):
        settings = load_settings(config_path)

    assert _unknown_warnings(caplog) == [
        "Unknown config key 'servee' is ignored",
        "Unknown config key 'job.top_K' is ignored",
        "Unknown config key 'job.automl.candidates[0].wieghts' is ignored",
        "Unknown config key 'job.trigger.enabeld' is ignored",
        "Unknown config key 'serve.prot' is ignored",
        "Unknown config key 'events.online.fit_partial_epoch' is ignored",
        "Unknown config key 'experiment.variants[0].trafic' is ignored",
    ]
    assert settings.top_k == 10
    assert settings.serve.port == 8000
    assert settings.trigger.enabled is False


def test_documented_keys_and_free_form_tables_do_not_warn():
    raw = {
        "job": {
            "top_k": 5,
            "explain": {"enabled": False},
            "eval": {"ks": [5]},
            "automl": {"candidates": [{"models": ["popular"], "weights": {"popular": 1.0}, "rrf_k": 60}]},
        },
        "input": {"kind": "dataset", "options": {"anything": 1}},
        "output": {"kind": "db", "options": {"url": "x"}, "artifact_hmac_key": "k"},
        "model": {"collaborative": {"cls": "x", "model": {"epochs": 3}}},
        "events": {"enabled": True, "options": {"queue": "q"}, "incremental": {"batch_size": 1}},
        "experiment": {
            "variants": [
                {"name": "a", "traffic": 1.0, "weights": {"x": 1.0}, "blending": {"anything": 1}},
            ]
        },
    }
    assert warn_unknown_keys(raw) == []


def test_loaded_defaults_match_dataclass_defaults(tmp_path):
    settings = load_settings(write_toml(tmp_path, MINIMAL_IO_TOML))
    defaults = make_settings()

    for section in ("serve", "trigger", "dashboard", "automl", "explain"):
        assert getattr(settings, section) == getattr(defaults, section), section
    assert replace(settings, input=defaults.input, output=defaults.output, model_configs={}) == replace(
        defaults, model_configs={}
    )


@pytest.mark.parametrize("name", ["cicerone.toml", "cicerone.serve.toml", "cicerone.dashboard.toml"])
def test_shipped_configs_have_no_unknown_keys(name):
    with (REPO_CONFIG / name).open("rb") as f:
        assert warn_unknown_keys(tomllib.load(f)) == []

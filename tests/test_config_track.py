from __future__ import annotations

import pytest
from support.toml_config import write_toml

from cicerone.config import (
    ConfigError,
    IOSettings,
    load_eval_settings,
    load_experiment_settings,
    load_settings,
    load_track_settings,
    make_settings,
)


def _minimal_io() -> str:
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


def test_load_track_and_eval_defaults(tmp_path):
    settings = load_settings(write_toml(tmp_path, "[job]\n" + _minimal_io()))
    assert settings.track.enabled is False
    assert settings.track.attribution_window_hours == 24.0
    assert settings.track.min_impressions == 100
    assert settings.eval.enabled is False
    assert settings.serve.log_impressions is False
    assert settings.experiment.attribution == "user"


def test_load_track_eval_and_log_impressions(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            """
            [job]
            """
            + _minimal_io()
            + """
            [job.eval]
            enabled = true
            event_types = ["purchase"]
            ks = [5, 10]

            [track]
            enabled = true
            attribution_window_hours = 12
            conversion_event_types = ["purchase"]
            min_impressions = 50

            [serve]
            log_impressions = true
            """,
        )
    )
    assert settings.track.enabled is True
    assert settings.track.attribution_window_hours == 12
    assert settings.track.conversion_event_types == ("purchase",)
    assert settings.track.min_impressions == 50
    assert settings.eval.enabled is True
    assert settings.eval.event_types == ("purchase",)
    assert settings.eval.ks == (5, 10)
    assert settings.serve.log_impressions is True


def test_log_impressions_requires_track(tmp_path):
    with pytest.raises(ConfigError, match="log_impressions requires track.enabled"):
        load_settings(
            write_toml(
                tmp_path,
                """
                [job]
                """
                + _minimal_io()
                + """
                [serve]
                log_impressions = true
                """,
            )
        )


def test_ctr_metric_requires_track(tmp_path):
    with pytest.raises(ConfigError, match="ctr.*requires track.enabled"):
        load_settings(
            write_toml(
                tmp_path,
                """
                [job]
                """
                + _minimal_io()
                + """
                [experiment]
                enabled = true
                id = "exp"
                primary_metric = "ctr"
                attribution = "click"
                [[experiment.variants]]
                name = "control"
                traffic = 0.5
                [[experiment.variants]]
                name = "treatment"
                traffic = 0.5
                """,
            )
        )


def test_load_settings_rejects_track_on_object_store(tmp_path):
    with pytest.raises(ConfigError, match="not atomic"):
        load_settings(
            write_toml(
                tmp_path,
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
                storage_backend = "s3"
                bucket = "recs"
                access_key_id = "id"
                secret_access_key = "secret"
                [track]
                enabled = true
                """,
            )
        )


def test_load_settings_rejects_track_with_ha_on_local_dataset(tmp_path):
    with pytest.raises(ConfigError, match="events.ha requires output kind"):
        load_settings(
            write_toml(
                tmp_path,
                """
                [job]
                mode = "serve"
                [job.trigger]
                lock_backend = "redis"
                redis_url = "redis://localhost:6379/0"
                [serve]
                auth_token = "tok"
                [events]
                enabled = true
                kind = "webhook"
                ha = true
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
                [track]
                enabled = true
                """,
            )
        )


def test_load_track_rejects_bad_window_and_ks():
    with pytest.raises(ConfigError, match="attribution_window_hours"):
        load_track_settings({"enabled": True, "attribution_window_hours": 0})
    with pytest.raises(ConfigError, match="job.eval.ks"):
        load_eval_settings({"enabled": True, "ks": [-1]})


def test_make_settings_track_and_eval_dicts(tmp_path):
    settings = make_settings(
        track={"enabled": True, "min_impressions": 10},
        eval={"enabled": True, "ks": [5]},
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    assert settings.track.enabled is True
    assert settings.track.min_impressions == 10
    assert settings.eval.enabled is True
    assert settings.eval.ks == (5,)


def test_load_experiment_attribution(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            """
            [job]
            """
            + _minimal_io()
            + """
            [track]
            enabled = true
            [experiment]
            enabled = true
            id = "exp"
            primary_metric = "ctr"
            attribution = "click"
            [[experiment.variants]]
            name = "control"
            traffic = 0.5
            [[experiment.variants]]
            name = "treatment"
            traffic = 0.5
            """,
        )
    )
    assert settings.experiment.primary_metric == "ctr"
    assert settings.experiment.attribution == "click"
    with pytest.raises(ConfigError, match="attribution"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "attribution": "pixel",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5},
                ],
            }
        )
    with pytest.raises(ConfigError, match="click"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "primary_metric": "ctr",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5},
                ],
            }
        )
    with pytest.raises(ConfigError, match="impression"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "primary_metric": "conversion",
                "attribution": "recommended",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5},
                ],
            }
        )

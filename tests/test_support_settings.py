from __future__ import annotations

import pytest
from support.settings import make_settings

from cicerone.config import ConfigError, ServeSettings


def test_flat_overrides_merge_into_nested_section():
    settings = make_settings(serve=ServeSettings(auth_token="secret"), serve_port=9000)
    assert (settings.serve.auth_token, settings.serve.port) == ("secret", 9000)


@pytest.mark.parametrize(
    "overrides",
    [
        {"track": {"attribution_window_hours": 0}, "track_min_impressions": 1},
        {"eval": {"ks": [0]}, "eval_enabled": True},
    ],
)
def test_dict_sections_with_flat_overrides_keep_production_validation(overrides):
    with pytest.raises(ConfigError):
        make_settings(**overrides)


def test_invalid_section_with_flat_override_is_still_rejected():
    with pytest.raises(TypeError):
        make_settings(serve="bad", serve_port=9000)

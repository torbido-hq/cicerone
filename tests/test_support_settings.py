from __future__ import annotations

import pytest
from support.settings import make_settings

from cicerone.config import ConfigError, ServeSettings


def test_flat_overrides_merge_into_nested_section():
    settings = make_settings(serve=ServeSettings(auth_token="secret"), serve_port=9000)
    assert (settings.serve.auth_token, settings.serve.port) == ("secret", 9000)


def test_flat_override_merges_into_dict_section_through_production_loader():
    settings = make_settings(track={"conversion_event_types": ["buy"]}, track_min_impressions=7)
    assert settings.track.min_impressions == 7
    assert settings.track.conversion_event_types == ("buy",)


@pytest.mark.parametrize(
    "overrides",
    [
        {"track": {"enabled": True}, "track_attribution_window_hours": 0},
        {"eval": {"enabled": True}, "eval_ks": [0]},
    ],
)
def test_invalid_flat_override_in_dict_section_is_validated(overrides):
    with pytest.raises(ConfigError):
        make_settings(**overrides)


def test_invalid_section_with_flat_override_is_still_rejected():
    with pytest.raises(TypeError):
        make_settings(serve="bad", serve_port=9000)

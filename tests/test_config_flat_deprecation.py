from __future__ import annotations

import re

import pytest

from cicerone.config import Settings, make_settings

_FLAT_PROPERTIES = sorted(name for name, value in vars(Settings).items() if isinstance(value, property))


def test_every_flat_property_is_covered():
    assert len(_FLAT_PROPERTIES) == 36


@pytest.mark.parametrize("name", _FLAT_PROPERTIES)
def test_flat_property_still_returns_nested_value_with_warning(name):
    settings = make_settings()
    section, _, field = name.partition("_")
    with pytest.warns(FutureWarning, match=rf"Settings\.{name} is deprecated.*Settings\.{section}\.{field}"):
        value = getattr(settings, name)
    assert value == getattr(getattr(settings, section), field)


def test_flat_make_settings_kwarg_still_applies_with_warning():
    with pytest.warns(
        FutureWarning,
        match=re.escape("make_settings(serve_port=...) is deprecated")
        + ".*serve=ServeSettings\\(port=...\\)",
    ):
        settings = make_settings(serve_port=9000)
    assert settings.serve.port == 9000

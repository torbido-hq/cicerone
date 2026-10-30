from __future__ import annotations

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


def test_deprecation_warnings_point_at_the_caller():
    settings = make_settings()
    with pytest.warns(FutureWarning) as property_warning:
        _ = settings.serve_host
    assert [w.filename for w in property_warning] == [__file__]

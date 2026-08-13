from __future__ import annotations

import pytest

from cicerone.config import IOSettings, make_settings
from cicerone.events.store import empty_recommendations_frame, load_recommendations_frame


def test_load_recommendations_missing_file(tmp_path):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert frame.empty


def test_load_recommendations_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendations_frame(IOSettings(kind="other", options={}))

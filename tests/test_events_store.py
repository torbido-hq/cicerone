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


def test_load_recommendations_schema_mismatch_treated_as_empty(tmp_path):
    import pandas as pd

    path = tmp_path / "recommendations.parquet"
    pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]).to_parquet(path, index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert frame.empty


def test_load_recommendations_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendations_frame(IOSettings(kind="other", options={}))

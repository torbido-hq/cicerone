from __future__ import annotations

import pandas as pd
import pytest

from cicerone.io.replace_users import normalize_replace_user_ids


def test_normalize_replace_user_ids_ok():
    df = pd.DataFrame([{"user_id": "u1", "item_id": "i1"}])
    assert normalize_replace_user_ids(df, ["u1", "u1"]) == ["u1"]
    assert normalize_replace_user_ids(pd.DataFrame(), ["u2", "u1"]) == ["u1", "u2"]
    assert normalize_replace_user_ids(pd.DataFrame(), []) == []


def test_normalize_replace_user_ids_rejects_bad_frames():
    with pytest.raises(ValueError, match="requires user_ids"):
        normalize_replace_user_ids(pd.DataFrame([{"user_id": "u1"}]), [])
    with pytest.raises(ValueError, match="missing user_id"):
        normalize_replace_user_ids(pd.DataFrame([{"item_id": "i1"}]), ["u1"])
    with pytest.raises(ValueError, match="outside user_ids"):
        normalize_replace_user_ids(pd.DataFrame([{"user_id": "u9"}]), ["u1"])

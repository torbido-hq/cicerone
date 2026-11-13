from __future__ import annotations

import numpy as np
import pandas as pd

from cicerone.io.sql_rows import frame_records, jsonish


def test_jsonish_numpy_scalar_and_na() -> None:
    assert jsonish(np.int64(3)) == 3
    assert jsonish(pd.NA) is None


def test_jsonish_item_raises() -> None:
    class _Boom:
        def item(self):
            raise ValueError("nope")

    boom = _Boom()
    assert jsonish(boom) is boom


def test_frame_records_of_missing_or_empty_frame() -> None:
    assert frame_records(None) == []
    assert frame_records(pd.DataFrame()) == []
    assert frame_records(pd.DataFrame([{"n": np.int64(1), "x": None}])) == [{"n": 1, "x": None}]

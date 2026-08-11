"""Tests for shared missing-value helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cicerone.values import MISSING, as_list, is_missing, is_sequence_attr, str_set


def test_is_missing_scalars_and_containers():
    assert is_missing(None)
    assert is_missing(MISSING)
    assert is_missing(float("nan"))
    assert is_missing(pd.NA)
    assert not is_missing(0)
    assert not is_missing("")
    assert not is_missing([])
    assert not is_missing(np.array([1, 2]))
    assert not is_missing(pd.Series([1.0]))


def test_as_list_and_str_set():
    assert as_list(None) == []
    assert as_list(MISSING) == []
    assert as_list("x") == ["x"]
    assert as_list([1, None, float("nan"), 2]) == [1, 2]
    assert as_list([1, MISSING, 2]) == [1, 2]

    # numpy array handling: 0-d scalar and 1-d array with missing values
    assert as_list(np.array(3)) == [3]
    assert as_list(np.array([1, np.nan])) == [1.0]

    # pandas Series handling with missing values
    assert as_list(pd.Series([1, None, float("nan")])) == [1.0]

    # str_set behavior with different container types and missing-value filtering
    assert str_set([1, None, "a"]) == {"1", "a"}
    assert str_set([1, MISSING, "a"]) == {"1", "a"}
    assert str_set(pd.Series([1, None, float("nan")])) == {"1.0"}
    assert str_set(np.array([1, np.nan])) == {"1.0"}

    assert is_sequence_attr([1])
    assert not is_sequence_attr(math.nan)

from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.model.combine import combine_by_priority, combine_by_weighted_fusion
from cicerone.reasons import SOURCE_CONTRIBS_COLUMN


def test_combine_by_priority_fills_from_earlier_strategies_first():
    frames = [
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "rank": 1, "score": 1.0, "source": "first", "_weight": 1.0},
                {"user_id": "u1", "item_id": "b", "rank": 2, "score": 0.5, "source": "first", "_weight": 1.0},
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "c",
                    "rank": 1,
                    "score": 9.0,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "b",
                    "rank": 2,
                    "score": 8.0,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
    ]
    out = combine_by_priority(frames, top_k=2)
    assert list(out[Columns.Item]) == ["a", "b"]
    assert list(out[Columns.Rank]) == [1, 2]
    assert list(out["source"]) == ["first", "first"]


def test_combine_by_priority_fills_from_later_strategies_when_earlier_insufficient():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "first",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "b",
                    "rank": 1,
                    "score": 0.8,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "c",
                    "rank": 2,
                    "score": 0.7,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "d",
                    "rank": 1,
                    "score": 0.9,
                    "source": "third",
                    "_weight": 1.0,
                },
            ]
        ),
    ]

    combined = combine_by_priority(frames, top_k=2)

    assert list(combined[Columns.User].unique()) == ["u1"]
    assert len(combined) == 2
    assert list(combined[Columns.Item]) == ["a", "b"]
    assert list(combined[Columns.Rank]) == [1, 2]


def test_combine_by_priority_recomputes_ranks_per_user():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "first",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "b",
                    "rank": 1,
                    "score": 0.9,
                    "source": "first",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "c",
                    "rank": 2,
                    "score": 0.8,
                    "source": "first",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "d",
                    "rank": 1,
                    "score": 0.7,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "e",
                    "rank": 2,
                    "score": 0.6,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "f",
                    "rank": 1,
                    "score": 0.85,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
    ]

    top_k = 2
    combined = combine_by_priority(frames, top_k=top_k)

    counts = combined.groupby(Columns.User)[Columns.Item].count().to_dict()
    assert counts == {"u1": 2, "u2": 2}

    ranks_per_user = (
        combined.sort_values([Columns.User, Columns.Rank])
        .groupby(Columns.User)[Columns.Rank]
        .apply(list)
        .to_dict()
    )
    assert ranks_per_user == {"u1": [1, 2], "u2": [1, 2]}

    items_per_user = (
        combined.sort_values([Columns.User, Columns.Rank])
        .groupby(Columns.User)[Columns.Item]
        .apply(list)
        .to_dict()
    )
    assert items_per_user == {"u1": ["a", "d"], "u2": ["b", "c"]}

    for _, group in combined.groupby(Columns.User):
        ordered = group.sort_values(Columns.Rank)
        assert list(ordered[Columns.Rank]) == list(range(1, top_k + 1))


def test_combine_by_priority_keeps_winning_source_contrib():
    frames = [
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "rank": 3, "score": 1.0, "source": "first", "_weight": 1.0},
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 9.0,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
    ]
    out = combine_by_priority(frames, top_k=1)
    contribs = out.iloc[0][SOURCE_CONTRIBS_COLUMN]
    assert contribs == [{"label": "first", "rank": 3, "weight": 1.0, "contribution": None}]


def test_combine_by_weighted_fusion_keeps_per_source_contribs():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "personalized",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 3,
                    "score": 0.5,
                    "source": "popular_fallback",
                    "_weight": 0.5,
                },
            ]
        ),
    ]
    out = combine_by_weighted_fusion(
        frames, top_k=1, rrf_k=1.0, source_label_order=["personalized", "popular_fallback"]
    )
    assert list(out["source"]) == ["personalized+popular_fallback"]
    contribs = out.iloc[0][SOURCE_CONTRIBS_COLUMN]
    assert [row["label"] for row in contribs] == ["personalized", "popular_fallback"]
    assert contribs[0]["rank"] == 1
    assert contribs[0]["contribution"] == pytest.approx(1.0 / 2.0)
    assert contribs[1]["rank"] == 3
    assert contribs[1]["contribution"] == pytest.approx(0.5 / 4.0)


def test_as_rank_and_float_parse_strings():
    from cicerone.model.combine import _as_float, _as_rank

    assert _as_rank("3") == 3
    assert _as_float("0.5") == pytest.approx(0.5)

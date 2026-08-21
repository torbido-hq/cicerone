from __future__ import annotations

import time

import numpy as np
import pandas as pd
from rectools import Columns

from cicerone.model.combine import combine_by_priority, combine_by_weighted_fusion
from cicerone.model.constants import WEIGHT_COLUMN


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


def test_combine_by_weighted_fusion_ties_break_on_item_id():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "b",
                    "rank": 1,
                    "score": 1.0,
                    "source": "first",
                    WEIGHT_COLUMN: 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "second",
                    WEIGHT_COLUMN: 1.0,
                },
            ]
        ),
    ]
    out = combine_by_weighted_fusion(frames, top_k=2, rrf_k=60.0, source_label_order=["first", "second"])
    assert list(out[Columns.Item]) == ["a", "b"]


def test_combine_by_weighted_fusion_large_group_baseline():
    n_users = 200
    n_items = 20
    users = np.repeat([f"u{u:04d}" for u in range(n_users)], n_items)
    items = np.tile([f"i{i:03d}" for i in range(n_items)], n_users)
    ranks = np.tile(np.arange(1, n_items + 1), n_users)

    def _frame(source: str, weight: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                Columns.User: users,
                Columns.Item: items,
                Columns.Rank: ranks,
                Columns.Score: 1.0,
                "source": source,
                WEIGHT_COLUMN: weight,
            }
        )

    started = time.perf_counter()
    out = combine_by_weighted_fusion(
        [_frame("first", 1.0), _frame("second", 0.5)],
        top_k=10,
        rrf_k=60.0,
        source_label_order=["first", "second"],
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
    assert len(out) == n_users * 10
    assert (out["source"] == "first+second").all()

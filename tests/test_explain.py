from __future__ import annotations

import pandas as pd
from rectools import Columns

from cicerone.config.settings import ExplainSettings
from cicerone.explain import attach_reasons, overlap_for_item
from cicerone.feature_config import FeatureColumn
from cicerone.io.recommendation_schema import REASONS_COLUMN
from cicerone.reasons import SOURCE_CONTRIBS_COLUMN, parse_reasons


def test_overlap_for_item_shared_tokens():
    token_index = {
        "rec": {"style=lager": 1.0, "origin=cz": 1.0},
        "hist": {"style=lager": 1.0, "origin=de": 1.0},
        "other": {"style=ipa": 1.0},
    }
    similar, matched = overlap_for_item(
        item_id="rec",
        history_ids=["hist", "other"],
        token_index=token_index,
        max_similar_items=3,
        max_attributes=5,
    )
    assert similar == [{"item_id": "hist", "score": 1 / 3}]
    assert matched == [{"column": "style", "value": "lager"}]


def test_overlap_without_features_is_empty():
    similar, matched = overlap_for_item(
        item_id="rec",
        history_ids=["hist"],
        token_index={},
        max_similar_items=3,
        max_attributes=5,
    )
    assert similar == []
    assert matched == []


def test_overlap_skips_self_and_unindexed_history():
    token_index = {
        "rec": {"style=lager": 1.0},
        "hist": {"style=lager": 1.0},
        "other": {"style=ipa": 1.0},
    }
    similar, matched = overlap_for_item(
        item_id="rec",
        history_ids=["rec", "missing", "other", "hist"],
        token_index=token_index,
        max_similar_items=3,
        max_attributes=5,
    )
    assert similar == [{"item_id": "hist", "score": 1.0}]
    assert matched == [{"column": "style", "value": "lager"}]


def test_attach_reasons_serializes_sources_and_overlap():
    recs = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "rec",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": "personalized",
                SOURCE_CONTRIBS_COLUMN: [
                    {"label": "personalized", "rank": 1, "weight": 1.0, "contribution": None}
                ],
            }
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "rec", "style": "lager"},
            {"item_id": "hist", "style": "lager"},
        ]
    )
    interactions = pd.DataFrame(
        [
            {Columns.User: "u1", Columns.Item: "hist"},
        ]
    )
    out = attach_reasons(
        recs,
        items=items,
        interactions=interactions,
        feature_columns=[FeatureColumn(column="style", type="categorical")],
        settings=ExplainSettings(enabled=True, max_similar_items=3, max_attributes=5),
    )
    assert SOURCE_CONTRIBS_COLUMN not in out.columns
    parsed = parse_reasons(out.iloc[0][REASONS_COLUMN])
    assert parsed is not None
    assert parsed.sources[0].label == "personalized"
    assert parsed.similar_items[0].item_id == "hist"
    assert parsed.matched_attributes[0].column == "style"
    assert parsed.matched_attributes[0].value == "lager"


def test_overlap_max_similar_zero_still_matches_attributes():
    token_index = {
        "rec": {"style=lager": 1.0, "origin=cz": 1.0},
        "hist": {"style=lager": 1.0, "origin=de": 1.0},
    }
    similar, matched = overlap_for_item(
        item_id="rec",
        history_ids=["hist"],
        token_index=token_index,
        max_similar_items=0,
        max_attributes=5,
    )
    assert similar == []
    assert matched == [{"column": "style", "value": "lager"}]


def test_attach_reasons_history_prefers_recent_datetime():
    recs = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "rec",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": "personalized",
            }
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "rec", "style": "lager"},
            {"item_id": "old", "style": "ipa"},
            {"item_id": "hist", "style": "lager"},
        ]
    )
    interactions = pd.DataFrame(
        [
            {Columns.User: "u1", Columns.Item: "old", Columns.Datetime: pd.Timestamp("2020-01-01")},
            {Columns.User: "u1", Columns.Item: "hist", Columns.Datetime: pd.Timestamp("2026-01-01")},
        ]
    )
    out = attach_reasons(
        recs,
        items=items,
        interactions=interactions,
        feature_columns=[FeatureColumn(column="style", type="categorical")],
        settings=ExplainSettings(enabled=True, max_similar_items=1, max_attributes=5),
    )
    parsed = parse_reasons(out.iloc[0][REASONS_COLUMN])
    assert parsed is not None
    assert parsed.similar_items[0].item_id == "hist"
    recs = pd.DataFrame(
        columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source", SOURCE_CONTRIBS_COLUMN]
    )
    out = attach_reasons(
        recs,
        items=None,
        interactions=None,
        feature_columns=[],
        settings=ExplainSettings(enabled=True),
    )
    assert out.empty
    assert SOURCE_CONTRIBS_COLUMN not in out.columns
    assert REASONS_COLUMN not in out.columns


def test_attach_reasons_uses_source_when_contribs_missing():
    recs = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "rec",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": "popular_fallback",
            }
        ]
    )
    out = attach_reasons(
        recs,
        items=None,
        interactions=None,
        feature_columns=[],
        settings=ExplainSettings(enabled=True),
    )
    parsed = parse_reasons(out.iloc[0][REASONS_COLUMN])
    assert parsed is not None
    assert parsed.sources[0].label == "popular_fallback"
    assert parsed.similar_items == []
    assert parsed.matched_attributes == []


def test_attach_reasons_uses_source_when_contribs_empty():
    recs = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "rec",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": "popular_fallback",
                SOURCE_CONTRIBS_COLUMN: [],
            }
        ]
    )
    out = attach_reasons(
        recs,
        items=None,
        interactions=None,
        feature_columns=[],
        settings=ExplainSettings(enabled=True),
    )
    parsed = parse_reasons(out.iloc[0][REASONS_COLUMN])
    assert parsed is not None
    assert parsed.sources[0].label == "popular_fallback"


def test_attach_reasons_disabled_drops_internal_columns():
    recs = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "rec",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": "personalized",
                SOURCE_CONTRIBS_COLUMN: [{"label": "personalized", "rank": 1, "weight": 1.0}],
            }
        ]
    )
    out = attach_reasons(
        recs,
        items=None,
        interactions=None,
        feature_columns=[],
        settings=ExplainSettings(enabled=False),
    )
    assert REASONS_COLUMN not in out.columns
    assert SOURCE_CONTRIBS_COLUMN not in out.columns

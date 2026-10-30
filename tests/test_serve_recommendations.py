from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_settings

from cicerone.serve.recommendations import RecommendationService


def _rows(items: list[str], variant: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "item_id": items,
            "rank": [10 * (i + 1) for i in range(len(items))],
            "score": [1.0] * len(items),
            "source": ["als"] * len(items),
        }
    )
    if variant is not None:
        frame["variant"] = variant
    return frame


class _Reader:
    def __init__(self, recs: dict, cold: dict) -> None:
        self._recs = recs
        self._cold = cold
        self.calls: list[tuple[str, str | None]] = []
        self.fetch_ks: list[int] = []

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        self.calls.append(("recs", variant))
        self.fetch_ks.append(k)
        return self._recs.get(variant, _rows([])).head(k)

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        self.calls.append(("cold", variant))
        return self._cold.get(variant, _rows([])).head(k)


class _NoItems:
    category_column = "category"
    availability_filters: tuple[str, ...] = ()

    def get(self):
        return None, None, {}


class _GenreItems:
    category_column = "genre"
    availability_filters = ("in_stock",)

    def get(self):
        items = pd.DataFrame({"item_id": list("abcdef"), "genre": ["pop", "rock"] * 3})
        ids_by_category = {"pop": frozenset("ace"), "rock": frozenset("bdf")}
        return items, frozenset("abcde"), ids_by_category


def _service(reader: _Reader, monkeypatch, variant: str | None, items_cache=None) -> RecommendationService:
    monkeypatch.setattr(
        "cicerone.serve.recommendations.resolve_assignment",
        lambda *args, **kwargs: ("exp", variant) if variant else (None, None),
    )
    return RecommendationService(
        make_settings(mode="serve", serve_auth_token="secret"),
        reader,
        items_cache=items_cache or _NoItems(),
        assignment_overlay=lambda: (None, None),
    )


def test_assigned_variant_rows_are_ranked_and_truncated(monkeypatch) -> None:
    reader = _Reader({"treatment": _rows(["a", "b", "c"], "treatment")}, {})
    served = _service(reader, monkeypatch, "treatment").get("u1", 2, category=None, exclude_unavailable=True)
    assert served is not None
    assert served.recs["item_id"].tolist() == ["a", "b"]
    assert served.recs["rank"].tolist() == [1, 2]
    assert (served.used_fallback, served.experiment_id, served.variant) == (False, "exp", "treatment")
    assert reader.calls == [("recs", "treatment")]


def test_missing_variant_falls_back_to_leftover_rows_and_picks_control(monkeypatch) -> None:
    leftover = pd.concat([_rows(["a"], "zeta"), _rows(["b"], "control")], ignore_index=True)
    reader = _Reader({None: leftover}, {})
    served = _service(reader, monkeypatch, "treatment").get("u1", 5, category=None, exclude_unavailable=True)
    assert served is not None
    assert served.recs["item_id"].tolist() == ["a", "b"]
    assert served.variant == "control"
    assert not served.used_fallback
    assert reader.calls == [("recs", "treatment"), ("recs", None)]


def test_cold_start_retries_without_variant(monkeypatch) -> None:
    reader = _Reader({}, {None: _rows(["x"], "control")})
    served = _service(reader, monkeypatch, "treatment").get("u1", 5, category=None, exclude_unavailable=True)
    assert served is not None
    assert served.recs["item_id"].tolist() == ["x"]
    assert served.used_fallback
    assert served.variant == "control"
    assert reader.calls == [
        ("recs", "treatment"),
        ("recs", None),
        ("cold", "treatment"),
        ("cold", None),
    ]


def test_rows_without_variant_column_clear_assignment(monkeypatch) -> None:
    reader = _Reader({"treatment": _rows(["a"])}, {})
    served = _service(reader, monkeypatch, "treatment").get("u1", 5, category=None, exclude_unavailable=True)
    assert served is not None
    assert (served.experiment_id, served.variant) == (None, None)


@pytest.mark.parametrize("variant", [None, "treatment"])
def test_no_rows_and_no_cold_start_returns_none(monkeypatch, variant) -> None:
    reader = _Reader({}, {})
    assert (
        _service(reader, monkeypatch, variant).get("u1", 5, category=None, exclude_unavailable=True) is None
    )


def test_filter_config_comes_from_items_cache(monkeypatch) -> None:
    reader = _Reader({None: _rows(list("abcdef"))}, {})
    service = _service(reader, monkeypatch, None, items_cache=_GenreItems())

    served = service.get("u1", 2, category="pop", exclude_unavailable=True)
    assert served is not None
    assert served.recs["item_id"].tolist() == ["a", "c"]
    assert served.recs["rank"].tolist() == [1, 2]

    served = service.get("u1", 1, category=None, exclude_unavailable=True)
    assert served is not None
    assert served.recs["item_id"].tolist() == ["a"]
    assert reader.fetch_ks == [10, 5]

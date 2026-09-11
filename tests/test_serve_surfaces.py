from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient
from test_serve import _FakeReader, _feature_config, _items_df, _recs_df, _settings

from cicerone.io.surfaces_reader import EmptySurfacesReader
from cicerone.serve import create_app


class _Surfaces:
    def __init__(self) -> None:
        self.popular = pd.DataFrame(
            [
                {"item_id": "i1", "rank": 1, "score": 10.0, "source": "popular_fallback"},
                {"item_id": "i2", "rank": 2, "score": 4.0, "source": "popular_fallback"},
            ]
        )
        self.latest = pd.DataFrame([{"item_id": "i2", "rank": 1, "score": 2.0, "source": "latest"}])
        self.neighbors = pd.DataFrame(
            [
                {"item_id": "i1", "neighbor_id": "i2", "rank": 1, "score": 0.8},
                {"item_id": "i1", "neighbor_id": "i3", "rank": 2, "score": 0.2},
            ]
        )

    def get_popular(self, k: int) -> pd.DataFrame:
        return self.popular.head(k).reset_index(drop=True)

    def get_latest(self, k: int) -> pd.DataFrame:
        return self.latest.head(k).reset_index(drop=True)

    def get_similar(self, item_id: str, k: int) -> pd.DataFrame:
        rows = self.neighbors[self.neighbors["item_id"] == item_id]
        return rows.head(k).reset_index(drop=True)

    def refresh(self) -> None:
        return


def _app(**kwargs):
    return create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
        surfaces=_Surfaces(),
        **kwargs,
    )


def test_popular_latest_similar_and_session():
    client = TestClient(_app())
    headers = {"Authorization": "Bearer secret"}
    popular = client.get("/popular", headers=headers).json()
    assert [row["item_id"] for row in popular["items"]] == ["i1", "i2"]
    latest = client.get("/latest", headers=headers).json()
    assert [row["item_id"] for row in latest["items"]] == ["i2"]
    similar = client.get("/similar/i1", headers=headers).json()
    assert similar["item_id"] == "i1"
    assert [row["item_id"] for row in similar["items"]] == ["i2"]
    session = client.post("/session/recommendations", json={"items": ["i1"]}, headers=headers).json()
    assert session["fallback"] is False
    assert [row["item_id"] for row in session["items"]] == ["i2"]


def test_session_falls_back_to_popular():
    client = TestClient(_app())
    body = client.post(
        "/session/recommendations",
        json={"items": ["unknown"]},
        headers={"Authorization": "Bearer secret"},
    ).json()
    assert body["fallback"] is True
    assert [row["item_id"] for row in body["items"]] == ["i1", "i2"]


def test_empty_surfaces_return_404():
    app = create_app(_settings(), _FakeReader(_recs_df()), surfaces=EmptySurfacesReader())
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/popular", headers=headers).status_code == 404
    assert client.get("/similar/i1", headers=headers).status_code == 404


def test_session_requires_an_item():
    client = TestClient(_app())
    response = client.post(
        "/session/recommendations",
        json={"items": []},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400

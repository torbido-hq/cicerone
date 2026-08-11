from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from conftest import make_settings
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from cicerone.config import Settings
from cicerone.io.recommendation_reader import DatasetRecommendationReader
from cicerone.serve import create_app
from cicerone.serve.metrics import METRICS_TOKEN_HEADER


def _settings(**overrides) -> Settings:
    return make_settings(**{"mode": "serve", "serve_auth_token": "secret", **overrides})


class _FakeReader:
    def __init__(self, recs: pd.DataFrame):
        self._recs = recs

    def refresh(self) -> None:
        pass

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._recs[self._recs["user_id"] == user_id].sort_values("rank")
        return rows.head(k).reset_index(drop=True)

    def get_items(self):
        return None

    def items_version(self) -> int:
        return 0

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return self._recs.iloc[0:0]

    def configure_item_filters(self, *, category_column=None, availability_filters=()) -> None:
        del category_column, availability_filters


def _recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "item_based"},
        ]
    )


def _metric_samples(body: str, name: str) -> list[tuple[dict[str, str], float]]:
    out: list[tuple[dict[str, str], float]] = []
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name != name:
                continue
            out.append((dict(sample.labels), sample.value))
    return out


def test_metrics_returns_prometheus_text_format():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "cicerone_up" in response.text
    text_string_to_metric_families(response.text)


def test_metrics_disabled_returns_not_found():
    app = create_app(_settings(serve_metrics_enabled=False), _FakeReader(_recs_df()))
    response = TestClient(app).get("/metrics")
    assert response.status_code == 404


def test_metrics_token_required_when_configured():
    app = create_app(_settings(serve_metrics_token="metrics-secret"), _FakeReader(_recs_df()))
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers={METRICS_TOKEN_HEADER: "metrics-secret"})
    assert response.status_code == 200


def test_metrics_does_not_require_bearer_token():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200


def test_recommend_request_increments_request_and_source_metrics():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)
    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200

    metrics = client.get("/metrics").text
    requests = _metric_samples(metrics, "cicerone_requests_total")
    assert any(
        labels.get("endpoint") == "/recommendations/{user_id}"
        and labels.get("method") == "GET"
        and labels.get("status") == "200"
        for labels, _ in requests
    )

    served = _metric_samples(metrics, "cicerone_recommendations_served_total")
    sources = {labels["source"] for labels, value in served if value >= 1}
    assert sources == {"collaborative", "item_based"}


def test_cache_refresh_metrics_on_success_and_failure(tmp_path: Path):
    recs = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)

    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})
    reader.refresh()
    time.sleep(0.01)

    app = create_app(_settings(), reader)
    metrics = TestClient(app).get("/metrics").text
    refresh = _metric_samples(metrics, "cicerone_cache_refresh_total")
    assert any(labels.get("status") == "success" and value >= 1 for labels, value in refresh)
    age = _metric_samples(metrics, "cicerone_cache_age_seconds")
    assert any(value >= 0 for _, value in age)

    (tmp_path / "recommendations.parquet").unlink()
    reader.refresh()
    metrics_after_failure = TestClient(app).get("/metrics").text
    refresh_after = _metric_samples(metrics_after_failure, "cicerone_cache_refresh_total")
    assert any(labels.get("status") == "failure" and value >= 1 for labels, value in refresh_after)


def test_cache_hit_and_miss_counters(tmp_path: Path):
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {
                "user_id": "__cold_start__",
                "item_id": "i9",
                "rank": 1,
                "score": 0.4,
                "source": "popular_fallback",
            },
        ]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)
    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})

    app = create_app(_settings(), reader)
    client = TestClient(app)
    client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    client.get("/recommendations/unknown", headers={"Authorization": "Bearer secret"})

    metrics = client.get("/metrics").text
    hits = sum(value for _, value in _metric_samples(metrics, "cicerone_cache_hits_total"))
    misses = sum(value for _, value in _metric_samples(metrics, "cicerone_cache_misses_total"))
    assert hits >= 1
    assert misses >= 1

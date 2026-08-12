from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from conftest import make_settings
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
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


def _metric_value(body: str, name: str, labels: dict[str, str] | None = None) -> float:
    total = 0.0
    for sample_labels, value in _metric_samples(body, name):
        if labels is not None and sample_labels != labels:
            continue
        total += value
    return total


def test_metrics_returns_prometheus_text_format():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
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


def test_metrics_endpoint_excluded_from_request_counters():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)
    before = client.get("/metrics").text
    before_metrics_requests = sum(
        value
        for labels, value in _metric_samples(before, "cicerone_requests_total")
        if labels.get("endpoint") == "/metrics"
    )

    for _ in range(3):
        assert client.get("/metrics").status_code == 200

    after = client.get("/metrics").text
    after_metrics_requests = sum(
        value
        for labels, value in _metric_samples(after, "cicerone_requests_total")
        if labels.get("endpoint") == "/metrics"
    )
    assert before_metrics_requests == 0
    assert after_metrics_requests == 0


def test_recommend_request_increments_request_and_source_metrics():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)
    before = client.get("/metrics").text
    before_requests = _metric_value(
        before,
        "cicerone_requests_total",
        {"endpoint": "/recommendations/{user_id}", "method": "GET", "status": "200"},
    )
    before_collab = _metric_value(
        before, "cicerone_recommendations_served_total", {"source": "collaborative"}
    )
    before_item = _metric_value(before, "cicerone_recommendations_served_total", {"source": "item_based"})

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200

    after = client.get("/metrics").text
    assert (
        _metric_value(
            after,
            "cicerone_requests_total",
            {"endpoint": "/recommendations/{user_id}", "method": "GET", "status": "200"},
        )
        == before_requests + 1
    )
    assert (
        _metric_value(after, "cicerone_recommendations_served_total", {"source": "collaborative"})
        == before_collab + 1
    )
    assert (
        _metric_value(after, "cicerone_recommendations_served_total", {"source": "item_based"})
        == before_item + 1
    )


def test_cache_refresh_metrics_on_success_and_failure(tmp_path: Path):
    recs = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)

    # Snapshot counters before constructing the reader (its __init__ calls refresh).
    probe = TestClient(create_app(_settings(), _FakeReader(_recs_df()))).get("/metrics").text
    before_success = _metric_value(probe, "cicerone_cache_refresh_total", {"status": "success"})
    before_failure = _metric_value(probe, "cicerone_cache_refresh_total", {"status": "failure"})

    reader = DatasetRecommendationReader({"storage_backend": "local", "path": str(tmp_path)})
    reader.refresh()
    time.sleep(0.01)

    app = create_app(_settings(), reader)
    metrics = TestClient(app).get("/metrics").text
    assert _metric_value(metrics, "cicerone_cache_refresh_total", {"status": "success"}) >= before_success + 2
    age = _metric_samples(metrics, "cicerone_cache_age_seconds")
    assert any(value >= 0 for _, value in age)

    (tmp_path / "recommendations.parquet").unlink()
    reader.refresh()
    metrics_after_failure = TestClient(app).get("/metrics").text
    assert (
        _metric_value(metrics_after_failure, "cicerone_cache_refresh_total", {"status": "failure"})
        == before_failure + 1
    )


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
    before = client.get("/metrics").text
    before_hits = _metric_value(before, "cicerone_cache_hits_total")
    before_misses = _metric_value(before, "cicerone_cache_misses_total")

    client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    client.get("/recommendations/unknown", headers={"Authorization": "Bearer secret"})

    after = client.get("/metrics").text
    assert _metric_value(after, "cicerone_cache_hits_total") == before_hits + 1
    assert _metric_value(after, "cicerone_cache_misses_total") == before_misses + 1


def test_unknown_stored_source_is_ignored():
    from prometheus_client import generate_latest

    from cicerone.serve.metrics import record_recommendations_served

    before = _metric_value(
        generate_latest().decode(),
        "cicerone_recommendations_served_total",
        {"source": "collaborative"},
    )
    record_recommendations_served({"content_fallback", "personalized", "blended"})
    after = _metric_value(
        generate_latest().decode(),
        "cicerone_recommendations_served_total",
        {"source": "collaborative"},
    )
    # content_fallback ignored; personalized+blended collapse to one collaborative inc
    assert after == before + 1


def test_cache_age_zero_before_successful_refresh(monkeypatch):
    from prometheus_client import generate_latest

    import cicerone.serve.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "_last_successful_refresh_at", None)
    metrics_mod.update_cache_age_gauge()
    assert _metric_value(generate_latest().decode(), "cicerone_cache_age_seconds") == 0.0

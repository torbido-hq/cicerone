"""Robot Framework keyword library backing the Cicerone system/E2E suites.

Wraps ``tests/support`` helpers (catalog fixtures, TOML config, serve/dashboard
app mounts) so .robot suites can seed a catalog, run the real batch job, then
exercise serve/dashboard exactly like production — the Robot replacement for
the former ``tests/test_system_*.py`` pytest modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from support.postgres_defaults import resolve_test_database_url
from support.system_db import reset_schema, seed_catalog
from support.system_spec import (
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    append_dataset_events,
    available_recommendation_ids,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    run_system_job,
    sample_system_catalog,
    seed_dataset_catalog,
    write_system_config,
)

from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.blending import COLD_START_USER_ID
from cicerone.config import load_settings
from cicerone.feature_config import load_feature_config
from cicerone.io.factory import build_manifest_reader, build_output_sink, build_recommendation_reader
from cicerone.track.store import TrackStore
from cicerone.track.store_common import EVAL_FILENAME, HISTORY_DIR, TRACK_FILENAME

_SERVE_HEADERS = {"Authorization": f"Bearer {SYSTEM_SERVE_TOKEN}"}
_DASHBOARD_AUTH = (SYSTEM_DASHBOARD_USER, SYSTEM_DASHBOARD_PASSWORD)
_OUTPUT_FILES = ("recommendations.parquet", "items_snapshot.parquet", "manifest.json", "model.artifact")


class CiceroneSystemLibrary:
    """One instance per suite — mirrors the module-scoped pytest fixtures it replaces."""

    ROBOT_LIBRARY_SCOPE = "TEST SUITE"

    def __init__(self) -> None:
        self.database_url = resolve_test_database_url()
        self.engine: Engine | None = None
        self.config_path: Path | None = None
        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.events: pd.DataFrame | None = None
        self.users: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.settings: Any = None

    # -- environment ---------------------------------------------------
    def postgres_test_database_available(self) -> bool:
        return bool(self.database_url)

    def reset_postgres_schema(self) -> None:
        if self.engine is None:
            self.engine = create_engine(self.database_url, pool_pre_ping=True)
        reset_schema(self.engine)

    # -- setup -----------------------------------------------------------
    def seed_and_train_postgres_system(self, triggered_by: str = "system-spec") -> None:
        self.events, self.users, self.items = sample_system_catalog()
        seed_catalog(self.engine, self.events, self.users, self.items)
        root = Path(tempfile.mkdtemp(prefix="cicerone-robot-db-"))
        self.config_path = write_system_config(root / "cicerone.toml", database_url=self.database_url)
        self.run_system_job_again(triggered_by)

    def seed_and_train_dataset_system(self, triggered_by: str = "system-spec") -> None:
        root = Path(tempfile.mkdtemp(prefix="cicerone-robot-dataset-"))
        self.input_path = root / "in"
        self.output_path = root / "out"
        self.output_path.mkdir(parents=True)
        self.events, self.users, self.items = sample_system_catalog()
        seed_dataset_catalog(self.input_path, self.events, self.users, self.items)
        self.config_path = write_system_config(
            root / "cicerone.toml",
            kind="dataset",
            input_path=self.input_path,
            output_path=self.output_path,
        )
        self.run_system_job_again(triggered_by)

    def run_system_job_again(self, triggered_by: str) -> None:
        run_system_job(self.config_path, triggered_by=triggered_by)
        self.settings = load_settings(str(self.config_path))

    def append_postgres_conversion_event(self, user_id: str, item_id: str) -> None:
        from support.system_db import postgres_ready

        conversion = pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": pd.Timestamp("2026-09-01T14:00:00Z"),
                }
            ]
        )
        from cicerone.io.db_store import DEFAULT_EVENTS_TABLE

        postgres_ready(conversion).to_sql(DEFAULT_EVENTS_TABLE, self.engine, if_exists="append", index=False)

    def append_dataset_conversion_event(self, user_id: str, item_id: str) -> None:
        conversion = pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": pd.Timestamp("2026-09-01T14:00:00Z"),
                }
            ]
        )
        append_dataset_events(self.input_path, conversion)

    # -- readers / clients -------------------------------------------------
    def _rec_reader(self):
        return build_recommendation_reader(self.settings.output)

    def _manifest_reader(self):
        return build_manifest_reader(self.settings.output)

    def _serve_client(self) -> TestClient:
        return TestClient(mount_serve_app(self.settings))

    def _dashboard_client(self) -> TestClient:
        return TestClient(mount_dashboard_app(self.settings, dashboard_users(), config_path=self.config_path))

    def _expected_ids(self, user_id: str, k: int | None = None) -> list[str]:
        feature_config = load_feature_config(self.settings.feature_config_path)
        raw = self._rec_reader().get_recommendations(user_id, k=k or self.settings.serve.default_k * 10)
        return available_recommendation_ids(
            raw,
            self.items,
            availability_filters=feature_config.item_availability_filters,
            category_column=self.settings.serve.category_column,
            k=k or self.settings.serve.default_k,
        )

    # -- job / reader assertions -------------------------------------------
    def recommendation_reader_should_cover_catalog_items(self) -> None:
        scores = self._rec_reader().get_item_scores()
        assert not scores.empty
        assert set(scores.columns) >= {"item_id", "popular_score", "latest_score", "n_users"}
        assert set(self.items["item_id"]) <= set(scores["item_id"])

    def recommendations_should_be_served_for_all_known_users(self) -> None:
        expected_users = set(self.events["user_id"]) | set(self.users["user_id"])
        reader = self._rec_reader()
        for user_id in sorted(expected_users):
            served = reader.get_recommendations(user_id, k=10)
            assert not served.empty, f"expected recommendations for {user_id}"
            assert set(served.columns) >= {"user_id", "item_id", "rank", "score", "source"}
            assert served["rank"].min() >= 1

    def user_u1_recommendations_should_respect_requested_count(self) -> None:
        served = self._rec_reader().get_recommendations("u1", k=2)
        assert len(served) == 2
        assert set(served["user_id"]) == {"u1"}
        assert list(served["rank"]) == sorted(served["rank"].tolist())

    def manifest_should_record_successful_run(self, triggered_by: str, expected_n_events: int = -1) -> None:
        latest = self._manifest_reader().read_latest()
        assert latest is not None
        assert latest["status"] == "success"
        assert latest["triggered_by"] == triggered_by
        if expected_n_events >= 0:
            assert int(latest["n_events"]) == expected_n_events
        assert bool(latest["artifact_written"]) is True
        assert int(latest["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION

    def artifact_should_be_loadable_and_produce_recommendations(self) -> None:
        payload = build_output_sink(self.settings.output).read_model_artifact()
        assert payload is not None
        loaded = loads_artifact(payload)
        assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
        assert "collaborative" in loaded.models or "popular" in loaded.models
        from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
        assert not from_artifact.empty
        artifact_users = set(from_artifact["user_id"].astype(str))
        assert {"u1", "u2"} <= artifact_users
        assert artifact_users <= {"u1", "u2", COLD_START_USER_ID}

    # -- serve assertions ----------------------------------------------------
    def serve_should_return_recommendations_for_user(self, user_id: str) -> None:
        response = self._serve_client().get(f"/recommendations/{user_id}", headers=_SERVE_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == user_id
        assert body["fallback"] is False
        assert body["items"]
        assert [row["item_id"] for row in body["items"]] == self._expected_ids(user_id)

    def serve_should_fallback_for_unknown_user(self) -> None:
        response = self._serve_client().get("/recommendations/u-unknown", headers=_SERVE_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["fallback"] is True
        assert body["items"]

    def serve_should_support_category_filter(self, user_id: str, category: str, *allowed_ids: str) -> None:
        response = self._serve_client().get(
            f"/recommendations/{user_id}?category={category}", headers=_SERVE_HEADERS
        )
        assert response.status_code == 200
        served_ids = {row["item_id"] for row in response.json()["items"]}
        assert served_ids <= set(allowed_ids)

    def serve_should_support_exclude_unavailable(self, user_id: str, *excluded_ids: str) -> None:
        response = self._serve_client().get(
            f"/recommendations/{user_id}?exclude_unavailable=true&limit=10", headers=_SERVE_HEADERS
        )
        assert response.status_code == 200
        served_ids = {row["item_id"] for row in response.json()["items"]}
        assert served_ids.isdisjoint(excluded_ids)

    # -- dashboard assertions --------------------------------------------
    def dashboard_should_show_latest_run_summary(self, triggered_by: str) -> None:
        response = self._dashboard_client().get("/dashboard", auth=_DASHBOARD_AUTH)
        assert response.status_code == 200
        assert "success" in response.text
        assert triggered_by in response.text
        assert str(len(self.events)) in response.text

    def dashboard_should_show_recommendations_for_user(self, user_id: str) -> None:
        expected_ids = self._expected_ids(user_id)
        response = self._dashboard_client().get(
            "/dashboard", params={"user_id": user_id}, auth=_DASHBOARD_AUTH
        )
        assert response.status_code == 200
        assert "Recommendations for" in response.text
        assert user_id in response.text
        for item_id in expected_ids:
            assert item_id in response.text

    # -- track / quality -----------------------------------------------------
    def track_impressions_and_click_for_user(self, user_id: str) -> dict[str, Any]:
        serve = self._serve_client()
        served = serve.get(f"/recommendations/{user_id}", headers=_SERVE_HEADERS)
        assert served.status_code == 200
        body = served.json()
        items = body["items"]
        assert items
        generated_at = body["generated_at"]
        first_item = str(items[0]["item_id"])
        occurred = "2026-09-01T13:00:00Z"
        impressions = [
            {
                "kind": "impression",
                "user_id": user_id,
                "item_id": str(row["item_id"]),
                "rank": int(row["rank"]),
                "occurred_at": occurred,
                "event_id": f"sys-imp-{row['rank']}",
                "generated_at": generated_at,
            }
            for row in items
        ]
        click = {
            "kind": "click",
            "user_id": user_id,
            "item_id": first_item,
            "rank": int(items[0]["rank"]),
            "occurred_at": "2026-09-01T13:05:00Z",
            "event_id": "sys-clk-1",
            "generated_at": generated_at,
        }
        tracked = serve.post("/track", headers=_SERVE_HEADERS, json={"events": [*impressions, click]})
        assert tracked.status_code == 202
        assert tracked.json()["accepted"] == len(impressions) + 1
        event_ids = [event["event_id"] for event in impressions] + ["sys-clk-1"]
        return {
            "first_item": first_item,
            "impression_count": len(impressions),
            "event_ids": event_ids,
        }

    def track_store_should_contain_tracked_events(self, event_ids: list[str]) -> None:
        store = TrackStore(self.settings.output)
        rows = store.read_rows()
        assert {row["event_id"] for row in rows} >= set(event_ids)

    def dashboard_quality_page_should_show_tracked_metrics(self, impression_count: int) -> None:
        import re

        client = self._dashboard_client()
        response = client.get("/dashboard/quality", auth=_DASHBOARD_AUTH)
        assert response.status_code == 200
        assert "Could not load quality metrics." not in response.text
        assert "No impressions yet." not in response.text
        assert re.search(rf"Impressions</dt><dd[^>]*>{impression_count}</dd>", response.text)
        assert re.search(r"Clicks</dt><dd[^>]*>1</dd>", response.text)

    # -- dataset-backend file contract --------------------------------------
    def dataset_output_files_should_match_contract(self) -> None:
        for name in ("events.parquet", "users.parquet", "items.parquet"):
            assert (self.input_path / name).is_file()
            assert not (self.output_path / name).exists()
        for name in _OUTPUT_FILES:
            assert (self.output_path / name).is_file()
            assert not (self.input_path / name).exists()

    def dataset_manifest_on_disk_should_match_reader(self) -> None:
        on_disk = json.loads((self.output_path / "manifest.json").read_text())
        latest = self._manifest_reader().read_latest()
        assert latest == on_disk
        recent = self._manifest_reader().read_recent(limit=5)
        assert recent == [on_disk]

    def dataset_track_files_should_exist_only_under_output(self) -> None:
        assert (self.output_path / TRACK_FILENAME).is_file()
        assert not (self.input_path / TRACK_FILENAME).exists()

    def dataset_eval_and_history_files_should_exist(self, impression_count: int) -> None:
        eval_path = self.output_path / EVAL_FILENAME
        assert eval_path.is_file()
        eval_file = json.loads(eval_path.read_text())
        assert int(eval_file["track_eval"]["overall"]["n_impressions"]) == impression_count
        history_parts = list((self.output_path / HISTORY_DIR).glob("*.parquet"))
        assert history_parts

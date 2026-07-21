from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cicerone import scheduler


def _write_config(tmp_path, cron_schedule: str) -> str:
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        cron_schedule = "{cron_schedule}"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """
    )
    return str(config_path)


def test_seconds_until_next_run_is_positive_and_bounded():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seconds = scheduler._seconds_until_next_run("0 3 * * *", now)
    assert 0 < seconds <= 24 * 3600


def test_seconds_until_next_run_at_exact_schedule_time_is_about_a_day():
    now = datetime(2026, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
    seconds = scheduler._seconds_until_next_run("0 3 * * *", now)
    assert seconds == pytest.approx(24 * 3600, abs=1)


def test_main_raises_on_invalid_cron_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, "not a cron expression"))
    with pytest.raises(RuntimeError, match="Invalid cron_schedule"):
        scheduler.main()


def test_main_runs_job_each_iteration_and_survives_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("CICERONE_CONFIG_PATH", _write_config(tmp_path, "* * * * *"))

    calls = {"sleep": 0, "run": 0}

    def fake_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] >= 2:
            raise SystemExit("stop the loop after two iterations")

    def fake_run():
        calls["run"] += 1
        raise ValueError("boom")  # main() must log and keep looping, not crash

    monkeypatch.setattr(scheduler.time, "sleep", fake_sleep)
    monkeypatch.setattr(scheduler.job, "run", fake_run)

    with pytest.raises(SystemExit):
        scheduler.main()

    assert calls["sleep"] == 2
    assert calls["run"] == 1

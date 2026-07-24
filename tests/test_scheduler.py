from __future__ import annotations

from datetime import UTC, datetime

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
    now = datetime(2026, 1, 1, tzinfo=UTC)
    seconds = scheduler._seconds_until_next_run("0 3 * * *", now)
    assert 0 < seconds <= 24 * 3600


def test_seconds_until_next_run_at_exact_schedule_time_is_about_a_day():
    now = datetime(2026, 1, 1, 3, 0, 0, tzinfo=UTC)
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

    def fake_run(triggered_by="cron"):
        calls["run"] += 1
        assert triggered_by == "cron"
        raise ValueError("boom")  # main() must log and keep looping, not crash

    monkeypatch.setattr(scheduler.time, "sleep", fake_sleep)
    monkeypatch.setattr(scheduler.job, "run", fake_run)

    with pytest.raises(SystemExit):
        scheduler.main()

    assert calls["sleep"] == 2
    assert calls["run"] == 1


def test_main_with_trigger_enabled_starts_cron_thread_and_serves_http(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        cron_schedule = "* * * * *"

        [job.trigger]
        enabled = true
        auth_token = "secret"
        poll_input_bucket = true

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    started_threads = []
    real_thread_init = scheduler.threading.Thread.__init__

    def fake_thread_init(self, *args, **kwargs):
        started_threads.append(kwargs.get("target"))
        real_thread_init(self, *args, **kwargs)

    uvicorn_calls = {}

    def fake_uvicorn_run(app, host, port):
        uvicorn_calls["app"] = app
        uvicorn_calls["host"] = host
        uvicorn_calls["port"] = port

    monkeypatch.setattr(scheduler.threading.Thread, "__init__", fake_thread_init)
    monkeypatch.setattr(scheduler.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(scheduler, "uvicorn", type("_U", (), {"run": staticmethod(fake_uvicorn_run)}))

    from cicerone.trigger import poll_input_forever

    scheduler.main()

    assert scheduler._cron_loop in started_threads
    assert poll_input_forever in started_threads
    assert uvicorn_calls["host"] == "0.0.0.0"
    assert uvicorn_calls["port"] == 8080

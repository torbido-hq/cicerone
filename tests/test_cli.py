from __future__ import annotations

import os

import pytest

from cicerone import __version__
from cicerone.cli import main


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    monkeypatch.delenv("CICERONE_CONFIG_PATH", raising=False)


def _write_config(tmp_path, *, mode: str = "batch", users_path: str | None = None) -> str:
    extra = ""
    if mode == "serve":
        extra += """
        [serve]
        auth_token = "test-token"
        """
    if users_path is not None:
        extra += f"""
        [dashboard]
        enabled = true
        users_path = "{users_path}"
        """
    path = tmp_path / "cicerone.toml"
    path.write_text(
        f"""
        [job]
        mode = "{mode}"
        cron_schedule = "0 3 * * *"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path / "in"}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path / "out"}"
        {extra}
        """
    )
    return str(path)


def test_version_prints_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_job_help_does_not_run_job(monkeypatch):
    monkeypatch.setattr("cicerone.job.run", lambda: pytest.fail("job must not run"))
    with pytest.raises(SystemExit) as exc_info:
        main(["job", "--help"])
    assert exc_info.value.code == 0


def test_start_rejects_extra_args(tmp_path):
    config = _write_config(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        main(["--config", config, "start", "nope"])
    assert exc_info.value.code == 2


def test_start_serve_mode_does_not_run_job(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))
    monkeypatch.setattr("cicerone.job.run", lambda: calls.append("job"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))

    assert main(["--config", config, "start"]) == 0
    assert calls == ["serve"]
    assert os.environ["CICERONE_CONFIG_PATH"] == config


def test_config_flag_after_command(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    monkeypatch.setattr("cicerone.serve.app.main", lambda: None)
    assert main(["start", "--config", config]) == 0
    assert os.environ["CICERONE_CONFIG_PATH"] == config


def test_config_equals_form_after_command(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    monkeypatch.setattr("cicerone.serve.app.main", lambda: None)
    assert main(["start", f"--config={config}"]) == 0
    assert os.environ["CICERONE_CONFIG_PATH"] == config


def test_config_flag_without_value_after_command_errors():
    with pytest.raises(SystemExit) as exc_info:
        main(["start", "--config"])
    assert exc_info.value.code == 2


def test_run_alias_batch_runs_job_then_scheduler(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="batch")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.job.run", lambda: calls.append("job"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))

    assert main(["-c", config, "run"]) == 0
    assert calls == ["job", "scheduler"]


def test_start_job_failure_skips_scheduler(tmp_path, monkeypatch):
    config = _write_config(tmp_path)

    def boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cicerone.job.run", boom)
    monkeypatch.setattr("cicerone.scheduler.main", lambda: pytest.fail("scheduler should not run"))

    assert main(["--config", config, "start"]) == 1


def test_job_success_and_failure(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    monkeypatch.setattr("cicerone.job.run", lambda: None)
    assert main(["--config", config, "job"]) == 0

    def boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cicerone.job.run", boom)
    assert main(["--config", config, "job"]) == 1


def test_serve_dashboard_scheduler_dispatch(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))
    monkeypatch.setattr("cicerone.dashboard.main", lambda: calls.append("dashboard"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))

    assert main(["--config", config, "serve"]) == 0
    assert main(["--config", config, "dashboard"]) == 0
    assert main(["--config", config, "scheduler"]) == 0
    assert calls == ["serve", "dashboard", "scheduler"]


def test_users_injects_users_path_from_config(tmp_path, monkeypatch):
    users_path = str(tmp_path / "dashboard_users.toml")
    config = _write_config(tmp_path, users_path=users_path)
    seen: list[list[str]] = []
    monkeypatch.setattr("cicerone.manage_dashboard_users.main", lambda argv: seen.append(list(argv)))

    assert main(["--config", config, "users", "list"]) == 0
    assert main(["--config", config, "users", "--users-path=/explicit", "add", "alice"]) == 0
    monkeypatch.delenv("CICERONE_CONFIG_PATH", raising=False)
    assert main(["users", "list"]) == 0
    monkeypatch.setenv("CICERONE_CONFIG_PATH", config)
    assert main(["users", "list"]) == 0

    assert seen == [
        ["--users-path", users_path, "list"],
        ["--users-path=/explicit", "add", "alice"],
        ["list"],
        ["--users-path", users_path, "list"],
    ]


def test_export_openapi_forwards_args(monkeypatch):
    seen: list[list[str] | None] = []

    def fake_main(argv=None):
        seen.append(argv)
        return 0

    monkeypatch.setattr("cicerone.export_serve_openapi.main", fake_main)
    assert main(["export-openapi", "-o", "docs/openapi/serve.openapi.json"]) == 0
    assert seen == [["-o", "docs/openapi/serve.openapi.json"]]


def test_package_main_exports_cli_main():
    import cicerone.__main__ as package_main

    assert package_main.main is main

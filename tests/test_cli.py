from __future__ import annotations

import os

import pytest

from cicerone import __version__
from cicerone.cli import main


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


def test_start_serve_mode_does_not_run_job(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))
    monkeypatch.setattr("cicerone.job.run", lambda: calls.append("job"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))

    main(["--config", config, "start"])

    assert calls == ["serve"]
    assert os.environ["CICERONE_CONFIG_PATH"] == config


def test_run_alias_batch_runs_job_then_scheduler(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="batch")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.job.run", lambda: calls.append("job"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))

    main(["-c", config, "run"])

    assert calls == ["job", "scheduler"]


def test_start_job_failure_skips_scheduler(tmp_path, monkeypatch):
    config = _write_config(tmp_path)

    def boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cicerone.job.run", boom)
    monkeypatch.setattr("cicerone.scheduler.main", lambda: pytest.fail("scheduler should not run"))

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", config, "start"])
    assert exc_info.value.code == 1


def test_job_success_and_failure(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    monkeypatch.setattr("cicerone.job.run", lambda: None)
    main(["--config", config, "job"])

    def boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cicerone.job.run", boom)
    with pytest.raises(SystemExit) as exc_info:
        main(["--config", config, "job"])
    assert exc_info.value.code == 1


def test_serve_dashboard_scheduler_dispatch(tmp_path, monkeypatch):
    config = _write_config(tmp_path, mode="serve")
    calls: list[str] = []
    monkeypatch.setattr("cicerone.serve.app.main", lambda: calls.append("serve"))
    monkeypatch.setattr("cicerone.dashboard.main", lambda: calls.append("dashboard"))
    monkeypatch.setattr("cicerone.scheduler.main", lambda: calls.append("scheduler"))

    main(["--config", config, "serve"])
    main(["--config", config, "dashboard"])
    main(["--config", config, "scheduler"])
    assert calls == ["serve", "dashboard", "scheduler"]


def test_users_injects_users_path_from_config(tmp_path, monkeypatch):
    users_path = str(tmp_path / "dashboard_users.toml")
    config = _write_config(tmp_path, users_path=users_path)
    seen: list[list[str]] = []
    monkeypatch.setattr("cicerone.manage_dashboard_users.main", lambda argv: seen.append(list(argv)))

    main(["--config", config, "users", "list"])
    main(["--config", config, "users", "--users-path", "/explicit", "add", "alice"])
    main(["users", "list"])

    assert seen == [
        ["--users-path", users_path, "list"],
        ["--users-path", "/explicit", "add", "alice"],
        ["list"],
    ]


def test_export_openapi_forwards_args(monkeypatch):
    seen: list[list[str] | None] = []

    def fake_main(argv=None):
        seen.append(argv)
        return 0

    monkeypatch.setattr("cicerone.export_serve_openapi.main", fake_main)
    with pytest.raises(SystemExit) as exc_info:
        main(["export-openapi", "-o", "docs/openapi/serve.openapi.json"])
    assert exc_info.value.code == 0
    assert seen == [["-o", "docs/openapi/serve.openapi.json"]]


def test_package_main_exports_cli_main():
    import cicerone.__main__ as package_main

    assert package_main.main is main

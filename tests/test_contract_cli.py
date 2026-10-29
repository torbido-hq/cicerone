from __future__ import annotations

import re

import pytest

from cicerone.cli import main

GLOBAL_OPTIONS = ["--config", "--help", "--log-format", "--log-level", "-c", "-h"]


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    monkeypatch.delenv("CICERONE_CONFIG_PATH", raising=False)


def _help(capsys, argv: list[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    return capsys.readouterr().out


def _options(text: str) -> list[str]:
    found: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^  (-\S.*?)(?:\s{2,}|$)", line)
        if match:
            found.update(re.findall(r"(?<![\w-])--?[A-Za-z][\w-]*", match.group(1)))
    return sorted(found)


def _choices(text: str) -> list[str]:
    match = re.search(r"\{([\w,-]+)\}", text)
    assert match is not None
    return match.group(1).split(",")


def test_top_level_commands_and_options(capsys):
    text = _help(capsys, ["--help"])
    assert _choices(text) == [
        "start",
        "run",
        "job",
        "serve",
        "dashboard",
        "scheduler",
        "users",
        "export-openapi",
    ]
    assert _options(text) == sorted([*GLOBAL_OPTIONS, "--version", "-V"])


@pytest.mark.parametrize("command", ["start", "run", "job", "serve", "dashboard", "scheduler"])
def test_command_options(capsys, command):
    assert _options(_help(capsys, [command, "--help"])) == GLOBAL_OPTIONS


def test_users_command(capsys):
    text = _help(capsys, ["users", "--help"])
    assert _choices(text) == ["add", "remove", "list"]
    assert _options(text) == ["--help", "--users-path", "-h"]


def test_export_openapi_command(capsys):
    assert _options(_help(capsys, ["export-openapi", "--help"])) == ["--help", "--output", "-h", "-o"]

from __future__ import annotations


def write_toml(tmp_path, content: str) -> str:
    path = tmp_path / "cicerone.toml"
    path.write_text(content)
    return str(path)

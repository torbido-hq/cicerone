"""Invariants for examples/serve/curl_examples.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "serve" / "curl_examples.sh"


def test_events_body_uses_json_dumps_not_interpolation():
    text = SCRIPT.read_text()
    assert "json.dumps" in text
    assert 'os.environ["USER_ID"]' in text
    assert '-d "$events_body"' in text
    assert "command -v python3" in text
    assert "PYTHON=python3" in text
    assert '"$PYTHON"' in text
    assert '"user_id":"${USER_ID}"' not in text
    assert '\\"${USER_ID}\\"' not in text

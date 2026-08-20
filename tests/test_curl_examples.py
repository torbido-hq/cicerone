"""Invariants for examples/serve/curl_examples.sh."""

from __future__ import annotations

from pathlib import Path

from cicerone.serve.code_samples import PYTHON_DETECT, PYTHON_DETECT_PATH

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "serve" / "curl_examples.sh"


def test_events_body_uses_json_dumps_not_interpolation():
    text = SCRIPT.read_text()
    assert "json.dumps" in text
    assert 'os.environ["USER_ID"]' in text
    assert '-d "$events_body"' in text
    assert "python_detect.sh" in text
    assert '"$PYTHON"' in text
    assert '"user_id":"${USER_ID}"' not in text
    assert '\\"${USER_ID}\\"' not in text


def test_curl_examples_sources_shared_python_detect():
    assert PYTHON_DETECT_PATH.is_file()
    assert PYTHON_DETECT_PATH.read_text(encoding="utf-8") == PYTHON_DETECT
    assert "python_detect.sh" in SCRIPT.read_text()

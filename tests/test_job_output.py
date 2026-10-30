from __future__ import annotations

import pytest

from cicerone.job_eval import retrain_fence
from cicerone.job_output import (
    JobManifest,
    ensure_publication_fence,
    skip_stale_job_manifest,
    truncate_job_error,
    write_job_manifest,
)
from cicerone.locks import LockLostError, WriteFence


def test_truncate_job_error_short() -> None:
    assert truncate_job_error(RuntimeError("boom")) == "boom"


def test_skip_stale_job_manifest_on_lock_loss() -> None:
    assert skip_stale_job_manifest(WriteFence(), exc=LockLostError("lost", kind="retrain")) is True
    assert skip_stale_job_manifest(WriteFence(check=lambda: False)) is True
    assert skip_stale_job_manifest(WriteFence(check=lambda: True)) is False
    assert skip_stale_job_manifest(WriteFence()) is False


def test_ensure_publication_fence_raises_when_retrain_lost() -> None:
    with pytest.raises(LockLostError) as exc_info:
        ensure_publication_fence(object(), retrain_fence(None, lambda: False))
    assert exc_info.value.kind == "retrain"
    ensure_publication_fence(object(), retrain_fence(None, None))
    ensure_publication_fence(object(), retrain_fence(None, lambda: True))


def test_job_manifest_dict_compat() -> None:
    manifest = JobManifest(triggered_by="cron", top_k=10)
    manifest["status"] = "success"
    manifest.update({"n_events": 3, "generated_at": "t"})
    assert manifest.get("status") == "success"
    assert manifest.get("missing") is None
    assert dict(manifest)["n_events"] == 3
    written: list[dict] = []

    class _Sink:
        def write_manifest(self, payload: dict, *, skip_if_newer_than=None) -> bool:
            written.append(payload)
            return True

    assert write_job_manifest(_Sink(), manifest) is True
    assert written[0]["triggered_by"] == "cron"
    assert written[0]["generated_at"] == "t"


def test_job_manifest_unknown_keys_raise_keyerror() -> None:
    manifest = JobManifest()
    try:
        manifest["unknown"]
    except KeyError as exc:
        assert exc.args == ("unknown",)
    else:
        raise AssertionError("expected KeyError")
    try:
        manifest["unknown"] = 1
    except KeyError as exc:
        assert exc.args == ("unknown",)
    else:
        raise AssertionError("expected KeyError")
    try:
        manifest.update({"status": "success", "unknown": 1})
    except KeyError as exc:
        assert exc.args == ("unknown",)
    else:
        raise AssertionError("expected KeyError")
    assert manifest.status == "failed"
    assert manifest.get("unknown") is None

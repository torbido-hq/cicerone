from __future__ import annotations

from cicerone.job_output import (
    JobManifest,
    ensure_fence,
    skip_stale_job_manifest,
    truncate_job_error,
    write_job_manifest,
    write_manifest_accepts_skip,
)
from cicerone.locks import LockLostError


def test_truncate_job_error_short() -> None:
    assert truncate_job_error(RuntimeError("boom")) == "boom"


def test_skip_stale_job_manifest_on_lock_loss() -> None:
    assert skip_stale_job_manifest(exc=LockLostError("lost", kind="retrain")) is True
    assert skip_stale_job_manifest(fence_check=lambda: False) is True
    assert skip_stale_job_manifest(fence_check=lambda: True) is False


def test_ensure_fence_raises_when_lost() -> None:
    try:
        ensure_fence(lambda: False)
    except LockLostError as exc:
        assert exc.kind == "retrain"
    else:
        raise AssertionError("expected LockLostError")
    ensure_fence(None)
    ensure_fence(lambda: True)


def test_write_job_manifest_legacy_and_skip() -> None:
    written: list[dict] = []

    class _Legacy:
        def write_manifest(self, manifest: dict) -> None:
            written.append(manifest)

    assert write_job_manifest(_Legacy(), {"status": "failed"}, skip_if_newer_than="x") is True
    assert written == [{"status": "failed"}]
    assert write_manifest_accepts_skip(object()) is False


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

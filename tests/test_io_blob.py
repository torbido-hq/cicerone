from __future__ import annotations

from botocore.exceptions import ClientError

from cicerone.io.blob import append_storage_bytes, read_storage_bytes, write_storage_bytes


def test_local_read_write_append_round_trip(tmp_path) -> None:
    options = {"storage_backend": "local", "path": str(tmp_path)}
    assert read_storage_bytes(options, "missing.bin") is None
    write_storage_bytes(options, "note.txt", b"hello", "text/plain")
    assert read_storage_bytes(options, "note.txt") == b"hello"
    append_storage_bytes(options, "note.txt", b" world")
    assert read_storage_bytes(options, "note.txt") == b"hello world"


def test_s3_read_closes_body(mocker) -> None:
    body = mocker.Mock()
    body.read.return_value = b"payload"
    client = mocker.Mock()
    client.get_object.return_value = {"Body": body}
    mocker.patch("cicerone.io.blob.build_s3_client", return_value=client)
    options = {
        "storage_backend": "s3",
        "access_key_id": "id",
        "secret_access_key": "secret",
        "bucket": "bucket",
    }
    assert read_storage_bytes(options, "file.bin") == b"payload"
    body.close.assert_called_once()


def test_s3_read_missing_returns_none(mocker) -> None:
    client = mocker.Mock()
    client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
        "GetObject",
    )
    mocker.patch("cicerone.io.blob.build_s3_client", return_value=client)
    options = {
        "storage_backend": "s3",
        "access_key_id": "id",
        "secret_access_key": "secret",
        "bucket": "bucket",
    }
    assert read_storage_bytes(options, "file.bin") is None

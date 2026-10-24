from __future__ import annotations

import pytest

from cicerone.events.base import ClosableSource, EphemeralIdSource, HeartbeatSource
from cicerone.events.db import DbEventSource
from cicerone.events.kafka import KafkaEventSource
from cicerone.events.online import OnlineTrainer
from cicerone.events.rabbitmq import RabbitMQEventSource
from cicerone.events.redis_streams import RedisStreamsEventSource
from cicerone.events.s3 import S3EventSource
from cicerone.events.updater import AbortingOnlineRefresher, CommittingOnlineRefresher
from cicerone.events.webhook import WebhookEventSource
from cicerone.experiment.assignment import snapshot_variant_names
from cicerone.io.base import (
    ArtifactFingerprintSink,
    ConditionalArtifactSink,
    GuardedRecommendationsSink,
    VariantSnapshotReader,
    WriterHeldSink,
    ensure_sink_writer_held,
    recommendations_write_guard,
)
from cicerone.io.dataset_recommendation_reader import DatasetRecommendationReader
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.db_recommendation_reader import DbRecommendationReader
from cicerone.io.db_store import DatabaseOutputSink
from cicerone.locks.hold import GenerationReleaseLock, TryAcquireLock
from cicerone.locks.redis import RedisLock

_SINK_CAPABILITIES = (
    GuardedRecommendationsSink,
    WriterHeldSink,
    ArtifactFingerprintSink,
    ConditionalArtifactSink,
)
_POLLING_SOURCES = [S3EventSource, KafkaEventSource, RedisStreamsEventSource, RabbitMQEventSource]


@pytest.mark.parametrize("sink", [DatasetOutputSink, DatabaseOutputSink])
@pytest.mark.parametrize("capability", _SINK_CAPABILITIES)
def test_builtin_sinks_declare_every_capability(sink, capability):
    assert issubclass(sink, capability)


@pytest.mark.parametrize("reader", [DatasetRecommendationReader, DbRecommendationReader])
def test_builtin_readers_report_snapshot_variants(reader):
    assert issubclass(reader, VariantSnapshotReader)


@pytest.mark.parametrize("source", _POLLING_SOURCES)
def test_polling_sources_heartbeat(source):
    assert issubclass(source, HeartbeatSource)


@pytest.mark.parametrize("source", [*_POLLING_SOURCES, DbEventSource, WebhookEventSource])
def test_sources_close(source):
    assert issubclass(source, ClosableSource)


def test_rabbitmq_source_regenerates_event_ids():
    assert isinstance(object.__new__(RabbitMQEventSource), EphemeralIdSource)
    assert RabbitMQEventSource.ephemeral_event_ids is True


def test_online_trainer_stages_commits():
    assert issubclass(OnlineTrainer, CommittingOnlineRefresher)
    assert issubclass(OnlineTrainer, AbortingOnlineRefresher)


def test_online_commit_and_abort_are_independent():
    class CommitOnly:
        def commit(self) -> None:
            return None

    class AbortOnly:
        def abort(self) -> None:
            return None

    assert isinstance(CommitOnly(), CommittingOnlineRefresher)
    assert not isinstance(CommitOnly(), AbortingOnlineRefresher)
    assert isinstance(AbortOnly(), AbortingOnlineRefresher)
    assert not isinstance(AbortOnly(), CommittingOnlineRefresher)


def test_redis_lock_fences_by_generation():
    assert issubclass(RedisLock, TryAcquireLock)
    assert issubclass(RedisLock, GenerationReleaseLock)


def test_plain_objects_fall_back_to_no_capability():
    plain = object()
    for capability in (*_SINK_CAPABILITIES, VariantSnapshotReader, HeartbeatSource, ClosableSource):
        assert not isinstance(plain, capability)
    with recommendations_write_guard(plain):
        pass
    ensure_sink_writer_held(plain)
    assert snapshot_variant_names(plain) is None


def test_capability_set_to_none_is_not_offered():
    class Disabled:
        recommendations_write = None
        ensure_writer_held = None

    with recommendations_write_guard(Disabled()):
        pass
    ensure_sink_writer_held(Disabled())


def test_a_non_callable_release_generation_falls_back_to_release():
    from cicerone.locks.hold import WriteFence

    class Lock:
        release_generation = 1
        released = 0

        def try_acquire(self) -> int:
            return 7

        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            self.released += 1

        def owned(self, generation: int | None = None) -> bool:
            return True

        def is_locked(self) -> bool:
            return True

    lock = Lock()
    assert not isinstance(lock, GenerationReleaseLock)
    with WriteFence(lock).hold():
        pass
    assert lock.released == 1


def test_capabilities_supplied_per_instance_or_by_getattr_are_offered():
    class Dynamic:
        def __getattr__(self, name: str):
            if name == "heartbeat":
                return lambda events: None
            raise AttributeError(name)

    class Bare:
        pass

    per_instance = Bare()
    per_instance.close = lambda: None  # type: ignore[attr-defined]
    assert isinstance(Dynamic(), HeartbeatSource)
    assert not isinstance(Dynamic(), ClosableSource)
    assert isinstance(per_instance, ClosableSource)
    assert not isinstance(Bare(), ClosableSource)


def test_a_combined_capability_protocol_checks_inherited_members():
    from typing import Protocol, runtime_checkable

    from cicerone.capability import CapabilityMeta

    @runtime_checkable
    class FullSink(GuardedRecommendationsSink, WriterHeldSink, Protocol, metaclass=CapabilityMeta):
        pass

    class HalfSink:
        def ensure_writer_held(self) -> None:
            return None

    assert not isinstance(HalfSink(), FullSink)
    assert isinstance(object.__new__(DatasetOutputSink), FullSink)

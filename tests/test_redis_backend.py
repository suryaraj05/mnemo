"""Phase 17: Redis backend — skipped unless MNEMO_REDIS_URL is set."""

from __future__ import annotations

import os

import pytest

from mnemo.models import MemoryTier

pytestmark = pytest.mark.skipif(
    not os.environ.get("MNEMO_REDIS_URL"),
    reason="Set MNEMO_REDIS_URL to run Redis integration tests",
)


@pytest.fixture
def redis_backend():
    from mnemo.backends.redis_backend import RedisBackend

    backend = RedisBackend(os.environ["MNEMO_REDIS_URL"], key_prefix="mnemo_test")
    yield backend
    for item in backend.list(MemoryTier.WORKING, {}):
        backend.delete(MemoryTier.WORKING, item.key)
    backend.close()


def test_redis_write_read_roundtrip(redis_backend) -> None:
    redis_backend.write(MemoryTier.WORKING, "k1", "hello", {"importance": 0.5})
    hits = redis_backend.read(MemoryTier.WORKING, "hello", top_k=1)
    assert len(hits) == 1
    assert hits[0].key == "k1"

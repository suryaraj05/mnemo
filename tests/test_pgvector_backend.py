"""Pgvector integration tests — skipped unless MNEMO_PGVECTOR_DSN is set."""

from __future__ import annotations

import os
import uuid

import pytest

from mnemo.embeddings import HashEmbedder
from mnemo.models import MemoryTier

pytestmark = pytest.mark.skipif(
    not os.environ.get("MNEMO_PGVECTOR_DSN"),
    reason="Set MNEMO_PGVECTOR_DSN to run pgvector integration tests",
)


@pytest.fixture
def pg_backend():
    from mnemo.backends.pgvector import PgvectorBackend

    ns = f"test_{uuid.uuid4().hex[:8]}"
    backend = PgvectorBackend(dimension=32, namespace=ns)
    yield backend
    backend.close()


def test_pgvector_write_and_semantic_search(pg_backend) -> None:
    emb = HashEmbedder(32)
    v_cat = emb.embed("the cat sat on the mat")
    v_query = emb.embed("kitten on the mat")

    pg_backend.write(
        MemoryTier.EPISODIC,
        "e1",
        "cat event",
        {"embedding": v_cat, "source": "user"},
    )

    results = pg_backend.search_by_vector(MemoryTier.EPISODIC, v_query, top_k=1)
    assert len(results) == 1
    assert results[0].key == "e1"

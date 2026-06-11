"""PostgreSQL + pgvector backend with HNSW cosine index."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any
from urllib.parse import urlparse

from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY, VectorBackend
from mnemo.models import MemoryItem, MemoryTier

_DEFAULT_DSN = "postgresql://localhost:5432/mnemo"


class PgvectorBackend(VectorBackend):
    """Production vector store using pgvector HNSW approximate nearest neighbors.

    Requires: ``CREATE EXTENSION vector`` on the database.

    Environment:
        MNEMO_PGVECTOR_DSN — connection string (default localhost mnemo)
        MNEMO_PGVECTOR_NAMESPACE — table name prefix (default ``mnemo``)
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        dimension: int = 384,
        namespace: str | None = None,
    ) -> None:
        self._dsn = dsn or os.environ.get("MNEMO_PGVECTOR_DSN", _DEFAULT_DSN)
        self._dimension = dimension
        self._namespace = namespace or os.environ.get("MNEMO_PGVECTOR_NAMESPACE", "mnemo")
        self._table = f"{self._namespace}_memory_items"

        self._warn_if_remote_dsn(self._dsn)

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ImportError(
                'PgvectorBackend requires: pip install "mnemo[pgvector]"'
            ) from exc

        self._psycopg = psycopg
        self._register_vector = register_vector
        self._conn = psycopg.connect(self._dsn)
        register_vector(self._conn)
        self._ensure_schema()

    @staticmethod
    def _warn_if_remote_dsn(dsn: str) -> None:
        host = urlparse(dsn).hostname or ""
        if host not in ("localhost", "127.0.0.1", ""):
            warnings.warn(
                f"PgvectorBackend DSN host {host!r} is not local — verbatim memory "
                "will be sent to that server. This is an explicit opt-in.",
                stacklevel=2,
            )

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    tier TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self._dimension}),
                    PRIMARY KEY (tier, key)
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw
                ON {self._table}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        self._conn.commit()

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        embedding = meta.pop(EMBEDDING_METADATA_KEY, None)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table} (tier, key, value, metadata, embedding)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (tier, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    tier.value,
                    key,
                    json.dumps(value),
                    json.dumps(meta),
                    embedding,
                ),
            )
        self._conn.commit()

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        if top_k <= 0:
            return []
        pattern = f"%{query}%"
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT key, value, metadata FROM {self._table}
                WHERE tier = %s AND (key LIKE %s OR value::text LIKE %s)
                LIMIT %s
                """,
                (tier.value, pattern, pattern, top_k),
            )
            rows = cur.fetchall()
        return [self._row_to_item(k, v, m) for k, v, m in rows]

    def delete(self, tier: MemoryTier, key: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE tier = %s AND key = %s",
                (tier.value, key),
            )
        self._conn.commit()

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value, metadata FROM {self._table} WHERE tier = %s",
                (tier.value,),
            )
            rows = cur.fetchall()
        active = filters or {}
        items = [self._row_to_item(k, v, m) for k, v, m in rows]
        return [
            item
            for item in items
            if all(item.metadata.get(k) == v for k, v in active.items())
        ]

    def search_by_vector(
        self,
        tier: MemoryTier,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if top_k <= 0:
            return []
        with self._conn.cursor() as cur:
            if filters:
                items = self.list(tier, filters)
                keys = {item.key for item in items}
                if not keys:
                    return []
                cur.execute(
                    f"""
                    SELECT key, value, metadata
                    FROM {self._table}
                    WHERE tier = %s AND key = ANY(%s) AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (tier.value, list(keys), query_vector, top_k),
                )
            else:
                cur.execute(
                    f"""
                    SELECT key, value, metadata
                    FROM {self._table}
                    WHERE tier = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (tier.value, query_vector, top_k),
                )
            rows = cur.fetchall()
        return [self._row_to_item(k, v, m) for k, v, m in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_item(key: str, value: Any, metadata: Any) -> MemoryItem:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return MemoryItem(key=key, value=value, metadata=metadata)

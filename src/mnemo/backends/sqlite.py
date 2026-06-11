"""SQLite persistence backend — same MemoryBackend contract as InMemory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    tier     TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (tier, key)
);
"""


class SQLiteBackend(MemoryBackend):
    """Disk-backed storage: one row per (tier, key), JSON payloads."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memory_items (tier, key, value, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (tier.value, key, json.dumps(value), json.dumps(meta)),
        )
        self._conn.commit()

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        if top_k <= 0:
            return []
        pattern = f"%{query}%"
        cursor = self._conn.execute(
            """
            SELECT key, value, metadata FROM memory_items
            WHERE tier = ? AND (key LIKE ? OR value LIKE ?)
            """,
            (tier.value, pattern, pattern),
        )
        rows = cursor.fetchall()[:top_k]
        return [self._row_to_item(key, value, metadata) for key, value, metadata in rows]

    def delete(self, tier: MemoryTier, key: str) -> None:
        self._conn.execute(
            "DELETE FROM memory_items WHERE tier = ? AND key = ?",
            (tier.value, key),
        )
        self._conn.commit()

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        cursor = self._conn.execute(
            "SELECT key, value, metadata FROM memory_items WHERE tier = ?",
            (tier.value,),
        )
        active = filters or {}
        items = [self._row_to_item(k, v, m) for k, v, m in cursor.fetchall()]
        return [
            item
            for item in items
            if all(item.metadata.get(k) == v for k, v in active.items())
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_item(key: str, value_json: str, metadata_json: str) -> MemoryItem:
        return MemoryItem(
            key=key,
            value=json.loads(value_json),
            metadata=json.loads(metadata_json),
        )

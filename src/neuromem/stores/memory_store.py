from __future__ import annotations

from dataclasses import replace

from neuromem.core.models import MemoryEdge, MemoryItem
from neuromem.stores.base import MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._memories: dict[str, MemoryItem] = {}
        self._edges: list[MemoryEdge] = []

    def upsert_memory(self, item: MemoryItem) -> None:
        self._memories[item.id] = item

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        return self._memories.get(memory_id)

    def list_memories(self, namespace: str | None = None) -> list[MemoryItem]:
        values = list(self._memories.values())
        if namespace is None:
            return values
        return [item for item in values if item.namespace == namespace]

    def add_edge(self, edge: MemoryEdge) -> None:
        self._edges.append(edge)

    def list_edges(self, source_id: str | None = None) -> list[MemoryEdge]:
        if source_id is None:
            return list(self._edges)
        return [edge for edge in self._edges if edge.source_id == source_id]


from __future__ import annotations

from abc import ABC, abstractmethod
from neuromem.core.models import MemoryEdge, MemoryItem


class MemoryStore(ABC):
    @abstractmethod
    def upsert_memory(self, item: MemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_memory(self, memory_id: str) -> MemoryItem | None:
        raise NotImplementedError

    @abstractmethod
    def list_memories(self, namespace: str | None = None) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: MemoryEdge) -> None:
        raise NotImplementedError

    def upsert_edge(self, edge: MemoryEdge) -> None:
        self.add_edge(edge)

    @abstractmethod
    def list_edges(self, source_id: str | None = None) -> list[MemoryEdge]:
        raise NotImplementedError

    def search_memory_cards(self, query: str, *, namespace: str | None = None, limit: int = 20) -> list[tuple[str, float]]:
        del query, namespace, limit
        return []

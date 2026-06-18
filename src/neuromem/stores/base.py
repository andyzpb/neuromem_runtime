from __future__ import annotations

from abc import ABC, abstractmethod
from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryEdge, MemoryFrame, MemoryItem


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

    @abstractmethod
    def add_associative_edge(self, edge: AssociativeEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_associative_edges(self, source_id: str | None = None, namespace: str | None = None) -> list[AssociativeEdge]:
        raise NotImplementedError

    @abstractmethod
    def add_logic_node(self, frame: MemoryFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_logic_node(self, frame_id: str) -> MemoryFrame | None:
        raise NotImplementedError

    @abstractmethod
    def list_logic_nodes(self, namespace: str | None = None, *, lifecycle_state: str | None = None) -> list[MemoryFrame]:
        raise NotImplementedError

    @abstractmethod
    def add_logic_edge(self, edge: LogicEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_logic_edges(self, source_frame_id: str | None = None, namespace: str | None = None) -> list[LogicEdge]:
        raise NotImplementedError

    def search_memory_cards(self, query: str, *, namespace: str | None = None, limit: int = 20) -> list[tuple[str, float]]:
        del query, namespace, limit
        return []

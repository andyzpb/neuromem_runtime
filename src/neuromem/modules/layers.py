from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.models import MemoryItem
from neuromem.stores.base import MemoryStore


@dataclass(slots=True)
class HippocampalStore:
    store: MemoryStore
    namespace: str

    def encode(self, item: MemoryItem) -> None:
        item.type = "episodic"
        self.store.upsert_memory(item)

    def timeline(self) -> list[MemoryItem]:
        return [item for item in self.store.list_memories(self.namespace) if item.type == "episodic"]


@dataclass(slots=True)
class NeocorticalStore:
    store: MemoryStore
    namespace: str

    def store_fact(self, item: MemoryItem) -> None:
        item.type = "semantic" if item.type not in {"preference", "schema"} else item.type
        self.store.upsert_memory(item)

    def stable_memories(self) -> list[MemoryItem]:
        return [item for item in self.store.list_memories(self.namespace) if item.type in {"semantic", "schema", "preference"}]


@dataclass(slots=True)
class ProceduralStore:
    store: MemoryStore
    namespace: str

    def store_rule(self, item: MemoryItem) -> None:
        item.type = "procedural"
        self.store.upsert_memory(item)

    def rules(self) -> list[MemoryItem]:
        return [item for item in self.store.list_memories(self.namespace) if item.type == "procedural"]


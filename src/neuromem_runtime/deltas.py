from __future__ import annotations

from dataclasses import asdict, dataclass, field

from neuromem.core.deltas import GraphDelta
from neuromem.core.policy import MemoryTrace
from neuromem_runtime.policy_v2 import ValidatedMutation


@dataclass(slots=True)
class MemoryDelta:
    memory_id: str
    field: str
    old: object
    new: object
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LifecycleDelta:
    memory_id: str
    from_state: str
    to_state: str
    trigger: str
    evidence: list[str] = field(default_factory=list)
    validator: str = ""
    reason: str = ""
    rollback_state: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IndexDelta:
    index: str
    status: str
    memory_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionDeltaPlan:
    memory_deltas: list[MemoryDelta] = field(default_factory=list)
    graph_deltas: list[GraphDelta] = field(default_factory=list)
    lifecycle_deltas: list[LifecycleDelta] = field(default_factory=list)
    index_deltas: list[IndexDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_deltas": [delta.to_dict() for delta in self.memory_deltas],
            "graph_deltas": [delta.to_dict() for delta in self.graph_deltas],
            "lifecycle_deltas": [delta.to_dict() for delta in self.lifecycle_deltas],
            "index_deltas": [delta.to_dict() for delta in self.index_deltas],
        }


@dataclass(slots=True)
class MemorySnapshot:
    memories: dict[str, dict[str, object]] = field(default_factory=dict)
    edges: dict[str, dict[str, object]] = field(default_factory=dict)
    frames: dict[str, dict[str, object]] = field(default_factory=dict)
    transaction_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MutationExecutionResult:
    trace: MemoryTrace
    validated_mutation: ValidatedMutation
    created_memory_ids: list[str] = field(default_factory=list)
    updated_memory_ids: list[str] = field(default_factory=list)
    deleted_memory_ids: list[str] = field(default_factory=list)
    memory_deltas: list[MemoryDelta] = field(default_factory=list)
    graph_deltas: list[GraphDelta] = field(default_factory=list)
    lifecycle_deltas: list[LifecycleDelta] = field(default_factory=list)
    index_deltas: list[IndexDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace": self.trace.to_dict(),
            "validated_mutation": self.validated_mutation.model_dump(),
            "created_memory_ids": list(self.created_memory_ids),
            "updated_memory_ids": list(self.updated_memory_ids),
            "deleted_memory_ids": list(self.deleted_memory_ids),
            "memory_deltas": [delta.to_dict() for delta in self.memory_deltas],
            "graph_deltas": [delta.to_dict() for delta in self.graph_deltas],
            "lifecycle_deltas": [delta.to_dict() for delta in self.lifecycle_deltas],
            "index_deltas": [delta.to_dict() for delta in self.index_deltas],
        }


__all__ = [
    "ExecutionDeltaPlan",
    "GraphDelta",
    "IndexDelta",
    "LifecycleDelta",
    "MemoryDelta",
    "MemorySnapshot",
    "MutationExecutionResult",
]

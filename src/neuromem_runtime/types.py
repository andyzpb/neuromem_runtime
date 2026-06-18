from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from neuromem.core.policy import MemoryPolicy, MemoryTrace, MemoryTransactionRecord
from neuromem_runtime.ledger import ExperienceEvent


RetrievalLens = str


@dataclass(slots=True)
class RuntimeConfig:
    namespace: str = "default"
    path: Path = Path(".neuromem")
    db_path: Path = Path(".neuromem/memory.sqlite3")
    traces_path: Path = Path(".neuromem/traces")
    agent_id: str = "local-agent"
    mode: str = "lite"
    model_policy_enabled: bool = False
    graph_mode: str = "governed_hybrid"
    crystallization_mode: str = "governed_progressive"
    graph_storage: str = "split"
    version: str = "0.2.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "path": str(self.path),
            "db_path": str(self.db_path),
            "traces_path": str(self.traces_path),
            "agent_id": self.agent_id,
            "mode": self.mode,
            "model_policy_enabled": self.model_policy_enabled,
            "graph_mode": self.graph_mode,
            "crystallization_mode": self.crystallization_mode,
            "graph_storage": self.graph_storage,
            "version": self.version,
        }


@dataclass(slots=True)
class MemoryEvent:
    content: str
    type: str = "task_result"
    source: str = "user"
    task: str | None = None
    evidence: str | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    outcome: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_internal(self) -> dict[str, object]:
        value: dict[str, object] = {
            "content": self.content,
            "type": self.type,
            "source": self.source,
            "entities": self.entities,
            "keywords": self.keywords,
            "tags": self.tags,
            **self.metadata,
        }
        if self.task is not None:
            value["task"] = self.task
        if self.evidence is not None:
            value["evidence"] = self.evidence
        if self.outcome is not None:
            value["outcome"] = self.outcome
        return value


@dataclass(slots=True)
class MemoryQuery:
    query: str
    budget_tokens: int = 800
    filters: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceBundle:
    memory_id: str | None
    content: str
    memory_type: str | None = None
    evidence: list[str] = field(default_factory=list)
    trace_id: str | None = None
    event_id: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MemoryContext:
    query: str
    text: str
    selected_memory_ids: list[str] = field(default_factory=list)
    trace_id: str | None = None
    results: list[dict[str, object]] = field(default_factory=list)
    transactions: list[dict[str, object]] = field(default_factory=list)

    def to_prompt(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MemoryTransaction = MemoryTransactionRecord


def event_to_dict(event: MemoryEvent | dict[str, object]) -> dict[str, object]:
    if isinstance(event, MemoryEvent):
        return event.to_internal()
    return dict(event)


__all__ = [
    "RuntimeConfig",
    "MemoryEvent",
    "MemoryQuery",
    "MemoryContext",
    "EvidenceBundle",
    "ExperienceEvent",
    "MemoryPolicy",
    "MemoryTransaction",
    "MemoryTrace",
    "RetrievalLens",
    "event_to_dict",
]

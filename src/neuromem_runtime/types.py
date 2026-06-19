from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

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
    mutation_mode: Literal["append_only_view", "strict_append_only"] = "append_only_view"
    embedding_cache_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 20
    retrieval_graph_commit: Literal["async", "off", "sync", "trace_only"] = "trace_only"
    retrieval_mode: Literal["auto", "full_debug"] = "auto"
    ollama_keep_alive: str = "30m"
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
            "mutation_mode": self.mutation_mode,
            "embedding_cache_enabled": self.embedding_cache_enabled,
            "retrieval_cache_ttl_seconds": self.retrieval_cache_ttl_seconds,
            "retrieval_graph_commit": self.retrieval_graph_commit,
            "retrieval_mode": self.retrieval_mode,
            "ollama_keep_alive": self.ollama_keep_alive,
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
    impact: dict[str, object] | None = None

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
    timing: dict[str, object] = field(default_factory=dict)
    cache: dict[str, object] = field(default_factory=dict)
    retrieval_metadata: dict[str, object] = field(default_factory=dict)
    worldview: dict[str, object] | None = None
    worldview_trace: dict[str, object] | None = None
    prompt_sections: dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        sections: list[str] = []
        if self.prompt_sections:
            for key in ["worldview", "facts", "preferences", "constraints", "procedures", "suppressions", "conflicts", "supporting_memories"]:
                value = self.prompt_sections.get(key)
                if value:
                    sections.append(value)
        elif self.worldview and self.worldview.get("prompt"):
            sections.append(str(self.worldview["prompt"]))
        if self.text:
            sections.append(f"[Supporting Memory Snippets]\n{self.text}")
        if sections:
            return "\n\n".join(sections)
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

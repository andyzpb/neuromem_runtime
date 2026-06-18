from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
class GraphDelta:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    old_weight: float
    new_weight: float
    delta: float
    eligibility: float = 1.0
    salience: float = 0.0
    outcome_reward: float = 0.0
    confidence: float = 0.0
    inhibition_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    provenance: list[str] = field(default_factory=list)
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


__all__ = ["MemoryDelta", "GraphDelta", "LifecycleDelta", "IndexDelta"]

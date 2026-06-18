from __future__ import annotations

from dataclasses import asdict, dataclass, field


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


__all__ = ["GraphDelta"]

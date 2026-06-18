from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.models import MemoryEdge
from neuromem_runtime.deltas import GraphDelta


@dataclass(slots=True)
class PlasticityEngine:
    learning_rate: float = 0.1
    provenance_trust: float = 1.0

    def update_edge(
        self,
        edge: MemoryEdge,
        *,
        salience: float,
        outcome_reward: float,
        confidence: float,
        reason: str = "outcome-shaped plasticity",
    ) -> GraphDelta:
        old_weight = edge.weight
        delta = (
            self.learning_rate
            * edge.eligibility_trace
            * salience
            * outcome_reward
            * confidence
            * self.provenance_trust
            - edge.inhibition_score
            - edge.contradiction_penalty
        )
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        return GraphDelta(
            edge_id=f"{edge.source_id}:{edge.relation}:{edge.target_id}",
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation,
            old_weight=old_weight,
            new_weight=edge.weight,
            delta=delta,
            eligibility=edge.eligibility_trace,
            salience=salience,
            outcome_reward=outcome_reward,
            confidence=confidence,
            inhibition_penalty=edge.inhibition_score,
            contradiction_penalty=edge.contradiction_penalty,
            provenance=edge.provenance,
            reason=reason,
        )


__all__ = ["PlasticityEngine"]

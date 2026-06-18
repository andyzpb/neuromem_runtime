from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.models import MemoryEdge, MemoryItem
from neuromem.modules.lifecycle import obsolete, promote


@dataclass(slots=True)
class ReconsolidationDecision:
    relation: str
    reason: str


class Reconsolidator:
    def judge(self, memory: MemoryItem, evidence: str) -> ReconsolidationDecision:
        text = evidence.lower()
        if any(term in text for term in ["obsolete", "supersede", "replaced", "now says", "changed to"]):
            return ReconsolidationDecision("contradicted", "new evidence supersedes this memory")
        if any(term in text for term in ["confirmed", "reinforced", "supports", "repeated"]):
            return ReconsolidationDecision("reinforced", "new evidence reinforces this memory")
        if any(term in text for term in ["generalize", "rule", "workflow"]):
            return ReconsolidationDecision("generalized", "new evidence suggests a reusable rule")
        return ReconsolidationDecision("touched", "memory retrieved with neutral evidence")

    def apply(self, memory: MemoryItem, evidence: str) -> tuple[MemoryItem, MemoryEdge | None]:
        decision = self.judge(memory, evidence)
        memory.evidence.append(evidence)
        if decision.relation == "contradicted":
            obsolete(memory, reason=decision.reason)
            return memory, None
        if decision.relation == "reinforced":
            memory.confidence = min(1.0, memory.confidence + 0.08)
            memory.consolidation_count += 1
            promote(memory, reason=decision.reason)
            return memory, None
        if decision.relation == "generalized":
            memory.confidence = min(1.0, memory.confidence + 0.05)
            memory.consolidation_count += 1
            memory.type = "procedural"
            promote(memory, reason=decision.reason)
            return memory, None
        return memory, None


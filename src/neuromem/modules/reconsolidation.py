from __future__ import annotations

import json
from dataclasses import dataclass

from neuromem.core.models import MemoryEdge, MemoryItem
from neuromem.modules.lifecycle import obsolete, promote


@dataclass(slots=True)
class ReconsolidationDecision:
    relation: str
    reason: str


class Reconsolidator:
    def judge(self, memory: MemoryItem, evidence: str) -> ReconsolidationDecision:
        del memory
        relation = _structured_relation(evidence)
        if relation in {"contradicted", "superseded", "inhibited"}:
            return ReconsolidationDecision("contradicted", "new evidence supersedes this memory")
        if relation in {"reinforced", "supported"}:
            return ReconsolidationDecision("reinforced", "new evidence reinforces this memory")
        if relation in {"generalized", "compiled"}:
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


def _structured_relation(evidence: str) -> str:
    try:
        payload = json.loads(evidence)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return ""
    value = payload.get("relation") or payload.get("event_type") or payload.get("lifecycle_relation")
    return str(value or "").strip().lower()

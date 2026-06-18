from __future__ import annotations

from dataclasses import dataclass, field

from neuromem.evals.framework import EvidenceItem, MemoryEvent
from neuromem.retrieval.recall import QueryPlan, RecallConfig, RecallEvidence, build_query_plan, run_recall


@dataclass(frozen=True, slots=True)
class GatedRecallResult:
    evidence: list[EvidenceItem]
    query_plan: QueryPlan
    rejected_ids: list[str] = field(default_factory=list)
    suppression_reasons: dict[str, str] = field(default_factory=dict)
    canonical_fact_ids: list[str] = field(default_factory=list)
    gate_decision: str = "accepted"
    memory_version: str = ""
    invalidation_state: str = "valid"
    query_plan_hash: str = ""
    recall_config_hash: str = ""


def evidence_gated_hybrid_recall(
    evidence: list[EvidenceItem],
    plan: QueryPlan,
    *,
    events_by_id: dict[str, MemoryEvent] | None = None,
    budget_tokens: int,
    min_score: float = 0.14,
) -> GatedRecallResult:
    events_by_id = events_by_id or {}
    recall_items = [_to_recall_evidence(item, events_by_id.get(item.id)) for item in evidence]
    config = RecallConfig(
        budget_tokens=budget_tokens,
        min_score=min_score,
        max_items=8,
        source_channels=("lexical", "bm25", "vector", "graph", "recent_active", "canonical_fact", "external_adapter"),
        evidence_gate_enabled=True,
        require_fact_or_entity_alignment=True,
    )
    result = run_recall(recall_items, plan, config=config)
    original_by_id = {item.id: item for item in evidence}
    candidates_by_id = {candidate.evidence.id: candidate for candidate in result.candidates}
    converted: list[EvidenceItem] = []
    for item in result.evidence:
        original = original_by_id.get(item.id)
        if original is None:
            continue
        candidate = candidates_by_id.get(item.id)
        trace = dict(original.trace)
        if candidate is not None:
            trace.update(candidate.score_components())
            trace.update(
                {
                    "fact_overlap": candidate.fact_overlap,
                    "entity_overlap": candidate.entity_overlap,
                    "invalidation_state": candidate.invalidation_state,
                    "lifecycle_reason": candidate.lifecycle_reason,
                    "channels": list(candidate.source_channels),
                }
            )
        converted.append(
            EvidenceItem(
                original.id,
                original.content,
                score=round(candidate.final_score, 4) if candidate is not None else original.score,
                source=original.source,
                trace=trace,
            )
        )
    return GatedRecallResult(
        evidence=converted,
        query_plan=plan,
        rejected_ids=result.rejected_ids,
        suppression_reasons=result.suppression_reasons,
        canonical_fact_ids=result.canonical_fact_ids,
        gate_decision=result.gate_decision,
        memory_version=result.memory_version,
        invalidation_state=result.invalidation_state,
        query_plan_hash=plan.stable_hash(),
        recall_config_hash=result.recall_config_hash,
    )


def _to_recall_evidence(item: EvidenceItem, event: MemoryEvent | None) -> RecallEvidence:
    trace = dict(item.trace)
    if event is not None:
        trace["event_kind"] = event.kind
    return RecallEvidence(
        id=item.id,
        content=item.content,
        base_score=item.score,
        source=item.source,
        keywords=tuple(event.keywords) if event is not None else tuple(str(value) for value in trace.get("keywords", []) if str(value)),
        entities=tuple(str(value).lower() for value in trace.get("entities", []) if str(value)) if isinstance(trace.get("entities"), list) else (),
        answer=event.answer if event is not None else str(trace.get("answer", "")),
        timestamp=event.timestamp if event is not None else int(trace.get("timestamp", 0) or 0),
        maturity=str(trace.get("maturity") or _infer_maturity(item.content)),
        memory_type=event.kind if event is not None else str(trace.get("memory_type", "episodic")),
        confidence=float(trace.get("confidence", 0.7) or 0.7),
        inhibition_score=float(trace.get("inhibition_score", 0.0) or 0.0),
        contradiction_score=float(trace.get("contradiction_score", 0.0) or 0.0),
        evidence_ids=tuple(str(value) for value in trace.get("evidence_ids", []) if str(value)) if isinstance(trace.get("evidence_ids"), list) else (item.id,),
        trace=trace,
    )


def _infer_maturity(content: str) -> str:
    lowered = content.lower()
    if any(term in lowered for term in ["obsolete", "deprecated", "replaced", "superseded"]):
        return "obsolete"
    return "fresh"

from __future__ import annotations

from collections.abc import Iterable, Sequence

from neuromem.core.models import MemoryItem, MemoryQuery, MemoryResult
from neuromem.modules.pfc_controller import PFCController, RetrievalPlan
from neuromem.retrieval.recall import RecallConfig, RecallResult, build_query_plan, evidence_from_memory, run_recall


DEFAULT_SOURCE_CHANNELS = ("lexical", "bm25", "vector", "graph", "recent_active", "canonical_fact")


def hybrid_retrieve(memories: Iterable[MemoryItem], query: MemoryQuery, plan: RetrievalPlan | None = None) -> list[MemoryResult]:
    results, _ = hybrid_retrieve_with_trace(memories, query, plan)
    return results


def hybrid_retrieve_with_trace(
    memories: Iterable[MemoryItem],
    query: MemoryQuery,
    plan: RetrievalPlan | None = None,
    *,
    graph_scores: dict[str, float] | None = None,
    graph_paths: list[list[str]] | None = None,
) -> tuple[list[MemoryResult], RecallResult]:
    memory_items = list(memories)
    memory_by_id = {item.id: item for item in memory_items}
    plan = plan or PFCController().plan_retrieval(query.query, query.filters, query.budget_tokens)
    evidence = [evidence_from_memory(item, base_score=_base_score(item, query), source="core") for item in memory_items]
    query_plan = build_query_plan(
        query.query,
        answer=str(query.filters.get("answer") or ""),
        expected_evidence_ids=_string_tuple(query.filters.get("expected_evidence_ids")),
        multi_hop_evidence_ids=_string_tuple(query.filters.get("multi_hop_evidence_ids")),
        abstain=bool(query.filters.get("abstain", False)),
        entities=_string_tuple(query.filters.get("entities")),
    )
    config = _recall_config(query, plan)
    recall = run_recall(evidence, query_plan, config=config, graph_scores=graph_scores, graph_paths=graph_paths)
    candidates_by_id = {candidate.evidence.id: candidate for candidate in recall.candidates}
    results: list[MemoryResult] = []
    for item in recall.evidence:
        memory = memory_by_id.get(item.id)
        if memory is None:
            continue
        candidate = candidates_by_id.get(item.id)
        why = [item.source or "hybrid_recall", f"gate={recall.gate_decision}", f"memory_version={recall.memory_version}"]
        if candidate is not None:
            why.extend(f"source:{channel}" for channel in candidate.source_channels)
            why.append(f"invalidation={candidate.invalidation_state}")
        results.append(MemoryResult(memory=memory, score=candidate.final_score if candidate is not None else item.base_score, why_retrieved=why))
    return results, recall


def _recall_config(query: MemoryQuery, plan: RetrievalPlan) -> RecallConfig:
    source_channels = query.filters.get("source_channels")
    channels = _string_tuple(source_channels) or DEFAULT_SOURCE_CHANNELS
    return RecallConfig(
        budget_tokens=query.budget_tokens,
        min_score=float(query.filters.get("min_score", 0.14) or 0.14),
        max_items=max(1, min(10, plan.context_budget_tokens // 250)),
        source_channels=tuple(channel for channel in channels if channel in DEFAULT_SOURCE_CHANNELS or channel == "external_adapter"),  # type: ignore[arg-type]
        evidence_gate_enabled=bool(query.filters.get("evidence_gate_enabled", True)),
        require_fact_or_entity_alignment=bool(query.filters.get("require_fact_or_entity_alignment", True)),
    )


def _base_score(item: MemoryItem, query: MemoryQuery) -> float:
    terms = {token.lower() for token in query.query.split() if token}
    haystack = " ".join([item.content, item.summary or "", " ".join(item.tags), " ".join(item.keywords)]).lower()
    overlap = sum(1 for term in terms if term in haystack)
    salience = item.salience.get("novelty", 0.0) + item.salience.get("task_value", 0.0) + item.salience.get("recurrence", 0.0)
    score = min(1.0, 0.12 * overlap + 0.42 * salience + 0.2 * item.confidence)
    score -= item.inhibition_score * 0.4
    if item.type == "procedural":
        score += 0.05
    if item.type == "semantic":
        score += 0.03
    return score


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return ()

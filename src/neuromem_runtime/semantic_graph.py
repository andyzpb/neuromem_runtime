from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from neuromem.core.models import MemoryEdge, MemoryItem, MemoryRelation, utcnow
from neuromem.stores.base import MemoryStore
from neuromem_runtime.policy_v2 import GraphDeltaProposal, ValidationStep


GraphMode = Literal["off", "deterministic", "governed_hybrid"]
GraphEdgeState = Literal["candidate", "provisional", "captured", "reinforced", "mature", "inhibited", "expired", "superseded"]

ASSOCIATIVE_RELATIONS = {"associated_with", "retrieved_with", "coactivated_with", "precedes", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}
SAFE_RELATIONS = {"evidence_for", "retrieved_with", "coactivated_with", "precedes", "derived_from", "compresses_to", "associated_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}
SEMANTIC_RELATIONS = {"supports", "same_as", "procedure_for", "generalizes", "specializes"}
HIGH_RISK_RELATIONS = {"causes", "contradicts", "supersedes", "inhibits"}
LOGIC_RELATIONS = {"supports", "same_as", "procedure_for", "generalizes", "specializes", "causes", "contradicts", "supersedes", "inhibits", "evidence_for", "derived_from", "compresses_to", "preference_of", "applies_to"}
RELATION_FAMILIES = {
    "associated_with": "association",
    "evidence_for": "evidence",
    "derived_from": "lifecycle",
    "compresses_to": "lifecycle",
    "retrieved_with": "activation",
    "coactivated_with": "activation",
    "precedes": "activation",
    "same_trace": "activation",
    "same_episode": "activation",
    "nearby_context": "activation",
    "used_with_success": "activation",
    "used_with_failure": "activation",
    "supports": "semantic",
    "same_as": "semantic",
    "procedure_for": "procedural",
    "preference_of": "preference",
    "applies_to": "semantic",
    "generalizes": "semantic",
    "specializes": "semantic",
    "causes": "causal",
    "contradicts": "suppression",
    "supersedes": "suppression",
    "inhibits": "suppression",
}
ALLOWED_GRAPH_RELATIONS = frozenset(SAFE_RELATIONS | SEMANTIC_RELATIONS | HIGH_RISK_RELATIONS | LOGIC_RELATIONS)


@dataclass(slots=True)
class GraphRelationCandidate:
    source_memory_id: str
    target_memory_id: str
    candidate_sources: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    suggested_relations: list[str] = field(default_factory=list)
    score: float = 0.0
    namespace: str = "default"
    reason: str = ""
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class GraphBuildContext:
    namespace: str
    memories: list[MemoryItem]
    selected_memory_ids: list[str] = field(default_factory=list)
    target_memory_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    retrieval_trace: dict[str, object] = field(default_factory=dict)
    mutation_trace: dict[str, object] = field(default_factory=dict)
    sleep_clusters: list[list[str]] = field(default_factory=list)
    outcome: str = "unknown"
    proposer: str = "deterministic"

    def memory_by_id(self) -> dict[str, MemoryItem]:
        return {memory.id: memory for memory in self.memories}


class GraphProposalProvider(Protocol):
    def propose_graph_deltas(self, context: GraphBuildContext, candidates: list[GraphRelationCandidate]) -> list[GraphDeltaProposal]:
        raise NotImplementedError


class GraphCandidateGenerator:
    def __init__(self, *, min_score: float = 0.18) -> None:
        self.min_score = min_score

    def generate(self, context: GraphBuildContext) -> list[GraphRelationCandidate]:
        by_id = context.memory_by_id()
        pairs: dict[tuple[str, str], GraphRelationCandidate] = {}

        def add_pair(left_id: str, right_id: str, source: str, score: float, suggested: list[str], evidence: list[str] | None = None) -> None:
            if left_id == right_id or left_id not in by_id or right_id not in by_id:
                return
            left = by_id[left_id]
            right = by_id[right_id]
            if left.namespace != context.namespace or right.namespace != context.namespace:
                return
            key = tuple(sorted([left_id, right_id]))
            existing = pairs.get(key)
            if existing is None:
                existing = GraphRelationCandidate(
                    source_memory_id=key[0],
                    target_memory_id=key[1],
                    namespace=context.namespace,
                    evidence_ids=list(dict.fromkeys([*(evidence or []), *context.evidence_ids, *left.evidence, *right.evidence])),
                    reason="candidate graph relation from bounded retrieval/mutation context",
                )
                pairs[key] = existing
            existing.score += score
            if source not in existing.candidate_sources:
                existing.candidate_sources.append(source)
            for relation in suggested:
                if relation not in existing.suggested_relations:
                    existing.suggested_relations.append(relation)

        selected = [memory_id for memory_id in context.selected_memory_ids if memory_id in by_id]
        targets = [memory_id for memory_id in context.target_memory_ids if memory_id in by_id]
        for source_id in selected:
            for target_id in targets:
                add_pair(source_id, target_id, "co_use_outcome", 0.42 if context.outcome == "success" else 0.18, ["supports", "retrieved_with"])
        for ids in [selected, targets, [*selected, *targets]]:
            for index, left_id in enumerate(ids):
                for right_id in ids[index + 1 :]:
                    add_pair(left_id, right_id, "same_query_retrieval", 0.22, ["retrieved_with", "coactivated_with"])
        for cluster in context.sleep_clusters:
            cluster_ids = [memory_id for memory_id in cluster if memory_id in by_id]
            for index, left_id in enumerate(cluster_ids):
                for right_id in cluster_ids[index + 1 :]:
                    add_pair(left_id, right_id, "same_sleep_cluster", 0.46, ["generalizes", "procedure_for", "derived_from"])

        memories = list(by_id.values())
        for index, left in enumerate(memories):
            for right in memories[index + 1 :]:
                shared_evidence = sorted(set(left.evidence) & set(right.evidence))
                if shared_evidence:
                    add_pair(left.id, right.id, "same_evidence_chain", 0.38, ["evidence_for", "supports"], shared_evidence)
                shared_entities = sorted((set(left.entities) | set(left.keywords)) & (set(right.entities) | set(right.keywords)))
                if shared_entities:
                    add_pair(left.id, right.id, "same_entity", min(0.28, 0.12 + 0.04 * len(shared_entities)), ["associated_with", "supports"])
                if _canonical_fact_key(left) and _canonical_fact_key(left) == _canonical_fact_key(right):
                    add_pair(left.id, right.id, "same_canonical_fact_key", 0.5, ["same_as", "supersedes", "contradicts"])
                lexical = _jaccard(left.content, right.content)
                if lexical >= 0.18:
                    add_pair(left.id, right.id, "lexical_overlap", lexical * 0.4, ["associated_with", "same_as"])
                if _looks_like_supersession(left.content, right.content):
                    add_pair(left.id, right.id, "same_failure_pattern", 0.48, ["supersedes", "contradicts"])

        return sorted(
            [candidate for candidate in pairs.values() if candidate.score >= self.min_score and candidate.evidence_ids],
            key=lambda candidate: (-candidate.score, candidate.source_memory_id, candidate.target_memory_id),
        )


class DeterministicRelationProposer:
    def propose_graph_deltas(self, context: GraphBuildContext, candidates: list[GraphRelationCandidate]) -> list[GraphDeltaProposal]:
        proposals: list[GraphDeltaProposal] = []
        by_id = context.memory_by_id()
        for candidate in candidates:
            relation = _choose_relation(candidate, by_id)
            if relation is None:
                continue
            confidence = _initial_confidence(candidate, relation)
            weight = min(0.95, max(0.08, confidence * (0.72 if relation in SAFE_RELATIONS else 0.5)))
            if relation in HIGH_RISK_RELATIONS:
                confidence = min(confidence, 0.6 if relation == "causes" else 0.68)
                weight = min(weight, 0.35)
            proposals.append(
                GraphDeltaProposal(
                    operation="add_edge",
                    source_memory_id=candidate.source_memory_id,
                    target_memory_id=candidate.target_memory_id,
                    relation=relation,
                    weight=weight,
                    confidence=confidence,
                    evidence_ids=candidate.evidence_ids,
                    candidate_sources=candidate.candidate_sources,
                    reason=f"{relation} proposed from {', '.join(candidate.candidate_sources)}",
                    proposer=context.proposer,
                    lifecycle_state="captured" if relation in SAFE_RELATIONS else "provisional",
                    valid_from=utcnow(),
                )
            )
        return proposals


class GraphDeltaValidator:
    def validate(self, proposal: GraphDeltaProposal, *, context: GraphBuildContext, store: MemoryStore | None = None) -> ValidationStep:
        if proposal.operation not in {"add_edge", "update_edge", "inhibit_edge", "expire_edge"}:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason=f"unsupported graph operation: {proposal.operation}")
        if proposal.relation not in ALLOWED_GRAPH_RELATIONS:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason=f"unsupported graph relation: {proposal.relation}")
        if not proposal.source_memory_id or not proposal.target_memory_id or proposal.source_memory_id == proposal.target_memory_id:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="graph deltas require distinct endpoints")
        if not proposal.evidence_ids:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="graph deltas require evidence ids")
        if proposal.confidence < _min_confidence(proposal.relation):
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason=f"confidence below threshold for {proposal.relation}")
        if proposal.valid_from and proposal.valid_to and proposal.valid_to < proposal.valid_from:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="graph edge valid_to is before valid_from")
        by_id = context.memory_by_id()
        source = by_id.get(proposal.source_memory_id) or (store.get_memory(proposal.source_memory_id) if store is not None else None)
        target = by_id.get(proposal.target_memory_id) or (store.get_memory(proposal.target_memory_id) if store is not None else None)
        if source is None or target is None:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="graph delta endpoint not found")
        if source.namespace != context.namespace or target.namespace != context.namespace:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="graph delta endpoint outside namespace")
        if proposal.relation == "supports" and not _has_shared_basis(source, target, proposal):
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="supports requires shared evidence, entity, fact key, trace, or sleep cluster")
        if proposal.relation == "causes" and proposal.confidence > 0.6 and "repeated_outcome" not in proposal.candidate_sources:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="causes confidence is capped without repeated outcome evidence")
        if proposal.relation == "generalizes" and "same_sleep_cluster" not in proposal.candidate_sources and proposal.proposer not in {"admin", "user"}:
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="generalizes requires sleep cluster or explicit user/admin proposer")
        if proposal.relation == "same_as" and _canonical_fact_key(source) != _canonical_fact_key(target):
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason="same_as requires matching canonical fact key")
        if proposal.relation in {"contradicts", "supersedes"} and not (_canonical_fact_key(source) and _canonical_fact_key(source) == _canonical_fact_key(target) or set(source.entities) & set(target.entities) or set(source.keywords) & set(target.keywords)):
            return ValidationStep(name="GraphDeltaValidator", passed=False, reason=f"{proposal.relation} requires shared fact key or entity")
        return ValidationStep(name="GraphDeltaValidator", passed=True)


class GraphMutationCommitter:
    def commit(self, proposal: GraphDeltaProposal, *, store: MemoryStore) -> MemoryEdge:
        source_id, target_id = proposal.source_memory_id, proposal.target_memory_id
        if proposal.operation in {"add_edge", "update_edge"}:
            existing = _find_edge(store, source_id, target_id, proposal.relation)
            edge = existing or MemoryEdge(source_id=source_id, target_id=target_id, relation=proposal.relation)  # type: ignore[arg-type]
            edge.weight = max(edge.weight, min(1.0, proposal.weight))
            edge.confidence = max(edge.confidence, min(1.0, proposal.confidence))
            edge.valid_from = proposal.valid_from or edge.valid_from
            edge.valid_to = proposal.valid_to
            edge.recorded_at = edge.recorded_at or utcnow()
            edge.lifecycle_state = proposal.lifecycle_state  # type: ignore[assignment]
            edge.provenance = list(dict.fromkeys([*edge.provenance, *proposal.evidence_ids]))
            if proposal.relation in {"contradicts", "inhibits"}:
                edge.inhibition_score = max(edge.inhibition_score, 0.55)
                edge.contradiction_penalty = max(edge.contradiction_penalty, 0.45)
                edge.lifecycle_state = "inhibited"
            store.add_edge(edge)
            _apply_endpoint_lists(store, proposal)
            return edge
        edge = _find_edge(store, source_id, target_id, proposal.relation)
        if edge is None:
            edge = MemoryEdge(source_id=source_id, target_id=target_id, relation=proposal.relation, provenance=list(proposal.evidence_ids))  # type: ignore[arg-type]
        if proposal.operation == "inhibit_edge":
            edge.inhibition_score = max(edge.inhibition_score, 0.85)
            edge.lifecycle_state = "inhibited"
        elif proposal.operation == "expire_edge":
            edge.valid_to = proposal.valid_to or utcnow()
            edge.lifecycle_state = "expired"
        store.add_edge(edge)
        return edge


def relation_family(relation: str) -> str:
    return RELATION_FAMILIES.get(relation, "semantic")


def graph_delta_from_edge(edge: MemoryEdge, *, old_weight: float, operation: str, proposer: str, reason: str) -> dict[str, object]:
    return {
        "edge_id": "|".join([edge.source_id, edge.target_id, edge.relation]),
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation": edge.relation,
        "old_weight": old_weight,
        "new_weight": edge.weight,
        "delta": edge.weight - old_weight,
        "operation": operation,
        "relation_family": relation_family(edge.relation),
        "eligibility": edge.eligibility_trace,
        "confidence": edge.confidence,
        "inhibition_penalty": edge.inhibition_score,
        "contradiction_penalty": edge.contradiction_penalty,
        "lifecycle_state": edge.lifecycle_state,
        "provenance": list(edge.provenance),
        "evidence_ids": list(edge.provenance),
        "proposer": proposer,
        "valid_from": edge.valid_from.isoformat() if edge.valid_from else None,
        "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
        "reason": reason,
    }


def _find_edge(store: MemoryStore, source_id: str, target_id: str, relation: str) -> MemoryEdge | None:
    for edge in store.list_edges():
        if edge.relation != relation:
            continue
        if {edge.source_id, edge.target_id} == {source_id, target_id}:
            return edge
    return None


def _apply_endpoint_lists(store: MemoryStore, proposal: GraphDeltaProposal) -> None:
    source = store.get_memory(proposal.source_memory_id)
    target = store.get_memory(proposal.target_memory_id)
    if source is None or target is None:
        return
    if proposal.relation == "supports" and target.id not in source.supports:
        source.supports.append(target.id)
    elif proposal.relation == "contradicts" and target.id not in source.contradicts:
        source.contradicts.append(target.id)
    elif proposal.relation == "supersedes" and target.id not in source.supersedes:
        source.supersedes.append(target.id)
        if target.maturity not in {"deleted", "archived"}:
            target.maturity = "obsolete"
    elif proposal.relation in {"derived_from", "compresses_to", "generalizes"} and target.id not in source.derived_from:
        source.derived_from.append(target.id)
    store.upsert_memory(source)
    store.upsert_memory(target)


def _choose_relation(candidate: GraphRelationCandidate, by_id: dict[str, MemoryItem]) -> str | None:
    suggestions = candidate.suggested_relations
    if "same_canonical_fact_key" in candidate.candidate_sources:
        source = by_id.get(candidate.source_memory_id)
        target = by_id.get(candidate.target_memory_id)
        if source and target and _looks_like_supersession(source.content, target.content):
            return "supersedes"
        if "same_as" in suggestions:
            return "same_as"
    for preferred in ["derived_from", "evidence_for", "procedure_for", "generalizes", "supports", "retrieved_with", "coactivated_with", "precedes", "contradicts", "supersedes"]:
        if preferred in suggestions and preferred in ALLOWED_GRAPH_RELATIONS:
            return preferred
    return None


def _initial_confidence(candidate: GraphRelationCandidate, relation: str) -> float:
    base = 0.58 + min(0.28, candidate.score * 0.18)
    if relation in SAFE_RELATIONS:
        base += 0.12
    if relation in HIGH_RISK_RELATIONS:
        base -= 0.08
    return min(0.92, max(0.5, base))


def _min_confidence(relation: str) -> float:
    if relation in SAFE_RELATIONS:
        return 0.45
    if relation in SEMANTIC_RELATIONS:
        return 0.58
    return 0.6


def _has_shared_basis(source: MemoryItem, target: MemoryItem, proposal: GraphDeltaProposal) -> bool:
    if set(source.evidence) & set(target.evidence):
        return True
    if set(source.entities) & set(target.entities) or set(source.keywords) & set(target.keywords):
        return True
    if _canonical_fact_key(source) and _canonical_fact_key(source) == _canonical_fact_key(target):
        return True
    return bool(set(proposal.candidate_sources) & {"same_query_retrieval", "same_sleep_cluster", "same_evidence_chain", "co_use_outcome"})


def _canonical_fact_key(item: MemoryItem) -> str:
    terms = sorted(set(item.entities or item.keywords))
    return "::".join(term.lower() for term in terms[:3])


def _jaccard(left: str, right: str) -> float:
    left_terms = {term.lower().strip(".,:;()[]`'\"?") for term in left.split() if term.strip()}
    right_terms = {term.lower().strip(".,:;()[]`'\"?") for term in right.split() if term.strip()}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _looks_like_supersession(left: str, right: str) -> bool:
    text = f"{left} {right}".lower()
    return any(term in text for term in ["now", "current", "instead", "replaces", "replaced", "supersedes", "obsolete", "deprecated", "contradict"])


__all__ = [
    "ALLOWED_GRAPH_RELATIONS",
    "DeterministicRelationProposer",
    "GraphBuildContext",
    "GraphCandidateGenerator",
    "GraphDeltaValidator",
    "GraphEdgeState",
    "GraphMode",
    "GraphMutationCommitter",
    "GraphProposalProvider",
    "GraphRelationCandidate",
    "HIGH_RISK_RELATIONS",
    "SAFE_RELATIONS",
    "SEMANTIC_RELATIONS",
    "graph_delta_from_edge",
    "relation_family",
]

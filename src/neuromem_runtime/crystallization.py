from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Literal, Protocol

from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryFrame, MemoryItem, utcnow
from neuromem.stores.base import MemoryStore
from neuromem_runtime.policy_v2 import AssociativeEdgeProposal, FrameDeltaProposal, GraphDeltaProposal, LogicEdgeProposal, ValidationStep
from neuromem_runtime.semantic_graph import GraphBuildContext


RetrievalLens = Literal["auto", "associative", "logical", "procedural", "historical", "audit"]

ASSOCIATIVE_RELATIONS = {"associated_with", "coactivated_with", "precedes", "retrieved_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}
LOGIC_RELATIONS = {"supports", "contradicts", "supersedes", "same_as", "generalizes", "specializes", "procedure_for", "preference_of", "applies_to", "evidence_for", "derived_from", "compresses_to", "causes", "inhibits"}
HIGH_RISK_LOGIC_RELATIONS = {"causes", "contradicts", "supersedes", "inhibits"}


class FrameExtractor(Protocol):
    def extract(self, context: GraphBuildContext, memory_ids: list[str] | None = None) -> list[FrameDeltaProposal]:
        raise NotImplementedError


class FrameValidator(Protocol):
    def validate_frame(self, proposal: FrameDeltaProposal, *, context: GraphBuildContext, store: MemoryStore | None = None) -> ValidationStep:
        raise NotImplementedError


class LogicRelationValidator(Protocol):
    def validate_logic_edge(self, proposal: LogicEdgeProposal, *, context: GraphBuildContext, store: MemoryStore | None = None) -> ValidationStep:
        raise NotImplementedError


class CrystallizationPlanner(Protocol):
    def plan_sleep_frames(self, context: GraphBuildContext) -> list[FrameDeltaProposal]:
        raise NotImplementedError


class DeterministicFrameExtractor:
    def extract(self, context: GraphBuildContext, memory_ids: list[str] | None = None) -> list[FrameDeltaProposal]:
        by_id = context.memory_by_id()
        ids = memory_ids or sorted(by_id)
        proposals: list[FrameDeltaProposal] = []
        for memory_id in ids:
            memory = by_id.get(memory_id)
            if memory is None or memory.namespace != context.namespace:
                continue
            proposals.append(frame_proposal_for_memory(memory, context=context))
        return proposals


class DefaultFrameValidator:
    def validate_frame(self, proposal: FrameDeltaProposal, *, context: GraphBuildContext, store: MemoryStore | None = None) -> ValidationStep:
        if not proposal.content.strip():
            return ValidationStep(name="FrameValidator", passed=False, reason="frame content is required")
        if not proposal.source_memory_ids and not proposal.source_event_ids:
            return ValidationStep(name="FrameValidator", passed=False, reason="frame requires source memory or source event ids")
        if not proposal.evidence_ids:
            return ValidationStep(name="FrameValidator", passed=False, reason="frame requires evidence ids")
        if proposal.valid_from and proposal.valid_to and proposal.valid_to < proposal.valid_from:
            return ValidationStep(name="FrameValidator", passed=False, reason="frame valid_to is before valid_from")
        if proposal.confidence < 0.5:
            return ValidationStep(name="FrameValidator", passed=False, reason="frame confidence below threshold")
        by_id = context.memory_by_id()
        for memory_id in proposal.source_memory_ids:
            memory = by_id.get(memory_id) or (store.get_memory(memory_id) if store is not None else None)
            if memory is None:
                return ValidationStep(name="FrameValidator", passed=False, reason=f"frame source memory not found: {memory_id}")
            if memory.namespace != context.namespace:
                return ValidationStep(name="FrameValidator", passed=False, reason="frame source memory outside namespace")
        return ValidationStep(name="FrameValidator", passed=True)


class DefaultLogicRelationValidator:
    def validate_logic_edge(self, proposal: LogicEdgeProposal, *, context: GraphBuildContext, store: MemoryStore | None = None) -> ValidationStep:
        if proposal.relation not in LOGIC_RELATIONS:
            return ValidationStep(name="LogicRelationValidator", passed=False, reason=f"unsupported logic relation: {proposal.relation}")
        if not proposal.evidence_ids:
            return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge requires evidence ids")
        if not proposal.proof_obligation.strip():
            return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge requires proof obligation")
        if proposal.source_frame_id == proposal.target_frame_id:
            return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge endpoints must be distinct frames")
        if proposal.valid_from and proposal.valid_to and proposal.valid_to < proposal.valid_from:
            return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge valid_to is before valid_from")
        if proposal.relation in HIGH_RISK_LOGIC_RELATIONS and proposal.confidence > 0.68:
            return ValidationStep(name="LogicRelationValidator", passed=False, reason=f"{proposal.relation} confidence is capped before repeated replay/outcome evidence")
        if store is not None:
            source = store.get_logic_node(proposal.source_frame_id)
            target = store.get_logic_node(proposal.target_frame_id)
            if source is None or target is None:
                return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge requires existing frame endpoints")
            if source.namespace != context.namespace or target.namespace != context.namespace:
                return ValidationStep(name="LogicRelationValidator", passed=False, reason="logic edge frame endpoint outside namespace")
        return ValidationStep(name="LogicRelationValidator", passed=True)


class DeterministicCrystallizationPlanner:
    def plan_sleep_frames(self, context: GraphBuildContext) -> list[FrameDeltaProposal]:
        by_id = context.memory_by_id()
        proposals: list[FrameDeltaProposal] = []
        for index, cluster in enumerate(context.sleep_clusters):
            memories = [by_id[memory_id] for memory_id in cluster if memory_id in by_id]
            if len(memories) < 2:
                continue
            content = _compile_cluster_content(memories)
            evidence_ids = sorted({evidence for memory in memories for evidence in memory.evidence} | set(context.evidence_ids))
            source_event_ids = sorted({event for memory in memories for event in memory.source_event_ids})
            source_memory_ids = [memory.id for memory in memories]
            proposals.append(
                FrameDeltaProposal(
                    operation="promote_frame",
                    frame_id=f"frame_sleep_{_short_hash(context.namespace, str(index), *source_memory_ids)}",
                    frame_type="procedure" if _looks_procedural(content) else "schema",
                    content=content,
                    canonical_key=_canonical_from_terms([term for memory in memories for term in [*memory.entities, *memory.keywords]]) or f"sleep_cluster_{index}",
                    payload={"cluster_size": len(memories), "source_types": sorted({memory.type for memory in memories})},
                    source_memory_ids=source_memory_ids,
                    source_event_ids=source_event_ids,
                    evidence_ids=evidence_ids,
                    confidence=0.78,
                    commitment_level="compiled_schema",
                    lifecycle_state="compiled",
                    reason="sleep replay crystallized repeated experience",
                    proposer=context.proposer,
                )
            )
        return proposals


def frame_proposal_for_memory(memory: MemoryItem, *, context: GraphBuildContext, validated: bool = False) -> FrameDeltaProposal:
    frame_type = infer_frame_type(memory)
    evidence_ids = list(dict.fromkeys([*memory.evidence, *context.evidence_ids, *memory.source_event_ids]))
    return FrameDeltaProposal(
        operation="validate_frame" if validated else "propose_frame",
        frame_id=frame_id_for_memory(memory, frame_type),
        frame_type=frame_type,
        content=memory.summary or memory.content,
        canonical_key=canonical_key_for_memory(memory),
        payload={"memory_type": memory.type, "entities": list(memory.entities), "keywords": list(memory.keywords)},
        source_memory_ids=[memory.id],
        source_event_ids=list(memory.source_event_ids),
        evidence_ids=evidence_ids,
        confidence=max(0.55, min(0.95, memory.confidence)),
        commitment_level="validated_logic" if validated else "candidate_frame",
        lifecycle_state="validated" if validated else "candidate",
        valid_from=memory.valid_from,
        valid_to=memory.valid_to,
        reason="deterministic frame extracted from memory",
        proposer=context.proposer,
    )


def graph_delta_to_structural(proposal: GraphDeltaProposal, *, context: GraphBuildContext) -> tuple[list[FrameDeltaProposal], list[AssociativeEdgeProposal], list[LogicEdgeProposal]]:
    if proposal.relation in ASSOCIATIVE_RELATIONS:
        return (
            [],
            [
                AssociativeEdgeProposal(
                    operation=proposal.operation,
                    source_memory_id=proposal.source_memory_id,
                    target_memory_id=proposal.target_memory_id,
                    relation=proposal.relation,
                    weight=min(proposal.weight, 0.45),
                    confidence=min(proposal.confidence, 0.62),
                    evidence_ids=list(proposal.evidence_ids),
                    candidate_sources=list(proposal.candidate_sources),
                    reason=proposal.reason,
                    proposer=proposal.proposer,
                    lifecycle_state=proposal.lifecycle_state if proposal.lifecycle_state in {"captured", "reinforced", "mature", "inhibited", "expired"} else "captured",
                )
            ],
            [],
        )
    by_id = context.memory_by_id()
    source = by_id.get(proposal.source_memory_id)
    target = by_id.get(proposal.target_memory_id)
    if source is None or target is None:
        return [], [], []
    source_frame = frame_proposal_for_memory(source, context=context, validated=False)
    target_frame = frame_proposal_for_memory(target, context=context, validated=False)
    logic = LogicEdgeProposal(
        operation=proposal.operation,
        source_frame_id=source_frame.frame_id or frame_id_for_memory(source, source_frame.frame_type),
        target_frame_id=target_frame.frame_id or frame_id_for_memory(target, target_frame.frame_type),
        source_memory_id=source.id,
        target_memory_id=target.id,
        relation=proposal.relation,
        weight=proposal.weight,
        confidence=min(proposal.confidence, 0.68 if proposal.relation in HIGH_RISK_LOGIC_RELATIONS else proposal.confidence),
        proof_obligation=proposal.reason or f"{proposal.relation} requires bounded candidate evidence",
        evidence_ids=list(proposal.evidence_ids),
        valid_from=proposal.valid_from,
        valid_to=proposal.valid_to,
        reason=proposal.reason,
        proposer=proposal.proposer,
        lifecycle_state="inhibited" if proposal.relation in {"contradicts", "inhibits"} else proposal.lifecycle_state,
    )
    return [source_frame, target_frame], [], [logic]


def frame_from_proposal(proposal: FrameDeltaProposal, *, namespace: str) -> MemoryFrame:
    frame_id = proposal.frame_id or f"frame_{_short_hash(namespace, proposal.frame_type, proposal.content, *proposal.source_memory_ids)}"
    record = {
        "frame_id": frame_id,
        "namespace": namespace,
        "frame_type": proposal.frame_type,
        "content": proposal.content,
        "canonical_key": proposal.canonical_key or _canonical_from_terms([proposal.content]),
        "payload": dict(proposal.payload),
        "source_memory_ids": list(proposal.source_memory_ids),
        "source_event_ids": list(proposal.source_event_ids),
        "evidence_ids": list(proposal.evidence_ids),
        "confidence": proposal.confidence,
        "commitment_level": proposal.commitment_level,
        "lifecycle_state": proposal.lifecycle_state,
        "valid_from": proposal.valid_from.isoformat() if proposal.valid_from else None,
        "valid_to": proposal.valid_to.isoformat() if proposal.valid_to else None,
        "provenance_hash": provenance_hash({"frame": proposal.model_dump(mode="json"), "namespace": namespace}),
    }
    return MemoryFrame.from_record(record)


def associative_edge_from_proposal(proposal: AssociativeEdgeProposal, *, namespace: str) -> AssociativeEdge:
    return AssociativeEdge(
        namespace=namespace,
        source_id=proposal.source_memory_id,
        target_id=proposal.target_memory_id,
        relation=proposal.relation,  # type: ignore[arg-type]
        weight=proposal.weight,
        confidence=proposal.confidence,
        salience=proposal.salience,
        outcome_reward=proposal.outcome_reward,
        lifecycle_state=proposal.lifecycle_state,  # type: ignore[arg-type]
        provenance=list(proposal.evidence_ids),
    )


def logic_edge_from_proposal(proposal: LogicEdgeProposal, *, namespace: str) -> LogicEdge:
    inhibition = 0.7 if proposal.relation in {"contradicts", "inhibits"} else 0.0
    contradiction = 0.6 if proposal.relation == "contradicts" else 0.0
    return LogicEdge(
        namespace=namespace,
        source_frame_id=proposal.source_frame_id,
        target_frame_id=proposal.target_frame_id,
        source_memory_id=proposal.source_memory_id,
        target_memory_id=proposal.target_memory_id,
        relation=proposal.relation,  # type: ignore[arg-type]
        weight=proposal.weight,
        confidence=proposal.confidence,
        proof_obligation=proposal.proof_obligation,
        evidence_ids=list(proposal.evidence_ids),
        valid_from=proposal.valid_from,
        valid_to=proposal.valid_to,
        lifecycle_state=proposal.lifecycle_state,  # type: ignore[arg-type]
        inhibition_score=inhibition,
        contradiction_penalty=contradiction,
        provenance_hash=provenance_hash(proposal.model_dump(mode="json")),
        proposer=proposal.proposer,
    )


def infer_frame_type(memory: MemoryItem) -> str:
    text = memory.content.lower()
    if memory.type == "procedural" or any(term in text for term in ["always", "workflow", "procedure", "step", "rule:"]):
        return "procedure"
    if memory.type == "preference" or any(term in text for term in ["prefers", "preference"]):
        return "preference"
    if memory.type == "constraint" or any(term in text for term in ["must", "must not", "constraint", "policy"]):
        return "constraint"
    if any(term in text for term in ["failed", "failure", "bug", "regression", "error"]):
        return "failure_pattern"
    if memory.type == "semantic" or any(term in text for term in [" is ", " are ", "current", "now"]):
        return "fact"
    if memory.type == "schema":
        return "schema"
    return "episode"


def frame_id_for_memory(memory: MemoryItem, frame_type: str) -> str:
    return f"frame_{_short_hash(memory.namespace, memory.id, frame_type)}"


def canonical_key_for_memory(memory: MemoryItem) -> str:
    return _canonical_from_terms([*memory.entities, *memory.keywords]) or _canonical_from_terms([memory.content]) or memory.id


def provenance_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _compile_cluster_content(memories: list[MemoryItem]) -> str:
    terms = _canonical_from_terms([term for memory in memories for term in [*memory.entities, *memory.keywords]])
    prefix = f"Repeated pattern for {terms}: " if terms else "Repeated experience pattern: "
    snippets = [memory.summary or memory.content for memory in memories[:3]]
    return prefix + " | ".join(snippet.strip() for snippet in snippets if snippet.strip())


def _looks_procedural(content: str) -> bool:
    lowered = content.lower()
    return any(term in lowered for term in ["fix", "run", "before", "after", "workflow", "procedure", "step", "rule"])


def _canonical_from_terms(terms: list[str]) -> str:
    normalized = []
    for term in terms:
        for piece in str(term).lower().replace(":", " ").replace("/", " ").split():
            cleaned = piece.strip(".,;()[]`'\"?")
            if len(cleaned) > 2 and cleaned not in normalized:
                normalized.append(cleaned)
    return "::".join(sorted(normalized)[:4])


def _short_hash(*values: str) -> str:
    return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ASSOCIATIVE_RELATIONS",
    "CrystallizationPlanner",
    "DefaultFrameValidator",
    "DefaultLogicRelationValidator",
    "DeterministicCrystallizationPlanner",
    "DeterministicFrameExtractor",
    "FrameExtractor",
    "FrameValidator",
    "HIGH_RISK_LOGIC_RELATIONS",
    "LOGIC_RELATIONS",
    "LogicRelationValidator",
    "RetrievalLens",
    "associative_edge_from_proposal",
    "frame_from_proposal",
    "frame_id_for_memory",
    "frame_proposal_for_memory",
    "graph_delta_to_structural",
    "logic_edge_from_proposal",
    "provenance_hash",
]

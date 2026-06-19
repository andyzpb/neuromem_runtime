from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProposalSource = Literal["deterministic", "small_llm", "user", "system", "tool", "admin"]
PolicyIntent = Literal["retrieve", "add", "update", "link", "suppress", "supersede", "consolidate", "delete_request", "noop"]
RiskLevel = Literal["low", "medium", "high", "critical"]
WriteGateDecision = Literal["commit", "defer", "noop"]
DurabilityHorizon = Literal["none", "thread", "cross_thread", "long_term"]
WriteGateBasis = Literal["current_user_message", "short_term_context", "retrieved_memory", "tool_result", "system"]
GraphDeltaOperation = Literal["add_edge", "update_edge", "inhibit_edge", "expire_edge"]
FrameDeltaOperation = Literal["propose_frame", "validate_frame", "promote_frame", "inhibit_frame", "archive_frame"]
EdgeDeltaOperation = Literal["add_edge", "update_edge", "inhibit_edge", "expire_edge"]
SemanticCommitmentLevel = Literal["raw_experience", "durable_memory", "associative_link", "candidate_frame", "validated_logic", "compiled_schema"]
FrameType = Literal["episode", "fact", "claim", "procedure", "preference", "constraint", "entity", "schema", "failure_pattern"]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    source: str = "unknown"
    content_hash: str | None = None
    signature: str | None = None


class ProposedDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    target_memory_id: str | None = None
    field: str | None = None
    value: object | None = None
    reason: str = ""


class GraphDeltaProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: GraphDeltaOperation = "add_edge"
    source_memory_id: str
    target_memory_id: str
    relation: str
    weight: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    candidate_sources: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    reason: str = ""
    proposer: str = "deterministic"
    lifecycle_state: str = "provisional"


class FrameDeltaProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: FrameDeltaOperation = "propose_frame"
    frame_id: str | None = None
    frame_type: FrameType
    content: str
    canonical_key: str = ""
    payload: dict[str, object] = Field(default_factory=dict)
    source_memory_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    commitment_level: SemanticCommitmentLevel = "candidate_frame"
    lifecycle_state: str = "candidate"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    reason: str = ""
    proposer: str = "deterministic"


class AssociativeEdgeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: EdgeDeltaOperation = "add_edge"
    source_memory_id: str
    target_memory_id: str
    relation: str
    weight: float = Field(default=0.12, ge=0.0, le=1.0)
    confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    candidate_sources: list[str] = Field(default_factory=list)
    reason: str = ""
    proposer: str = "deterministic"
    lifecycle_state: str = "captured"


class LogicEdgeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: EdgeDeltaOperation = "add_edge"
    source_frame_id: str
    target_frame_id: str
    relation: str
    source_memory_id: str | None = None
    target_memory_id: str | None = None
    weight: float = Field(default=0.25, ge=0.0, le=1.0)
    confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    proof_obligation: str
    evidence_ids: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    reason: str = ""
    proposer: str = "deterministic"
    lifecycle_state: str = "provisional"


class TargetSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_ids: list[str] = Field(default_factory=list)
    query: str | None = None
    namespace: str | None = None


class WriteGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: WriteGateDecision = "noop"
    durability_horizon: DurabilityHorizon = "none"
    commitment_level: SemanticCommitmentLevel = "raw_experience"
    basis: WriteGateBasis = "current_user_message"
    signals: list[str] = Field(default_factory=list)
    rationale: str = ""


class MemoryPolicyV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(default_factory=lambda: f"policy_{uuid4().hex}")
    proposer: str = "deterministic"
    proposal_source: ProposalSource = "deterministic"
    intent: PolicyIntent = "noop"
    risk_level: RiskLevel = "low"
    evidence_chain: list[EvidenceRef] = Field(default_factory=list)
    target_selector: TargetSelector = Field(default_factory=TargetSelector)
    proposed_deltas: list[ProposedDelta] = Field(default_factory=list)
    grounded_claims: list[dict[str, object]] = Field(default_factory=list)
    frame_deltas: list[FrameDeltaProposal] = Field(default_factory=list)
    associative_deltas: list[AssociativeEdgeProposal] = Field(default_factory=list)
    logic_deltas: list[LogicEdgeProposal] = Field(default_factory=list)
    graph_deltas: list[GraphDeltaProposal] = Field(default_factory=list)
    safety_annotations: dict[str, object] = Field(default_factory=dict)
    write_gate: WriteGate | None = None
    temporal_scope: str = "all_valid"
    retention_policy: str = "keep_for_audit"
    rollback_plan: str | None = None

    @model_validator(mode="after")
    def evidence_required_for_mutation(self):
        structural_deltas = [*self.frame_deltas, *self.associative_deltas, *self.logic_deltas, *self.graph_deltas]
        if self.intent in {"add", "update", "link", "suppress", "supersede", "consolidate", "delete_request"} and not self.evidence_chain and not structural_deltas:
            raise ValueError("mutation policies require evidence_chain")
        for delta in structural_deltas:
            evidence_ids = getattr(delta, "evidence_ids", None)
            if isinstance(evidence_ids, list) and not evidence_ids and self.evidence_chain:
                evidence_ids.extend(evidence.event_id for evidence in self.evidence_chain)
        return self

    @model_validator(mode="after")
    def normalize_write_gate(self):
        if self.write_gate is None:
            raw = self.safety_annotations.get("write_gate")
            if isinstance(raw, dict):
                self.write_gate = WriteGate.model_validate(raw)
            elif self.proposed_deltas and self.intent in {"add", "update", "link"}:
                self.write_gate = WriteGate(decision="commit", durability_horizon="long_term", commitment_level="durable_memory")
            elif self.intent == "noop":
                self.write_gate = WriteGate(decision="noop")
        if self.write_gate is not None:
            self.safety_annotations["write_gate"] = self.write_gate.model_dump(mode="json")
        return self


class ValidationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    reason: str = ""


class ValidatedMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    approved_deltas: list[ProposedDelta] = Field(default_factory=list)
    rejected_deltas: list[ProposedDelta] = Field(default_factory=list)
    required_human_review: bool = False
    risk_score: float = 0.0
    validator_trace: list[ValidationStep] = Field(default_factory=list)


__all__ = [
    "AssociativeEdgeProposal",
    "EvidenceRef",
    "FrameDeltaProposal",
    "GraphDeltaProposal",
    "LogicEdgeProposal",
    "MemoryPolicyV2",
    "ProposedDelta",
    "TargetSelector",
    "ValidatedMutation",
    "ValidationStep",
    "WriteGate",
]

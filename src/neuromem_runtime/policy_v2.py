from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProposalSource = Literal["deterministic", "small_llm", "user", "system", "tool", "admin"]
PolicyIntent = Literal["retrieve", "add", "update", "link", "suppress", "supersede", "consolidate", "delete_request", "noop"]
RiskLevel = Literal["low", "medium", "high", "critical"]
GraphDeltaOperation = Literal["add_edge", "update_edge", "inhibit_edge", "expire_edge"]


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


class TargetSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_ids: list[str] = Field(default_factory=list)
    query: str | None = None
    namespace: str | None = None


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
    graph_deltas: list[GraphDeltaProposal] = Field(default_factory=list)
    safety_annotations: dict[str, object] = Field(default_factory=dict)
    temporal_scope: str = "all_valid"
    retention_policy: str = "keep_for_audit"
    rollback_plan: str | None = None

    @model_validator(mode="after")
    def evidence_required_for_mutation(self):
        if self.intent in {"add", "update", "link", "suppress", "supersede", "consolidate", "delete_request"} and not self.evidence_chain and not self.graph_deltas:
            raise ValueError("mutation policies require evidence_chain")
        for delta in self.graph_deltas:
            if not delta.evidence_ids and self.evidence_chain:
                delta.evidence_ids.extend(evidence.event_id for evidence in self.evidence_chain)
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
    "EvidenceRef",
    "GraphDeltaProposal",
    "MemoryPolicyV2",
    "ProposedDelta",
    "TargetSelector",
    "ValidatedMutation",
    "ValidationStep",
]

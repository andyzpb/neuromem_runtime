from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProposalSource = Literal["deterministic", "small_llm", "user", "system", "tool", "admin"]
PolicyIntent = Literal["retrieve", "add", "update", "link", "suppress", "supersede", "consolidate", "delete_request", "noop"]
RiskLevel = Literal["low", "medium", "high", "critical"]


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
    safety_annotations: dict[str, object] = Field(default_factory=dict)
    temporal_scope: str = "all_valid"
    retention_policy: str = "keep_for_audit"
    rollback_plan: str | None = None

    @model_validator(mode="after")
    def evidence_required_for_mutation(self):
        if self.intent in {"add", "update", "link", "suppress", "supersede", "consolidate", "delete_request"} and not self.evidence_chain:
            raise ValueError("mutation policies require evidence_chain")
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
    "MemoryPolicyV2",
    "ProposedDelta",
    "TargetSelector",
    "ValidatedMutation",
    "ValidationStep",
]

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


MemoryOperation = Literal["ADD", "UPDATE", "LINK", "NOOP"]
ForgetOperation = Literal["NOOP", "DECAY", "INHIBIT", "INVALIDATE", "ARCHIVE", "DELETE_REQUEST"]
ConsolidationObjective = Literal["deduplicate", "generalize", "extract_rule", "resolve_conflict", "compress"]
TemporalScope = Literal["current", "recent", "all_valid", "as_of_date", "all_including_obsolete"]
MemoryTTL = Literal["working", "session", "long_term", "until_invalidated"]
TransactionPhase = Literal["PROPOSED", "VALIDATED", "COMMITTED", "REJECTED", "ROLLED_BACK", "AUDITED"]
TransactionOperation = Literal[
    "RETRIEVE",
    "ADD",
    "UPDATE",
    "LINK",
    "INHIBIT",
    "INVALIDATE",
    "ARCHIVE",
    "DELETE_REQUEST",
    "CONSOLIDATE",
    "NOOP",
]


@dataclass(slots=True)
class RetrievalPlan:
    enabled: bool = True
    query: str = ""
    query_rewrites: list[str] = field(default_factory=list)
    memory_types: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    temporal_scope: TemporalScope = "all_valid"
    max_items: int = 8
    graph_expansion: bool = True
    graph_depth: int = 2
    graph_restart_prob: float = 0.25
    graph_min_score: float = 0.03
    require_provenance: bool = True


@dataclass(slots=True)
class WritePlan:
    operation: MemoryOperation = "NOOP"
    memory_type: str | None = None
    content: str | None = None
    target_memory_id: str | None = None
    salience_estimate: float = 0.0
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    ttl: MemoryTTL = "session"


@dataclass(slots=True)
class ForgetPlan:
    operation: ForgetOperation = "NOOP"
    target_memory_id: str | None = None
    reason: str | None = None
    keep_for_audit: bool = True


@dataclass(slots=True)
class ConsolidationPlan:
    enabled: bool = False
    cluster_ids: list[str] = field(default_factory=list)
    target_type: str | None = None
    objective: ConsolidationObjective | None = None


@dataclass(slots=True)
class MemoryPolicy:
    retrieval: RetrievalPlan = field(default_factory=RetrievalPlan)
    write: WritePlan = field(default_factory=WritePlan)
    forget: ForgetPlan = field(default_factory=ForgetPlan)
    consolidation: ConsolidationPlan = field(default_factory=ConsolidationPlan)
    reason: str = ""
    source: Literal["deterministic", "small_llm"] = "deterministic"
    write_gate: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ValidatedPolicy:
    policy: MemoryPolicy
    approved: bool
    approved_actions: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryTransactionRecord:
    transaction_id: str
    phase: TransactionPhase
    operation: TransactionOperation
    proposed_by: str = "deterministic"
    validator_decision: str = "not_applicable"
    evidence: list[str] = field(default_factory=list)
    target_memories: list[str] = field(default_factory=list)
    graph_effects: list[str] = field(default_factory=list)
    lifecycle_effects: list[str] = field(default_factory=list)
    audit_trace: dict[str, object] = field(default_factory=dict)
    rollback: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MemoryTrace:
    task_id: str
    query: str
    retrieval_plan: RetrievalPlan
    policy_source: Literal["deterministic", "small_llm"] = "deterministic"
    approved_actions: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)
    executed_actions: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    selected_memory_ids: list[str] = field(default_factory=list)
    rejected_memory_ids: list[str] = field(default_factory=list)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    graph_paths: list[list[str]] = field(default_factory=list)
    pfc_reason: str = ""
    validator_decision: str = ""
    final_context_tokens: int = 0
    outcome: Literal["success", "failure", "partial", "unknown"] | None = None
    trace_id: str = field(default_factory=lambda: f"trace_{uuid4().hex}")
    events: list[dict[str, object]] = field(default_factory=list)
    baseline_scores: dict[str, float] = field(default_factory=dict)
    diffusion_scores: dict[str, float] = field(default_factory=dict)
    suppression_reasons: dict[str, str] = field(default_factory=dict)
    consolidation_links: dict[str, list[str]] = field(default_factory=dict)
    query_plan: dict[str, object] = field(default_factory=dict)
    source_channels: list[str] = field(default_factory=list)
    gate_decision: str = ""
    canonical_fact_ids: list[str] = field(default_factory=list)
    memory_version: str = ""
    invalidation_state: str = "valid"
    recall_config_hash: str = ""

    def _audit_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "retrieval_plan": asdict(self.retrieval_plan),
            "policy_source": self.policy_source,
            "approved_actions": self.approved_actions,
            "rejected_reasons": self.rejected_reasons,
            "executed_actions": self.executed_actions,
            "fallback_reason": self.fallback_reason,
            "selected_memory_ids": self.selected_memory_ids,
            "rejected_memory_ids": self.rejected_memory_ids,
            "scores": self.scores,
            "graph_paths": self.graph_paths,
            "pfc_reason": self.pfc_reason,
            "validator_decision": self.validator_decision,
            "final_context_tokens": self.final_context_tokens,
            "outcome": self.outcome,
            "trace_id": self.trace_id,
            "events": self.events,
            "baseline_scores": self.baseline_scores,
            "diffusion_scores": self.diffusion_scores,
            "suppression_reasons": self.suppression_reasons,
            "consolidation_links": self.consolidation_links,
            "query_plan": self.query_plan,
            "source_channels": self.source_channels,
            "gate_decision": self.gate_decision,
            "canonical_fact_ids": self.canonical_fact_ids,
            "memory_version": self.memory_version,
            "invalidation_state": self.invalidation_state,
            "recall_config_hash": self.recall_config_hash,
        }

    def to_transactions(self) -> list[MemoryTransactionRecord]:
        operation = "NOOP"
        if "RETRIEVE" in self.executed_actions or self.selected_memory_ids:
            operation = "RETRIEVE"
        elif self.executed_actions:
            first = self.executed_actions[0]
            operation = first if first in TransactionOperation.__args__ else "NOOP"  # type: ignore[attr-defined]
        phase: TransactionPhase = "COMMITTED" if self.executed_actions else "REJECTED" if self.rejected_reasons else "AUDITED"
        effects = [f"suppressed:{memory_id}" for memory_id in self.rejected_memory_ids]
        effects.extend(f"consolidated:{source}->{','.join(targets)}" for source, targets in self.consolidation_links.items())
        graph_effects = ["graph_diffusion"] if self.graph_paths or self.diffusion_scores else []
        return [
            MemoryTransactionRecord(
                transaction_id=f"txn_{self.trace_id}",
                phase=phase,
                operation=operation,  # type: ignore[arg-type]
                proposed_by=self.policy_source,
                validator_decision=self.validator_decision or ("; ".join(self.rejected_reasons) if self.rejected_reasons else "not_applicable"),
                evidence=list(self.selected_memory_ids),
                target_memories=list(self.selected_memory_ids),
                graph_effects=graph_effects,
                lifecycle_effects=effects,
                audit_trace=self._audit_dict(),
                rollback=self.fallback_reason,
            )
        ]

    def to_dict(self) -> dict[str, object]:
        value = self._audit_dict()
        value["transactions"] = [transaction.to_dict() for transaction in self.to_transactions()]
        return value

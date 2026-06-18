from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from neuromem.core.policy import (
    ConsolidationPlan,
    ForgetPlan,
    MemoryPolicy,
    MemoryTrace,
    RetrievalPlan,
    ValidatedPolicy,
    WritePlan,
)
from neuromem.core.runtime import NeuroMemRuntime
from neuromem_runtime.deltas import GraphDelta, IndexDelta, LifecycleDelta, MemoryDelta, MutationExecutionResult
from neuromem_runtime.ledger import LedgerEvent, MemoryLedger
from neuromem_runtime.lifecycle import LifecycleStateMachine
from neuromem_runtime.policy_v2 import MemoryPolicyV2, ValidatedMutation
from neuromem_runtime.validators import ValidationContext, ValidatorStack


@dataclass(slots=True)
class PolicyExecutionContext:
    phase: str
    task: str
    query: str
    state: dict[str, object]
    authorize_delete: bool = False
    retrieved_memory_ids: list[str] | None = None
    user_id: str | None = None
    namespace: str = "default"
    historical: bool = False


class PolicyExecutor:
    def __init__(self, runtime: NeuroMemRuntime, ledger: MemoryLedger, validator_stack: ValidatorStack | None = None) -> None:
        self.runtime = runtime
        self.ledger = ledger
        self.validator_stack = validator_stack or ValidatorStack()
        self.lifecycle = LifecycleStateMachine()

    def execute(self, policy: MemoryPolicy | MemoryPolicyV2, context: PolicyExecutionContext) -> MutationExecutionResult:
        assert self.runtime.store is not None
        legacy_policy = self._coerce_policy(policy)
        before_memories = self._snapshot_memories(context.namespace)
        before_edges = self._snapshot_edges()
        pre_validation = self.validator_stack.validate(
            legacy_policy,
            ValidationContext(
                store=self.runtime.store,
                ledger=self.ledger,
                phase=context.phase,
                authorize_delete=context.authorize_delete,
                user_id=context.user_id,
                namespace=context.namespace,
                historical=context.historical,
            ),
        )
        transaction_id = f"txn_{uuid4().hex}"
        self._append_stage(
            transaction_id,
            None,
            "PROPOSED",
            "proposal_recorded",
            legacy_policy,
            pre_validation,
            audit={"policy": self._policy_dict(legacy_policy)},
        )
        if not pre_validation.approved:
            trace = self._rejected_trace(legacy_policy, context, pre_validation)
            self.runtime.last_trace = trace
            self.runtime.traces[trace.trace_id] = trace
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "REJECTED",
                "validation_rejected",
                legacy_policy,
                pre_validation,
                audit=trace.to_dict(),
            )
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "AUDITED",
                "audit_finalized",
                legacy_policy,
                pre_validation,
                audit=trace.to_dict(),
            )
            return MutationExecutionResult(trace=trace, validated_mutation=pre_validation)

        approved = ValidatedPolicy(
            policy=legacy_policy,
            approved=True,
            approved_actions=self._approved_actions(legacy_policy),
            rejected_reasons=[],
        )
        self._append_stage(transaction_id, None, "VALIDATED", "validation_approved", legacy_policy, pre_validation, audit=pre_validation.model_dump())
        self.runtime._execute_validated_policy(  # noqa: SLF001 - product executor owns validation and audit around core mutation primitive.
            legacy_policy,
            approved,
            phase=context.phase,
            task=context.task,
            query=context.query,
            state=context.state,
            retrieved_memory_ids=context.retrieved_memory_ids,
        )
        trace = self.runtime.last_trace
        if trace is None:
            raise RuntimeError("policy execution did not produce a trace")
        trace.query_plan["governed_transaction_id"] = transaction_id
        after_memories = self._snapshot_memories(context.namespace)
        after_edges = self._snapshot_edges()
        memory_deltas = self._memory_deltas(before_memories, after_memories)
        lifecycle_deltas = self._lifecycle_deltas(before_memories, after_memories)
        graph_deltas = self._graph_deltas(before_edges, after_edges)
        affected_ids = sorted({delta.memory_id for delta in memory_deltas})
        index_deltas = [IndexDelta(index="sqlite_memory_cards", status="updated", memory_id=memory_id).to_dict() for memory_id in affected_ids]
        post_validation = self.validator_stack.validate(
            legacy_policy,
            ValidationContext(
                store=self.runtime.store,
                ledger=self.ledger,
                phase=context.phase,
                authorize_delete=context.authorize_delete,
                user_id=context.user_id,
                namespace=context.namespace,
                historical=context.historical,
                post_commit=True,
                affected_memory_ids=affected_ids,
            ),
        )
        created_ids = sorted(set(after_memories) - set(before_memories))
        deleted_ids = sorted(set(before_memories) - set(after_memories))
        updated_ids = sorted((set(before_memories) & set(after_memories)) & {delta.memory_id for delta in memory_deltas})
        trace.query_plan["mutation_execution_result"] = {
            "created_memory_ids": created_ids,
            "updated_memory_ids": updated_ids,
            "deleted_memory_ids": deleted_ids,
            "memory_deltas": [delta.to_dict() for delta in memory_deltas],
            "graph_deltas": [delta.to_dict() for delta in graph_deltas],
            "lifecycle_deltas": [delta.to_dict() for delta in lifecycle_deltas],
            "index_deltas": index_deltas,
            "validator_trace": [step.model_dump() for step in post_validation.validator_trace],
        }
        evidence = self._evidence(legacy_policy)
        targets = created_ids or updated_ids or deleted_ids or self._target_ids(legacy_policy)
        self._append_stage(
            transaction_id,
            trace.trace_id,
            "COMMITTED",
            "memory_delta_committed",
            legacy_policy,
            post_validation,
            evidence=evidence,
            targets=targets,
            memory_delta=[delta.to_dict() for delta in memory_deltas],
            audit=trace.to_dict(),
        )
        self._append_stage(
            transaction_id,
            trace.trace_id,
            "COMMITTED",
            "index_updated",
            legacy_policy,
            post_validation,
            evidence=evidence,
            targets=targets,
            index_delta=index_deltas,
            audit={"affected_memory_ids": affected_ids},
        )
        if graph_deltas:
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "COMMITTED",
                "graph_delta_committed",
                legacy_policy,
                post_validation,
                evidence=evidence,
                targets=targets,
                graph_delta=[delta.to_dict() for delta in graph_deltas],
                audit={"edge_count": len(graph_deltas)},
            )
        if lifecycle_deltas:
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "COMMITTED",
                "lifecycle_delta_committed",
                legacy_policy,
                post_validation,
                evidence=evidence,
                targets=targets,
                lifecycle_delta=[delta.to_dict() for delta in lifecycle_deltas],
                audit={"transition_count": len(lifecycle_deltas)},
            )
        self._append_stage(
            transaction_id,
            trace.trace_id,
            "AUDITED",
            "audit_finalized",
            legacy_policy,
            post_validation,
            evidence=evidence,
            targets=targets,
            audit=trace.to_dict(),
        )
        for memory_id in targets:
            state = after_memories.get(memory_id)
            if state is not None:
                self.ledger.record_memory_version(memory_id, transaction_id, state)
        for edge_id, state in after_edges.items():
            if before_edges.get(edge_id) != state:
                self.ledger.record_edge_version(edge_id, transaction_id, state)
        return MutationExecutionResult(
            trace=trace,
            validated_mutation=post_validation,
            created_memory_ids=created_ids,
            updated_memory_ids=updated_ids,
            deleted_memory_ids=deleted_ids,
            memory_deltas=memory_deltas,
            graph_deltas=graph_deltas,
            lifecycle_deltas=lifecycle_deltas,
            index_deltas=[IndexDelta(index="sqlite_memory_cards", status="updated", memory_id=memory_id) for memory_id in affected_ids],
        )

    def _coerce_policy(self, policy: MemoryPolicy | MemoryPolicyV2) -> MemoryPolicy:
        if isinstance(policy, MemoryPolicy):
            return policy
        evidence_ids = [evidence.event_id for evidence in policy.evidence_chain]
        first_delta = policy.proposed_deltas[0] if policy.proposed_deltas else None
        operation = (first_delta.operation if first_delta is not None else policy.intent).upper()
        if policy.intent == "delete_request":
            operation = "DELETE_REQUEST"
        if operation in {"INHIBIT", "INVALIDATE", "ARCHIVE", "DELETE_REQUEST", "DECAY"}:
            return MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=policy.target_selector.query or ""),
                write=WritePlan(operation="NOOP"),
                forget=ForgetPlan(operation=operation, target_memory_id=first_delta.target_memory_id if first_delta else (policy.target_selector.memory_ids[0] if policy.target_selector.memory_ids else None), reason=first_delta.reason if first_delta else policy.rollback_plan),
                consolidation=ConsolidationPlan(enabled=False),
                reason=first_delta.reason if first_delta else policy.intent,
                source="small_llm" if policy.proposal_source == "small_llm" else "deterministic",
            )
        write_operation = "ADD" if policy.intent == "add" else "UPDATE" if policy.intent in {"update", "supersede"} else "LINK" if policy.intent == "link" else "NOOP"
        value = first_delta.value if first_delta is not None else None
        memory_type = str(value.get("memory_type")) if isinstance(value, dict) and value.get("memory_type") else None
        content = str(value if not isinstance(value, dict) else value.get("content", "")) if value is not None else None
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=policy.target_selector.query or ""),
            write=WritePlan(
                operation=write_operation,
                memory_type=memory_type,
                content=content,
                target_memory_id=first_delta.target_memory_id if first_delta else (policy.target_selector.memory_ids[0] if policy.target_selector.memory_ids else None),
                confidence=0.75,
                evidence_ids=evidence_ids,
            ),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=policy.intent == "consolidate"),
            reason=first_delta.reason if first_delta else policy.intent,
            source="small_llm" if policy.proposal_source == "small_llm" else "deterministic",
        )

    def _snapshot_memories(self, namespace: str) -> dict[str, dict[str, Any]]:
        assert self.runtime.store is not None
        return {item.id: item.to_record() for item in self.runtime.store.list_memories(namespace=namespace)}

    def _snapshot_edges(self) -> dict[str, dict[str, Any]]:
        assert self.runtime.store is not None
        return {self._edge_id(edge.to_record()): edge.to_record() for edge in self.runtime.store.list_edges()}

    def _memory_deltas(self, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[MemoryDelta]:
        deltas: list[MemoryDelta] = []
        for memory_id in sorted(set(before) | set(after)):
            if memory_id not in before:
                deltas.append(MemoryDelta(memory_id=memory_id, field="created", old=None, new=after[memory_id], reason="policy execution"))
                continue
            if memory_id not in after:
                deltas.append(MemoryDelta(memory_id=memory_id, field="deleted", old=before[memory_id], new=None, reason="policy execution"))
                continue
            old = before[memory_id]
            new = after[memory_id]
            for field in sorted(set(old) | set(new)):
                if old.get(field) != new.get(field):
                    deltas.append(MemoryDelta(memory_id=memory_id, field=field, old=old.get(field), new=new.get(field), reason="policy execution"))
        return deltas

    def _lifecycle_deltas(self, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[LifecycleDelta]:
        deltas: list[LifecycleDelta] = []
        for memory_id in sorted(set(before) & set(after)):
            old = str(before[memory_id].get("maturity", ""))
            new = str(after[memory_id].get("maturity", ""))
            if old != new:
                deltas.append(self.lifecycle.transition(memory_id, from_state=old, to_state=new, trigger="policy execution", evidence=list(after[memory_id].get("evidence", [])), reason="maturity changed"))
        for memory_id in sorted(set(after) - set(before)):
            new = str(after[memory_id].get("maturity", "captured"))
            from_state = "provisional" if new != "fresh" else "observed"
            if self.lifecycle.validate_transition(from_state, new):
                deltas.append(self.lifecycle.transition(memory_id, from_state=from_state, to_state=new, trigger="validated write", evidence=list(after[memory_id].get("evidence", [])), reason="memory created"))
        return deltas

    def _graph_deltas(self, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[GraphDelta]:
        deltas: list[GraphDelta] = []
        for edge_id in sorted(set(before) | set(after)):
            old = before.get(edge_id)
            new = after.get(edge_id)
            if new is None:
                continue
            old_weight = float(old.get("weight", 0.0)) if old else 0.0
            new_weight = float(new.get("weight", 0.0))
            if old != new:
                deltas.append(
                    GraphDelta(
                        edge_id=edge_id,
                        source_id=str(new.get("source_id", "")),
                        target_id=str(new.get("target_id", "")),
                        relation=str(new.get("relation", "")),
                        old_weight=old_weight,
                        new_weight=new_weight,
                        delta=new_weight - old_weight,
                        eligibility=float(new.get("eligibility_trace", 1.0) or 1.0),
                        confidence=float(new.get("confidence", 0.0) or 0.0),
                        inhibition_penalty=float(new.get("inhibition_score", 0.0) or 0.0),
                        contradiction_penalty=float(new.get("contradiction_penalty", 0.0) or 0.0),
                        provenance=list(new.get("provenance", [])),
                        reason="policy execution plasticity",
                    )
                )
        return deltas

    def _append_stage(
        self,
        transaction_id: str,
        trace_id: str | None,
        phase: str,
        event_type: str,
        policy: MemoryPolicy,
        validated: ValidatedMutation,
        *,
        evidence: list[dict[str, object]] | None = None,
        targets: list[str] | None = None,
        memory_delta: list[dict[str, object]] | None = None,
        graph_delta: list[dict[str, object]] | None = None,
        lifecycle_delta: list[dict[str, object]] | None = None,
        index_delta: list[dict[str, object]] | None = None,
        audit: dict[str, object] | None = None,
    ) -> None:
        self.ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                trace_id=trace_id,
                phase=phase,
                event_type=event_type,
                operation=self._operation(policy),
                proposer=policy.source,
                validator_decision="approved" if validated.approved else "; ".join(step.reason for step in validated.validator_trace if not step.passed),
                evidence=evidence if evidence is not None else self._evidence(policy),
                targets=targets if targets is not None else self._target_ids(policy),
                memory_delta=memory_delta or [],
                graph_delta=graph_delta or [],
                lifecycle_delta=lifecycle_delta or [],
                index_delta=index_delta or [],
                audit={
                    **(audit or {}),
                    "validated_mutation": validated.model_dump(),
                },
            )
        )

    def _rejected_trace(self, policy: MemoryPolicy, context: PolicyExecutionContext, validated: ValidatedMutation) -> MemoryTrace:
        reasons = [step.reason for step in validated.validator_trace if not step.passed]
        return MemoryTrace(
            task_id=context.task,
            query=context.query,
            retrieval_plan=policy.retrieval,
            policy_source=policy.source,
            rejected_reasons=reasons,
            pfc_reason=policy.reason,
            validator_decision="; ".join(reasons),
        )

    def _approved_actions(self, policy: MemoryPolicy) -> list[str]:
        actions: list[str] = []
        if policy.write.operation != "NOOP":
            actions.append(policy.write.operation)
        if policy.forget.operation != "NOOP":
            actions.append(policy.forget.operation)
        if policy.consolidation.enabled:
            actions.append("CONSOLIDATE")
        return actions

    def _operation(self, policy: MemoryPolicy) -> str:
        if policy.write.operation != "NOOP":
            return policy.write.operation
        if policy.forget.operation != "NOOP":
            return policy.forget.operation
        if policy.consolidation.enabled:
            return "CONSOLIDATE"
        return "NOOP"

    def _evidence(self, policy: MemoryPolicy) -> list[dict[str, object]]:
        return [{"event_id": evidence_id, "source": "policy"} for evidence_id in policy.write.evidence_ids]

    def _target_ids(self, policy: MemoryPolicy) -> list[str]:
        return [value for value in [policy.write.target_memory_id, policy.forget.target_memory_id] if value]

    def _policy_dict(self, policy: MemoryPolicy) -> dict[str, object]:
        return {
            "retrieval": asdict(policy.retrieval),
            "write": asdict(policy.write),
            "forget": asdict(policy.forget),
            "consolidation": asdict(policy.consolidation),
            "reason": policy.reason,
            "source": policy.source,
        }

    def _edge_id(self, record: dict[str, Any]) -> str:
        return "|".join(str(record.get(key, "")) for key in ["source_id", "target_id", "relation"])


__all__ = ["PolicyExecutionContext", "PolicyExecutor"]

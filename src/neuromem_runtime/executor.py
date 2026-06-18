from __future__ import annotations

from dataclasses import asdict, dataclass
from sqlite3 import Connection
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
from neuromem_runtime.policy_v2 import GraphDeltaProposal, MemoryPolicyV2, ValidatedMutation, ValidationStep
from neuromem_runtime.semantic_graph import (
    GraphBuildContext,
    GraphDeltaValidator,
    GraphMutationCommitter,
    graph_delta_from_edge,
)
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
    agent_id: str | None = None
    allow_cross_namespace: bool = False


class _PostCommitValidationError(RuntimeError):
    def __init__(self, reason: str, validated: ValidatedMutation) -> None:
        super().__init__(reason)
        self.reason = reason
        self.validated = validated


class PolicyExecutor:
    def __init__(self, runtime: NeuroMemRuntime, ledger: MemoryLedger, validator_stack: ValidatorStack | None = None) -> None:
        self.runtime = runtime
        self.ledger = ledger
        self.validator_stack = validator_stack or ValidatorStack()
        self.lifecycle = LifecycleStateMachine()
        self.graph_validator = GraphDeltaValidator()
        self.graph_committer = GraphMutationCommitter()

    def execute(self, policy: MemoryPolicy | MemoryPolicyV2, context: PolicyExecutionContext) -> MutationExecutionResult:
        assert self.runtime.store is not None
        legacy_policy = self._coerce_policy(policy)
        graph_proposals = self._graph_proposals(policy)
        if isinstance(policy, MemoryPolicyV2):
            unsupported = self._unsupported_v2_reason(policy)
            if unsupported:
                rejected = self._manual_rejection(legacy_policy, context, unsupported)
                return self._record_rejection(legacy_policy, context, rejected)

        before_memories = self._snapshot_memories(context.namespace)
        before_edges = self._snapshot_edges()
        pre_validation = self.validator_stack.validate(legacy_policy, self._validation_context(context))
        if not pre_validation.approved:
            return self._record_rejection(legacy_policy, context, pre_validation)

        transaction_id = f"txn_{uuid4().hex}"
        try:
            with self.runtime.store.transaction() as conn:  # type: ignore[attr-defined]
                self._append_stage(
                    transaction_id,
                    None,
                    "PROPOSED",
                    "proposal_recorded",
                    legacy_policy,
                    pre_validation,
                    context=context,
                    conn=conn,
                    audit={"policy": self._policy_dict(legacy_policy)},
                )
                approved = ValidatedPolicy(
                    policy=legacy_policy,
                    approved=True,
                    approved_actions=self._approved_actions(legacy_policy),
                    rejected_reasons=[],
                )
                self._append_stage(
                    transaction_id,
                    None,
                    "VALIDATED",
                    "validation_approved",
                    legacy_policy,
                    pre_validation,
                    context=context,
                    conn=conn,
                    audit=pre_validation.model_dump(),
                )
                self.runtime._execute_validated_policy(  # noqa: SLF001 - executor wraps core primitive in validation, transaction, and audit.
                    legacy_policy,
                    approved,
                    phase=context.phase,
                    task=context.task,
                    query=context.query,
                    state=context.state,
                    retrieved_memory_ids=context.retrieved_memory_ids,
                )
                proposal_validation = self._validate_graph_proposals(graph_proposals, context)
                if proposal_validation and not proposal_validation.approved:
                    reason = "; ".join(step.reason for step in proposal_validation.validator_trace if not step.passed) or "graph delta validation failed"
                    raise _PostCommitValidationError(reason, proposal_validation)
                explicit_graph_deltas = self._commit_graph_proposals(graph_proposals, context)
                trace = self.runtime.last_trace
                if trace is None:
                    raise RuntimeError("policy execution did not produce a trace")
                trace.query_plan["governed_transaction_id"] = transaction_id
                if graph_proposals:
                    trace.query_plan["graph_delta_proposals"] = [proposal.model_dump(mode="json") for proposal in graph_proposals]
                if explicit_graph_deltas:
                    trace.query_plan["governed_graph_deltas"] = [delta.to_dict() for delta in explicit_graph_deltas]

                after_memories = self._snapshot_memories(context.namespace)
                after_edges = self._snapshot_edges()
                memory_deltas = self._memory_deltas(before_memories, after_memories)
                lifecycle_deltas = self._lifecycle_deltas(before_memories, after_memories)
                graph_deltas = [*self._policy_graph_deltas(trace, before_edges, after_edges), *explicit_graph_deltas]
                affected_ids = sorted({delta.memory_id for delta in memory_deltas})
                index_delta_objs = [IndexDelta(index="sqlite_memory_cards", status="updated", memory_id=memory_id) for memory_id in affected_ids]
                index_deltas = [delta.to_dict() for delta in index_delta_objs]
                post_validation = self.validator_stack.validate(
                    legacy_policy,
                    self._validation_context(context, post_commit=True, affected_memory_ids=affected_ids),
                )
                if not post_validation.approved:
                    reason = "; ".join(step.reason for step in post_validation.validator_trace if not step.passed) or "post-commit assertion failed"
                    raise _PostCommitValidationError(reason, post_validation)

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
                    context=context,
                    conn=conn,
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
                    context=context,
                    conn=conn,
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
                        context=context,
                        conn=conn,
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
                        context=context,
                        conn=conn,
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
                    context=context,
                    conn=conn,
                    evidence=evidence,
                    targets=targets,
                    audit=trace.to_dict(),
                )
                for memory_id in targets:
                    state = after_memories.get(memory_id)
                    if state is not None:
                        self.ledger.record_memory_version(memory_id, transaction_id, state, conn=conn)
                for edge_id, state in after_edges.items():
                    if before_edges.get(edge_id) != state:
                        self.ledger.record_edge_version(edge_id, transaction_id, state, conn=conn)

                return MutationExecutionResult(
                    trace=trace,
                    validated_mutation=post_validation,
                    created_memory_ids=created_ids,
                    updated_memory_ids=updated_ids,
                    deleted_memory_ids=deleted_ids,
                    memory_deltas=memory_deltas,
                    graph_deltas=graph_deltas,
                    lifecycle_deltas=lifecycle_deltas,
                    index_deltas=index_delta_objs,
                )
        except _PostCommitValidationError as exc:
            trace = self._rolled_back_trace(legacy_policy, context, exc.validated, exc.reason)
            self.runtime.last_trace = trace
            self.runtime.traces[trace.trace_id] = trace
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "ROLLED_BACK",
                "transaction_rolled_back",
                legacy_policy,
                exc.validated,
                context=context,
                rollback_reason=exc.reason,
                audit=trace.to_dict(),
            )
            self._append_stage(
                transaction_id,
                trace.trace_id,
                "AUDITED",
                "audit_finalized",
                legacy_policy,
                exc.validated,
                context=context,
                rollback_reason=exc.reason,
                audit=trace.to_dict(),
            )
            return MutationExecutionResult(trace=trace, validated_mutation=exc.validated)

    def _record_rejection(self, policy: MemoryPolicy, context: PolicyExecutionContext, validated: ValidatedMutation) -> MutationExecutionResult:
        transaction_id = f"txn_{uuid4().hex}"
        trace = self._rejected_trace(policy, context, validated)
        self.runtime.last_trace = trace
        self.runtime.traces[trace.trace_id] = trace
        self._append_stage(transaction_id, None, "PROPOSED", "proposal_recorded", policy, validated, context=context, audit={"policy": self._policy_dict(policy)})
        self._append_stage(transaction_id, trace.trace_id, "REJECTED", "validation_rejected", policy, validated, context=context, audit=trace.to_dict())
        self._append_stage(transaction_id, trace.trace_id, "AUDITED", "audit_finalized", policy, validated, context=context, audit=trace.to_dict())
        return MutationExecutionResult(trace=trace, validated_mutation=validated)

    def _coerce_policy(self, policy: MemoryPolicy | MemoryPolicyV2) -> MemoryPolicy:
        if isinstance(policy, MemoryPolicy):
            return policy
        if policy.graph_deltas and not policy.proposed_deltas:
            return MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=policy.target_selector.query or ""),
                write=WritePlan(operation="NOOP"),
                forget=ForgetPlan(operation="NOOP"),
                consolidation=ConsolidationPlan(enabled=False),
                reason=policy.rollback_plan or "governed graph mutation",
                source="small_llm" if policy.proposal_source == "small_llm" else "deterministic",
            )
        evidence_ids = [evidence.event_id for evidence in policy.evidence_chain]
        first_delta = policy.proposed_deltas[0] if policy.proposed_deltas else None
        operation = (first_delta.operation if first_delta is not None else policy.intent).upper()
        if policy.intent == "suppress" or operation == "SUPPRESS":
            operation = "INHIBIT"
        if policy.intent == "delete_request":
            operation = "DELETE_REQUEST"
        target_id = first_delta.target_memory_id if first_delta else (policy.target_selector.memory_ids[0] if policy.target_selector.memory_ids else None)
        reason = first_delta.reason if first_delta else (policy.rollback_plan or policy.intent)
        if operation in {"INHIBIT", "INVALIDATE", "ARCHIVE", "DELETE_REQUEST", "DECAY"}:
            return MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=policy.target_selector.query or ""),
                write=WritePlan(operation="NOOP"),
                forget=ForgetPlan(operation=operation, target_memory_id=target_id, reason=reason),
                consolidation=ConsolidationPlan(enabled=False),
                reason=reason,
                source="small_llm" if policy.proposal_source == "small_llm" else "deterministic",
            )
        value = first_delta.value if first_delta is not None else None
        memory_type = str(value.get("memory_type")) if isinstance(value, dict) and value.get("memory_type") else None
        content = str(value if not isinstance(value, dict) else value.get("content", "")) if value is not None else None
        write_operation = "ADD" if policy.intent == "add" else "UPDATE" if policy.intent in {"update", "supersede"} else "LINK" if policy.intent == "link" else "NOOP"
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=policy.target_selector.query or ""),
            write=WritePlan(
                operation=write_operation,
                memory_type=memory_type,
                content=content,
                target_memory_id=target_id,
                confidence=0.75,
                evidence_ids=evidence_ids,
            ),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=policy.intent == "consolidate"),
            reason=reason,
            source="small_llm" if policy.proposal_source == "small_llm" else "deterministic",
        )

    def _unsupported_v2_reason(self, policy: MemoryPolicyV2) -> str | None:
        if policy.intent == "supersede" and not policy.graph_deltas:
            return "MemoryPolicyV2 supersede requires multi-delta graph or lifecycle execution; legacy one-delta supersede is not supported"
        if len(policy.proposed_deltas) <= 1:
            return None
        operations = {(delta.operation or policy.intent).lower() for delta in policy.proposed_deltas}
        if len(operations) == 1 and operations <= {"link", "update"}:
            return None
        if policy.graph_deltas:
            return None
        return "multi-delta MemoryPolicyV2 transactions are not yet supported by this executor"

    def _graph_proposals(self, policy: MemoryPolicy | MemoryPolicyV2) -> list[GraphDeltaProposal]:
        if isinstance(policy, MemoryPolicyV2):
            return list(policy.graph_deltas)
        return []

    def _graph_context(self, context: PolicyExecutionContext) -> GraphBuildContext:
        assert self.runtime.store is not None
        return GraphBuildContext(
            namespace=context.namespace,
            memories=self.runtime.store.list_memories(namespace=context.namespace),
            selected_memory_ids=context.retrieved_memory_ids or [],
            target_memory_ids=self._target_ids_from_context(context),
            evidence_ids=[],
            outcome=str(context.state.get("status", "unknown")),
            proposer="small_llm",
        )

    def _target_ids_from_context(self, context: PolicyExecutionContext) -> list[str]:
        ids: list[str] = []
        for value in context.state.get("target_memory_ids", []) if isinstance(context.state.get("target_memory_ids", []), list) else []:
            ids.append(str(value))
        return list(dict.fromkeys(ids))

    def _validate_graph_proposals(self, proposals: list[GraphDeltaProposal], context: PolicyExecutionContext) -> ValidatedMutation | None:
        if not proposals:
            return None
        graph_context = self._graph_context(context)
        steps = [self.graph_validator.validate(proposal, context=graph_context, store=self.runtime.store) for proposal in proposals]
        approved = all(step.passed for step in steps)
        return ValidatedMutation(
            approved=approved,
            approved_deltas=[],
            rejected_deltas=[],
            required_human_review=any(not step.passed and "high-risk" in step.reason for step in steps),
            risk_score=0.25 if approved else 0.85,
            validator_trace=steps,
        )

    def _commit_graph_proposals(self, proposals: list[GraphDeltaProposal], context: PolicyExecutionContext) -> list[GraphDelta]:
        assert self.runtime.store is not None
        deltas: list[GraphDelta] = []
        for proposal in proposals:
            before = self._edge_for_proposal(proposal)
            old_weight = before.weight if before is not None else 0.0
            edge = self.graph_committer.commit(proposal, store=self.runtime.store)
            deltas.append(
                GraphDelta(
                    **graph_delta_from_edge(
                        edge,
                        old_weight=old_weight,
                        operation=proposal.operation,
                        proposer=proposal.proposer,
                        reason=proposal.reason or "governed graph delta committed",
                    )
                )
            )
        return deltas

    def _edge_for_proposal(self, proposal: GraphDeltaProposal) -> object | None:
        assert self.runtime.store is not None
        for edge in self.runtime.store.list_edges():
            if edge.relation == proposal.relation and {edge.source_id, edge.target_id} == {proposal.source_memory_id, proposal.target_memory_id}:
                return edge
        return None

    def _manual_rejection(self, policy: MemoryPolicy, context: PolicyExecutionContext, reason: str) -> ValidatedMutation:
        del context
        delta = self.validator_stack.validate(policy, ValidationContext()).rejected_deltas
        return ValidatedMutation(
            approved=False,
            approved_deltas=[],
            rejected_deltas=delta,
            required_human_review=False,
            risk_score=0.8,
            validator_trace=[ValidationStep(name="MemoryPolicyV2SupportValidator", passed=False, reason=reason)],
        )

    def _validation_context(
        self,
        context: PolicyExecutionContext,
        *,
        post_commit: bool = False,
        affected_memory_ids: list[str] | None = None,
    ) -> ValidationContext:
        return ValidationContext(
            store=self.runtime.store,
            ledger=self.ledger,
            phase=context.phase,
            authorize_delete=context.authorize_delete,
            user_id=context.user_id,
            namespace=context.namespace,
            historical=context.historical,
            post_commit=post_commit,
            affected_memory_ids=affected_memory_ids,
            allow_cross_namespace=context.allow_cross_namespace,
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

    def _policy_graph_deltas(self, trace: MemoryTrace, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[GraphDelta]:
        captured = trace.query_plan.get("plasticity_graph_deltas", [])
        if isinstance(captured, list) and captured:
            deltas: list[GraphDelta] = []
            for item in captured:
                if isinstance(item, dict):
                    deltas.append(GraphDelta(**item))
            if deltas:
                return deltas
        return self._graph_deltas(before, after)

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
        context: PolicyExecutionContext,
        conn: Connection | None = None,
        evidence: list[dict[str, object]] | None = None,
        targets: list[str] | None = None,
        memory_delta: list[dict[str, object]] | None = None,
        graph_delta: list[dict[str, object]] | None = None,
        lifecycle_delta: list[dict[str, object]] | None = None,
        index_delta: list[dict[str, object]] | None = None,
        rollback_reason: str | None = None,
        audit: dict[str, object] | None = None,
    ) -> None:
        self.ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                trace_id=trace_id,
                namespace=context.namespace,
                agent_id=context.agent_id,
                user_id=context.user_id,
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
                rollback_reason=rollback_reason,
                audit={
                    **(audit or {}),
                    "validated_mutation": validated.model_dump(),
                },
            ),
            conn=conn,
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

    def _rolled_back_trace(self, policy: MemoryPolicy, context: PolicyExecutionContext, validated: ValidatedMutation, reason: str) -> MemoryTrace:
        trace = self._rejected_trace(policy, context, validated)
        trace.fallback_reason = reason
        trace.validator_decision = reason
        trace.query_plan["rollback_reason"] = reason
        return trace

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

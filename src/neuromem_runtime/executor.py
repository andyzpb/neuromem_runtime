from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuromem.core.policy import MemoryPolicy, MemoryTrace
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.core.validator import PolicyValidator
from neuromem_runtime.deltas import IndexDelta, LifecycleDelta, MemoryDelta
from neuromem_runtime.ledger import LedgerEvent, MemoryLedger


@dataclass(slots=True)
class PolicyExecutionContext:
    phase: str
    task: str
    query: str
    state: dict[str, object]
    authorize_delete: bool = False
    retrieved_memory_ids: list[str] | None = None


class PolicyExecutor:
    def __init__(self, runtime: NeuroMemRuntime, ledger: MemoryLedger, validator: PolicyValidator | None = None) -> None:
        self.runtime = runtime
        self.ledger = ledger
        self.validator = validator or PolicyValidator()

    def execute(self, policy: MemoryPolicy, context: PolicyExecutionContext) -> MemoryTrace:
        assert self.runtime.store is not None
        before_states = self._capture_targets(policy)
        validated = self.validator.validate(
            policy,
            {
                "phase": context.phase,
                "user_explicit_delete_request": context.authorize_delete,
                **context.state,
            },
        )
        self.runtime._execute_validated_policy(  # noqa: SLF001 - core owns mutation semantics until PolicyExecutor fully subsumes it.
            policy,
            validated,
            phase=context.phase,
            task=context.task,
            query=context.query,
            state=context.state,
            retrieved_memory_ids=context.retrieved_memory_ids,
        )
        trace = self.runtime.last_trace
        if trace is None:
            raise RuntimeError("policy execution did not produce a trace")
        after_states = self._capture_targets(policy)
        memory_deltas = self._memory_deltas(before_states, after_states)
        lifecycle_deltas = self._lifecycle_deltas(before_states, after_states)
        index_delta = [IndexDelta(index="sqlite", status="updated" if validated.approved else "unchanged").to_dict()]
        txn = trace.to_transactions()[0]
        ledger_event = LedgerEvent(
            transaction_id=txn.transaction_id,
            trace_id=trace.trace_id,
            phase=txn.phase,
            event_type="policy_committed" if validated.approved else "policy_rejected",
            operation=txn.operation,
            proposer=txn.proposed_by,
            validator_decision=txn.validator_decision,
            evidence=[{"event_id": evidence_id, "source": "policy"} for evidence_id in txn.evidence],
            targets=txn.target_memories,
            lifecycle_delta=[delta.to_dict() for delta in lifecycle_deltas],
            index_delta=index_delta,
            memory_delta=[delta.to_dict() for delta in memory_deltas],
            rollback_reason=txn.rollback,
            audit=txn.audit_trace,
        )
        self.ledger.append(ledger_event)
        for memory_id, state in after_states.items():
            self.ledger.record_memory_version(memory_id, txn.transaction_id, state)
        return trace

    def _capture_targets(self, policy: MemoryPolicy) -> dict[str, dict[str, Any]]:
        assert self.runtime.store is not None
        ids = {
            value
            for value in [
                policy.write.target_memory_id,
                policy.forget.target_memory_id,
            ]
            if value
        }
        states: dict[str, dict[str, Any]] = {}
        for memory_id in ids:
            item = self.runtime.store.get_memory(memory_id)
            if item is not None:
                states[memory_id] = item.to_record()
        return states

    def _memory_deltas(self, before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[MemoryDelta]:
        deltas: list[MemoryDelta] = []
        for memory_id in sorted(set(before) | set(after)):
            old = before.get(memory_id, {})
            new = after.get(memory_id, {})
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
                deltas.append(
                    LifecycleDelta(
                        memory_id=memory_id,
                        from_state=old,
                        to_state=new,
                        trigger="policy execution",
                        evidence=list(after[memory_id].get("evidence", [])),
                        validator="LifecycleTransitionValidator",
                        reason="maturity changed",
                        rollback_state=old,
                    )
                )
        return deltas


__all__ = ["PolicyExecutionContext", "PolicyExecutor"]

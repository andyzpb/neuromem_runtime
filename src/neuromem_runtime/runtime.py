from __future__ import annotations

import json
import tomllib
from pathlib import Path

from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.stores.sqlite_store import SQLiteMemoryStore

from neuromem_runtime.executor import PolicyExecutionContext, PolicyExecutor
from neuromem_runtime.ledger import ExperienceEvent, LedgerEvent, MemoryLedger
from neuromem_runtime.policy_v2 import MemoryPolicyV2
from neuromem_runtime.providers import DeterministicPolicyProvider, PolicyProvider
from neuromem_runtime.retrieval import RetrievalTraceMetadata
from neuromem_runtime.sleep import LedgerReport, ReplayBatch, SleepPlanner, SleepReport
from neuromem_runtime.types import EvidenceBundle, MemoryContext, MemoryEvent, MemoryQuery, RuntimeConfig, event_to_dict
from neuromem_runtime.validators import ValidatorStack


class MemoryRuntime:
    """Product facade over the research NeuroMem runtime."""

    def __init__(self, config: RuntimeConfig, runtime: NeuroMemRuntime, policy_provider: PolicyProvider | None = None) -> None:
        self.config = config
        self._runtime = runtime
        self._validator_stack = ValidatorStack()
        self._policy_provider = policy_provider or DeterministicPolicyProvider()
        self._ledger = MemoryLedger(config.db_path)
        self._executor = PolicyExecutor(runtime, self._ledger, self._validator_stack)
        self._sleep_planner = SleepPlanner()

    @classmethod
    async def local(
        cls,
        namespace: str = "default",
        path: str | Path = ".neuromem",
        agent_id: str = "local-agent",
        mode: str = "lite",
        policy_provider: PolicyProvider | None = None,
    ) -> "MemoryRuntime":
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        traces = root / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        db_path = root / "memory.sqlite3"
        config = RuntimeConfig(
            namespace=namespace,
            path=root,
            db_path=db_path,
            traces_path=traces,
            agent_id=agent_id,
            mode=mode,
            model_policy_enabled=policy_provider is not None,
        )
        _write_config(root / "config.toml", config)
        runtime = NeuroMemRuntime(
            agent_id=agent_id,
            namespace=namespace,
            store=SQLiteMemoryStore(db_path),
            db_path=db_path,
        )
        return cls(config=config, runtime=runtime, policy_provider=policy_provider)

    @classmethod
    async def from_config(cls, path: str | Path = ".neuromem") -> "MemoryRuntime":
        root = Path(path)
        config_path = root / "config.toml"
        if not config_path.exists():
            return await cls.local(path=root)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return await cls.local(
            namespace=str(data.get("namespace", "default")),
            path=root,
            agent_id=str(data.get("agent_id", "local-agent")),
            mode=str(data.get("mode", "lite")),
        )

    @property
    def internal_runtime(self) -> NeuroMemRuntime:
        return self._runtime

    @property
    def ledger(self) -> MemoryLedger:
        return self._ledger

    async def observe(self, event: MemoryEvent | dict[str, object], *, auto_commit: bool = False) -> EvidenceBundle:
        payload = event_to_dict(event)
        content = str(payload.get("content", ""))
        if not content:
            raise ValueError("observe() requires a non-empty content field")
        experience = self._ledger.record_experience(
            ExperienceEvent(
                content=content,
                namespace=self.config.namespace,
                source=str(payload.get("source", "user")),
                metadata={key: value for key, value in payload.items() if key != "content"},
            )
        )
        self._ledger.append(
            LedgerEvent(
                transaction_id=f"txn_{experience.event_id}",
                phase="AUDITED",
                event_type="experience_observed",
                operation="NOOP",
                proposer="system",
                evidence=[experience.to_dict()],
                audit={"experience_event": experience.to_dict(), "auto_commit": auto_commit},
            )
        )
        if not auto_commit:
            return EvidenceBundle(memory_id=None, content=content, evidence=[], event_id=experience.event_id, content_hash=experience.content_hash)
        return await self.observe_and_commit({**payload, "evidence": experience.event_id}, experience=experience)

    async def observe_and_commit(self, event: MemoryEvent | dict[str, object], *, experience: ExperienceEvent | None = None) -> EvidenceBundle:
        payload = event_to_dict(event)
        content = str(payload.get("content", ""))
        if not content:
            raise ValueError("observe_and_commit() requires a non-empty content field")
        if experience is None:
            observed = await self.observe(payload, auto_commit=False)
            if observed.event_id is None:
                raise RuntimeError("observe_and_commit() could not record experience event")
            stored = self._ledger.get_experience(observed.event_id)
            if stored is None:
                raise RuntimeError("observe_and_commit() could not load recorded experience event")
            experience = stored
        memory_type = "episodic"
        if payload.get("type") == "user_preference":
            memory_type = "preference"
        elif payload.get("type") == "rule":
            memory_type = "procedural"
        elif payload.get("type") == "fact":
            memory_type = "semantic"
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=content),
            write=WritePlan(
                operation="ADD",
                memory_type=memory_type,
                content=content,
                confidence=float(payload.get("confidence", 0.85) or 0.85),
                evidence_ids=[experience.event_id],
            ),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="explicit observe_and_commit",
            source="deterministic",
        )
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase="after_step",
                task=str(payload.get("task") or content),
                query=content,
                state={
                    "status": str(payload.get("outcome", "success")),
                    "confidence": float(payload.get("confidence", 0.85) or 0.85),
                    "prediction_error": float(payload.get("prediction_error", 0.0) or 0.0),
                    "future_utility": float(payload.get("future_utility", 0.0) or 0.0),
                },
                namespace=self.config.namespace,
            ),
        )
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        memory_id = result.created_memory_ids[0] if result.created_memory_ids else None
        stored_item = self._runtime.store.get_memory(memory_id) if self._runtime.store is not None and memory_id else None
        return EvidenceBundle(
            memory_id=memory_id,
            content=stored_item.content if stored_item is not None else content,
            memory_type=stored_item.type if stored_item is not None else memory_type,
            evidence=list(stored_item.evidence) if stored_item is not None else [experience.event_id],
            event_id=experience.event_id,
            content_hash=experience.content_hash,
        )

    async def query(
        self,
        query: str | MemoryQuery,
        budget_tokens: int = 800,
        filters: dict[str, object] | None = None,
    ) -> MemoryContext:
        query_obj = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query, budget_tokens=budget_tokens, filters=filters or {})
        before_states = self._memory_states()
        results, trace = self._runtime.retrieve_with_trace(query_obj.query, filters=query_obj.filters, budget_tokens=query_obj.budget_tokens, task_id=query_obj.query)
        after_states = self._memory_states()
        text = "\n".join(f"- [{result.score:.2f}] {result.memory.content}" for result in results)
        trace_id = trace.trace_id
        transactions: list[dict[str, object]] = []
        trace.selected_memory_ids = [result.memory.id for result in results]
        trace.final_context_tokens = len(text.split())
        candidate_details = trace.query_plan.get("candidate_details", {}) if isinstance(trace.query_plan.get("candidate_details", {}), dict) else {}
        retrieval_ledger = trace.query_plan.get("retrieval_ledger", {}) if isinstance(trace.query_plan.get("retrieval_ledger", {}), dict) else {}
        query_plan_v2 = trace.query_plan.get("query_plan_v2", {}) if isinstance(trace.query_plan.get("query_plan_v2", {}), dict) else {}
        source_channels = list(retrieval_ledger.get("channel_candidates", {}).keys()) if isinstance(retrieval_ledger.get("channel_candidates", {}), dict) else list(trace.source_channels)
        trace.query_plan.update(
            {
                "retrieval_metadata": RetrievalTraceMetadata(
                    retrieval_mode=str(retrieval_ledger.get("retrieval_mode") or query_plan_v2.get("mode") or "local_activation"),
                    candidate_sources=[str(item) for item in source_channels],
                    fusion_strategy="rrf+ppr+lite_rerank",
                    rank_before_fusion=[result.memory.id for result in results],
                    rank_after_fusion=[result.memory.id for result in results],
                ).to_dict()
            }
        )
        txn = trace.to_transactions()[0]
        access_deltas = _memory_deltas_for_fields(
            before_states,
            after_states,
            fields={"access_count", "activation_count", "last_accessed_at"},
            reason="retrieval access update",
        )
        trace.query_plan["retrieval_access_deltas"] = access_deltas
        trace_dict = trace.to_dict()
        self._ledger.append(
            LedgerEvent(
                transaction_id=txn.transaction_id,
                trace_id=trace.trace_id,
                phase=txn.phase,
                event_type="memory_retrieved",
                operation="RETRIEVE",
                proposer=txn.proposed_by,
                validator_decision=txn.validator_decision,
                evidence=[{"memory_id": memory_id, "source": "retrieval"} for memory_id in trace.selected_memory_ids],
                targets=trace.selected_memory_ids,
                graph_delta=[{"paths": trace.graph_paths, "scores": trace.diffusion_scores}] if trace.graph_paths or trace.diffusion_scores else [],
                lifecycle_delta=[{"memory_id": memory_id, "reason": reason} for memory_id, reason in trace.suppression_reasons.items()],
                index_delta=[{"index": "sqlite_fts5", "status": "read"}, {"index": "memory_graph", "status": "activated"}],
                memory_delta=access_deltas,
                audit=trace_dict,
            )
        )
        transactions = self._ledger.events_for_trace(trace.trace_id) or [transaction.to_dict() for transaction in trace.to_transactions()]
        self._persist_trace(trace_id, trace_dict)
        return MemoryContext(
            query=query_obj.query,
            text=text,
            selected_memory_ids=[result.memory.id for result in results],
            trace_id=trace_id,
            results=[
                {
                    "memory_id": result.memory.id,
                    "score": result.score,
                    "content": result.memory.content,
                    "type": result.memory.type,
                    "why_retrieved": list(result.why_retrieved),
                    "score_components": dict(candidate_details.get(result.memory.id, {}).get("channel_scores", {})) if isinstance(candidate_details.get(result.memory.id, {}), dict) else {},
                    "graph_paths": candidate_details.get(result.memory.id, {}).get("graph_paths", []) if isinstance(candidate_details.get(result.memory.id, {}), dict) else [],
                    "reranker_score": candidate_details.get(result.memory.id, {}).get("reranker_score", result.score) if isinstance(candidate_details.get(result.memory.id, {}), dict) else result.score,
                    "lifecycle_reason": candidate_details.get(result.memory.id, {}).get("lifecycle_reason", "active") if isinstance(candidate_details.get(result.memory.id, {}), dict) else "active",
                    "provenance_ids": list(result.memory.evidence),
                }
                for result in results
            ],
            transactions=transactions,
        )

    async def propose(self, value: str | dict[str, object]) -> MemoryPolicy:
        data = {"query": value} if isinstance(value, str) else dict(value)
        return self._policy_provider.propose(data)

    async def commit(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        legacy_policy = self._executor._coerce_policy(policy)  # noqa: SLF001 - public facade needs task derivation for both policy generations.
        phase = "after_step" if legacy_policy.write.operation != "NOOP" or legacy_policy.forget.operation != "NOOP" or legacy_policy.consolidation.enabled else "before_step"
        task = legacy_policy.retrieval.query or legacy_policy.write.content or legacy_policy.forget.target_memory_id or "memory mutation"
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase=phase,
                task=task,
                query=legacy_policy.retrieval.query or task,
                state={"status": "success", "confidence": legacy_policy.write.confidence or 0.75},
                authorize_delete=authorize_delete,
                namespace=self.config.namespace,
            ),
        )
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        value = result.trace.to_dict()
        value["mutation_execution_result"] = result.to_dict()
        return value

    async def mutate(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        return await self.commit(policy, authorize_delete=authorize_delete)

    async def sleep(self) -> dict[str, object]:
        memories_before = self._memory_states()
        consolidation_report = self._runtime.neuro_sleep().to_dict()
        memories_after = self._memory_states()
        trace = self._runtime.last_trace
        if trace is not None:
            self._persist_trace(trace.trace_id, trace.to_dict())
        deltas = _memory_deltas_for_fields(memories_before, memories_after, fields=None, reason="governed sleep")
        transaction_id = f"txn_sleep_{trace.trace_id if trace is not None else 'manual'}"
        self._ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                trace_id=trace.trace_id if trace is not None else None,
                phase="COMMITTED",
                event_type="sleep_committed",
                operation="CONSOLIDATE",
                proposer="deterministic",
                validator_decision="approved",
                targets=sorted({str(delta["memory_id"]) for delta in deltas if delta.get("memory_id")}),
                memory_delta=deltas,
                lifecycle_delta=[delta for delta in deltas if delta.get("field") == "maturity"],
                audit={"consolidation_report": consolidation_report},
            )
        )
        sleep_report = SleepReport(
            plan=self._sleep_planner.plan(policy="manual", replay_trace_ids=[trace.trace_id] if trace is not None else []),
            replay=ReplayBatch(trace_ids=[trace.trace_id] if trace is not None else []),
            ledger=LedgerReport(transaction_ids=[transaction_id]),
        ).to_dict()
        return {**consolidation_report, "sleep": sleep_report, "ledger_transaction_id": transaction_id}

    async def forget(
        self,
        memory_id: str,
        action: str = "inhibit",
        reason: str = "user-requested forgetting",
        authorize_delete: bool = False,
    ) -> dict[str, object]:
        normalized = action.lower()
        if normalized == "delete" and not authorize_delete:
            policy = MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=memory_id),
                write=WritePlan(operation="NOOP"),
                forget=ForgetPlan(operation="DELETE_REQUEST", target_memory_id=memory_id, reason=reason),
                consolidation=ConsolidationPlan(enabled=False),
                reason="delete rejected without explicit authorization",
                source="deterministic",
            )
            return await self.commit(policy, authorize_delete=False)
        if normalized not in {"decay", "inhibit", "invalidate", "archive", "compress", "delete"}:
            raise ValueError(f"unsupported forget action: {action}")
        item = self._runtime.store.get_memory(memory_id) if self._runtime.store is not None else None
        if item is None:
            raise ValueError(f"memory not found: {memory_id}")
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=memory_id),
            write=WritePlan(operation="NOOP"),
            forget=ForgetPlan(operation=_policy_forget_operation(normalized), target_memory_id=memory_id, reason=reason),
            consolidation=ConsolidationPlan(enabled=False),
            reason="deterministic product forgetting",
            source="deterministic",
        )
        return await self.commit(policy, authorize_delete=authorize_delete)

    async def replay_trace(self, trace_id: str) -> dict[str, object] | None:
        live = self._runtime.replay_trace(trace_id)
        if live is not None:
            ledger = self._ledger.replay_trace(trace_id)
            if ledger is not None:
                merged = dict(live)
                merged.update({key: value for key, value in ledger.items() if key not in merged or key.endswith("_deltas") or key == "ledger_events"})
                return merged
            return live
        path = self.config.traces_path / f"{trace_id}.json"
        if not path.exists():
            return self._ledger.replay_trace(trace_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        ledger = self._ledger.replay_trace(trace_id)
        if ledger is not None:
            value.update({key: item for key, item in ledger.items() if key not in value or key.endswith("_deltas") or key == "ledger_events"})
        return value

    def _persist_trace(self, trace_id: str, data: dict[str, object]) -> None:
        self.config.traces_path.mkdir(parents=True, exist_ok=True)
        (self.config.traces_path / f"{trace_id}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _memory_states(self) -> dict[str, dict[str, object]]:
        if self._runtime.store is None:
            return {}
        return {item.id: item.to_record() for item in self._runtime.store.list_memories(namespace=self.config.namespace)}


def _policy_forget_operation(action: str) -> str:
    return {
        "decay": "DECAY",
        "inhibit": "INHIBIT",
        "invalidate": "INVALIDATE",
        "archive": "ARCHIVE",
        "delete": "DELETE_REQUEST",
        "compress": "ARCHIVE",
    }[action]


def _write_config(path: Path, config: RuntimeConfig) -> None:
    if path.exists():
        return
    values = config.to_dict()
    lines = []
    for key in ["namespace", "db_path", "traces_path", "agent_id", "mode", "version"]:
        lines.append(f'{key} = "{values[key]}"')
    lines.append(f"model_policy_enabled = {str(config.model_policy_enabled).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["MemoryRuntime"]


def _memory_deltas_for_fields(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    *,
    fields: set[str] | None,
    reason: str,
) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for memory_id in sorted(set(before) | set(after)):
        old = before.get(memory_id, {})
        new = after.get(memory_id, {})
        keys = (set(old) | set(new)) if fields is None else fields
        for field in sorted(keys):
            if old.get(field) != new.get(field):
                deltas.append({"memory_id": memory_id, "field": field, "old": old.get(field), "new": new.get(field), "reason": reason})
    return deltas

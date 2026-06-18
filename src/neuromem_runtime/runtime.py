from __future__ import annotations

import json
import tomllib
from pathlib import Path

from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.core.validator import PolicyValidator
from neuromem.stores.sqlite_store import SQLiteMemoryStore

from neuromem_runtime.executor import PolicyExecutionContext, PolicyExecutor
from neuromem_runtime.ledger import ExperienceEvent, LedgerEvent, MemoryLedger
from neuromem_runtime.providers import DeterministicPolicyProvider, PolicyProvider
from neuromem_runtime.retrieval import RetrievalTraceMetadata
from neuromem_runtime.sleep import LedgerReport, ReplayBatch, SleepPlanner, SleepReport
from neuromem_runtime.types import EvidenceBundle, MemoryContext, MemoryEvent, MemoryQuery, RuntimeConfig, event_to_dict


class MemoryRuntime:
    """Product facade over the research NeuroMem runtime."""

    def __init__(self, config: RuntimeConfig, runtime: NeuroMemRuntime, policy_provider: PolicyProvider | None = None) -> None:
        self.config = config
        self._runtime = runtime
        self._validator = PolicyValidator()
        self._policy_provider = policy_provider or DeterministicPolicyProvider()
        self._ledger = MemoryLedger(config.db_path)
        self._executor = PolicyExecutor(runtime, self._ledger, self._validator)
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

    async def observe(self, event: MemoryEvent | dict[str, object], *, auto_commit: bool = True) -> EvidenceBundle:
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
        if not auto_commit:
            event_txn = f"txn_{experience.event_id}"
            self._ledger.append(
                LedgerEvent(
                    transaction_id=event_txn,
                    phase="AUDITED",
                    event_type="experience_observed",
                    operation="NOOP",
                    proposer="system",
                    evidence=[experience.to_dict()],
                    audit={"experience_event": experience.to_dict()},
                )
            )
            return EvidenceBundle(
                memory_id=None,
                content=content,
                evidence=[],
                event_id=experience.event_id,
                content_hash=experience.content_hash,
            )
        payload.setdefault("evidence", experience.event_id)
        item = self._runtime.observe(payload)
        if item is None:
            return EvidenceBundle(memory_id=None, content=content, evidence=[], event_id=experience.event_id, content_hash=experience.content_hash)
        transaction_id = f"txn_observe_{item.id}"
        self._ledger.record_memory_version(item.id, transaction_id, item.to_record())
        self._ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                phase="COMMITTED",
                event_type="experience_committed",
                operation="ADD",
                proposer="deterministic",
                validator_decision="auto_commit_compat",
                evidence=[experience.to_dict()],
                targets=[item.id],
                memory_delta=[{"memory_id": item.id, "field": "created", "old": None, "new": item.to_record()}],
                index_delta=[{"index": "sqlite", "status": "updated", "memory_id": item.id}],
                audit={"compatibility": "observe(auto_commit=True)", "experience_event": experience.to_dict()},
            )
        )
        return EvidenceBundle(memory_id=item.id, content=item.content, memory_type=item.type, evidence=list(item.evidence), event_id=experience.event_id, content_hash=experience.content_hash)

    async def query(
        self,
        query: str | MemoryQuery,
        budget_tokens: int = 800,
        filters: dict[str, object] | None = None,
    ) -> MemoryContext:
        query_obj = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query, budget_tokens=budget_tokens, filters=filters or {})
        results, trace = self._runtime.retrieve_with_trace(query_obj.query, filters=query_obj.filters, budget_tokens=query_obj.budget_tokens, task_id=query_obj.query)
        text = "\n".join(f"- [{result.score:.2f}] {result.memory.content}" for result in results)
        trace_id = trace.trace_id
        transactions: list[dict[str, object]] = []
        trace.selected_memory_ids = [result.memory.id for result in results]
        trace.final_context_tokens = len(text.split())
        trace.query_plan.update(
            {
                "retrieval_metadata": RetrievalTraceMetadata(
                    rank_before_fusion=[result.memory.id for result in results],
                    rank_after_fusion=[result.memory.id for result in results],
                ).to_dict()
            }
        )
        transactions = [transaction.to_dict() for transaction in trace.to_transactions()]
        self._persist_trace(trace_id, trace.to_dict())
        txn = trace.to_transactions()[0]
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
                index_delta=[{"index": "sqlite", "status": "read"}],
                audit=trace.to_dict(),
            )
        )
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
                }
                for result in results
            ],
            transactions=transactions,
        )

    async def propose(self, value: str | dict[str, object]) -> MemoryPolicy:
        data = {"query": value} if isinstance(value, str) else dict(value)
        return self._policy_provider.propose(data)

    async def commit(self, policy: MemoryPolicy, *, authorize_delete: bool = False) -> dict[str, object]:
        phase = "after_step" if policy.write.operation != "NOOP" or policy.forget.operation != "NOOP" or policy.consolidation.enabled else "before_step"
        task = policy.retrieval.query or policy.write.content or policy.forget.target_memory_id or "memory mutation"
        trace = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase=phase,
                task=task,
                query=policy.retrieval.query or task,
                state={"status": "success", "confidence": policy.write.confidence or 0.75},
                authorize_delete=authorize_delete,
            ),
        )
        self._persist_trace(trace.trace_id, trace.to_dict())
        return trace.to_dict()

    async def mutate(self, policy: MemoryPolicy, *, authorize_delete: bool = False) -> dict[str, object]:
        return await self.commit(policy, authorize_delete=authorize_delete)

    async def sleep(self) -> dict[str, object]:
        consolidation_report = self._runtime.neuro_sleep().to_dict()
        trace = self._runtime.last_trace
        if trace is not None:
            self._persist_trace(trace.trace_id, trace.to_dict())
        sleep_report = SleepReport(
            plan=self._sleep_planner.plan(policy="manual", replay_trace_ids=[trace.trace_id] if trace is not None else []),
            replay=ReplayBatch(trace_ids=[trace.trace_id] if trace is not None else []),
            ledger=LedgerReport(transaction_ids=[f"txn_{trace.trace_id}"] if trace is not None else []),
        ).to_dict()
        return {**consolidation_report, "sleep": sleep_report}

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
            return live
        path = self.config.traces_path / f"{trace_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _persist_trace(self, trace_id: str, data: dict[str, object]) -> None:
        self.config.traces_path.mkdir(parents=True, exist_ok=True)
        (self.config.traces_path / f"{trace_id}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

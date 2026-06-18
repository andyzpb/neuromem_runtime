from __future__ import annotations

import asyncio
import sys

import neuromem_runtime as nmem
from neuromem.core.models import MemoryEdge
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan


def test_public_api_exports() -> None:
    assert nmem.MemoryRuntime
    assert nmem.RuntimeConfig
    assert nmem.MemoryContext
    assert nmem.ExperienceEvent
    assert nmem.MemoryPolicyV2
    assert nmem.MemoryLedger
    assert nmem.RetrievalConfig
    assert nmem.QueryPlanV2
    assert nmem.MemoryCard
    assert nmem.RetrievalCandidate
    assert nmem.ActivationResult
    assert nmem.RetrievalLedgerRecord
    assert "neuromem_runtime.langgraph" not in sys.modules
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "faiss" not in sys.modules


def test_async_local_workflow(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        assert (tmp_path / ".neuromem" / "config.toml").exists()
        assert (tmp_path / ".neuromem" / "memory.sqlite3").exists()
        assert (tmp_path / ".neuromem" / "traces").is_dir()

        bundle = await memory.observe(
            {
                "type": "task_result",
                "content": "Login redirect bug was fixed by changing session refresh order.",
                "task": "Fix login redirect",
                "evidence": "trace-1",
                "keywords": ["login", "session"],
            }
        )
        assert bundle.memory_id is not None
        assert bundle.event_id is not None
        assert bundle.content_hash is not None

        context = await memory.query("session refresh login", budget_tokens=800)
        assert isinstance(context, nmem.MemoryContext)
        assert bundle.memory_id in context.selected_memory_ids
        assert context.trace_id is not None
        assert "session refresh" in context.to_prompt().lower()
        assert context.results[0]["why_retrieved"]
        assert "reranker_score" in context.results[0]
        assert "lifecycle_reason" in context.results[0]

        report = await memory.sleep()
        assert "processed" in report

        trace = await memory.replay_trace(context.trace_id)
        assert trace is not None
        assert "transactions" in trace
        assert trace["selected_memory_ids"] == context.selected_memory_ids

        ledger_events = memory.ledger.events_for_trace(context.trace_id)
        assert ledger_events
        assert ledger_events[-1]["event_hash"]
        assert ledger_events[-1]["audit"]["query_plan"]["retrieval_metadata"]["retrieval_mode"] == "local_activation"
        assert ledger_events[-1]["audit"]["query_plan"]["retrieval_ledger"]["selected_ids"] == context.selected_memory_ids

    asyncio.run(run())


def test_observe_can_record_experience_without_long_term_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe({"content": "Keep as an immutable observation only."}, auto_commit=False)
        assert bundle.event_id is not None
        assert bundle.memory_id is None
        assert memory.internal_runtime.store is not None
        assert memory.internal_runtime.store.list_memories(namespace="demo") == []
        replay = memory.ledger.replay()
        assert replay[-1]["event_type"] == "experience_observed"

    asyncio.run(run())


def test_sync_wrapper(tmp_path) -> None:
    from neuromem_runtime.sync import MemoryRuntime

    memory = MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
    bundle = memory.observe({"content": "User prefers concise answers.", "type": "user_preference", "evidence": "pref-1"})
    context = memory.query("concise answers")
    assert bundle.memory_id in context.selected_memory_ids


def test_delete_requires_authorization(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe({"content": "Temporary secret should be removed.", "evidence": "trace-1"})
        assert bundle.memory_id is not None
        rejected = await memory.forget(bundle.memory_id, action="delete", reason="test delete")
        assert rejected["validator_decision"] == "DELETE_REQUEST requires explicit user authorization"
        assert rejected["rejected_reasons"]
        assert rejected["transactions"][0]["phase"] == "REJECTED"
        stored = memory.internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert stored is not None
        assert stored.maturity != "deleted"

    asyncio.run(run())


def test_forget_decay_applies_once_after_validation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe({"content": "Temporary command can decay.", "evidence": "trace-1"})
        assert bundle.memory_id is not None
        before = memory.internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert before is not None
        result = await memory.forget(bundle.memory_id, action="decay", reason="low utility")
        after = memory.internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert after is not None
        assert round(after.decay_score - before.decay_score, 3) == 0.2
        assert result["transactions"][0]["phase"] == "COMMITTED"
        assert memory.ledger.why_written(bundle.memory_id)

    asyncio.run(run())


def test_commit_policy_records_trace(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query="write rule"),
            write=WritePlan(operation="ADD", memory_type="procedural", content="Always run tests after schema changes.", confidence=0.9, evidence_ids=["trace-1"]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="test commit",
            source="deterministic",
        )
        trace = await memory.commit(policy)
        assert "ADD" in trace["executed_actions"]
        assert trace["transactions"]

    asyncio.run(run())


def test_custom_policy_provider_is_used(tmp_path) -> None:
    class Provider:
        def propose(self, payload):
            return MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=str(payload["task"])),
                write=WritePlan(operation="ADD", memory_type="semantic", content="Provider generated policy.", confidence=0.9, evidence_ids=["provider"]),
                forget=ForgetPlan(operation="NOOP"),
                consolidation=ConsolidationPlan(enabled=False),
                reason="provider test",
                source="small_llm",
            )

    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", policy_provider=Provider())
        policy = await memory.propose({"task": "provider task"})
        assert policy.source == "small_llm"
        assert policy.reason == "provider test"
        trace = await memory.commit(policy)
        assert trace["policy_source"] == "small_llm"

    asyncio.run(run())


def test_edge_upsert_prevents_duplicate_edges(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        assert memory.internal_runtime.store is not None
        edge = MemoryEdge(source_id="a", target_id="b", relation="coactivated_with", weight=0.1, confidence=0.5)
        memory.internal_runtime.store.add_edge(edge)
        memory.internal_runtime.store.add_edge(MemoryEdge(source_id="a", target_id="b", relation="coactivated_with", weight=0.9, confidence=0.8))
        edges = memory.internal_runtime.store.list_edges("a")
        assert len(edges) == 1
        assert edges[0].weight == 0.9

    asyncio.run(run())

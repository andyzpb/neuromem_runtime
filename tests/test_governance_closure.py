from __future__ import annotations

import asyncio

import neuromem_runtime as nmem
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan


def test_missing_evidence_rejects_without_store_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query="unsafe"),
            write=WritePlan(operation="ADD", memory_type="semantic", content="No provenance.", confidence=0.9, evidence_ids=[]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="missing evidence",
        )
        result = await memory.commit(policy)
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert memory.internal_runtime.store is not None
        assert memory.internal_runtime.store.list_memories(namespace="demo") == []
        events = memory.ledger.events_for_trace(result["trace_id"])
        assert [event["event_type"] for event in events] == ["validation_rejected", "audit_finalized"]
        assert memory.ledger.verify_hash_chain()

    asyncio.run(run())


def test_policy_v2_commit_and_replay_trace(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        observed = await memory.observe({"content": "Evidence for v2 write."})
        policy = nmem.MemoryPolicyV2(
            intent="add",
            evidence_chain=[{"event_id": observed.event_id, "source": "test", "content_hash": observed.content_hash}],
            proposed_deltas=[{"operation": "ADD", "value": {"content": "V2 policy created this memory.", "memory_type": "semantic"}, "reason": "test v2"}],
        )
        result = await memory.commit(policy)
        created_id = result["mutation_execution_result"]["created_memory_ids"][0]
        replay = await memory.replay_trace(result["trace_id"])
        assert replay is not None
        assert created_id in replay["ledger_events"][-1]["targets"]
        assert memory.ledger.reconstruct().memories[created_id]["content"] == "V2 policy created this memory."

    asyncio.run(run())


def test_private_memory_acl_rejects_forget_without_authorized_user(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe_and_commit({"content": "Private memory.", "evidence": "seed"})
        assert bundle.memory_id is not None
        item = memory.internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert item is not None
        item.privacy_level = "sensitive"
        item.acl = ["owner"]
        memory.internal_runtime.store.upsert_memory(item)  # type: ignore[union-attr]
        result = await memory.forget(bundle.memory_id, action="archive")
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        after = memory.internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert after is not None
        assert after.maturity != "archived"

    asyncio.run(run())


def test_query_ledgers_retrieval_access_delta(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe_and_commit({"content": "Query access should be ledgered.", "evidence": "seed"})
        context = await memory.query("ledgered access")
        assert bundle.memory_id in context.selected_memory_ids
        replay = await memory.replay_trace(context.trace_id)
        assert replay is not None
        fields = {delta["field"] for delta in replay["memory_deltas"]}
        assert {"access_count", "activation_count", "last_accessed_at"} <= fields

    asyncio.run(run())

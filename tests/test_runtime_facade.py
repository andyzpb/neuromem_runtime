from __future__ import annotations

import asyncio
import sys

import neuromem_runtime as nmem
import pytest
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
    assert nmem.ExecutionDeltaPlan
    assert nmem.MemorySnapshot
    assert nmem.MutationExecutionResult
    assert nmem.WorldviewImpactAssessment
    assert nmem.WorldviewPacket
    assert nmem.EdgeEvidenceEvent
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

        bundle = await memory.observe_and_commit(
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
        assert report["ledger_transaction_id"]
        assert memory.ledger.verify_hash_chain()
        sleep_events = memory.ledger.show_transaction(report["ledger_transaction_id"], namespace="demo")
        assert [event["event_type"] for event in sleep_events] == [
            "sleep_plan_proposed",
            "sleep_validation_approved",
            "replay_batch_selected",
            "consolidation_delta_committed",
            "suppression_delta_committed",
            "compilation_delta_committed",
            "sleep_audit_finalized",
        ]
        assert "memory_deltas" in report["sleep"]
        assert "lifecycle" in report["sleep"]

        trace = await memory.replay_trace(context.trace_id)
        assert trace is not None
        assert "transactions" in trace
        assert trace["selected_memory_ids"] == context.selected_memory_ids

        ledger_events = memory.ledger.events_for_trace(context.trace_id)
        assert ledger_events
        retrieval_event = next(event for event in ledger_events if event["event_type"] == "memory_retrieved")
        assert retrieval_event["event_hash"]
        assert retrieval_event["audit"]["query_plan"]["retrieval_metadata"]["retrieval_mode"] == "local_activation"
        assert retrieval_event["audit"]["query_plan"]["retrieval_ledger"]["selected_ids"] == context.selected_memory_ids

    asyncio.run(run())


def test_observe_can_record_experience_without_long_term_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe({"content": "Keep as an immutable observation only."})
        assert bundle.event_id is not None
        assert bundle.memory_id is None
        assert memory.unsafe_internal_runtime.store is not None
        assert memory.unsafe_internal_runtime.store.list_memories(namespace="demo") == []
        replay = memory.ledger.replay()
        assert replay[-1]["event_type"] == "worldview_impact_assessed"
        assert bundle.impact is not None
        assert bundle.impact["decision"] in {"ledger_only", "propose_frame", "sleep_priority", "append_evidence", "propose_worldview_candidate"}

    asyncio.run(run())


def test_internal_runtime_requires_explicit_unsafe_opt_in(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        with pytest.raises(RuntimeError):
            _ = memory.internal_runtime
        with pytest.raises(RuntimeError):
            _ = memory.unsafe_internal_runtime
        unsafe = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem-unsafe", allow_unsafe_internal=True)
        assert unsafe.unsafe_internal_runtime.store is not None

    asyncio.run(run())


def test_sync_wrapper(tmp_path) -> None:
    from neuromem_runtime.sync import MemoryRuntime

    memory = MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
    bundle = memory.observe_and_commit({"content": "User prefers concise answers.", "type": "user_preference", "evidence": "pref-1"})
    context = memory.query("concise answers")
    assert bundle.memory_id in context.selected_memory_ids
    assert context.worldview is not None
    assert "Worldview Snapshot" in context.to_prompt()


def test_observe_and_route_uses_worldview_impact_gate(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_route(
            {
                "content": "User prefers concise answers.",
                "type": "user_preference",
                "grounded_claims": [
                    {
                        "claim_type": "preference",
                        "canonical_statement": "The user prefers concise answers.",
                        "canonical_slot_key": "user.preference.answer_style",
                        "source_kind": "llm_canonicalization",
                        "commitment_level": "candidate_frame",
                        "confidence": 0.86,
                    }
                ],
            }
        )
        assert first["impact"]["decision"] == "propose_candidate"
        assert first["committed"] is False
        assert first["bundle"]["memory_id"] is None
        assert first.get("candidate_events")

        duplicate = await memory.observe_and_route({"content": "User prefers concise answers.", "type": "user_preference"})
        assert duplicate["impact"]["impact_type"] == "redundant"
        assert duplicate["decision"] == "ledger_only"
        assert duplicate["committed"] is False

        changed = await memory.observe_and_route(
            {
                "content": "Actually, from now on use detailed structured answers instead.",
                "type": "user_preference",
                "grounded_claims": [
                    {
                        "claim_type": "preference",
                        "canonical_statement": "The user prefers detailed structured answers.",
                        "canonical_slot_key": "user.preference.answer_style",
                        "source_kind": "llm_canonicalization",
                        "commitment_level": "candidate_frame",
                        "confidence": 0.88,
                        "target_candidate_ids": [first["grounded_claims"][0]["claim_id"]],
                        "metadata": {"correction": True},
                    }
                ],
            }
        )
        assert changed["impact"]["impact_type"] == "supersession"
        assert changed["decision"] == "append_supersession"

    asyncio.run(run())


def test_query_returns_worldview_packet_and_trace_only_graph_default(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "User prefers concise answers.", "type": "user_preference", "keywords": ["style"]})
        await memory.observe_and_commit({"content": "Run pytest after storage schema changes.", "type": "rule", "keywords": ["tests"]})

        context = await memory.query("how should storage schema work be handled", lens="procedural")
        assert context.worldview is not None
        assert context.worldview["procedures"]
        assert "Worldview Snapshot" in context.to_prompt()
        assert memory.config.retrieval_graph_commit == "trace_only"
        assert memory.unsafe_internal_runtime.store is not None
        assert memory.unsafe_internal_runtime.store.list_edges() == []

    asyncio.run(run())


def test_delete_is_not_supported_by_append_only_runtime(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "Temporary secret should be removed.", "evidence": "trace-1"})
        assert bundle.memory_id is not None
        rejected = await memory.forget(bundle.memory_id, action="delete", reason="test delete")
        assert "destructive memory delete is not supported" in rejected["validator_decision"]
        assert rejected["rejected_reasons"]
        assert rejected["mutation_execution_result"]["validated_mutation"]["approved"] is False
        stored = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert stored is not None
        assert stored.maturity != "deleted"

    asyncio.run(run())


def test_forget_appends_suppression_without_mutating_memory_by_default(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "Temporary command can decay.", "evidence": "trace-1"})
        assert bundle.memory_id is not None
        before = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert before is not None
        result = await memory.forget(bundle.memory_id, action="inhibit", reason="low utility")
        after = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert after is not None
        assert after.maturity == before.maturity
        assert after.inhibition_score == before.inhibition_score
        assert result["append_only"] is True
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is True
        assert bundle.memory_id in memory.ledger.active_suppressed_memory_ids("demo")
        current = await memory.query("temporary command", lens="associative")
        historical = await memory.query("temporary command", lens="historical")
        assert bundle.memory_id not in current.selected_memory_ids
        assert bundle.memory_id in historical.selected_memory_ids

    asyncio.run(run())


def test_commit_policy_records_trace(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        evidence = await memory.observe({"content": "Evidence for test commit."})
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query="write rule"),
            write=WritePlan(operation="ADD", memory_type="procedural", content="Always run tests after schema changes.", confidence=0.9, evidence_ids=[evidence.event_id]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="test commit",
            source="deterministic",
        )
        trace = await memory.commit(policy)
        assert "ADD" in trace["executed_actions"]
        assert trace["transactions"]
        execution = trace["mutation_execution_result"]
        created_id = execution["created_memory_ids"][0]
        events = memory.ledger.why_written(created_id)
        assert any(event["event_type"] == "memory_delta_committed" for event in events)
        assert any(delta["field"] == "created" for event in events for delta in event["memory_delta"])
        assert memory.ledger.reconstruct().memories[created_id]["content"] == "Always run tests after schema changes."

    asyncio.run(run())


def test_custom_policy_provider_is_used(tmp_path) -> None:
    class Provider:
            def propose(self, payload):
                return MemoryPolicy(
                    retrieval=RetrievalPlan(enabled=False, query=str(payload["task"])),
                    write=WritePlan(operation="ADD", memory_type="semantic", content="Provider generated policy.", confidence=0.9, evidence_ids=[payload["evidence_id"]]),
                forget=ForgetPlan(operation="NOOP"),
                consolidation=ConsolidationPlan(enabled=False),
                reason="provider test",
                source="small_llm",
            )

    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", policy_provider=Provider())
        evidence = await memory.observe({"content": "Provider evidence."})
        policy = await memory.propose({"task": "provider task", "evidence_id": evidence.event_id})
        assert policy.source == "small_llm"
        assert policy.reason == "provider test"
        trace = await memory.commit(policy)
        assert trace["policy_source"] == "small_llm"

    asyncio.run(run())


def test_edge_upsert_prevents_duplicate_edges(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        assert memory.unsafe_internal_runtime.store is not None
        edge = MemoryEdge(source_id="a", target_id="b", relation="coactivated_with", weight=0.1, confidence=0.5)
        memory.unsafe_internal_runtime.store.add_edge(edge)
        memory.unsafe_internal_runtime.store.add_edge(MemoryEdge(source_id="a", target_id="b", relation="coactivated_with", weight=0.9, confidence=0.8))
        edges = memory.unsafe_internal_runtime.store.list_edges("a")
        assert len(edges) == 1
        assert edges[0].weight == 0.9

    asyncio.run(run())

from __future__ import annotations

import asyncio

import neuromem_runtime as nmem
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem_runtime.executor import PolicyExecutionContext
from neuromem_runtime.policy_v2 import ValidationStep
from neuromem_runtime.validators import MutationValidator, ValidationContext, ValidatorStack


class FailingPostCommitValidator(MutationValidator):
    name = "FailingPostCommitValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        del policy
        if context.post_commit:
            return ValidationStep(name=self.name, passed=False, reason="forced post-commit failure")
        return ValidationStep(name=self.name, passed=True)


def test_missing_evidence_rejects_without_store_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query="unsafe"),
            write=WritePlan(operation="ADD", memory_type="semantic", content="No provenance.", confidence=0.9, evidence_ids=[]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="missing evidence",
        )
        result = await memory.commit(policy)
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert memory.unsafe_internal_runtime.store is not None
        assert memory.unsafe_internal_runtime.store.list_memories(namespace="demo") == []
        events = memory.ledger.events_for_trace(result["trace_id"])
        assert [event["event_type"] for event in events] == ["validation_rejected", "audit_finalized"]
        assert memory.ledger.verify_hash_chain()

    asyncio.run(run())


def test_policy_v2_commit_and_replay_trace(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
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
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "Private memory.", "evidence": "seed"})
        assert bundle.memory_id is not None
        item = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
        assert item is not None
        item.privacy_level = "sensitive"
        item.acl = ["owner"]
        memory.unsafe_internal_runtime.store.upsert_memory(item)  # type: ignore[union-attr]
        result = await memory.forget(bundle.memory_id, action="archive")
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        after = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[union-attr]
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


def test_post_commit_failure_rolls_back_store_and_audits(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        memory._validator_stack = ValidatorStack(validators=[FailingPostCommitValidator()])  # noqa: SLF001
        memory._executor.validator_stack = memory._validator_stack  # noqa: SLF001
        evidence = await memory.observe({"content": "Evidence for rollback."})
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False),
            write=WritePlan(operation="ADD", memory_type="semantic", content="Rollback target.", confidence=0.9, evidence_ids=[evidence.event_id]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="rollback test",
        )
        result = await memory.commit(policy)
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert result["fallback_reason"] == "forced post-commit failure"
        assert memory.unsafe_internal_runtime.store.list_memories(namespace="demo") == []
        events = memory.ledger.events_for_trace(result["trace_id"], namespace="demo")
        assert [event["event_type"] for event in events] == ["transaction_rolled_back", "audit_finalized"]
        assert events[0]["rollback_reason"] == "forced post-commit failure"
        assert memory.ledger.verify_hash_chain()

    asyncio.run(run())


def test_namespace_scope_blocks_cross_namespace_forget_and_filters_replay(tmp_path) -> None:
    async def run() -> None:
        root = tmp_path / ".neuromem"
        namespace_a = await nmem.MemoryRuntime.local(namespace="a", path=root)
        namespace_b = await nmem.MemoryRuntime.local(namespace="b", path=root, allow_unsafe_internal=True)
        bundle_b = await namespace_b.observe_and_commit({"content": "Namespace B memory."})
        assert bundle_b.memory_id is not None
        result = await namespace_a.forget(bundle_b.memory_id, action="archive")
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert "outside namespace" in result["validator_decision"]
        assert namespace_b.unsafe_internal_runtime.store.get_memory(bundle_b.memory_id).maturity != "archived"  # type: ignore[union-attr]
        replay_a = namespace_a.ledger.replay(namespace="a")
        replay_b = namespace_b.ledger.replay(namespace="b")
        assert all(event["namespace"] == "a" for event in replay_a)
        assert all(event["namespace"] == "b" for event in replay_b)
        assert namespace_a.ledger.reconstruct(namespace="a").memories == {}
        assert bundle_b.memory_id in namespace_b.ledger.reconstruct(namespace="b").memories

    asyncio.run(run())


def test_v2_suppress_inhibits_target_memory(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "Suppressible memory."})
        evidence = await memory.observe({"content": "User asked to suppress it."})
        policy = nmem.MemoryPolicyV2(
            intent="suppress",
            evidence_chain=[{"event_id": evidence.event_id, "source": "test", "content_hash": evidence.content_hash}],
            target_selector={"memory_ids": [bundle.memory_id]},
            proposed_deltas=[{"operation": "suppress", "target_memory_id": bundle.memory_id, "reason": "not useful now"}],
        )
        result = await memory.commit(policy)
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is True
        stored = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[arg-type,union-attr]
        assert stored is not None
        assert stored.maturity == "inhibited"

    asyncio.run(run())


def test_v2_supersede_rejects_without_partial_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "Old current fact."})
        evidence = await memory.observe({"content": "New fact should supersede old."})
        policy = nmem.MemoryPolicyV2(
            intent="supersede",
            evidence_chain=[{"event_id": evidence.event_id, "source": "test", "content_hash": evidence.content_hash}],
            target_selector={"memory_ids": [bundle.memory_id]},
            proposed_deltas=[{"operation": "UPDATE", "target_memory_id": bundle.memory_id, "value": {"content": "New fact."}, "reason": "supersede"}],
        )
        result = await memory.commit(policy)
        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert "supersede requires multi-delta" in result["validator_decision"]
        stored = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id)  # type: ignore[arg-type,union-attr]
        assert stored is not None
        assert stored.content == "Old current fact."
        assert stored.maturity != "obsolete"

    asyncio.run(run())


def test_plasticity_delta_preserves_three_factor_details(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "First related memory."})
        evidence = await memory.observe({"content": "Evidence for second related memory."})
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False),
            write=WritePlan(operation="ADD", memory_type="semantic", content="Second related memory.", confidence=0.9, salience_estimate=0.8, evidence_ids=[evidence.event_id]),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="plasticity test",
        )
        result = memory._executor.execute(  # noqa: SLF001
            policy,
            PolicyExecutionContext(
                phase="after_step",
                task="plasticity",
                query="plasticity",
                state={"status": "success", "confidence": 0.9},
                retrieved_memory_ids=[first.memory_id],
                namespace="demo",
                agent_id=memory.config.agent_id,
            ),
        )
        assert result.graph_deltas
        delta = result.graph_deltas[0].to_dict()
        assert delta["salience"] == 0.8
        assert delta["outcome_reward"] == 1.0
        assert delta["confidence"] == 0.9
        assert first.memory_id in delta["provenance"]

    asyncio.run(run())

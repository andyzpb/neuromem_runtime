from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import time

import pytest

import neuromem_runtime as nmem
from neuromem.core.models import MemoryEdge
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem_runtime.policy_v2 import EvidenceRef
from neuromem_runtime.providers import memory_policy_from_payload
from neuromem_runtime.validators import ValidationContext


class CountingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float((sum(text.encode("utf-8")) + index) % 17) / 17.0 for index in range(8)] for text in texts]


class SlowCountingEmbeddingProvider(CountingEmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        time.sleep(0.1)
        return super().embed(texts)


def test_memory_policy_v2_requires_evidence_for_mutation() -> None:
    with pytest.raises(ValueError):
        nmem.MemoryPolicyV2(intent="add")

    policy = nmem.MemoryPolicyV2(
        intent="add",
        evidence_chain=[EvidenceRef(event_id="evt-1", source="tool_result")],
    )
    assert policy.intent == "add"


def test_policy_payload_normalizer_accepts_wrapped_llm_v2_shape() -> None:
    policy = memory_policy_from_payload(
        {
            "after_step": {
                "policy_id": "pol_001",
                "proposer": "NeuroMem PFC",
                "proposal_source": "after_step",
                "intent": "add",
                "risk_level": "low",
                "evidence_chain": ["evt_1"],
                "target_selector": "none",
                "proposed_deltas": [
                    {
                        "memory_type": "semantic",
                        "content": "The user's name is Andy. The user lives in Denmark.",
                        "salience_estimate": 0.8,
                        "confidence": 0.95,
                        "evidence_ids": ["evt_1"],
                    }
                ],
                "write_gate": {
                    "decision": "commit",
                    "durability_horizon": "long_term",
                    "commitment_level": "high",
                    "basis": "explicit user fact",
                    "signals": ["first_person_introduction"],
                    "rationale": "User explicitly stated stable personal facts.",
                },
            }
        },
        source="small_llm",
    )

    assert isinstance(policy, nmem.MemoryPolicyV2)
    assert policy.proposal_source == "small_llm"
    assert policy.evidence_chain[0].event_id == "evt_1"
    assert policy.write_gate is not None
    assert policy.write_gate.commitment_level == "durable_memory"
    assert policy.write_gate.basis == "current_user_message"
    assert policy.proposed_deltas[0].operation == "ADD"
    assert policy.proposed_deltas[0].value == {
        "content": "The user's name is Andy. The user lives in Denmark.",
        "memory_type": "semantic",
    }


def test_validator_stack_blocks_poisoning_and_unauthorized_delete() -> None:
    stack = nmem.ValidatorStack()
    poisoned = MemoryPolicy(
        retrieval=RetrievalPlan(enabled=False),
        write=WritePlan(operation="ADD", memory_type="semantic", content="Untrusted external memory proposal.", confidence=0.9, evidence_ids=["evt-1"]),
        forget=ForgetPlan(operation="NOOP"),
        consolidation=ConsolidationPlan(enabled=False),
        reason="poison",
        write_gate={"decision": "commit", "risk_score": 0.91},
    )
    result = stack.validate(poisoned, ValidationContext())
    assert not result.approved
    assert any(step.name == "PoisoningRiskValidator" and not step.passed for step in result.validator_trace)

    delete = MemoryPolicy(
        retrieval=RetrievalPlan(enabled=False),
        write=WritePlan(operation="NOOP"),
        forget=ForgetPlan(operation="DELETE_REQUEST", target_memory_id="mem-1", reason="delete"),
        consolidation=ConsolidationPlan(enabled=False),
        reason="delete",
    )
    delete_result = stack.validate(delete, ValidationContext(authorize_delete=False))
    assert not delete_result.approved
    assert delete_result.required_human_review


def test_raw_observation_is_ledger_only_without_grounded_claims(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        route = await memory.observe_and_route({"type": "chat_turn", "content": "你好，我叫 Andy，我住在丹麦。", "source": "user"})
        worldview = await memory.resolve_worldview(lens="audit")

        assert route["decision"] == "ledger_only"
        assert worldview["slots"] == []

    import asyncio

    asyncio.run(run())


def test_structured_grounded_claims_drive_worldview_not_assistant_text(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        route = await memory.observe_and_route(
            {
                "type": "chat_turn",
                "source": "memory_dashboard_chat",
                "content": "你好，我叫 Andy，我住在丹麦。",
                "assistant_answer": "你好 Andy，很高兴认识你。丹麦很适合骑车。",
                "grounded_claims": [
                    {
                        "claim_type": "fact",
                        "canonical_statement": "The user's name is Andy.",
                        "canonical_slot_key": "user.identity.name",
                        "source_kind": "llm_canonicalization",
                        "commitment_level": "candidate_frame",
                        "confidence": 0.82,
                    }
                ],
            }
        )
        worldview = await memory.resolve_worldview(lens="audit")
        statements = [candidate["statement"] for slot in worldview["slots"] for candidate in slot["candidates"]]

        assert route["decision"] == "propose_candidate"
        assert "The user's name is Andy." in statements
        assert all("丹麦很适合骑车" not in statement for statement in statements)
        assert all("你好 Andy" not in statement for statement in statements)

    import asyncio

    asyncio.run(run())


def test_retrieval_and_plasticity_surfaces_are_operational() -> None:
    provider = nmem.DeterministicEmbeddingProvider(dims=4)
    vectors = provider.embed(["alpha", "beta"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4

    metadata = nmem.RetrievalTraceMetadata(rank_before_fusion=["a"], rank_after_fusion=["a"])
    assert metadata.to_dict()["embedding_mode"] == "disabled"
    assert metadata.to_dict()["retrieval_mode"] == "local_activation"
    assert metadata.to_dict()["fusion_strategy"] == "rrf+ppr+lite_rerank"

    edge = MemoryEdge(source_id="a", target_id="b", relation="supports", weight=0.2, confidence=0.8)
    delta = nmem.PlasticityEngine().update_edge(edge, salience=0.8, outcome_reward=1.0, confidence=0.9)
    assert delta.old_weight == 0.2
    assert delta.new_weight > delta.old_weight


def test_sleep_planner_surface() -> None:
    plan = nmem.SleepPlanner().plan(policy="manual", replay_trace_ids=["trace-1"])
    assert plan.to_dict()["replay_trace_ids"] == ["trace-1"]


def test_embedding_cache_avoids_reembedding_memory_cards(tmp_path) -> None:
    async def run() -> None:
        provider = CountingEmbeddingProvider()
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", embedding_provider=provider)
        for index in range(5):
            await memory.observe_and_commit({"content": f"Andy visited Ginza Tokyo memory {index}.", "type": "fact", "keywords": ["ginza", "tokyo"]})

        first = await memory.query("日本旅游地点", top_k=3)
        first_call_count = len(provider.calls)
        assert first.selected_memory_ids
        first_embedding = first.cache["embedding_cache"]
        assert first_embedding["memory_hits"] + first_embedding["memory_misses"] >= 5

        second = await memory.query("另一个日本旅游问题", top_k=3)
        assert second.selected_memory_ids
        del first_call_count
        assert second.cache["embedding_cache"]["memory_hits"] >= 5
        assert second.cache["embedding_cache"]["memory_misses"] == 0

    import asyncio

    asyncio.run(run())


def test_retrieval_cache_uses_semantic_store_version(tmp_path) -> None:
    async def run() -> None:
        provider = CountingEmbeddingProvider()
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", embedding_provider=provider, allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Neutral project note about cache behavior.", "type": "fact", "keywords": ["cache"]})

        version_before = memory._store_version("demo")  # noqa: SLF001 - regression checks cache invalidation semantics.
        first = await memory.query("cache behavior", top_k=3)
        version_after_access = memory._store_version("demo")  # noqa: SLF001
        second = await memory.query("cache behavior", top_k=3)

        assert first.cache["retrieval_cache"] == "miss"
        assert first.cache["namespace"] == "demo"
        assert isinstance(first.cache["semantic_version"], int)
        assert isinstance(first.cache["worldview_version"], int)
        assert "semantic_version_short" not in first.cache
        assert version_after_access == version_before
        assert second.cache["retrieval_cache"] == "hit"

        await memory.observe_and_commit({"content": "Second neutral cache note.", "type": "fact", "keywords": ["cache"]})
        third = await memory.query("cache behavior", top_k=3)

        assert third.cache["retrieval_cache"] == "miss"
        assert third.cache["miss_reason"] in {"invalidated_by_mutation", "semantic_version_changed"}

    import asyncio

    asyncio.run(run())


def test_embedding_cache_batch_reads_and_wal(tmp_path) -> None:
    cache = nmem.EmbeddingCache(tmp_path / "memory.sqlite3")

    cache.set_many(namespace="demo", provider_model="provider:model", vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
    values = cache.get_many(namespace="demo", provider_model="provider:model", text_hashes=["a", "b", "missing"])

    assert values == {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    with sqlite3.connect(tmp_path / "memory.sqlite3") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_retrieval_cache_keeps_old_entries_after_version_bump(tmp_path) -> None:
    async def run() -> None:
        provider = CountingEmbeddingProvider()
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", embedding_provider=provider, allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Cache version note.", "type": "fact", "keywords": ["cache"]})
        first = await memory.query("cache version", top_k=3)
        stats_after_first = memory.performance_stats()["retrieval_cache"]
        await memory.observe_and_commit({"content": "Cache version note two.", "type": "fact", "keywords": ["cache"]})
        second = await memory.query("cache version", top_k=3)
        stats_after_second = memory.performance_stats()["retrieval_cache"]

        assert first.cache["retrieval_cache"] == "miss"
        assert second.cache["retrieval_cache"] == "miss"
        assert stats_after_second["entries"] >= stats_after_first["entries"]
        assert memory.performance_stats()["semantic_versions"]["demo"] >= 2
        assert memory.performance_stats()["cache_versions"]["demo"]["semantic_version"] >= 2
        assert "embedding_batcher" in memory.performance_stats()

    import asyncio

    asyncio.run(run())


def test_concurrent_identical_retrieval_misses_share_singleflight(tmp_path) -> None:
    async def setup():
        provider = SlowCountingEmbeddingProvider()
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", embedding_provider=provider, allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Concurrent cache behavior note.", "type": "fact", "keywords": ["cache", "concurrent"]})
        return memory

    import asyncio

    memory = asyncio.run(setup())

    def query_once():
        return asyncio.run(memory.query("concurrent cache behavior", top_k=3)).cache["retrieval_cache"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: query_once(), range(2)))

    stats = memory.performance_stats()["retrieval_singleflight"]
    assert "miss" in results
    assert "hit" in results
    assert stats["join_count"] >= 1


def test_write_gate_validator_rejects_inconsistent_policies(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        event = await memory.observe({"content": "A neutral reusable implementation detail was stated.", "source": "test"})
        evidence = [EvidenceRef(event_id=event.event_id, source="current_user_message")]

        commit_without_delta = nmem.MemoryPolicyV2(
            intent="noop",
            evidence_chain=evidence,
            safety_annotations={
                "write_gate": {
                    "decision": "commit",
                    "durability_horizon": "long_term",
                    "commitment_level": "durable_memory",
                    "basis": "current_user_message",
                    "signals": ["future_utility"],
                    "rationale": "Useful durable memory should be remembered.",
                }
            },
        )
        rejected_no_delta = await memory.commit(commit_without_delta)
        assert rejected_no_delta["mutation_execution_result"]["validated_mutation"]["approved"] is False

        noop_with_delta = nmem.MemoryPolicyV2(
            intent="add",
            evidence_chain=evidence,
            proposed_deltas=[
                {
                    "operation": "ADD",
                    "value": {"content": "Neutral reusable implementation detail.", "memory_type": "semantic"},
                    "reason": "canonical content",
                }
            ],
            safety_annotations={
                "write_gate": {
                    "decision": "noop",
                    "durability_horizon": "none",
                    "commitment_level": "raw_experience",
                    "basis": "current_user_message",
                    "signals": ["uncertainty"],
                    "rationale": "Do not commit this yet.",
                }
            },
        )
        rejected_delta = await memory.commit(noop_with_delta)
        assert rejected_delta["mutation_execution_result"]["validated_mutation"]["approved"] is False

        consistent = nmem.MemoryPolicyV2(
            intent="add",
            evidence_chain=evidence,
            grounded_claims=[
                {
                    "claim_type": "fact",
                    "canonical_statement": "Neutral reusable implementation detail.",
                    "canonical_slot_key": "implementation.detail.reusable",
                    "truth_source_event_ids": [event.event_id],
                    "evidence_ids": [event.event_id],
                    "source_kind": "llm_canonicalization",
                    "metadata": {"source_channel": "current_user_message"},
                }
            ],
            proposed_deltas=[
                {
                    "operation": "ADD",
                    "value": {"content": "Neutral reusable implementation detail.", "memory_type": "semantic"},
                    "reason": "canonical content",
                }
            ],
            safety_annotations={
                "write_gate": {
                    "decision": "commit",
                    "durability_horizon": "long_term",
                    "commitment_level": "durable_memory",
                    "basis": "current_user_message",
                    "signals": ["future_utility", "stability"],
                    "rationale": "The statement is stable and reusable.",
                }
            },
        )
        accepted = await memory.commit(consistent)
        assert accepted["mutation_execution_result"]["validated_mutation"]["approved"] is True
        assert accepted["mutation_execution_result"]["created_memory_ids"]

    import asyncio

    asyncio.run(run())


def test_multilingual_fact_queries_prioritize_entity_facts_over_style(tmp_path) -> None:
    async def run() -> None:
        provider = nmem.DeterministicEmbeddingProvider(dims=12)
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", embedding_provider=provider)
        ginza = await memory.observe_and_commit({"content": "Andy went to Ginza, Tokyo five years ago with his friend Ben to eat oden.", "type": "fact", "keywords": ["Ginza", "Tokyo", "Japan", "Ben", "travel"]})
        sushi = await memory.observe_and_commit({"content": "Andy's friend Ben prefers sushi without wasabi.", "type": "fact", "keywords": ["Ben", "sushi", "wasabi"]})
        await memory.observe_and_commit({"content": "Andy prefers the assistant to roleplay a catgirl and end short replies with meow.", "type": "user_preference", "keywords": ["style"]})

        travel = await memory.query("我们是去日本哪里旅游来着", top_k=3)
        ben = await memory.query("Ben 有什么癖好，关于寿司的", top_k=3)

        assert ginza.memory_id in travel.selected_memory_ids
        assert sushi.memory_id in ben.selected_memory_ids

    import asyncio

    asyncio.run(run())

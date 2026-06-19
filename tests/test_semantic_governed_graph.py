from __future__ import annotations

import asyncio

import neuromem_runtime as nmem
from neuromem.core.models import MemoryItem
from neuromem.retrieval.activation import build_memory_card
from neuromem_runtime.policy_v2 import EvidenceRef


class FakeMultilingualEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in ["login", "登录", "auth", "redirect", "跳转", "session", "会话"]):
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif any(term in lowered for term in ["billing", "invoice", "账单"]):
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return vectors


class TopicEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in ["espresso", "cafe", "coffee"]):
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif any(term in lowered for term in ["bicycle", "train", "commute"]):
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return vectors


class FakeHyDE:
    def generate(self, query: str, *, namespace: str = "default") -> str | None:
        del query, namespace
        return "Previous fix involved session refresh order and redirect validation."


class FakeRewrite:
    def rewrite(self, query: str, *, namespace: str = "default") -> list[str]:
        del query, namespace
        return ["auth session redirect fix", "登录 跳转 会话 修复"]


def _truth_claim(event_id: str, statement: str, slot_key: str = "test.graph.claim") -> dict[str, object]:
    return {
        "claim_type": "fact",
        "canonical_statement": statement,
        "canonical_slot_key": slot_key,
        "truth_source_event_ids": [event_id],
        "evidence_ids": [event_id],
        "source_kind": "llm_canonicalization",
        "metadata": {"source_channel": "current_user_message"},
    }


def test_semantic_recall_cross_lingual_and_hyde_trace(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(
            namespace="demo",
            path=tmp_path / ".neuromem",
            embedding_provider=FakeMultilingualEmbedding(),
            query_rewrite_provider=FakeRewrite(),
            hyde_provider=FakeHyDE(),
        )
        auth = await memory.observe_and_commit({"content": "Login redirect bug was fixed by changing session refresh order.", "keywords": ["login", "redirect", "session"]})
        await memory.observe_and_commit({"content": "Billing invoice export uses CSV.", "keywords": ["billing", "invoice"]})

        context = await memory.query("登录跳转问题之前怎么修的？")

        assert auth.memory_id in context.selected_memory_ids
        replay = await memory.replay_trace(context.trace_id)
        assert replay is not None
        ledger = replay["retrieval_ledger"]
        assert "dense" in ledger["channel_candidates"]
        assert "hyde" in ledger["channel_candidates"] or "rewrite" in ledger["channel_candidates"]
        assert replay["query_plan"]["retrieval_metadata"]["embedding_mode"] == "enabled"

    asyncio.run(run())


def test_graph_candidate_generator_outputs_candidates_not_edges() -> None:
    left = MemoryItem(id="mem-a", namespace="demo", content="Login redirect fixed by session refresh.", keywords=["login", "redirect"], evidence=["evt-1"])
    right = MemoryItem(id="mem-b", namespace="demo", content="Always refresh session before redirect validation.", keywords=["session", "redirect"], evidence=["evt-1"])
    context = nmem.GraphBuildContext(namespace="demo", memories=[left, right], selected_memory_ids=[left.id], target_memory_ids=[right.id], evidence_ids=["evt-1"], outcome="success")

    candidates = nmem.GraphCandidateGenerator().generate(context)

    assert candidates
    assert candidates[0].source_memory_id in {left.id, right.id}
    assert candidates[0].target_memory_id in {left.id, right.id}
    assert candidates[0].evidence_ids


def test_graph_candidate_generator_does_not_fully_connect_bulk_import_noise() -> None:
    memories = [
        MemoryItem(id=f"mem-{index}", namespace="demo", content=content, evidence=[f"evt-{index}"])
        for index, content in enumerate(
            [
                "espresso beans roast profile",
                "bicycle derailleur tuning",
                "coffee grinder burr size",
                "train timetable platform change",
                "garden soil moisture check",
                "route planner commute shortcut",
                "menu layout cafe breakfast",
                "repair kit bike tire patch",
            ]
        )
    ]
    context = nmem.GraphBuildContext(
        namespace="demo",
        memories=memories,
        selected_memory_ids=[memory.id for memory in memories],
        target_memory_ids=[memory.id for memory in memories],
        evidence_ids=[f"evt-{index}" for index in range(8)],
        mutation_trace={"operation": "dashboard_bulk_import"},
        outcome="success",
    )

    candidates = nmem.GraphCandidateGenerator().generate(context)

    assert len(candidates) < 8


def test_graph_candidate_generator_uses_embedding_similarity_when_available() -> None:
    left = MemoryItem(id="mem-left", namespace="demo", content="Coffee shop notes about espresso beans and cafe flow.", evidence=["evt-1"])
    right = MemoryItem(id="mem-right", namespace="demo", content="Cafe workflow for espresso service and tasting.", evidence=["evt-2"])
    noise = MemoryItem(id="mem-noise", namespace="demo", content="Bicycle repair checklist for commute setup.", evidence=["evt-3"])
    context = nmem.GraphBuildContext(
        namespace="demo",
        memories=[left, right, noise],
        selected_memory_ids=[left.id],
        target_memory_ids=[right.id],
        evidence_ids=["evt-1", "evt-2", "evt-3"],
        outcome="success",
        embedding_provider=TopicEmbedding(),
    )

    candidates = nmem.GraphCandidateGenerator().generate(context)

    assert any(
        "embedding_similarity" in candidate.candidate_sources
        and {candidate.source_memory_id, candidate.target_memory_id} == {left.id, right.id}
        for candidate in candidates
    )
    assert all(
        {candidate.source_memory_id, candidate.target_memory_id} != {left.id, noise.id}
        for candidate in candidates
    )


def test_graph_delta_policy_commits_edge_and_rejects_missing_evidence(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "Successful login redirect fix used session refresh.", "keywords": ["login", "redirect"]})
        second = await memory.observe_and_commit({"content": "Rule: refresh session before redirect validation.", "type": "rule", "keywords": ["session", "redirect"]})
        assert first.memory_id is not None and second.memory_id is not None

        evidence = await memory.observe({"content": "Trace shows the episode supports the rule."})
        policy = nmem.MemoryPolicyV2(
            intent="link",
            evidence_chain=[EvidenceRef(event_id=evidence.event_id, source="test")],
            grounded_claims=[_truth_claim(evidence.event_id, "Trace shows the episode supports the rule.")],
            graph_deltas=[
                {
                    "operation": "add_edge",
                    "source_memory_id": first.memory_id,
                    "target_memory_id": second.memory_id,
                    "relation": "supports",
                    "weight": 0.35,
                    "confidence": 0.7,
                    "evidence_ids": [evidence.event_id],
                    "candidate_sources": ["same_evidence_chain"],
                    "reason": "episode supports procedural rule",
                }
            ],
        )
        committed = await memory.commit(policy)
        assert committed["mutation_execution_result"]["validated_mutation"]["approved"] is True
        edges = memory.unsafe_internal_runtime.store.list_edges()  # type: ignore[union-attr]
        assert any(edge.relation == "supports" and {edge.source_id, edge.target_id} == {first.memory_id, second.memory_id} for edge in edges)

        rejected = await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                grounded_claims=[_truth_claim(evidence.event_id, "Trace shows the episode supports the rule.")],
                graph_deltas=[
                    {
                        "operation": "add_edge",
                        "source_memory_id": first.memory_id,
                        "target_memory_id": second.memory_id,
                        "relation": "causes",
                        "weight": 0.9,
                        "confidence": 0.9,
                        "reason": "unsafe causal claim",
                    }
                ],
            )
        )
        assert rejected["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert "evidence ids" in rejected["validator_decision"]

    asyncio.run(run())


def test_automatic_retrieval_graph_builder_uses_bounded_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NEUROMEM_RETRIEVAL_GRAPH_COMMIT", "sync")

    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "Login redirect bug fixed by session refresh.", "keywords": ["login", "redirect"], "evidence": "evt-a"})
        second = await memory.observe_and_commit({"content": "Session refresh depends on redirect validation.", "keywords": ["session", "redirect"], "evidence": "evt-b"})

        context = await memory.query("login redirect session")
        replay = await memory.replay_trace(context.trace_id)
        assert replay is not None
        assert "semantic_graph_builder" in replay["query_plan"]
        store = memory.unsafe_internal_runtime.store
        edges = store.list_edges()  # type: ignore[union-attr]
        assert any({edge.source_id, edge.target_id} == {first.memory_id, second.memory_id} for edge in edges)
        associative_edges = store.list_associative_edges(namespace="demo")  # type: ignore[union-attr]
        assert any({edge.source_id, edge.target_id} == {first.memory_id, second.memory_id} for edge in associative_edges)

    asyncio.run(run())


def test_default_retrieval_graph_commit_is_trace_only(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NEUROMEM_RETRIEVAL_GRAPH_COMMIT", raising=False)

    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Login redirect bug fixed by session refresh.", "keywords": ["login", "redirect"], "evidence": "evt-a"})
        await memory.observe_and_commit({"content": "Session refresh depends on redirect validation.", "keywords": ["session", "redirect"], "evidence": "evt-b"})

        context = await memory.query("login redirect session")
        replay = await memory.replay_trace(context.trace_id)

        assert replay is not None
        builder = replay["query_plan"]["semantic_graph_builder"]
        assert builder["status"] == "trace_only"

    asyncio.run(run())


def test_sleep_graph_compiler_populates_graph_report(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Login redirect fixed by refreshing session before validation.", "keywords": ["auth", "redirect"], "evidence": "evt-1"})
        await memory.observe_and_commit({"content": "OAuth redirect fixed by refreshing session before validation.", "keywords": ["auth", "redirect"], "evidence": "evt-2"})
        await memory.query("auth redirect session")

        report = await memory.sleep()

        assert report["sleep"]["graph"]["proposed_deltas"]
        assert report["sleep"]["graph"]["approved_deltas"]
        assert report["sleep"]["graph"]["compiled_nodes"]

    asyncio.run(run())


def test_hyde_text_never_becomes_evidence_or_memory(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(
            namespace="demo",
            path=tmp_path / ".neuromem",
            embedding_provider=FakeMultilingualEmbedding(),
            hyde_provider=FakeHyDE(),
            allow_unsafe_internal=True,
        )
        await memory.observe_and_commit({"content": "Login redirect bug was fixed by changing session refresh order.", "keywords": ["login", "redirect"]})
        await memory.query("中文登录问题")

        memories = memory.unsafe_internal_runtime.store.list_memories(namespace="demo")  # type: ignore[union-attr]
        assert all("Previous fix involved session refresh order" not in item.content for item in memories)
        assert all("Previous fix involved session refresh order" not in evidence for item in memories for evidence in item.evidence)

    asyncio.run(run())

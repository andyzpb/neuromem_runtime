from __future__ import annotations

import asyncio
import sqlite3

import neuromem_runtime as nmem


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


def test_split_graph_storage_is_new_baseline(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "Login redirect fix used session refresh.", "keywords": ["login", "redirect"], "evidence": "evt-a"})
        second = await memory.observe_and_commit({"content": "Session refresh was retrieved with redirect validation.", "keywords": ["session", "redirect"], "evidence": "evt-b"})
        evidence = await memory.observe({"content": "Both memories were co-retrieved during the same query."})
        assert first.memory_id and second.memory_id and evidence.event_id

        result = await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                evidence_chain=[{"event_id": evidence.event_id, "source": "test"}],
                grounded_claims=[_truth_claim(evidence.event_id, "Both memories were co-retrieved during the same query.")],
                associative_deltas=[
                    {
                        "source_memory_id": first.memory_id,
                        "target_memory_id": second.memory_id,
                        "relation": "retrieved_with",
                        "weight": 0.18,
                        "confidence": 0.52,
                        "evidence_ids": [evidence.event_id],
                    }
                ],
            )
        )

        assert result["mutation_execution_result"]["validated_mutation"]["approved"] is True
        store = memory.unsafe_internal_runtime.store
        assert store is not None
        assert len(store.list_associative_edges(namespace="demo")) == 1
        assert store.list_logic_edges(namespace="demo") == []
        with sqlite3.connect(memory.config.db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"associative_edges", "logic_nodes", "logic_edges"} <= tables
        assert "edges" not in tables

    asyncio.run(run())


def test_logic_edge_requires_existing_frame_endpoints(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "Successful login redirect fix used session refresh.", "keywords": ["login", "redirect"]})
        second = await memory.observe_and_commit({"content": "Rule: refresh session before redirect validation.", "type": "rule", "keywords": ["session", "redirect"]})
        evidence = await memory.observe({"content": "Trace proves the episode supports the rule."})
        assert first.memory_id and second.memory_id and evidence.event_id

        rejected = await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                evidence_chain=[{"event_id": evidence.event_id, "source": "test"}],
                grounded_claims=[_truth_claim(evidence.event_id, "Trace proves the episode supports the rule.")],
                logic_deltas=[
                    {
                        "source_frame_id": "missing-source",
                        "target_frame_id": "missing-target",
                        "relation": "supports",
                        "proof_obligation": "must fail without frame endpoints",
                        "evidence_ids": [evidence.event_id],
                    }
                ],
            )
        )
        assert rejected["mutation_execution_result"]["validated_mutation"]["approved"] is False
        assert "existing frame endpoints" in rejected["validator_decision"]
        store = memory.unsafe_internal_runtime.store
        assert store is not None
        assert store.list_logic_edges(namespace="demo") == []

        source_frame = {
            "frame_id": "frame-source",
            "frame_type": "episode",
            "content": "Successful login redirect fix used session refresh.",
            "source_memory_ids": [first.memory_id],
            "evidence_ids": [evidence.event_id],
            "confidence": 0.72,
            "commitment_level": "validated_logic",
            "lifecycle_state": "validated",
        }
        target_frame = {
            "frame_id": "frame-target",
            "frame_type": "procedure",
            "content": "Refresh session before redirect validation.",
            "source_memory_ids": [second.memory_id],
            "evidence_ids": [evidence.event_id],
            "confidence": 0.74,
            "commitment_level": "validated_logic",
            "lifecycle_state": "validated",
        }
        approved = await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                evidence_chain=[{"event_id": evidence.event_id, "source": "test"}],
                grounded_claims=[_truth_claim(evidence.event_id, "Trace proves the episode supports the rule.")],
                frame_deltas=[source_frame, target_frame],
                logic_deltas=[
                    {
                        "source_frame_id": "frame-source",
                        "target_frame_id": "frame-target",
                        "source_memory_id": first.memory_id,
                        "target_memory_id": second.memory_id,
                        "relation": "supports",
                        "proof_obligation": "episode provides successful evidence for the rule",
                        "evidence_ids": [evidence.event_id],
                        "confidence": 0.66,
                    }
                ],
            )
        )

        assert approved["mutation_execution_result"]["validated_mutation"]["approved"] is True
        assert len(store.list_logic_nodes(namespace="demo")) == 2
        assert len(store.list_logic_edges(namespace="demo")) == 1

    asyncio.run(run())


def test_candidate_frame_does_not_enter_logical_lens(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem")
        bundle = await memory.observe_and_commit({"content": "Current test command is python -m pytest -q.", "type": "fact", "keywords": ["pytest", "command"]})
        evidence = await memory.observe({"content": "Candidate extraction evidence."})
        assert bundle.memory_id and evidence.event_id

        await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                evidence_chain=[{"event_id": evidence.event_id, "source": "test"}],
                grounded_claims=[_truth_claim(evidence.event_id, "Candidate extraction evidence.")],
                frame_deltas=[
                    {
                        "frame_id": "candidate-test-command",
                        "frame_type": "fact",
                        "content": "Current test command is python -m pytest -q.",
                        "source_memory_ids": [bundle.memory_id],
                        "evidence_ids": [evidence.event_id],
                        "confidence": 0.7,
                        "commitment_level": "candidate_frame",
                        "lifecycle_state": "candidate",
                    }
                ],
            )
        )

        candidate_context = await memory.query("current test command", lens="logical")
        assert bundle.memory_id not in candidate_context.selected_memory_ids

        await memory.commit(
            nmem.MemoryPolicyV2(
                intent="link",
                evidence_chain=[{"event_id": evidence.event_id, "source": "test"}],
                grounded_claims=[_truth_claim(evidence.event_id, "Candidate extraction evidence.")],
                frame_deltas=[
                    {
                        "frame_id": "validated-test-command",
                        "frame_type": "fact",
                        "content": "Current test command is python -m pytest -q.",
                        "source_memory_ids": [bundle.memory_id],
                        "evidence_ids": [evidence.event_id],
                        "confidence": 0.82,
                        "commitment_level": "validated_logic",
                        "lifecycle_state": "validated",
                    }
                ],
            )
        )
        validated_context = await memory.query("current test command", lens="logical")
        assert bundle.memory_id in validated_context.selected_memory_ids

    asyncio.run(run())


def test_sleep_crystallization_frames_are_reported_and_reconstructed(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        await memory.observe_and_commit({"content": "Login redirect fixed by refreshing session before validation.", "keywords": ["auth", "redirect"], "evidence": "evt-1"})
        await memory.observe_and_commit({"content": "OAuth redirect fixed by refreshing session before validation.", "keywords": ["auth", "redirect"], "evidence": "evt-2"})
        await memory.query("auth redirect session")

        report = await memory.sleep()

        graph = report["sleep"]["graph"]
        assert graph["frame_candidates"]
        assert graph["validated_frames"]
        assert graph["logic_promotions"]
        assert graph["compiled_schemas"]
        snapshot = memory.ledger.reconstruct(namespace="demo")
        assert snapshot.frames
        assert any(frame["commitment_level"] == "compiled_schema" for frame in snapshot.frames.values())

    asyncio.run(run())

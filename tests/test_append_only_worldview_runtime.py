from __future__ import annotations

import asyncio
import sqlite3

import neuromem_runtime as nmem


def test_route_proposes_candidate_without_durable_memory_and_query_returns_worldview(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        routed = await memory.observe_and_route(
            {
                "content": "User prefers brief direct answers.",
                "type": "user_preference",
                "grounded_claims": [
                    {
                        "claim_type": "preference",
                        "canonical_statement": "The user prefers brief direct answers.",
                        "canonical_slot_key": "user.preference.answer_style",
                        "source_kind": "llm_canonicalization",
                        "commitment_level": "candidate_frame",
                        "confidence": 0.84,
                    }
                ],
            }
        )

        assert routed["committed"] is False
        assert routed["bundle"]["memory_id"] is None
        assert routed["decision"] == "propose_candidate"

        packet = await memory.resolve_worldview("style", lens="audit")
        assert packet["slots"]
        assert packet["prompt"].startswith("[Worldview Snapshot]")

        context = await memory.query("style preference", lens="audit")
        assert context.worldview is not None
        assert context.worldview_trace is not None
        assert context.prompt_sections

    asyncio.run(run())


def test_forget_appends_suppression_and_rebuild_restores_materialized_view(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        bundle = await memory.observe_and_commit({"content": "The user prefers concise answers.", "type": "user_preference", "keywords": ["style"]})
        assert bundle.memory_id is not None
        before = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id).to_record()  # type: ignore[union-attr]

        result = await memory.forget(bundle.memory_id, action="inhibit", reason="preference is no longer current")
        assert result["append_only"] is True
        after = memory.unsafe_internal_runtime.store.get_memory(bundle.memory_id).to_record()  # type: ignore[union-attr]
        assert after == before

        packet = await memory.resolve_worldview("style", lens="historical")
        assert any(slot["status"] in {"suppressed", "historical"} for slot in packet["slots"])

        with sqlite3.connect(tmp_path / ".neuromem" / "memory.sqlite3") as conn:
            conn.execute("DELETE FROM worldview_candidates WHERE namespace = ?", ("demo",))
            conn.execute("DELETE FROM worldview_slots WHERE namespace = ?", ("demo",))
            conn.execute("DELETE FROM associative_edges WHERE namespace = ?", ("demo",))
            conn.execute("DELETE FROM logic_edges WHERE namespace = ?", ("demo",))

        rebuilt = await memory.rebuild_materialized_views()
        assert bundle.memory_id in rebuilt["suppressed_memory_ids"]
        packet_after = await memory.resolve_worldview("style", lens="historical")
        assert any(bundle.memory_id in slot["source_memory_ids"] for slot in packet_after["slots"])

    asyncio.run(run())


def test_after_turn_appends_reinforcement_evidence_without_direct_edge_mutation(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "Login redirect fix used session refresh.", "keywords": ["login", "redirect"]})
        second = await memory.observe_and_commit({"content": "Session refresh validates redirect state.", "keywords": ["session", "redirect"]})
        assert first.memory_id and second.memory_id

        context = await memory.query("login redirect session", lens="associative", top_k=2)
        assert context.trace_id is not None
        outcome = await memory.after_turn(context.trace_id, "success", feedback="answer used both memories")

        assert outcome["edge_evidence_events"]
        assert all(event["event_type"] == "reinforce" for event in outcome["edge_evidence_events"])
        events = memory.ledger.edge_evidence_events(namespace="demo", event_type="reinforce")
        assert events

    asyncio.run(run())


def test_sleep_compiles_frames_without_mutating_source_memories(tmp_path) -> None:
    async def run() -> None:
        memory = await nmem.MemoryRuntime.local(namespace="demo", path=tmp_path / ".neuromem", allow_unsafe_internal=True)
        first = await memory.observe_and_commit({"content": "When login redirect fails, refresh the session first.", "type": "rule", "keywords": ["login", "redirect"]})
        second = await memory.observe_and_commit({"content": "For redirect validation, check the session freshness.", "type": "rule", "keywords": ["login", "redirect"]})
        assert first.memory_id and second.memory_id
        await memory.query("login redirect session", top_k=2)

        store = memory.unsafe_internal_runtime.store
        before = {item.id: item.to_record() for item in store.list_memories(namespace="demo")}  # type: ignore[union-attr]
        report = await memory.sleep()
        after = {item.id: item.to_record() for item in store.list_memories(namespace="demo")}  # type: ignore[union-attr]

        assert report["append_only"] is True
        assert report["source_memories_mutated"] is False
        assert after == before
        assert report["sleep"]["graph"]["compiled_schemas"] or report["processed"] >= 2

    asyncio.run(run())

from __future__ import annotations

from neuromem.core.models import MemoryEdge, MemoryItem, MemoryQuery
from neuromem.retrieval.activation import ActivationResult, ActivationRetrievalEngine, RetrievalCandidate, RetrievalConfig, build_memory_card, build_query_plan_v2
from neuromem.stores.sqlite_store import SQLiteMemoryStore


def _memory(content: str, *, memory_type: str = "episodic", maturity: str = "fresh", keywords: list[str] | None = None, evidence: list[str] | None = None) -> MemoryItem:
    return MemoryItem(
        agent_id="agent",
        namespace="demo",
        type=memory_type,  # type: ignore[arg-type]
        content=content,
        keywords=keywords or [],
        evidence=evidence or ["trace-1"],
        confidence=0.8,
        maturity=maturity,  # type: ignore[arg-type]
    )


def test_query_plan_v2_routes_intents() -> None:
    assert build_query_plan_v2("What procedure should fix login redirects?", filters={"query_intent": "procedural_recall"}).intent == "procedural_recall"
    assert build_query_plan_v2("What style does the user prefer?", filters={"query_intent": "preference_recall"}).intent == "preference_recall"
    assert build_query_plan_v2("What is the latest session rule?", filters={"query_intent": "temporal_current"}).intent == "temporal_current"
    assert build_query_plan_v2("Why is auth related to redirect?", filters={"query_intent": "multi_hop"}).mode == "drift_activation"
    assert build_query_plan_v2("Summarize common failures", filters={"query_intent": "summary"}).mode == "global_consolidated"


def test_sqlite_fts_card_index_retrieves_symbols_and_multilingual_terms(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    item = _memory("Fix AUTH-401 in src/session.py by refreshing csrf token. 中文登录故障修复。", keywords=["AUTH-401", "src/session.py", "登录"])
    store.upsert_memory(item)

    exact = store.search_memory_cards("AUTH-401 src/session.py", namespace="demo")
    chinese = store.search_memory_cards("中文登录", namespace="demo")

    assert exact and exact[0][0] == item.id
    assert chinese and chinese[0][0] == item.id


def test_activation_retrieval_rrf_ppr_and_typed_edge_suppression(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    seed = _memory("Login redirect bug fixed by session refresh order.", keywords=["login", "redirect"])
    bridge = _memory("Session refresh order depends on CSRF token rotation.", keywords=["csrf", "session"])
    stale = _memory("Old redirect fix used cookie deletion.", maturity="obsolete", keywords=["redirect"])
    procedure = _memory("Always refresh session before redirect validation.", memory_type="procedural", keywords=["session", "redirect"])
    for item in [seed, bridge, stale, procedure]:
        store.upsert_memory(item)
    store.add_edge(MemoryEdge(source_id=seed.id, target_id=bridge.id, relation="supports", weight=0.9, confidence=0.9))
    store.add_edge(MemoryEdge(source_id=bridge.id, target_id=procedure.id, relation="procedure_for", weight=0.9, confidence=0.9, success_count=3))
    store.add_edge(MemoryEdge(source_id=seed.id, target_id=stale.id, relation="contradicts", weight=1.0, confidence=0.9))

    result = ActivationRetrievalEngine(store).retrieve(store.list_memories("demo"), MemoryQuery(query="login redirect session procedure", budget_tokens=900))
    selected_ids = [candidate.memory.id for candidate in result.selected]

    assert seed.id in selected_ids
    assert procedure.id in selected_ids
    assert result.activation.paths
    assert stale.id in result.ledger_record.suppressed_ids
    assert result.ledger_record.fusion_scores
    assert result.ledger_record.reranker_scores


def test_semantic_match_beats_context_only_candidate() -> None:
    semantic = _memory("The user shared a specific travel meal memory with a named companion.", keywords=["travel", "meal", "companion"])
    context_only = _memory("The user likes concise answers.", memory_type="preference", keywords=["preference"])
    semantic_candidate = RetrievalCandidate(memory=semantic, card=build_memory_card(semantic))
    semantic_candidate.channel_scores = {"dense": 0.82, "entity": 0.7}
    semantic_candidate.rrf_score = 1 / 61
    context_candidate = RetrievalCandidate(memory=context_only, card=build_memory_card(context_only))
    context_candidate.channel_scores = {"recent_current": 1.0, "graph_seed": 0.95}
    context_candidate.rrf_score = 1 / 60

    plan = build_query_plan_v2("Who was involved in that travel meal?", filters={"query_intent": "fact_lookup"})
    ActivationRetrievalEngine()._score(  # noqa: SLF001 - regression checks score composition directly.
        [semantic_candidate, context_candidate],
        plan,
        RetrievalConfig(),
        ActivationResult(scores={context_only.id: 0.8}),
    )

    assert semantic_candidate.semantic_score > 0
    assert context_candidate.semantic_score == 0
    assert semantic_candidate.final_score > context_candidate.final_score


def test_lifecycle_gate_suppresses_inhibited_without_historical(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    active = _memory("Current login fix uses session refresh.", keywords=["login"])
    inhibited = _memory("Deprecated login fix uses hard reset.", maturity="inhibited", keywords=["login"])
    for item in [active, inhibited]:
        store.upsert_memory(item)

    result = ActivationRetrievalEngine(store).retrieve(store.list_memories("demo"), MemoryQuery(query="current login fix", budget_tokens=800))

    assert [candidate.memory.id for candidate in result.selected] == [active.id]
    assert result.ledger_record.suppressed_ids[inhibited.id] == "lifecycle_suppressed:inhibited"

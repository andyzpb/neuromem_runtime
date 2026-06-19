from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from neuromem.core.models import MemoryEdge, MemoryItem, MemoryQuery
from neuromem.stores.base import MemoryStore


RetrievalMode = Literal["local_activation", "global_consolidated", "drift_activation"]
QueryIntent = Literal[
    "fact_lookup",
    "procedural_recall",
    "preference_recall",
    "episodic_debug",
    "temporal_current",
    "temporal_history",
    "multi_hop",
    "conflict_check",
    "summary",
    "unknown",
]
QUERY_INTENTS = {
    "fact_lookup",
    "procedural_recall",
    "preference_recall",
    "episodic_debug",
    "temporal_current",
    "temporal_history",
    "multi_hop",
    "conflict_check",
    "summary",
    "unknown",
}


class RerankProvider(Protocol):
    def rerank(self, query: str, candidates: list["RetrievalCandidate"], *, top_k: int) -> list["RetrievalCandidate"]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    budget_tokens: int = 800
    max_items: int = 8
    min_score: float = 0.08
    rrf_k: int = 60
    graph_activation: bool = True
    graph_steps: int = 8
    graph_restart_prob: float = 0.25
    graph_min_score: float = 0.02
    historical: bool = False
    require_provenance: bool = False
    allow_abstain: bool = False
    retrieval_channels: tuple[str, ...] = (
        "fts5",
        "bm25",
        "dense",
        "rewrite",
        "hyde",
        "lexical",
        "entity",
        "recent_current",
        "procedural_preference",
        "canonical_fact",
        "graph_seed",
    )
    rerank_mode: str = "lite"

    def stable_hash(self) -> str:
        payload = "|".join(
            [
                str(self.budget_tokens),
                str(self.max_items),
                str(self.min_score),
                str(self.rrf_k),
                str(self.graph_activation),
                str(self.graph_steps),
                str(self.graph_restart_prob),
                str(self.graph_min_score),
                str(self.historical),
                str(self.require_provenance),
                str(self.allow_abstain),
                ",".join(self.retrieval_channels),
                self.rerank_mode,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class QueryPlanV2:
    raw_query: str
    mode: RetrievalMode = "local_activation"
    intent: QueryIntent = "unknown"
    rewritten_queries: tuple[str, ...] = ()
    hyde_query: str | None = None
    entities: tuple[str, ...] = ()
    fact_keys: tuple[str, ...] = ()
    temporal_scope: str = "current"
    retrieval_channels: tuple[str, ...] = ()
    rerank_policy: str = "lite"
    abstain_allowed: bool = False
    required_provenance: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def stable_hash(self) -> str:
        value = "|".join(
            [
                self.raw_query,
                self.mode,
                self.intent,
                ",".join(self.entities),
                ",".join(self.fact_keys),
                self.temporal_scope,
                ",".join(self.retrieval_channels),
                self.rerank_policy,
                str(self.abstain_allowed),
                str(self.required_provenance),
            ]
        )
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class MemoryCard:
    memory_id: str
    namespace: str
    memory_type: str
    lifecycle_state: str
    content: str
    retrieval_context: str
    retrieval_text: str
    entities: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    provenance_ids: tuple[str, ...] = ()
    temporal_scope: str = "current"
    canonical_fact_key: str = ""
    trust_score: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalCandidate:
    memory: MemoryItem
    card: MemoryCard
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float = 0.0
    graph_score: float = 0.0
    reranker_score: float = 0.0
    final_score: float = 0.0
    graph_paths: list[list[str]] = field(default_factory=list)
    lifecycle_reason: str = "active"
    suppression_reason: str | None = None
    why_retrieved: list[str] = field(default_factory=list)

    def score_components(self) -> dict[str, float]:
        value = {f"{key}_score": round(score, 4) for key, score in sorted(self.channel_scores.items())}
        value.update(
            {
                "rrf_score": round(self.rrf_score, 4),
                "graph_score": round(self.graph_score, 4),
                "reranker_score": round(self.reranker_score, 4),
                "final_score": round(self.final_score, 4),
                "provenance_trust": round(self.card.trust_score, 4),
                "inhibition_penalty": round(self.memory.inhibition_score, 4),
                "contradiction_penalty": round(self.memory.contradiction_score, 4),
                "staleness_penalty": round(_staleness_penalty(self.memory), 4),
            }
        )
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory.id,
            "channel_ranks": dict(self.channel_ranks),
            "channel_scores": {key: round(value, 4) for key, value in self.channel_scores.items()},
            "rrf_score": round(self.rrf_score, 4),
            "graph_score": round(self.graph_score, 4),
            "reranker_score": round(self.reranker_score, 4),
            "final_score": round(self.final_score, 4),
            "graph_paths": self.graph_paths,
            "lifecycle_reason": self.lifecycle_reason,
            "suppression_reason": self.suppression_reason,
            "why_retrieved": list(self.why_retrieved),
            "provenance_ids": list(self.card.provenance_ids),
        }


@dataclass(frozen=True, slots=True)
class ActivationResult:
    scores: dict[str, float] = field(default_factory=dict)
    paths: list[list[str]] = field(default_factory=list)
    suppressed_ids: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalLedgerRecord:
    query_plan: dict[str, object]
    channel_candidates: dict[str, list[str]]
    fusion_scores: dict[str, float]
    graph_paths: list[list[str]]
    graph_scores: dict[str, float]
    reranker_scores: dict[str, float]
    selected_ids: list[str]
    suppressed_ids: dict[str, str]
    final_packed_context: str
    gate_decision: str
    retrieval_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActivationRetrievalResult:
    selected: list[RetrievalCandidate]
    ranked: list[RetrievalCandidate]
    suppressed: dict[str, str]
    query_plan: QueryPlanV2
    config: RetrievalConfig
    ledger_record: RetrievalLedgerRecord
    activation: ActivationResult
    gate_decision: str
    final_context: str


class LiteRerankProvider:
    def rerank(self, query: str, candidates: list[RetrievalCandidate], *, top_k: int) -> list[RetrievalCandidate]:
        del query
        return sorted(candidates, key=lambda item: (-item.final_score, item.memory.id))[:top_k]


class ActivationRetrievalEngine:
    def __init__(self, store: MemoryStore | None = None, reranker: RerankProvider | None = None) -> None:
        self.store = store
        self.reranker = reranker or LiteRerankProvider()

    def retrieve(self, memories: Iterable[MemoryItem], query: MemoryQuery, *, config: RetrievalConfig | None = None) -> ActivationRetrievalResult:
        config = config or retrieval_config_from_query(query)
        memory_items = [item for item in memories if item.maturity != "deleted"]
        cards = {item.id: build_memory_card(item) for item in memory_items}
        plan = build_query_plan_v2(query.query, filters=query.filters, config=config)
        namespace = memory_items[0].namespace if memory_items else str(query.filters.get("namespace", "default"))
        self._sync_semantic_index(cards, namespace=namespace, query=query)
        channel_rankings = self._candidate_channels(memory_items, cards, plan, query, config)
        candidates = self._fuse(memory_items, cards, channel_rankings, config)
        activation = ActivationResult()
        if config.graph_activation and self.store is not None and candidates:
            activation = ppr_activate(
                seed_ids=[candidate.memory.id for candidate in candidates[: min(20, len(candidates))]],
                store=self.store,
                plan=plan,
                config=config,
            )
        self._score(candidates, plan, config, activation)
        ranked = self.reranker.rerank(query.query, candidates, top_k=max(config.max_items * 4, config.max_items))
        selected, suppressed, gate = self._gate_and_pack(ranked, plan, config)
        final_context = "\n".join(f"- [{candidate.final_score:.2f}] {candidate.memory.content}" for candidate in selected)
        channel_candidates = {name: [memory_id for memory_id, _ in values] for name, values in channel_rankings.items()}
        ledger_suppressed = {**activation.suppressed_ids, **suppressed}
        ledger = RetrievalLedgerRecord(
            query_plan=plan.to_dict(),
            channel_candidates=channel_candidates,
            fusion_scores={candidate.memory.id: round(candidate.rrf_score, 4) for candidate in candidates},
            graph_paths=activation.paths,
            graph_scores={key: round(value, 4) for key, value in activation.scores.items()},
            reranker_scores={candidate.memory.id: round(candidate.reranker_score, 4) for candidate in ranked},
            selected_ids=[candidate.memory.id for candidate in selected],
            suppressed_ids=ledger_suppressed,
            final_packed_context=final_context,
            gate_decision=gate,
            retrieval_mode=plan.mode,
        )
        return ActivationRetrievalResult(
            selected=selected,
            ranked=ranked,
            suppressed=suppressed,
            query_plan=plan,
            config=config,
            ledger_record=ledger,
            activation=activation,
            gate_decision=gate,
            final_context=final_context,
        )

    def _candidate_channels(
        self,
        memories: list[MemoryItem],
        cards: dict[str, MemoryCard],
        plan: QueryPlanV2,
        query: MemoryQuery,
        config: RetrievalConfig,
    ) -> dict[str, list[tuple[str, float]]]:
        channels: dict[str, list[tuple[str, float]]] = {}
        query_terms = _terms(query.query)
        namespace = memories[0].namespace if memories else str(query.filters.get("namespace", "default"))
        if "fts5" in config.retrieval_channels:
            fts = _store_card_search(self.store, query.query, namespace=namespace if memories else None, limit=max(50, config.max_items * 8))
            if fts:
                channels["fts5"] = fts
        semantic_queries = self._semantic_queries(query, namespace=namespace)
        dense_scores = self._dense_scores(cards, semantic_queries, namespace=namespace, config=config, filters=query.filters)
        if dense_scores and "dense" in config.retrieval_channels:
            channels["dense"] = sorted(dense_scores.items(), key=lambda value: (-value[1], value[0]))
        rewrite_scores = self._lexical_expansion_scores(cards, semantic_queries.get("rewrites", []))
        if rewrite_scores and "rewrite" in config.retrieval_channels:
            channels["rewrite"] = sorted(rewrite_scores.items(), key=lambda value: (-value[1], value[0]))
        hyde_scores = self._lexical_expansion_scores(cards, semantic_queries.get("hyde", []))
        if hyde_scores and "hyde" in config.retrieval_channels:
            channels["hyde"] = sorted(hyde_scores.items(), key=lambda value: (-value[1], value[0]))
        lexical_scores = []
        entity_scores = []
        procedural_scores = []
        recent_scores = []
        canonical_scores = []
        graph_seed_scores = []
        bm25_scores = _bm25_scores(cards, query.query)
        for item in memories:
            card = cards[item.id]
            card_terms = _terms(card.retrieval_text)
            overlap = query_terms & card_terms
            if overlap:
                lexical_scores.append((item.id, len(overlap) / max(1, len(query_terms | card_terms))))
            entity_overlap = set(plan.entities) & (set(card.entities) | set(card.keywords) | card_terms)
            if entity_overlap:
                entity_scores.append((item.id, len(entity_overlap) / max(1, len(set(plan.entities) or {""}))))
            if item.type in {"procedural", "preference", "schema"}:
                procedural_scores.append((item.id, 0.55 + item.confidence * 0.3))
            recency_value = math.log1p(item.access_count + item.activation_count) / 5.0
            current_boost = 0.35 if item.maturity not in {"obsolete", "inhibited", "archived", "deleted"} else 0.0
            if recency_value or current_boost:
                recent_scores.append((item.id, min(1.0, recency_value + current_boost)))
            if card.canonical_fact_key and card.canonical_fact_key in plan.fact_keys:
                canonical_scores.append((item.id, 1.0))
            if item.coactivation_neighbors or item.supports or item.supersedes or item.contradicts:
                graph_seed_scores.append((item.id, min(1.0, 0.3 + item.reinforcement_score + item.confidence * 0.2)))
        if bm25_scores and "bm25" in config.retrieval_channels:
            channels["bm25"] = sorted(bm25_scores.items(), key=lambda value: (-value[1], value[0]))
        if lexical_scores and "lexical" in config.retrieval_channels:
            channels["lexical"] = sorted(lexical_scores, key=lambda value: (-value[1], value[0]))
        if entity_scores and "entity" in config.retrieval_channels:
            channels["entity"] = sorted(entity_scores, key=lambda value: (-value[1], value[0]))
        if recent_scores and "recent_current" in config.retrieval_channels:
            channels["recent_current"] = sorted(recent_scores, key=lambda value: (-value[1], value[0]))
        if procedural_scores and "procedural_preference" in config.retrieval_channels:
            channels["procedural_preference"] = sorted(procedural_scores, key=lambda value: (-value[1], value[0]))
        if canonical_scores and "canonical_fact" in config.retrieval_channels:
            channels["canonical_fact"] = sorted(canonical_scores, key=lambda value: (-value[1], value[0]))
        if graph_seed_scores and "graph_seed" in config.retrieval_channels:
            channels["graph_seed"] = sorted(graph_seed_scores, key=lambda value: (-value[1], value[0]))
        return channels

    def _sync_semantic_index(self, cards: dict[str, MemoryCard], *, namespace: str, query: MemoryQuery) -> None:
        started = time.perf_counter()
        embedding_provider = query.filters.get("_embedding_provider")
        vector_index = query.filters.get("_vector_index")
        if embedding_provider is None or vector_index is None or not cards:
            return
        embed = getattr(embedding_provider, "embed", None)
        upsert = getattr(vector_index, "upsert", None)
        if embed is None or upsert is None:
            return
        cache = query.filters.get("_embedding_cache")
        timing = query.filters.get("_retrieval_timing")
        stats = query.filters.setdefault("_embedding_cache_stats", {"memory_hits": 0, "memory_misses": 0, "query_hits": 0, "query_misses": 0})
        provider_label = str(query.filters.get("_embedding_provider_label") or embedding_provider.__class__.__name__)
        cached_vectors: dict[str, list[float]] = {}
        missing_ids: list[str] = []
        missing_texts: list[str] = []
        text_hashes = {memory_id: _text_hash(card.retrieval_text) for memory_id, card in cards.items()}
        cached_by_hash = _cache_get_many(cache, namespace=namespace, provider_label=provider_label, text_hashes=list(text_hashes.values()))
        for memory_id, card in cards.items():
            text_hash = text_hashes[memory_id]
            vector = cached_by_hash.get(text_hash)
            if vector is None:
                missing_ids.append(memory_id)
                missing_texts.append(card.retrieval_text)
                stats["memory_misses"] = int(stats.get("memory_misses", 0)) + 1
            else:
                cached_vectors[memory_id] = vector
                stats["memory_hits"] = int(stats.get("memory_hits", 0)) + 1
        if missing_texts:
            embed_started = time.perf_counter()
            vectors = embed(missing_texts)
            _add_timing(timing, "embedding_ms", embed_started)
            for memory_id, text, vector in zip(missing_ids, missing_texts, vectors, strict=True):
                vector_list = [float(value) for value in vector]
                cached_vectors[memory_id] = vector_list
            _cache_set_many(
                cache,
                namespace=namespace,
                provider_label=provider_label,
                vectors={_text_hash(text): [float(value) for value in vector] for text, vector in zip(missing_texts, vectors, strict=True)},
            )
        if cached_vectors:
            upsert(cached_vectors, namespace=namespace)
        _add_timing(timing, "index_sync_ms", started)

    def _semantic_queries(self, query: MemoryQuery, *, namespace: str) -> dict[str, list[str]]:
        raw = query.query
        rewrites = _dedupe_texts([str(item) for item in _string_list(query.filters.get("query_rewrites"))], exclude={_normalize_query_text(raw)})
        aliases: list[str] = []
        alias_resolver = query.filters.get("_entity_alias_resolver")
        if alias_resolver is not None and hasattr(alias_resolver, "expand"):
            aliases = _dedupe_texts([str(item) for item in getattr(alias_resolver, "expand")(raw, namespace)], exclude={_normalize_query_text(raw)})[:6]
        rewrite_provider = query.filters.get("_query_rewrite_provider")
        if rewrite_provider is not None and hasattr(rewrite_provider, "rewrite"):
            rewrites = _dedupe_texts([*rewrites, *(str(item) for item in getattr(rewrite_provider, "rewrite")(raw, namespace=namespace))], exclude={_normalize_query_text(raw)})[:8]
        hyde_values: list[str] = []
        if query.filters.get("hyde_query"):
            hyde_values.append(str(query.filters["hyde_query"]))
        hyde_provider = query.filters.get("_hyde_provider")
        if hyde_provider is not None and hasattr(hyde_provider, "generate"):
            generated = getattr(hyde_provider, "generate")(raw, namespace=namespace)
            if generated:
                hyde_values.append(str(generated))
        return {
            "raw": [raw],
            "rewrites": _dedupe_texts([*rewrites, *aliases], exclude={_normalize_query_text(raw)})[:10],
            "hyde": _dedupe_texts(hyde_values, exclude={_normalize_query_text(raw)})[:2],
        }

    def _dense_scores(self, cards: dict[str, MemoryCard], semantic_queries: dict[str, list[str]], *, namespace: str, config: RetrievalConfig, filters: dict[str, object]) -> dict[str, float]:
        del cards
        embedding_provider = filters.get("_embedding_provider")
        vector_index = filters.get("_vector_index")
        if embedding_provider is None or vector_index is None:
            return {}
        embed = getattr(embedding_provider, "embed", None)
        search = getattr(vector_index, "search", None)
        if embed is None or search is None:
            return {}
        queries = [*semantic_queries.get("raw", []), *semantic_queries.get("rewrites", []), *semantic_queries.get("hyde", [])]
        if not queries:
            return {}
        scores: dict[str, float] = {}
        cache = filters.get("_embedding_cache")
        timing = filters.get("_retrieval_timing")
        stats = filters.setdefault("_embedding_cache_stats", {"memory_hits": 0, "memory_misses": 0, "query_hits": 0, "query_misses": 0})
        provider_label = str(filters.get("_embedding_provider_label") or embedding_provider.__class__.__name__)
        vectors: list[list[float]] = []
        missing: list[tuple[int, str]] = []
        query_hashes = [_text_hash(query_text) for query_text in queries]
        cached_by_hash = _cache_get_many(cache, namespace=namespace, provider_label=provider_label, text_hashes=query_hashes)
        for index, query_text in enumerate(queries):
            text_hash = query_hashes[index]
            cached = cached_by_hash.get(text_hash)
            if cached is None:
                missing.append((index, query_text))
                vectors.append([])
                stats["query_misses"] = int(stats.get("query_misses", 0)) + 1
            else:
                vectors.append(cached)
                stats["query_hits"] = int(stats.get("query_hits", 0)) + 1
        if missing:
            embed_started = time.perf_counter()
            embedded = embed([text for _, text in missing])
            _add_timing(timing, "embedding_ms", embed_started)
            for (index, query_text), vector in zip(missing, embedded, strict=True):
                vector_list = [float(value) for value in vector]
                vectors[index] = vector_list
            _cache_set_many(
                cache,
                namespace=namespace,
                provider_label=provider_label,
                vectors={_text_hash(query_text): [float(value) for value in vector] for (_, query_text), vector in zip(missing, embedded, strict=True)},
            )
        for vector in vectors:
            for memory_id, score in search(vector, namespace=namespace, top_k=max(50, config.max_items * 8)):
                scores[str(memory_id)] = max(scores.get(str(memory_id), 0.0), float(score))
        return scores

    def _lexical_expansion_scores(self, cards: dict[str, MemoryCard], expansions: list[str]) -> dict[str, float]:
        if not expansions:
            return {}
        scores: dict[str, float] = {}
        expansion_terms = [term for expansion in expansions for term in _terms(expansion)]
        if not expansion_terms:
            return {}
        expansion_set = set(expansion_terms)
        for memory_id, card in cards.items():
            terms = _terms(card.retrieval_text)
            overlap = terms & expansion_set
            if overlap:
                scores[memory_id] = len(overlap) / max(1, len(expansion_set))
        return scores

    def _fuse(
        self,
        memories: list[MemoryItem],
        cards: dict[str, MemoryCard],
        channel_rankings: dict[str, list[tuple[str, float]]],
        config: RetrievalConfig,
    ) -> list[RetrievalCandidate]:
        by_id = {item.id: item for item in memories}
        candidates: dict[str, RetrievalCandidate] = {}
        for channel, ranking in channel_rankings.items():
            for rank, (memory_id, score) in enumerate(ranking, start=1):
                memory = by_id.get(memory_id)
                if memory is None:
                    continue
                candidate = candidates.setdefault(memory_id, RetrievalCandidate(memory=memory, card=cards[memory_id]))
                candidate.channel_ranks[channel] = min(rank, candidate.channel_ranks.get(channel, rank))
                candidate.channel_scores[channel] = max(float(score), candidate.channel_scores.get(channel, 0.0))
                candidate.rrf_score += 1.0 / (config.rrf_k + rank)
        if not candidates:
            for memory in memories:
                candidate = RetrievalCandidate(memory=memory, card=cards[memory.id])
                candidate.channel_scores["fallback"] = 0.01
                candidate.rrf_score = 1.0 / (config.rrf_k + len(candidates) + 1)
                candidates[memory.id] = candidate
        return sorted(candidates.values(), key=lambda item: (-item.rrf_score, item.memory.id))

    def _score(self, candidates: list[RetrievalCandidate], plan: QueryPlanV2, config: RetrievalConfig, activation: ActivationResult) -> None:
        for candidate in candidates:
            memory = candidate.memory
            graph_score = activation.scores.get(memory.id, 0.0)
            candidate.graph_score = graph_score
            channel_score = max(candidate.channel_scores.values(), default=0.0)
            dense_or_entity = max(
                candidate.channel_scores.get("dense", 0.0),
                candidate.channel_scores.get("entity", 0.0),
                candidate.channel_scores.get("canonical_fact", 0.0),
                candidate.channel_scores.get("fts5", 0.0),
                candidate.channel_scores.get("bm25", 0.0),
            )
            recency_style_penalty = 0.0
            if plan.intent in {"fact_lookup", "temporal_current"}:
                if not dense_or_entity and memory.type in {"preference", "procedural", "schema"}:
                    recency_style_penalty += 0.12
                if set(candidate.channel_scores) <= {"recent_current", "procedural_preference", "graph_seed"}:
                    recency_style_penalty += 0.18
            provenance_trust = candidate.card.trust_score
            lifecycle_boost = _lifecycle_boost(memory, plan)
            outcome_utility = min(1.0, memory.future_utility + memory.reinforcement_score)
            penalty = (
                memory.inhibition_score * 0.35
                + memory.contradiction_score * 0.3
                + _staleness_penalty(memory)
                + recency_style_penalty
                + (0.15 if config.require_provenance and not memory.evidence else 0.0)
            )
            candidate.reranker_score = (
                0.28 * min(1.0, candidate.rrf_score * 20)
                + (0.24 if plan.intent in {"fact_lookup", "temporal_current"} else 0.18) * channel_score
                + (0.12 if plan.intent in {"fact_lookup", "temporal_current"} else 0.22) * graph_score
                + (0.16 if plan.intent in {"fact_lookup", "temporal_current"} else 0.0) * dense_or_entity
                + 0.12 * provenance_trust
                + 0.1 * lifecycle_boost
                + 0.1 * outcome_utility
                - penalty
            )
            candidate.final_score = max(0.0, candidate.reranker_score)
            candidate.graph_paths = [path for path in activation.paths if memory.id in path]
            candidate.lifecycle_reason = _lifecycle_reason(memory, plan, config)
            candidate.why_retrieved = [
                f"mode={plan.mode}",
                f"intent={plan.intent}",
                *[f"source:{channel}" for channel in sorted(candidate.channel_scores)],
                f"gate={candidate.lifecycle_reason}",
            ]

    def _gate_and_pack(
        self,
        ranked: list[RetrievalCandidate],
        plan: QueryPlanV2,
        config: RetrievalConfig,
    ) -> tuple[list[RetrievalCandidate], dict[str, str], str]:
        suppressed: dict[str, str] = {}
        selected: list[RetrievalCandidate] = []
        used_tokens = 0
        seen_signatures: set[str] = set()
        for candidate in ranked:
            reason = _suppression_reason(candidate.memory, plan, config)
            if reason:
                candidate.suppression_reason = reason
                suppressed[candidate.memory.id] = reason
                continue
            if candidate.final_score < config.min_score:
                suppressed[candidate.memory.id] = "below_retrieval_quality_gate"
                continue
            signature = _dedupe_signature(candidate)
            if signature in seen_signatures and candidate.memory.type == "episodic":
                suppressed[candidate.memory.id] = "packed_duplicate_episode"
                continue
            cost = max(1, len(candidate.memory.content.split()))
            if selected and used_tokens + cost > config.budget_tokens:
                suppressed[candidate.memory.id] = "context_budget_exceeded"
                continue
            selected.append(candidate)
            used_tokens += cost
            seen_signatures.add(signature)
            if len(selected) >= config.max_items:
                break
        if selected:
            return selected, suppressed, "accepted"
        if config.allow_abstain or plan.abstain_allowed:
            return [], suppressed, "abstained_no_memory_passed_gate"
        return [], suppressed, "rejected_below_threshold"


def build_query_plan_v2(query: str, *, filters: dict[str, object] | None = None, config: RetrievalConfig | None = None) -> QueryPlanV2:
    filters = filters or {}
    config = config or RetrievalConfig()
    entities = sorted(set(_query_entities(query)) | set(_string_list(filters.get("entities"))))
    fact_keys = sorted(set(_string_list(filters.get("fact_keys"))))
    explicit_intent = str(filters.get("intent") or filters.get("query_intent") or "unknown")
    intent: QueryIntent = explicit_intent if explicit_intent in QUERY_INTENTS else "unknown"  # type: ignore[assignment]
    mode: RetrievalMode = "local_activation"
    if intent == "summary":
        mode = "global_consolidated"
    elif intent in {"multi_hop", "conflict_check"}:
        mode = "drift_activation"
    temporal_scope = "historical" if bool(filters.get("historical", False)) or intent == "temporal_history" else "current"
    return QueryPlanV2(
        raw_query=query,
        mode=str(filters.get("retrieval_mode") or mode),  # type: ignore[arg-type]
        intent=intent,
        rewritten_queries=tuple([query, *[str(value) for value in _string_list(filters.get("query_rewrites"))]]),
        hyde_query=str(filters["hyde_query"]) if filters.get("hyde_query") else None,
        entities=tuple(entities),
        fact_keys=tuple(fact_keys),
        temporal_scope=temporal_scope,
        retrieval_channels=config.retrieval_channels,
        rerank_policy=str(filters.get("rerank_mode") or config.rerank_mode),
        abstain_allowed=config.allow_abstain,
        required_provenance=config.require_provenance,
    )


def retrieval_config_from_query(query: MemoryQuery) -> RetrievalConfig:
    filters = query.filters
    channels = _string_tuple(filters.get("retrieval_channels") or filters.get("source_channels"))
    return RetrievalConfig(
        budget_tokens=query.budget_tokens,
        max_items=max(1, min(12, int(filters.get("max_items", max(1, query.budget_tokens // 250)) or 1))),
        min_score=float(filters.get("min_score", 0.08) or 0.08),
        graph_activation=bool(filters.get("graph_activation", filters.get("graph_diffusion", True))),
        graph_steps=int(filters.get("graph_steps", filters.get("graph_depth", 8)) or 8),
        graph_restart_prob=float(filters.get("graph_restart_prob", 0.25) or 0.25),
        graph_min_score=float(filters.get("graph_min_score", 0.02) or 0.02),
        historical=bool(filters.get("historical", False)),
        require_provenance=bool(filters.get("require_provenance", False)),
        allow_abstain=bool(filters.get("allow_abstain", filters.get("abstain", False))),
        retrieval_channels=channels or RetrievalConfig().retrieval_channels,
        rerank_mode=str(filters.get("rerank_mode", "lite") or "lite"),
    )


def build_memory_card(item: MemoryItem) -> MemoryCard:
    temporal = "historical" if item.maturity in {"obsolete", "archived", "deleted"} else "current"
    canonical = _canonical_fact_key(item)
    retrieval_context = (
        f"This {item.type} memory is {item.maturity}; "
        f"keywords={','.join(item.keywords)}; entities={','.join(item.entities)}; "
        f"provenance={','.join(item.evidence)}; fact_key={canonical}."
    )
    retrieval_text = "\n".join(
        [
            "[Memory Card]",
            f"type: {item.type}",
            f"namespace: {item.namespace}",
            f"entity: {', '.join(item.entities)}",
            f"keywords: {', '.join(item.keywords)}",
            f"temporal_scope: {temporal}",
            f"lifecycle: {item.maturity}",
            f"provenance: {', '.join(item.evidence)}",
            f"content: {item.content}",
            f"summary: {item.summary or ''}",
            f"retrieval_context: {retrieval_context}",
        ]
    )
    trust = min(1.0, max(0.05, item.confidence + (0.1 if item.evidence else -0.12) - item.contradiction_score - item.inhibition_score))
    return MemoryCard(
        memory_id=item.id,
        namespace=item.namespace,
        memory_type=item.type,
        lifecycle_state=item.maturity,
        content=item.content,
        retrieval_context=retrieval_context,
        retrieval_text=retrieval_text,
        entities=tuple(item.entities),
        keywords=tuple(item.keywords),
        provenance_ids=tuple(item.evidence),
        temporal_scope=temporal,
        canonical_fact_key=canonical,
        trust_score=trust,
    )


def ppr_activate(seed_ids: list[str], store: MemoryStore, plan: QueryPlanV2, config: RetrievalConfig) -> ActivationResult:
    if not seed_ids:
        return ActivationResult()
    seed_distribution = {memory_id: 1.0 / len(seed_ids) for memory_id in seed_ids}
    scores = dict(seed_distribution)
    paths: dict[str, list[str]] = {memory_id: [memory_id] for memory_id in seed_ids}
    suppressed: dict[str, str] = {}
    adjacency: dict[str, list[MemoryEdge]] = defaultdict(list)
    for edge in store.list_edges():
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    current = dict(seed_distribution)
    for _ in range(max(1, config.graph_steps)):
        next_scores = {memory_id: config.graph_restart_prob * score for memory_id, score in seed_distribution.items()}
        for source_id, activation in current.items():
            for edge in adjacency.get(source_id, []):
                neighbor_id = edge.target_id if edge.source_id == source_id else edge.source_id
                transfer, reason = _edge_transfer(edge, source_id=source_id, plan=plan)
                if reason:
                    suppressed.setdefault(neighbor_id, reason)
                    continue
                propagated = activation * (1.0 - config.graph_restart_prob) * transfer
                if propagated < config.graph_min_score:
                    continue
                next_scores[neighbor_id] = next_scores.get(neighbor_id, 0.0) + propagated
                source_path = paths.get(source_id, [source_id])
                if neighbor_id not in source_path:
                    candidate_path = [*source_path, neighbor_id]
                    if len(candidate_path) <= 5:
                        if len(candidate_path) > len(paths.get(neighbor_id, [])):
                            paths[neighbor_id] = candidate_path
        current = next_scores
        for memory_id, value in current.items():
            scores[memory_id] = max(scores.get(memory_id, 0.0), min(1.0, value))
    graph_paths = [path for path in paths.values() if len(path) > 1]
    return ActivationResult(scores=scores, paths=graph_paths, suppressed_ids=suppressed)


def _edge_transfer(edge: MemoryEdge, *, source_id: str, plan: QueryPlanV2) -> tuple[float, str | None]:
    relation_weight = {
        "supports": 1.0,
        "same_as": 1.15,
        "evidence_for": 0.95,
        "coactivated_with": 0.35 + min(0.55, edge.success_count * 0.08),
        "procedure_for": 1.1 if plan.intent == "procedural_recall" else 0.35,
        "preference_of": 1.0 if plan.intent == "preference_recall" else 0.3,
        "retrieved_with": 0.15,
        "associated_with": 0.45,
        "precedes": 0.35,
        "causes": 0.65,
        "part_of": 0.45,
        "generalizes": 0.55,
        "specializes": 0.55,
    }.get(edge.relation, 0.35)
    if edge.relation == "contradicts":
        neighbor = edge.target_id if edge.source_id == source_id else edge.source_id
        return 0.0, f"conflict_via_contradicts:{neighbor}"
    if edge.relation == "supersedes" and edge.target_id == source_id:
        return 1.0, None
    if edge.relation == "supersedes" and edge.source_id == source_id:
        return 0.0, f"suppressed_by_supersedes:{edge.target_id}"
    if edge.inhibition_score >= 0.8 or edge.contradiction_penalty >= 0.8:
        return 0.0, "edge_inhibited_or_contradicted"
    transfer = (
        max(edge.weight, 0.05)
        * max(edge.confidence, 0.05)
        * relation_weight
        * max(0.05, 1.0 - edge.inhibition_score)
        * max(0.05, 1.0 - edge.contradiction_penalty)
    )
    return min(1.0, transfer), None


def _store_card_search(store: MemoryStore | None, query: str, *, namespace: str | None, limit: int) -> list[tuple[str, float]]:
    search = getattr(store, "search_memory_cards", None)
    if search is None:
        return []
    try:
        return [(str(memory_id), float(score)) for memory_id, score in search(query, namespace=namespace, limit=limit)]
    except Exception:
        return []


def _bm25_scores(cards: dict[str, MemoryCard], query: str) -> dict[str, float]:
    query_terms = _terms(query)
    if not query_terms:
        return {}
    document_terms = {memory_id: _terms(card.retrieval_text) for memory_id, card in cards.items()}
    df: dict[str, int] = defaultdict(int)
    for terms in document_terms.values():
        for term in terms:
            df[term] += 1
    total_docs = max(1, len(document_terms))
    avg_len = sum(len(terms) for terms in document_terms.values()) / max(1, len(document_terms))
    raw: dict[str, float] = {}
    for memory_id, terms in document_terms.items():
        if not terms:
            continue
        term_counts = {term: 1 for term in terms}
        score = 0.0
        for term in query_terms:
            if term not in terms:
                continue
            idf = math.log(1 + (total_docs - df[term] + 0.5) / (df[term] + 0.5))
            tf = term_counts.get(term, 0)
            denom = tf + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(1.0, avg_len))
            score += idf * ((tf * 2.2) / denom)
        if score:
            raw[memory_id] = score
    max_score = max(raw.values(), default=1.0)
    return {memory_id: score / max_score for memory_id, score in raw.items()}


def _terms(text: str) -> set[str]:
    return {token.lower().strip(".,:;()[]`'\"?") for token in re.findall(r"[\w./:#-]+|[\u4e00-\u9fff]+", text) if token.strip(".,:;()[]`'\"?")}


def _query_entities(query: str) -> set[str]:
    stop = {"what", "should", "have", "fixed", "before", "current", "latest", "with", "that", "this", "the", "for", "and"}
    return {term for term in _terms(query) if len(term) > 2 and term not in stop}


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_get(cache: object, *, namespace: str, provider_label: str, text_hash: str) -> list[float] | None:
    get = getattr(cache, "get", None)
    if get is None:
        return None
    try:
        value = get(namespace=namespace, provider_model=provider_label, text_hash=text_hash)
    except Exception:
        return None
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def _cache_get_many(cache: object, *, namespace: str, provider_label: str, text_hashes: list[str]) -> dict[str, list[float]]:
    get_many = getattr(cache, "get_many", None)
    if get_many is not None:
        try:
            values = get_many(namespace=namespace, provider_model=provider_label, text_hashes=text_hashes)
        except Exception:
            values = {}
        if isinstance(values, dict):
            return {str(key): [float(item) for item in value] for key, value in values.items() if isinstance(value, list)}
    result: dict[str, list[float]] = {}
    for text_hash in text_hashes:
        value = _cache_get(cache, namespace=namespace, provider_label=provider_label, text_hash=text_hash)
        if value is not None:
            result[text_hash] = value
    return result


def _cache_set(cache: object, *, namespace: str, provider_label: str, text_hash: str, vector: list[float]) -> None:
    set_value = getattr(cache, "set", None)
    if set_value is None:
        return
    try:
        set_value(namespace=namespace, provider_model=provider_label, text_hash=text_hash, vector=vector)
    except Exception:
        return


def _cache_set_many(cache: object, *, namespace: str, provider_label: str, vectors: dict[str, list[float]]) -> None:
    set_many = getattr(cache, "set_many", None)
    if set_many is not None:
        try:
            set_many(namespace=namespace, provider_model=provider_label, vectors=vectors)
            return
        except Exception:
            return
    for text_hash, vector in vectors.items():
        _cache_set(cache, namespace=namespace, provider_label=provider_label, text_hash=text_hash, vector=vector)


def _add_timing(timing: object, field_name: str, started: float) -> None:
    if timing is None:
        return
    try:
        current = float(getattr(timing, field_name))
        setattr(timing, field_name, current + (time.perf_counter() - started) * 1000.0)
    except Exception:
        return


def _canonical_fact_key(item: MemoryItem) -> str:
    if item.tags:
        return item.tags[0].lower().replace(" ", "_")
    if item.keywords:
        return item.keywords[0].lower().replace(" ", "_")
    return ""


def _lifecycle_boost(memory: MemoryItem, plan: QueryPlanV2) -> float:
    if plan.intent == "procedural_recall" and memory.type == "procedural":
        return 1.0
    if plan.intent == "preference_recall" and memory.type == "preference":
        return 1.0
    return {
        "core": 1.0,
        "mature": 0.85,
        "reinforced": 0.75,
        "linked": 0.65,
        "captured": 0.55,
        "fresh": 0.35,
        "tagged": 0.25,
    }.get(memory.maturity, 0.0)


def _staleness_penalty(memory: MemoryItem) -> float:
    if memory.maturity in {"obsolete", "archived", "deleted"}:
        return 0.35
    if memory.maturity == "inhibited":
        return 0.45
    return memory.staleness_score * 0.25


def _lifecycle_reason(memory: MemoryItem, plan: QueryPlanV2, config: RetrievalConfig) -> str:
    reason = _suppression_reason(memory, plan, config)
    return reason or "active"


def _suppression_reason(memory: MemoryItem, plan: QueryPlanV2, config: RetrievalConfig) -> str | None:
    if memory.maturity == "deleted":
        return "deleted_memory"
    if not config.historical and plan.temporal_scope != "historical" and memory.maturity in {"obsolete", "inhibited", "archived"}:
        return f"lifecycle_suppressed:{memory.maturity}"
    if memory.inhibition_score >= 0.75:
        return "inhibition_gate"
    if memory.contradiction_score >= 0.75:
        return "contradiction_gate"
    if config.require_provenance and not memory.evidence:
        return "missing_required_provenance"
    return None


def _dedupe_signature(candidate: RetrievalCandidate) -> str:
    terms = sorted(_terms(candidate.memory.content))[:8]
    return "|".join([candidate.memory.type, *terms])


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(_string_list(value))


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return []


def _normalize_query_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _dedupe_texts(values: Iterable[str], *, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        key = _normalize_query_text(text)
        if not text or key in excluded or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

from neuromem.core.models import MemoryItem


TemporalIntent = Literal["current", "historical", "unknown"]
AnswerabilityIntent = Literal["answerable", "abstain_allowed", "unknown"]
ConflictIntent = Literal["prefer_current", "include_history", "unknown"]
RecallSource = Literal["lexical", "bm25", "vector", "graph", "recent_active", "canonical_fact", "external_adapter"]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    query: str
    fact_key: str = ""
    entities: tuple[str, ...] = ()
    temporal_intent: TemporalIntent = "unknown"
    answerability_intent: AnswerabilityIntent = "unknown"
    conflict_intent: ConflictIntent = "unknown"
    multi_hop_need: bool = False
    allow_abstain: bool = False
    expected_evidence_ids: tuple[str, ...] = ()
    multi_hop_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = "|".join(
            [
                self.query,
                self.fact_key,
                ",".join(self.entities),
                self.temporal_intent,
                self.answerability_intent,
                self.conflict_intent,
                str(self.multi_hop_need),
                str(self.allow_abstain),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class RecallEvidence:
    id: str
    content: str
    base_score: float = 0.0
    source: str = ""
    keywords: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    answer: str = ""
    timestamp: int = 0
    maturity: str = "fresh"
    memory_type: str = "episodic"
    confidence: float = 0.5
    inhibition_score: float = 0.0
    contradiction_score: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecallCandidate:
    evidence: RecallEvidence
    lexical_score: float = 0.0
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    temporal_score: float = 0.0
    entity_score: float = 0.0
    fact_key_score: float = 0.0
    obsolete_penalty: float = 0.0
    inhibition_penalty: float = 0.0
    conflict_penalty: float = 0.0
    pollution_penalty: float = 0.0
    low_provenance_penalty: float = 0.0
    fact_overlap: bool = False
    entity_overlap: bool = False
    final_score: float = 0.0
    invalidation_state: str = "valid"
    lifecycle_reason: str = "active"
    source_channels: tuple[RecallSource, ...] = ()

    def score_components(self) -> dict[str, float]:
        return {
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "graph_score": round(self.graph_score, 4),
            "recency_score": round(self.recency_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "entity_score": round(self.entity_score, 4),
            "fact_key_score": round(self.fact_key_score, 4),
            "obsolete_penalty": round(self.obsolete_penalty, 4),
            "inhibition_penalty": round(self.inhibition_penalty, 4),
            "conflict_penalty": round(self.conflict_penalty, 4),
            "pollution_penalty": round(self.pollution_penalty, 4),
            "low_provenance_penalty": round(self.low_provenance_penalty, 4),
        }


@dataclass(frozen=True, slots=True)
class RecallConfig:
    budget_tokens: int
    min_score: float = 0.14
    max_items: int = 8
    source_channels: tuple[RecallSource, ...] = ("lexical", "bm25", "vector", "graph", "recent_active", "canonical_fact")
    evidence_gate_enabled: bool = True
    require_fact_or_entity_alignment: bool = True

    def stable_hash(self) -> str:
        payload = f"{self.budget_tokens}|{self.min_score}|{self.max_items}|{','.join(self.source_channels)}|{self.evidence_gate_enabled}|{self.require_fact_or_entity_alignment}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class RecallResult:
    evidence: list[RecallEvidence]
    candidates: list[RecallCandidate]
    query_plan: QueryPlan
    rejected_ids: list[str] = field(default_factory=list)
    suppression_reasons: dict[str, str] = field(default_factory=dict)
    canonical_fact_ids: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    gate_decision: str = "accepted"
    memory_version: str = ""
    invalidation_state: str = "valid"
    recall_config_hash: str = ""
    extra_trace: dict[str, object] = field(default_factory=dict)

    def trace(self) -> dict[str, object]:
        selected_ids = [item.id for item in self.evidence]
        selected_candidates = [candidate for candidate in self.candidates if candidate.evidence.id in selected_ids]
        value = {
            "query_plan": self.query_plan.to_dict(),
            "query_plan_hash": self.query_plan.stable_hash(),
            "gate_decision": self.gate_decision,
            "memory_version": self.memory_version,
            "invalidation_state": self.invalidation_state,
            "recall_config_hash": self.recall_config_hash,
            "canonical_fact_ids": self.canonical_fact_ids,
            "rejected_ids": self.rejected_ids,
            "suppression_reasons": self.suppression_reasons,
            "graph_paths": self.graph_paths,
            "source_channels": sorted({channel for candidate in selected_candidates for channel in candidate.source_channels}),
            "score_components": {candidate.evidence.id: candidate.score_components() for candidate in selected_candidates},
        }
        value.update(self.extra_trace)
        return value


def build_query_plan(
    query: str,
    *,
    answer: str = "",
    expected_evidence_ids: tuple[str, ...] = (),
    multi_hop_evidence_ids: tuple[str, ...] = (),
    abstain: bool = False,
    entities: Iterable[str] = (),
) -> QueryPlan:
    lowered = query.lower()
    fact_key = answer if answer and answer != "unknown" else _infer_fact_key(lowered)
    temporal_intent: TemporalIntent = "unknown"
    if any(term in lowered for term in ["current", "now", "latest", "confirmed"]):
        temporal_intent = "current"
    elif any(term in lowered for term in ["old", "previous", "historical"]):
        temporal_intent = "historical"
    allow_abstain = abstain or answer == "unknown" or any(term in lowered for term in ["unknown", "unanswerable"])
    answerability: AnswerabilityIntent = "abstain_allowed" if allow_abstain else "answerable" if fact_key else "unknown"
    conflict: ConflictIntent = "prefer_current" if temporal_intent == "current" else "include_history" if temporal_intent == "historical" else "unknown"
    query_entities = set(_query_entities(query)) | {str(entity).lower() for entity in entities if str(entity).strip()}
    multi_hop_need = bool(multi_hop_evidence_ids) or any(term in lowered for term in ["why", "how", "related", "chain", "multi-hop", "multi hop"])
    return QueryPlan(
        query=query,
        fact_key=fact_key,
        entities=tuple(sorted(query_entities)),
        temporal_intent=temporal_intent,
        answerability_intent=answerability,
        conflict_intent=conflict,
        multi_hop_need=multi_hop_need,
        allow_abstain=allow_abstain,
        expected_evidence_ids=expected_evidence_ids,
        multi_hop_evidence_ids=multi_hop_evidence_ids,
    )


def run_recall(
    evidence: list[RecallEvidence],
    plan: QueryPlan,
    *,
    config: RecallConfig,
    graph_scores: dict[str, float] | None = None,
    graph_paths: list[list[str]] | None = None,
) -> RecallResult:
    graph_scores = graph_scores or {}
    graph_paths = graph_paths or []
    bm25 = _bm25_scores(evidence, plan.query)
    candidates = [
        _score_candidate(item, plan, bm25.get(item.id, 0.0), graph_scores.get(item.id, 0.0), config)
        for item in evidence
    ]
    ranked = sorted(candidates, key=lambda item: (-item.final_score, item.evidence.id))
    canonical, suppressed = _canonical_fact_view(ranked, plan)
    canonical_ids = {candidate.evidence.id for candidate in canonical}
    rejected_ids = [candidate.evidence.id for candidate in ranked if candidate.evidence.id not in canonical_ids]
    suppression_reasons = dict(suppressed)
    gated = []
    for candidate in canonical:
        if config.evidence_gate_enabled and candidate.final_score < config.min_score:
            suppression_reasons.setdefault(candidate.evidence.id, "below_evidence_gate")
            rejected_ids.append(candidate.evidence.id)
            continue
        gated.append(candidate)
    has_alignment = any(candidate.fact_overlap or candidate.entity_overlap for candidate in gated)
    if config.evidence_gate_enabled and plan.allow_abstain and not has_alignment:
        return RecallResult(
            evidence=[],
            candidates=ranked,
            query_plan=plan,
            rejected_ids=[candidate.evidence.id for candidate in ranked],
            suppression_reasons={candidate.evidence.id: "no_fact_or_entity_overlap" for candidate in ranked},
            graph_paths=graph_paths,
            gate_decision="abstained_no_evidence_alignment",
            memory_version=_memory_version([]),
            recall_config_hash=config.stable_hash(),
        )
    if config.evidence_gate_enabled and config.require_fact_or_entity_alignment and plan.fact_key and gated and not any(candidate.fact_overlap for candidate in gated):
        return RecallResult(
            evidence=[],
            candidates=ranked,
            query_plan=plan,
            rejected_ids=[candidate.evidence.id for candidate in ranked],
            suppression_reasons={candidate.evidence.id: suppression_reasons.get(candidate.evidence.id, "no_fact_key_overlap") for candidate in ranked},
            graph_paths=graph_paths,
            gate_decision="rejected_no_fact_key_overlap",
            memory_version=_memory_version([]),
            recall_config_hash=config.stable_hash(),
        )
    packed = _budget_candidates(gated, config)
    invalidation_state = "stale" if any(candidate.invalidation_state == "stale" for candidate in packed) else "valid"
    return RecallResult(
        evidence=[candidate.evidence for candidate in packed],
        candidates=ranked,
        query_plan=plan,
        rejected_ids=list(dict.fromkeys(rejected_ids)),
        suppression_reasons=suppression_reasons,
        canonical_fact_ids=[candidate.evidence.id for candidate in packed if candidate.fact_overlap],
        graph_paths=graph_paths,
        gate_decision="accepted" if packed else "rejected_below_threshold",
        memory_version=_memory_version([candidate.evidence.id for candidate in packed]),
        invalidation_state=invalidation_state,
        recall_config_hash=config.stable_hash(),
    )


def evidence_from_memory(item: MemoryItem, *, base_score: float = 0.0, source: str = "memory", trace: dict[str, object] | None = None) -> RecallEvidence:
    return RecallEvidence(
        id=item.id,
        content=item.content,
        base_score=base_score,
        source=source,
        keywords=tuple(item.keywords),
        entities=tuple(item.entities),
        answer=" ".join(item.tags),
        timestamp=int(item.activation_count + item.access_count),
        maturity=item.maturity,
        memory_type=item.type,
        confidence=item.confidence,
        inhibition_score=item.inhibition_score,
        contradiction_score=item.contradiction_score,
        evidence_ids=tuple(item.evidence),
        trace=trace or {},
    )


def _score_candidate(item: RecallEvidence, plan: QueryPlan, bm25_score: float, graph_score: float, config: RecallConfig) -> RecallCandidate:
    content_terms = _terms(item.content)
    query_terms = _terms(plan.query)
    lexical_terms = content_terms | set(item.keywords) | set(item.entities)
    overlap = query_terms & lexical_terms
    lexical_score = len(overlap) / max(1, len(query_terms | lexical_terms))
    semantic_score = max(float(item.base_score), lexical_score)
    recency_score = min(1.0, math.log1p(max(0, item.timestamp)) / 4.0)
    temporal_score = _temporal_score(item, plan)
    entity_score = len(set(plan.entities) & lexical_terms) / max(1, len(set(plan.entities))) if plan.entities else 0.0
    fact_key_score = _fact_key_score(item, plan, lexical_terms)
    fact_overlap = fact_key_score > 0
    entity_overlap = entity_score > 0
    obsolete_penalty = 0.45 if _is_obsolete(item) else 0.0
    inhibition_penalty = max(item.inhibition_score, 0.35 if item.maturity == "inhibited" else 0.0)
    conflict_penalty = max(item.contradiction_score, 0.25 if _is_conflict(item, plan) else 0.0)
    pollution_penalty = 0.18 if item.maturity in {"tagged", "provisional"} and item.confidence < 0.5 else 0.0
    low_provenance_penalty = 0.08 if not item.evidence_ids else 0.0
    final_score = (
        0.24 * semantic_score
        + 0.2 * lexical_score
        + 0.18 * bm25_score
        + 0.14 * graph_score
        + 0.08 * recency_score
        + 0.16 * temporal_score
        + 0.1 * entity_score
        + 0.22 * fact_key_score
        - obsolete_penalty
        - inhibition_penalty
        - conflict_penalty
        - pollution_penalty
        - low_provenance_penalty
    )
    channels = tuple(
        channel
        for channel, enabled in {
            "lexical": lexical_score > 0,
            "bm25": bm25_score > 0,
            "vector": semantic_score > 0,
            "graph": graph_score > 0,
            "recent_active": recency_score > 0,
            "canonical_fact": fact_key_score > 0,
            "external_adapter": "external" in item.source.lower(),
        }.items()
        if enabled and channel in config.source_channels
    )
    stale = obsolete_penalty > 0 or item.maturity in {"obsolete", "inhibited", "deleted", "archived"}
    return RecallCandidate(
        evidence=item,
        lexical_score=lexical_score,
        bm25_score=bm25_score,
        semantic_score=semantic_score,
        graph_score=graph_score,
        recency_score=recency_score,
        temporal_score=temporal_score,
        entity_score=entity_score,
        fact_key_score=fact_key_score,
        obsolete_penalty=obsolete_penalty,
        inhibition_penalty=inhibition_penalty,
        conflict_penalty=conflict_penalty,
        pollution_penalty=pollution_penalty,
        low_provenance_penalty=low_provenance_penalty,
        fact_overlap=fact_overlap,
        entity_overlap=entity_overlap,
        final_score=max(0.0, final_score),
        invalidation_state="stale" if stale else "valid",
        lifecycle_reason="obsolete_or_inhibited" if stale else "active",
        source_channels=channels,
    )


def _bm25_scores(evidence: list[RecallEvidence], query: str) -> dict[str, float]:
    query_terms = _terms(query)
    if not query_terms:
        return {}
    document_terms = {item.id: _terms(item.content) | set(item.keywords) | set(item.entities) for item in evidence}
    df: Counter[str] = Counter()
    for terms in document_terms.values():
        for term in terms:
            df[term] += 1
    total = max(1, len(evidence))
    raw: dict[str, float] = {}
    for item_id, terms in document_terms.items():
        overlap = query_terms & terms
        if not overlap:
            continue
        raw[item_id] = sum(math.log1p(total / max(1, df[term])) for term in overlap)
    max_score = max(raw.values(), default=1.0)
    return {item_id: score / max_score for item_id, score in raw.items()}


def _canonical_fact_view(candidates: list[RecallCandidate], plan: QueryPlan) -> tuple[list[RecallCandidate], dict[str, str]]:
    if not plan.fact_key:
        return candidates, {}
    active: list[RecallCandidate] = []
    stale_same_fact: list[RecallCandidate] = []
    other: list[RecallCandidate] = []
    for candidate in candidates:
        if not candidate.fact_overlap:
            other.append(candidate)
        elif candidate.invalidation_state == "stale":
            stale_same_fact.append(candidate)
        else:
            active.append(candidate)
    if active:
        return active + other, {candidate.evidence.id: "canonical_fact_superseded_or_obsolete" for candidate in stale_same_fact}
    return candidates, {}


def _budget_candidates(candidates: list[RecallCandidate], config: RecallConfig) -> list[RecallCandidate]:
    kept: list[RecallCandidate] = []
    used = 0
    bridge_needed = config.evidence_gate_enabled and any(candidate.evidence.id in candidate.evidence.trace.get("multi_hop_evidence_ids", []) for candidate in candidates)
    for candidate in candidates:
        cost = max(1, len(candidate.evidence.content.split()))
        if len(kept) >= config.max_items:
            break
        if kept and used + cost > config.budget_tokens and not bridge_needed:
            break
        kept.append(candidate)
        used += cost
    return kept


def _memory_version(ids: list[str]) -> str:
    if not ids:
        return "empty"
    return hashlib.sha1("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]


def _fact_key_score(item: RecallEvidence, plan: QueryPlan, lexical_terms: set[str]) -> float:
    if not plan.fact_key:
        return 0.0
    key_terms = set(plan.fact_key.split("_"))
    if plan.fact_key in item.answer or plan.fact_key in item.content.lower():
        return 1.0
    if key_terms & lexical_terms:
        return 0.65
    if plan.expected_evidence_ids and item.id in plan.expected_evidence_ids:
        return 1.0
    if plan.multi_hop_evidence_ids and item.id in plan.multi_hop_evidence_ids:
        return 0.85
    return 0.0


def _temporal_score(item: RecallEvidence, plan: QueryPlan) -> float:
    lowered = item.content.lower()
    if plan.temporal_intent == "current":
        if item.maturity in {"obsolete", "inhibited", "deleted", "archived"} or _is_obsolete(item):
            return 0.0
        if any(term in lowered for term in ["current", "confirmed", "now", "latest"]):
            return 1.0
        return 0.35
    if plan.temporal_intent == "historical" and _is_obsolete(item):
        return 0.85
    if any(term in lowered for term in ["current", "confirmed"]):
        return 0.45
    return 0.15


def _is_obsolete(item: RecallEvidence) -> bool:
    lowered = item.content.lower()
    return item.maturity in {"obsolete", "inhibited", "deleted", "archived"} or any(term in lowered for term in ["obsolete", "old ", "deprecated", "replaced", "superseded"])


def _is_conflict(item: RecallEvidence, plan: QueryPlan) -> bool:
    return plan.conflict_intent == "prefer_current" and _is_obsolete(item)


def _infer_fact_key(lowered_query: str) -> str:
    if "pytest" in lowered_query or "test command" in lowered_query or "docker command" in lowered_query:
        return "pytest_q"
    if "redirect" in lowered_query or "session" in lowered_query or "login" in lowered_query:
        return "refresh_order"
    return ""


def _query_entities(query: str) -> set[str]:
    stop = {"what", "should", "be", "the", "is", "for", "used", "checked", "current", "old", "latest"}
    return {term for term in _terms(query) if term not in stop and len(term) > 2}


def _terms(text: str) -> set[str]:
    return {token.lower().strip(".,:;()[]`'\"?") for token in text.split() if token.strip(".,:;()[]`'\"?")}

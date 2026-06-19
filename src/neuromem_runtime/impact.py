from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal
from uuid import uuid4

from neuromem.core.models import MemoryFrame, MemoryItem
from neuromem_runtime.ledger import ExperienceEvent


ImpactType = Literal[
    "redundant",
    "confirmation",
    "novel_local",
    "worldview_update",
    "conflict",
    "supersession",
    "high_risk",
    "needs_clarification",
    "sleep_worthy",
]
ImpactDecision = Literal[
    "ledger_only",
    "append_support",
    "append_evidence",
    "propose_frame",
    "propose_candidate",
    "propose_worldview_candidate",
    "append_derivation",
    "append_supersession",
    "append_contradiction",
    "append_inhibition",
    "append_suppression",
    "ask_clarification",
    "quarantine",
    "sleep_priority",
]


@dataclass(slots=True)
class WorldviewImpactVector:
    novelty: float = 0.0
    belief_delta: float = 0.0
    entropy_delta: float = 0.0
    contradiction: float = 0.0
    supersession: float = 0.0
    utility: float = 0.0
    propagation: float = 0.0
    source_reliability: float = 0.8
    risk: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SlotImpact:
    slot_key: str
    prior_candidates: list[dict[str, object]] = field(default_factory=list)
    posterior_candidates: list[dict[str, object]] = field(default_factory=list)
    belief_delta: float = 0.0
    entropy_delta: float = 0.0
    top_candidate_changed: bool = False
    contradiction_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class WorldviewImpactAssessment:
    event_id: str
    namespace: str
    input_hash: str
    impacted_slots: list[SlotImpact]
    vector: WorldviewImpactVector
    impact_score: float
    impact_type: ImpactType
    decision: ImpactDecision
    reason: str
    impact_id: str = field(default_factory=lambda: f"imp_{uuid4().hex}")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["impacted_slots"] = [slot.to_dict() for slot in self.impacted_slots]
        value["vector"] = self.vector.to_dict()
        return value


class WorldviewImpactMeter:
    """Deterministic write-pressure estimate over current namespace memory."""

    def assess(
        self,
        event: ExperienceEvent,
        memories: list[MemoryItem],
        *,
        worldview_candidates: list[dict[str, object]] | None = None,
        frames: list[MemoryFrame] | None = None,
        edge_events: list[dict[str, object]] | None = None,
        grounded_claims: list[dict[str, object]] | None = None,
    ) -> WorldviewImpactAssessment:
        if grounded_claims:
            return self._assess_claims(event, grounded_claims)
        return _assess_unstructured_event(event)

    def _assess_claims(self, event: ExperienceEvent, claims: list[dict[str, object]]) -> WorldviewImpactAssessment:
        slots: list[SlotImpact] = []
        max_contradiction = 0.0
        max_supersession = 0.0
        max_utility = 0.0
        has_derivation = False
        has_candidate = False
        for claim in claims:
            slot_key = str(claim.get("canonical_slot_key") or "claim.general")
            statement = str(claim.get("canonical_statement") or "")
            confidence = max(0.0, min(1.0, float(claim.get("confidence", 0.5) or 0.5)))
            target_memory_ids = [str(item) for item in claim.get("target_memory_ids", [])] if isinstance(claim.get("target_memory_ids"), list) else []
            target_candidate_ids = [str(item) for item in claim.get("target_candidate_ids", [])] if isinstance(claim.get("target_candidate_ids"), list) else []
            metadata = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}
            is_correction = bool(metadata.get("correction")) or bool(target_memory_ids or target_candidate_ids)
            is_derivation = str(claim.get("source_kind")) == "assistant_derivation"
            has_derivation = has_derivation or is_derivation
            has_candidate = has_candidate or not is_derivation
            contradiction = 0.86 if is_correction else 0.0
            supersession = 0.82 if is_correction else 0.0
            max_contradiction = max(max_contradiction, contradiction)
            max_supersession = max(max_supersession, supersession)
            max_utility = max(max_utility, confidence * 0.65)
            slots.append(
                SlotImpact(
                    slot_key=slot_key,
                    prior_candidates=[{"candidate_id": target, "score": 1.0, "statement": "targeted prior belief"} for target in [*target_memory_ids, *target_candidate_ids]],
                    posterior_candidates=[{"candidate_id": str(claim.get("claim_id") or "new_claim"), "score": confidence, "statement": statement}],
                    belief_delta=0.72 if is_correction else 0.38,
                    entropy_delta=0.0,
                    top_candidate_changed=is_correction,
                    contradiction_score=contradiction,
                )
            )
        vector = WorldviewImpactVector(
            novelty=0.0 if not claims else 0.55,
            belief_delta=max((slot.belief_delta for slot in slots), default=0.0),
            entropy_delta=0.0,
            contradiction=max_contradiction,
            supersession=max_supersession,
            utility=round(max_utility, 4),
            propagation=min(1.0, len(claims) / 4.0),
            source_reliability=_source_reliability(event.source),
            risk=_risk_score(event.content, event.metadata),
        )
        if vector.risk >= 0.75:
            impact_type: ImpactType = "high_risk"
            decision: ImpactDecision = "quarantine"
        elif max_supersession > 0:
            impact_type = "supersession"
            decision = "append_supersession"
        elif max_contradiction > 0:
            impact_type = "conflict"
            decision = "append_contradiction"
        elif has_derivation and not has_candidate:
            impact_type = "worldview_update"
            decision = "append_derivation"
        elif claims:
            impact_type = "worldview_update"
            decision = "propose_candidate"
        else:
            impact_type = "redundant"
            decision = "ledger_only"
        score = max(0.0, min(1.0, (0.24 + vector.belief_delta * 0.35 + vector.supersession * 0.25 + vector.utility * 0.16) * vector.source_reliability))
        return WorldviewImpactAssessment(
            event_id=event.event_id,
            namespace=event.namespace,
            input_hash=event.content_hash,
            impacted_slots=slots or [SlotImpact(slot_key="claim.general")],
            vector=vector,
            impact_score=round(score, 4),
            impact_type=impact_type,
            decision=decision,
            reason=f"{impact_type} from {len(claims)} structured grounded claim(s)",
        )


def _assess_unstructured_event(event: ExperienceEvent) -> WorldviewImpactAssessment:
    risk = _risk_score(event.content, event.metadata)
    vector = WorldviewImpactVector(source_reliability=_source_reliability(event.source), risk=round(risk, 4))
    if risk >= 0.75:
        impact_type: ImpactType = "high_risk"
        decision: ImpactDecision = "quarantine"
        score = risk
        reason = "structured risk metadata requires quarantine"
    else:
        impact_type = "redundant"
        decision = "ledger_only"
        score = 0.0
        reason = "raw experience recorded without structured grounded claims"
    return WorldviewImpactAssessment(
        event_id=event.event_id,
        namespace=event.namespace,
        input_hash=event.content_hash,
        impacted_slots=[SlotImpact(slot_key="unstructured:ledger_only")],
        vector=vector,
        impact_score=round(score, 4),
        impact_type=impact_type,
        decision=decision,
        reason=reason,
    )


def _slot_key(event: ExperienceEvent) -> str:
    metadata = event.metadata
    explicit = metadata.get("slot_key") or metadata.get("canonical_key")
    if explicit:
        return str(explicit).strip().lower().replace(" ", "_")
    kind = str(metadata.get("type") or "observation").lower()
    return f"{kind or 'observation'}:unresolved"


def _candidate_memories(slot_key: str, memories: list[MemoryItem]) -> list[MemoryItem]:
    del slot_key, memories
    return []


def _candidate_frames(slot_key: str, frames: list[MemoryFrame]) -> list[MemoryFrame]:
    return [frame for frame in frames if frame.canonical_key == slot_key]


def _candidate_rows(slot_key: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("slot_key")) == slot_key]


def _candidate_distribution(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return [{"candidate_id": "none", "score": 1.0, "statement": "no current candidate"}]
    weights = [max(0.03, min(1.0, float(row.get("score", 0.0)) or float(row.get("confidence", 0.5)))) for row in rows]
    total = sum(weights) or 1.0
    return [
        {"candidate_id": str(row.get("candidate_id")), "score": round(weight / total, 4), "statement": str(row.get("statement", ""))[:240]}
        for row, weight in zip(rows, weights, strict=False)
    ]


def _posterior_from_candidates(prior: list[dict[str, object]], content: str, contradiction: float, supersession: float) -> list[dict[str, object]]:
    adjusted: list[float] = []
    for item in prior:
        score = float(item["score"])
        similarity = _jaccard(content, str(item.get("statement", "")))
        score += similarity * 0.22
        score -= contradiction * 0.28
        score -= supersession * 0.38
        adjusted.append(max(0.01, score))
    adjusted.append(max(0.01, 0.16 + supersession * 0.58 + contradiction * 0.28))
    total = sum(adjusted) or 1.0
    values = [
        {"candidate_id": str(item["candidate_id"]), "score": round(score / total, 4), "statement": str(item["statement"])}
        for item, score in zip(prior, adjusted[:-1], strict=False)
    ]
    values.append({"candidate_id": "new_candidate", "score": round(adjusted[-1] / total, 4), "statement": content[:240]})
    return values


def _candidate_contradiction_score(content: str, rows: list[dict[str, object]]) -> float:
    del content, rows
    return 0.0


def _candidate_supersession_score(content: str, rows: list[dict[str, object]], contradiction: float) -> float:
    del content, rows
    return min(1.0, contradiction * 0.55)


def _edge_contradiction_score(rows: list[dict[str, object]], edge_events: list[dict[str, object]]) -> float:
    targets = {str(row.get("candidate_id")) for row in rows} | {str(item) for row in rows for item in row.get("source_memory_ids", []) if str(item)}
    if not targets:
        return 0.0
    active = [event for event in edge_events if event.get("event_type") == "contradict" and str(event.get("target_id")) in targets]
    return min(1.0, 0.18 * len(active))


def _memory_overlap(event: ExperienceEvent, memories: list[MemoryItem]) -> float:
    event_terms = _terms(event.content)
    metadata_terms = set()
    for key in ["keywords", "entities", "tags"]:
        raw = event.metadata.get(key)
        if isinstance(raw, list):
            metadata_terms.update(str(item).lower() for item in raw if str(item).strip())
    event_terms |= metadata_terms
    if not event_terms:
        return 0.0
    best = 0.0
    for memory in memories:
        terms = _terms(" ".join([memory.content, *memory.keywords, *memory.entities, *memory.tags]))
        if terms:
            best = max(best, len(event_terms & terms) / len(event_terms | terms))
    return best


def _distribution(candidates: list[MemoryItem]) -> list[dict[str, object]]:
    if not candidates:
        return [{"candidate_id": "none", "score": 1.0, "statement": "no current candidate"}]
    weights = [max(0.05, min(1.0, _num(memory.confidence) + _num(memory.salience) * 0.2 - _num(memory.decay_score) * 0.2 - _num(memory.inhibition_score) * 0.4)) for memory in candidates]
    total = sum(weights) or 1.0
    return [
        {"candidate_id": memory.id, "score": round(weight / total, 4), "statement": memory.content[:240]}
        for memory, weight in zip(candidates, weights, strict=False)
    ]


def _posterior(prior: list[dict[str, object]], content: str, candidates: list[MemoryItem], contradiction: float, supersession: float) -> list[dict[str, object]]:
    if not candidates:
        return [{"candidate_id": "new_candidate", "score": 1.0, "statement": content[:240]}]
    adjusted: list[float] = []
    for item, memory in zip(prior, candidates, strict=False):
        score = float(item["score"])
        similarity = _jaccard(content, memory.content)
        score += similarity * 0.25
        score -= contradiction * 0.25
        score -= supersession * 0.35
        adjusted.append(max(0.01, score))
    adjusted.append(max(0.01, 0.18 + supersession * 0.55 + contradiction * 0.25))
    total = sum(adjusted) or 1.0
    values = [
        {"candidate_id": str(item["candidate_id"]), "score": round(score / total, 4), "statement": str(item["statement"])}
        for item, score in zip(prior, adjusted[:-1], strict=False)
    ]
    values.append({"candidate_id": "new_candidate", "score": round(adjusted[-1] / total, 4), "statement": content[:240]})
    return values


def _classify(vector: WorldviewImpactVector, score: float, candidates: list[MemoryItem], *, has_worldview_candidates: bool = False) -> tuple[ImpactType, ImpactDecision]:
    if vector.risk >= 0.75:
        return "high_risk", "quarantine"
    if vector.contradiction >= 0.55 and vector.source_reliability < 0.7:
        return "needs_clarification", "ask_clarification"
    if vector.supersession >= 0.55:
        return "supersession", "propose_worldview_candidate"
    if vector.contradiction >= 0.5:
        return "conflict", "ask_clarification"
    if not candidates and not has_worldview_candidates and vector.novelty >= 0.65:
        return "novel_local", "propose_frame"
    if score >= 0.45:
        return "worldview_update", "propose_worldview_candidate"
    if vector.novelty <= 0.2 and vector.belief_delta <= 0.15:
        return "redundant", "ledger_only"
    if vector.entropy_delta > 0.05:
        return "confirmation", "append_evidence"
    if score >= 0.28:
        return "sleep_worthy", "sleep_priority"
    return "redundant", "ledger_only"


def _reason(impact_type: ImpactType, vector: WorldviewImpactVector, slot_key: str) -> str:
    return (
        f"{impact_type} for slot {slot_key}: novelty={vector.novelty:.2f}, "
        f"belief_delta={vector.belief_delta:.2f}, contradiction={vector.contradiction:.2f}, "
        f"supersession={vector.supersession:.2f}, risk={vector.risk:.2f}"
    )


def _source_reliability(source: str) -> float:
    normalized = source.lower()
    if normalized in {"system", "admin"}:
        return 0.95
    if normalized in {"user", "tool", "tool_result"}:
        return 0.85
    if normalized in {"small_llm", "model", "assistant"}:
        return 0.65
    return 0.75


def _utility_score(event: ExperienceEvent) -> float:
    metadata = event.metadata
    explicit = metadata.get("future_utility")
    if isinstance(explicit, int | float):
        return max(0.0, min(1.0, float(explicit)))
    return 0.0


def _propagation_score(event: ExperienceEvent) -> float:
    metadata = event.metadata
    tags = metadata.get("tags", [])
    keywords = metadata.get("keywords", [])
    count = (len(tags) if isinstance(tags, list) else 0) + (len(keywords) if isinstance(keywords, list) else 0)
    return max(0.0, min(1.0, count / 6.0))


def _risk_score(content: str, metadata: dict[str, object]) -> float:
    del content
    explicit = metadata.get("risk")
    if isinstance(explicit, int | float):
        return max(0.0, min(1.0, float(explicit)))
    return 0.0


def _contradiction_score(content: str, candidates: list[MemoryItem]) -> float:
    del content, candidates
    return 0.0


def _supersession_score(content: str, candidates: list[MemoryItem], contradiction: float) -> float:
    del content, candidates
    return min(1.0, contradiction * 0.55)


def _jaccard(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(text: str) -> set[str]:
    return {term.strip(".,:;!?()[]{}\"'").lower() for term in text.split() if len(term.strip(".,:;!?()[]{}\"'")) >= 2}


def _js_divergence(left: list[float], right: list[float]) -> float:
    size = max(len(left), len(right))
    p = _pad_distribution(left, size)
    q = _pad_distribution(right, size)
    m = [(a + b) / 2.0 for a, b in zip(p, q, strict=False)]
    return min(1.0, (_kl(p, m) + _kl(q, m)) / 2.0)


def _pad_distribution(values: list[float], size: int) -> list[float]:
    padded = [max(0.0001, value) for value in values]
    while len(padded) < size:
        padded.append(0.0001)
    total = sum(padded) or 1.0
    return [value / total for value in padded]


def _kl(left: list[float], right: list[float]) -> float:
    return sum(p * math.log(p / max(q, 0.0001), 2) for p, q in zip(left, right, strict=False) if p > 0)


def _entropy(values: list[float]) -> float:
    distribution = _pad_distribution(values, len(values))
    return -sum(value * math.log(value, 2) for value in distribution if value > 0)


def _top_id(values: list[dict[str, object]]) -> str | None:
    if not values:
        return None
    return str(max(values, key=lambda item: float(item["score"]))["candidate_id"])


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        nums = [float(item) for item in value.values() if isinstance(item, int | float)]
        return max(nums) if nums else default
    return default


__all__ = [
    "SlotImpact",
    "WorldviewImpactAssessment",
    "WorldviewImpactMeter",
    "WorldviewImpactVector",
]

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
    "append_evidence",
    "propose_frame",
    "propose_worldview_candidate",
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
    ) -> WorldviewImpactAssessment:
        slot_key = _slot_key(event)
        candidate_rows = _candidate_rows(slot_key, worldview_candidates or [])
        candidates = _candidate_memories(slot_key, memories)
        frame_candidates = _candidate_frames(slot_key, frames or [])
        similarity = max((_jaccard(event.content, memory.content) for memory in memories), default=0.0)
        slot_similarity = max((_jaccard(event.content, memory.content) for memory in candidates), default=0.0)
        frame_similarity = max((_jaccard(event.content, frame.content) for frame in frame_candidates), default=0.0)
        candidate_similarity = max((_jaccard(event.content, str(row.get("statement", ""))) for row in candidate_rows), default=0.0)
        slot_similarity = max(slot_similarity, frame_similarity, candidate_similarity)
        contradiction = _contradiction_score(event.content, candidates)
        contradiction = max(contradiction, _candidate_contradiction_score(event.content, candidate_rows), _edge_contradiction_score(candidate_rows, edge_events or []))
        supersession = _supersession_score(event.content, candidates, contradiction)
        supersession = max(supersession, _candidate_supersession_score(event.content, candidate_rows, contradiction))
        memory_overlap = _memory_overlap(event, memories)
        novelty = max(0.0, min(1.0, 1.0 - max(similarity, slot_similarity * 0.82, memory_overlap * 0.75)))
        utility = _utility_score(event)
        propagation = _propagation_score(event)
        risk = _risk_score(event.content, event.metadata)
        source_reliability = _source_reliability(event.source)
        prior = _candidate_distribution(candidate_rows) if candidate_rows else _distribution(candidates)
        posterior = _posterior_from_candidates(prior, event.content, contradiction, supersession) if candidate_rows else _posterior(prior, event.content, candidates, contradiction, supersession)
        belief_delta = _js_divergence([item["score"] for item in prior], [item["score"] for item in posterior])
        entropy_delta = _entropy([item["score"] for item in prior]) - _entropy([item["score"] for item in posterior])
        if not candidates and not candidate_rows and novelty > 0.35:
            belief_delta = max(belief_delta, min(0.65, 0.25 + novelty * 0.35))
        vector = WorldviewImpactVector(
            novelty=round(novelty, 4),
            belief_delta=round(belief_delta, 4),
            entropy_delta=round(entropy_delta, 4),
            contradiction=round(contradiction, 4),
            supersession=round(supersession, 4),
            utility=round(utility, 4),
            propagation=round(propagation, 4),
            source_reliability=round(source_reliability, 4),
            risk=round(risk, 4),
        )
        raw_score = (
            0.20 * vector.novelty
            + 0.25 * vector.belief_delta
            + 0.15 * abs(vector.entropy_delta)
            + 0.15 * vector.contradiction
            + 0.10 * vector.supersession
            + 0.10 * vector.utility
            + 0.05 * vector.propagation
        )
        effective_score = max(0.0, min(1.0, raw_score * vector.source_reliability))
        impact_type, decision = _classify(vector, effective_score, candidates, has_worldview_candidates=bool(candidate_rows))
        slot = SlotImpact(
            slot_key=slot_key,
            prior_candidates=prior,
            posterior_candidates=posterior,
            belief_delta=vector.belief_delta,
            entropy_delta=vector.entropy_delta,
            top_candidate_changed=_top_id(prior) != _top_id(posterior),
            contradiction_score=vector.contradiction,
        )
        return WorldviewImpactAssessment(
            event_id=event.event_id,
            namespace=event.namespace,
            input_hash=event.content_hash,
            impacted_slots=[slot],
            vector=vector,
            impact_score=round(effective_score, 4),
            impact_type=impact_type,
            decision=decision,
            reason=_reason(impact_type, vector, slot_key),
        )


def _slot_key(event: ExperienceEvent) -> str:
    metadata = event.metadata
    explicit = metadata.get("slot_key") or metadata.get("canonical_key")
    if explicit:
        return str(explicit).strip().lower().replace(" ", "_")
    kind = str(metadata.get("type") or "observation").lower()
    keywords = [str(item).lower() for item in metadata.get("keywords", []) if str(item).strip()] if isinstance(metadata.get("keywords"), list) else []
    entities = [str(item).lower() for item in metadata.get("entities", []) if str(item).strip()] if isinstance(metadata.get("entities"), list) else []
    if kind in {"user_preference", "preference"}:
        return "user_preference:" + (keywords[0] if keywords else "general")
    if kind in {"rule", "procedure", "procedural"}:
        return "procedure:" + (keywords[0] if keywords else "general")
    if kind in {"constraint"}:
        return "constraint:" + (keywords[0] if keywords else "general")
    if keywords:
        return f"{kind}:{keywords[0]}"
    if entities:
        return f"{kind}:{entities[0]}"
    return f"{kind}:general"


def _candidate_memories(slot_key: str, memories: list[MemoryItem]) -> list[MemoryItem]:
    key_terms = set(slot_key.replace(":", " ").replace("_", " ").split())
    candidates: list[MemoryItem] = []
    for memory in memories:
        haystack = " ".join([memory.type, memory.content, *memory.keywords, *memory.entities, *memory.tags]).lower()
        if key_terms and key_terms & set(haystack.replace("_", " ").split()):
            candidates.append(memory)
    return candidates


def _candidate_frames(slot_key: str, frames: list[MemoryFrame]) -> list[MemoryFrame]:
    key_terms = set(slot_key.replace(":", " ").replace("_", " ").split())
    candidates: list[MemoryFrame] = []
    for frame in frames:
        haystack = " ".join([frame.frame_type, frame.content, frame.canonical_key]).lower()
        if key_terms and key_terms & set(haystack.replace("_", " ").split()):
            candidates.append(frame)
    return candidates


def _candidate_rows(slot_key: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    key_terms = set(slot_key.replace(":", " ").replace("_", " ").split())
    candidates: list[dict[str, object]] = []
    for row in rows:
        haystack = " ".join([str(row.get("slot_key", "")), str(row.get("slot_kind", "")), str(row.get("statement", ""))]).lower()
        if str(row.get("slot_key")) == slot_key or (key_terms and key_terms & set(haystack.replace("_", " ").split())):
            candidates.append(row)
    return candidates


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
    text = content.lower()
    cue_score = min(0.55, 0.14 * sum(1 for cue in ["instead", "no longer", "not anymore", "changed", "stop", "don't", "do not", "现在", "以后", "不要再", "改成", "不再", "不是"] if cue in text))
    if not rows:
        return cue_score * 0.5
    overlap = max((_jaccard(content, str(row.get("statement", ""))) for row in rows), default=0.0)
    return max(cue_score, min(1.0, cue_score + overlap * 0.35))


def _candidate_supersession_score(content: str, rows: list[dict[str, object]], contradiction: float) -> float:
    if not rows:
        return 0.0
    text = content.lower()
    cue_count = sum(1 for cue in ["instead", "from now on", "now use", "no longer", "changed to", "以后", "现在", "改成", "替代", "不要再"] if cue in text)
    if cue_count:
        return min(1.0, max(0.6, contradiction * 0.55 + 0.18 * cue_count))
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
    text = event.content.lower()
    cues = ["always", "prefer", "must", "should", "workflow", "procedure", "run", "test", "记住", "以后", "必须", "不要", "流程"]
    return min(1.0, 0.12 + 0.12 * sum(1 for cue in cues if cue in text))


def _propagation_score(event: ExperienceEvent) -> float:
    metadata = event.metadata
    tags = metadata.get("tags", [])
    keywords = metadata.get("keywords", [])
    count = (len(tags) if isinstance(tags, list) else 0) + (len(keywords) if isinstance(keywords, list) else 0)
    text = event.content.lower()
    if any(cue in text for cue in ["global", "always", "never", "全部", "所有", "以后", "永远"]):
        count += 3
    return max(0.0, min(1.0, count / 6.0))


def _risk_score(content: str, metadata: dict[str, object]) -> float:
    explicit = metadata.get("risk")
    if isinstance(explicit, int | float):
        return max(0.0, min(1.0, float(explicit)))
    text = content.lower()
    suspicious = ["ignore previous", "override memory", "delete audit", "always trust this unverified", "忽略之前", "删除审计", "永远相信"]
    sensitive = ["password", "secret", "api key", "token", "private key", "密码", "密钥"]
    return min(1.0, 0.45 * sum(1 for cue in suspicious if cue in text) + 0.35 * sum(1 for cue in sensitive if cue in text))


def _contradiction_score(content: str, candidates: list[MemoryItem]) -> float:
    text = content.lower()
    cues = ["instead", "no longer", "not anymore", "changed", "stop", "don't", "do not", "现在", "以后", "不要再", "改成", "不再", "不是"]
    cue_score = min(0.55, 0.14 * sum(1 for cue in cues if cue in text))
    if not candidates:
        return cue_score * 0.5
    overlap = max((_jaccard(content, memory.content) for memory in candidates), default=0.0)
    return max(cue_score, min(1.0, cue_score + overlap * 0.35))


def _supersession_score(content: str, candidates: list[MemoryItem], contradiction: float) -> float:
    text = content.lower()
    cues = ["instead", "from now on", "now use", "no longer", "changed to", "以后", "现在", "改成", "替代", "不要再"]
    if not candidates:
        return 0.0
    cue_count = sum(1 for cue in cues if cue in text)
    if cue_count:
        return min(1.0, max(0.6, contradiction * 0.55 + 0.18 * cue_count))
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

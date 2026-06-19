from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryFrame, MemoryItem
from neuromem.stores.sqlite_store import SQLiteMemoryStore
from neuromem_runtime.ledger import EdgeEvidenceEvent, MemoryLedger, WorldviewCandidateRecord, WorldviewSlotRecord


POSITIVE_EDGE_EVENTS = {"support", "reinforce", "restore", "generalize", "derive"}
NEGATIVE_EDGE_EVENTS = {"contradict", "supersede", "inhibit", "suppress", "decay", "expire"}
SUPPORTED_EDGE_EVENTS = POSITIVE_EDGE_EVENTS | NEGATIVE_EDGE_EVENTS
ASSOCIATIVE_RELATIONS = {"associated_with", "coactivated_with", "precedes", "retrieved_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}
LOGIC_RELATIONS = {"supports", "contradicts", "supersedes", "same_as", "generalizes", "specializes", "procedure_for", "preference_of", "applies_to", "evidence_for", "derived_from", "compresses_to", "causes", "inhibits"}


@dataclass(slots=True)
class EdgeAggregation:
    namespace: str
    source_kind: str
    source_id: str
    target_kind: str
    target_id: str
    relation: str
    relation_family: str
    weight: float = 0.0
    confidence: float = 0.0
    event_types: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    lifecycle_state: str = "captured"
    inhibition_score: float = 0.0
    contradiction_penalty: float = 0.0
    proof_obligations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MaterializedWorldview:
    namespace: str
    edge_count: int = 0
    slot_count: int = 0
    candidate_count: int = 0
    suppressed_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EdgeEvidenceAppender:
    def __init__(self, ledger: MemoryLedger) -> None:
        self._ledger = ledger

    def append(self, event: EdgeEvidenceEvent) -> EdgeEvidenceEvent:
        if event.event_type not in SUPPORTED_EDGE_EVENTS:
            raise ValueError(f"unsupported edge evidence event_type: {event.event_type}")
        return self._ledger.append_edge_evidence(event)


class EdgeWeightAggregator:
    def aggregate(self, events: list[dict[str, object]]) -> list[EdgeAggregation]:
        grouped: dict[tuple[str, str, str, str, str, str], EdgeAggregation] = {}
        for event in events:
            key = (
                str(event["namespace"]),
                str(event["source_kind"]),
                str(event["source_id"]),
                str(event["target_kind"]),
                str(event["target_id"]),
                str(event["relation"]),
            )
            aggregate = grouped.get(key)
            if aggregate is None:
                aggregate = EdgeAggregation(
                    namespace=key[0],
                    source_kind=key[1],
                    source_id=key[2],
                    target_kind=key[3],
                    target_id=key[4],
                    relation=key[5],
                    relation_family=str(event.get("relation_family") or "association"),
                )
                grouped[key] = aggregate
            event_type = str(event["event_type"])
            confidence = _clamp(float(event.get("confidence", 0.5)))
            delta = float(event.get("delta_weight", 0.0))
            if event_type in POSITIVE_EDGE_EVENTS and delta <= 0:
                delta = 0.35
            if event_type in NEGATIVE_EDGE_EVENTS and delta >= 0:
                delta = -0.45
            aggregate.weight += delta * confidence
            aggregate.confidence = max(aggregate.confidence, confidence)
            aggregate.event_types.append(event_type)
            aggregate.evidence_ids.extend(str(item) for item in event.get("evidence_ids", []) if str(item))
            proof = event.get("proof_obligation")
            if proof:
                aggregate.proof_obligations.append(str(proof))
            if event_type in {"inhibit", "suppress", "expire", "supersede"}:
                aggregate.lifecycle_state = "superseded" if event_type == "supersede" else "inhibited"
                aggregate.inhibition_score = max(aggregate.inhibition_score, abs(delta) * confidence)
            if event_type == "contradict":
                aggregate.lifecycle_state = "inhibited"
                aggregate.contradiction_penalty = max(aggregate.contradiction_penalty, abs(delta) * confidence)
            if event_type in {"restore", "reinforce"} and aggregate.lifecycle_state in {"inhibited", "expired", "superseded"}:
                aggregate.lifecycle_state = "reinforced"
        for aggregate in grouped.values():
            aggregate.weight = round(_clamp_signed(aggregate.weight), 4)
            aggregate.confidence = round(_clamp(aggregate.confidence), 4)
            aggregate.evidence_ids = sorted(set(aggregate.evidence_ids))
            aggregate.proof_obligations = sorted(set(aggregate.proof_obligations))
            if aggregate.weight <= -0.2 and aggregate.lifecycle_state == "captured":
                aggregate.lifecycle_state = "inhibited"
        return sorted(grouped.values(), key=lambda item: (item.namespace, item.source_kind, item.source_id, item.target_kind, item.target_id, item.relation))


class WorldviewMaterializer:
    def __init__(self, ledger: MemoryLedger) -> None:
        self._ledger = ledger
        self._aggregator = EdgeWeightAggregator()

    def rebuild(
        self,
        *,
        namespace: str,
        store: SQLiteMemoryStore | None,
        memories: list[MemoryItem] | None = None,
        frames: list[MemoryFrame] | None = None,
        clear_edges: bool = True,
    ) -> MaterializedWorldview:
        memory_items = memories if memories is not None else (store.list_memories(namespace=namespace) if store is not None else [])
        frame_items = frames if frames is not None else (store.list_logic_nodes(namespace=namespace) if store is not None else [])
        events = self._ledger.edge_evidence_events(namespace=namespace)
        aggregations = self._aggregator.aggregate(events)
        if store is not None and clear_edges:
            self._clear_materialized_edges(store.path, namespace)
            self._write_materialized_edges(store, aggregations)
        self._ledger.clear_worldview_materialization(namespace)
        suppressed = self._suppressed_memory_ids(events)
        suppressed_claims = self._suppressed_claim_ids(events)
        claim_candidates = self._materialize_claim_candidates(namespace=namespace, claims=self._ledger.grounded_claims(namespace=namespace), events=events, suppressed_claim_ids=suppressed_claims)
        candidates = claim_candidates + self._materialize_candidates(namespace=namespace, memories=memory_items, frames=frame_items, events=events, suppressed_memory_ids=suppressed)
        return MaterializedWorldview(
            namespace=namespace,
            edge_count=len([item for item in aggregations if abs(item.weight) > 0.001]),
            slot_count=len(self._ledger.worldview_slots(namespace=namespace)),
            candidate_count=candidates,
            suppressed_memory_ids=sorted(suppressed),
        )

    def _clear_materialized_edges(self, db_path: Path, namespace: str) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM associative_edges WHERE namespace = ?", (namespace,))
            conn.execute("DELETE FROM logic_edges WHERE namespace = ?", (namespace,))

    def _write_materialized_edges(self, store: SQLiteMemoryStore, aggregations: list[EdgeAggregation]) -> None:
        for aggregate in aggregations:
            if abs(aggregate.weight) <= 0.001:
                continue
            if aggregate.source_kind == "memory" and aggregate.target_kind == "memory":
                if aggregate.relation in LOGIC_RELATIONS and aggregate.relation not in ASSOCIATIVE_RELATIONS:
                    store.add_logic_edge(
                        LogicEdge(
                            namespace=aggregate.namespace,
                            source_frame_id=aggregate.source_id,
                            target_frame_id=aggregate.target_id,
                            source_memory_id=aggregate.source_id,
                            target_memory_id=aggregate.target_id,
                            relation=aggregate.relation,  # type: ignore[arg-type]
                            weight=max(0.0, aggregate.weight),
                            confidence=aggregate.confidence,
                            proof_obligation="; ".join(aggregate.proof_obligations[:3]) or "append-only memory relation evidence",
                            evidence_ids=list(aggregate.evidence_ids),
                            lifecycle_state=aggregate.lifecycle_state,  # type: ignore[arg-type]
                            inhibition_score=aggregate.inhibition_score,
                            contradiction_penalty=aggregate.contradiction_penalty,
                            proposer="worldview_materializer",
                        )
                    )
                else:
                    relation = aggregate.relation if aggregate.relation in ASSOCIATIVE_RELATIONS else "associated_with"
                    store.add_associative_edge(
                        AssociativeEdge(
                            namespace=aggregate.namespace,
                            source_id=aggregate.source_id,
                            target_id=aggregate.target_id,
                            relation=relation,  # type: ignore[arg-type]
                            weight=max(0.0, aggregate.weight),
                            confidence=aggregate.confidence,
                            success_count=aggregate.event_types.count("reinforce"),
                            failure_count=aggregate.event_types.count("contradict") + aggregate.event_types.count("inhibit"),
                            salience=max(0.0, min(1.0, abs(aggregate.weight))),
                            outcome_reward=aggregate.weight,
                            inhibition_score=aggregate.inhibition_score,
                            lifecycle_state=aggregate.lifecycle_state,  # type: ignore[arg-type]
                            provenance=list(aggregate.evidence_ids),
                        )
                    )
                continue
            if aggregate.source_kind == "frame" and aggregate.target_kind == "frame" and aggregate.relation in LOGIC_RELATIONS:
                store.add_logic_edge(
                    LogicEdge(
                        namespace=aggregate.namespace,
                        source_frame_id=aggregate.source_id,
                        target_frame_id=aggregate.target_id,
                        relation=aggregate.relation,  # type: ignore[arg-type]
                        weight=max(0.0, aggregate.weight),
                        confidence=aggregate.confidence,
                        proof_obligation="; ".join(aggregate.proof_obligations[:3]) or "append-only edge evidence",
                        evidence_ids=list(aggregate.evidence_ids),
                        lifecycle_state=aggregate.lifecycle_state,  # type: ignore[arg-type]
                        inhibition_score=aggregate.inhibition_score,
                        contradiction_penalty=aggregate.contradiction_penalty,
                        proposer="worldview_materializer",
                    )
                )

    def _materialize_candidates(
        self,
        *,
        namespace: str,
        memories: list[MemoryItem],
        frames: list[MemoryFrame],
        events: list[dict[str, object]],
        suppressed_memory_ids: set[str],
    ) -> int:
        count = 0
        event_index = _event_index(events)
        for frame in frames:
            slot_kind = _slot_kind_for_frame(frame)
            slot_key = frame.canonical_key or _canonical_key(frame.content, slot_kind)
            slot = self._ledger.upsert_worldview_slot(WorldviewSlotRecord(namespace=namespace, key=slot_key, kind=slot_kind, scope="global"))
            status = _candidate_status_for_frame(frame)
            if any(memory_id in suppressed_memory_ids for memory_id in frame.source_memory_ids):
                status = "suppressed"
            score_components = _score_components_for_frame(frame, event_index)
            score = _candidate_score(score_components)
            candidate = WorldviewCandidateRecord(
                namespace=namespace,
                slot_id=slot.slot_id,
                candidate_id=_candidate_id(namespace, slot.slot_id, "frame", frame.frame_id),
                statement=frame.content,
                value=json.dumps(frame.payload, ensure_ascii=False, sort_keys=True) if frame.payload else None,
                status=status,
                confidence=frame.confidence,
                valid_from=frame.valid_from.isoformat() if frame.valid_from else None,
                valid_to=frame.valid_to.isoformat() if frame.valid_to else None,
                source_frame_ids=[frame.frame_id],
                source_memory_ids=list(frame.source_memory_ids),
                evidence_ids=list(dict.fromkeys([*frame.evidence_ids, *frame.source_event_ids])),
                score=score,
                score_components=score_components,
            )
            self._ledger.upsert_worldview_candidate(candidate)
            count += 1
        framed_memory_ids = {memory_id for frame in frames for memory_id in frame.source_memory_ids}
        for memory in memories:
            if memory.id in framed_memory_ids and memory.id not in suppressed_memory_ids:
                continue
            slot_kind = _slot_kind_for_memory(memory)
            slot_key = _slot_key_for_memory(memory, slot_kind)
            slot = self._ledger.upsert_worldview_slot(WorldviewSlotRecord(namespace=namespace, key=slot_key, kind=slot_kind, scope="global"))
            status = "suppressed" if memory.id in suppressed_memory_ids else ("historical" if memory.maturity in {"obsolete", "archived", "deleted"} else "active")
            score_components = _score_components_for_memory(memory, event_index, suppressed=status == "suppressed")
            score = _candidate_score(score_components)
            candidate = WorldviewCandidateRecord(
                namespace=namespace,
                slot_id=slot.slot_id,
                candidate_id=_candidate_id(namespace, slot.slot_id, "memory", memory.id),
                statement=memory.summary or memory.content,
                status=status,
                confidence=memory.confidence,
                valid_from=memory.valid_from.isoformat() if memory.valid_from else None,
                valid_to=memory.valid_to.isoformat() if memory.valid_to else None,
                source_memory_ids=[memory.id],
                evidence_ids=list(memory.evidence),
                score=score,
                score_components=score_components,
            )
            self._ledger.upsert_worldview_candidate(candidate)
            count += 1
        return count

    def _materialize_claim_candidates(
        self,
        *,
        namespace: str,
        claims: list[dict[str, object]],
        events: list[dict[str, object]],
        suppressed_claim_ids: set[str],
    ) -> int:
        count = 0
        event_index = _event_index(events)
        for claim in claims:
            slot_kind = _slot_kind_for_claim(claim)
            slot_key = str(claim.get("canonical_slot_key") or "claim.general")
            slot = self._ledger.upsert_worldview_slot(WorldviewSlotRecord(namespace=namespace, key=slot_key, kind=slot_kind, scope="global"))
            status = _candidate_status_for_claim(claim, suppressed_claim_ids)
            score_components = _score_components_for_claim(claim, event_index, suppressed=status in {"suppressed", "historical", "rejected"})
            candidate = WorldviewCandidateRecord(
                namespace=namespace,
                slot_id=slot.slot_id,
                candidate_id=str(claim["claim_id"]),
                statement=str(claim.get("canonical_statement") or ""),
                value=json.dumps({"claim": claim}, ensure_ascii=False, sort_keys=True),
                status=status,
                confidence=float(claim.get("confidence", 0.5) or 0.5),
                source_frame_ids=[],
                source_memory_ids=[str(item) for item in claim.get("target_memory_ids", [])],
                evidence_ids=[str(item) for item in claim.get("evidence_ids", [])],
                score=_candidate_score(score_components),
                score_components=score_components,
            )
            self._ledger.upsert_worldview_candidate(candidate)
            count += 1
        return count

    def _suppressed_memory_ids(self, events: list[dict[str, object]]) -> set[str]:
        suppressed: set[str] = set()
        for event in events:
            if event.get("target_kind") != "memory":
                continue
            target_id = str(event["target_id"])
            if event["event_type"] in {"inhibit", "suppress", "supersede", "expire"}:
                suppressed.add(target_id)
            elif event["event_type"] in {"restore", "reinforce"}:
                suppressed.discard(target_id)
        return suppressed

    def _suppressed_claim_ids(self, events: list[dict[str, object]]) -> set[str]:
        suppressed: set[str] = set()
        for event in events:
            if event["target_kind"] != "claim":
                continue
            target_id = str(event["target_id"])
            if event["event_type"] in {"inhibit", "suppress", "supersede", "expire"}:
                suppressed.add(target_id)
            elif event["event_type"] in {"restore", "reinforce"}:
                suppressed.discard(target_id)
        return suppressed


def _event_index(events: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for event in events:
        target_id = str(event.get("target_id"))
        if not target_id:
            continue
        bucket = index.setdefault(target_id, {"support": 0.0, "contradiction": 0.0, "inhibition": 0.0, "supersession": 0.0, "utility_success": 0.0, "decay": 0.0})
        event_type = str(event.get("event_type"))
        strength = abs(float(event.get("delta_weight", 0.0)) or 0.35) * _clamp(float(event.get("confidence", 0.5)))
        if event_type in {"support", "derive", "generalize"}:
            bucket["support"] += strength
        elif event_type == "reinforce":
            bucket["support"] += strength
            bucket["utility_success"] += strength
        elif event_type == "contradict":
            bucket["contradiction"] += strength
        elif event_type in {"inhibit", "suppress", "expire"}:
            bucket["inhibition"] += strength
        elif event_type == "supersede":
            bucket["supersession"] += strength
        elif event_type == "decay":
            bucket["decay"] += strength
    return index


def _score_components_for_frame(frame: MemoryFrame, event_index: dict[str, dict[str, float]]) -> dict[str, object]:
    source_scores = [_target_scores(event_index, source_id) for source_id in [frame.frame_id, *frame.source_memory_ids]]
    merged = _merge_scores(source_scores)
    lifecycle = 0.28 if frame.commitment_level in {"validated_logic", "compiled_schema"} else 0.08
    user_confirmation = 0.16 if any(str(evidence).startswith("user:") for evidence in frame.evidence_ids) else 0.0
    return {
        "support_strength": min(1.0, frame.confidence * 0.45 + merged["support"]),
        "provenance_strength": min(1.0, 0.12 * len(set(frame.evidence_ids + frame.source_event_ids)) + 0.1 * len(set(frame.source_memory_ids))),
        "recency_validity": 0.85 if frame.lifecycle_state not in {"archived", "superseded", "inhibited"} else 0.2,
        "utility_success": min(1.0, merged["utility_success"]),
        "lifecycle_commitment": lifecycle,
        "user_confirmation": user_confirmation,
        "contradiction": min(1.0, merged["contradiction"]),
        "inhibition": min(1.0, merged["inhibition"]),
        "supersession": min(1.0, merged["supersession"]),
        "staleness": 0.45 if frame.lifecycle_state in {"archived", "superseded"} else 0.0,
    }


def _score_components_for_memory(memory: MemoryItem, event_index: dict[str, dict[str, float]], *, suppressed: bool) -> dict[str, object]:
    scores = _target_scores(event_index, memory.id)
    return {
        "support_strength": min(1.0, memory.confidence * 0.55 + scores["support"]),
        "provenance_strength": min(1.0, 0.12 * len(set(memory.evidence + memory.source_event_ids))),
        "recency_validity": max(0.0, 1.0 - float(memory.staleness_score)),
        "utility_success": min(1.0, float(memory.future_utility) + scores["utility_success"]),
        "lifecycle_commitment": 0.18 if memory.maturity in {"consolidated", "mature", "core"} else 0.08,
        "user_confirmation": 0.16 if memory.source_event_ids or any(str(evidence).startswith("evt_") for evidence in memory.evidence) else 0.0,
        "contradiction": min(1.0, float(memory.contradiction_score) + scores["contradiction"]),
        "inhibition": min(1.0, float(memory.inhibition_score) + scores["inhibition"] + (0.65 if suppressed else 0.0)),
        "supersession": min(1.0, scores["supersession"]),
        "staleness": min(1.0, float(memory.decay_score) + float(memory.staleness_score) + scores["decay"]),
    }


def _score_components_for_claim(claim: dict[str, object], event_index: dict[str, dict[str, float]], *, suppressed: bool) -> dict[str, object]:
    scores = _target_scores(event_index, str(claim.get("claim_id") or ""))
    confidence = _clamp(float(claim.get("confidence", 0.5) or 0.5))
    source_kind = str(claim.get("source_kind") or "")
    commitment = str(claim.get("commitment_level") or "")
    evidence_ids = [str(item) for item in claim.get("evidence_ids", [])] if isinstance(claim.get("evidence_ids"), list) else []
    derived_from_ids = [str(item) for item in claim.get("derived_from_ids", [])] if isinstance(claim.get("derived_from_ids"), list) else []
    source_reliability = {
        "observed_user_fact": 0.82,
        "tool_fact": 0.9,
        "llm_canonicalization": 0.68,
        "assistant_derivation": 0.42,
    }.get(source_kind, 0.5)
    lifecycle = {
        "raw_experience": 0.04,
        "candidate_frame": 0.08,
        "durable_memory": 0.2,
        "validated_logic": 0.3,
        "compiled_schema": 0.34,
    }.get(commitment, 0.08)
    return {
        "support_strength": min(1.0, confidence * source_reliability * 0.62 + scores["support"]),
        "provenance_strength": min(1.0, 0.18 * len(set(evidence_ids)) + 0.08 * len(set(derived_from_ids))),
        "recency_validity": 0.85 if not suppressed else 0.2,
        "utility_success": min(1.0, scores["utility_success"]),
        "lifecycle_commitment": lifecycle,
        "user_confirmation": 0.18 if source_kind == "observed_user_fact" else 0.0,
        "contradiction": min(1.0, scores["contradiction"]),
        "inhibition": min(1.0, scores["inhibition"] + (0.65 if suppressed else 0.0)),
        "supersession": min(1.0, scores["supersession"]),
        "staleness": min(1.0, scores["decay"]),
    }


def _candidate_score(components: dict[str, object]) -> float:
    positive = (
        0.24 * float(components["support_strength"])
        + 0.18 * float(components["provenance_strength"])
        + 0.14 * float(components["recency_validity"])
        + 0.14 * float(components["utility_success"])
        + 0.16 * float(components["lifecycle_commitment"])
        + 0.14 * float(components["user_confirmation"])
    )
    negative = (
        0.24 * float(components["contradiction"])
        + 0.22 * float(components["inhibition"])
        + 0.24 * float(components["supersession"])
        + 0.14 * float(components["staleness"])
    )
    return round(_clamp(positive - negative), 4)


def _target_scores(event_index: dict[str, dict[str, float]], target_id: str) -> dict[str, float]:
    return dict(event_index.get(target_id, {"support": 0.0, "contradiction": 0.0, "inhibition": 0.0, "supersession": 0.0, "utility_success": 0.0, "decay": 0.0}))


def _merge_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    merged = {"support": 0.0, "contradiction": 0.0, "inhibition": 0.0, "supersession": 0.0, "utility_success": 0.0, "decay": 0.0}
    for score in scores:
        for key in merged:
            merged[key] += float(score.get(key, 0.0))
    return merged


def _slot_kind_for_frame(frame: MemoryFrame) -> str:
    if frame.frame_type in {"fact", "claim", "entity"}:
        return "fact" if frame.frame_type != "claim" else "hypothesis"
    if frame.frame_type in {"preference", "constraint", "procedure", "schema", "failure_pattern"}:
        return "procedure" if frame.frame_type == "failure_pattern" else frame.frame_type
    return "hypothesis"


def _slot_kind_for_memory(memory: MemoryItem) -> str:
    if memory.type in {"preference", "constraint", "procedural", "schema"}:
        return "procedure" if memory.type == "procedural" else memory.type
    if memory.maturity in {"inhibited", "obsolete", "archived", "deleted"}:
        return "suppression"
    if memory.type in {"semantic", "fact"}:
        return "fact"
    return "hypothesis" if memory.type in {"working", "provisional"} else "fact"


def _slot_kind_for_claim(claim: dict[str, object]) -> str:
    kind = str(claim.get("claim_type") or "fact").lower()
    if kind in {"fact", "preference", "constraint", "procedure", "schema", "hypothesis", "suppression"}:
        return kind
    if kind in {"procedural", "rule"}:
        return "procedure"
    if kind in {"claim", "episode", "episodic", "entity"}:
        return "fact"
    return "hypothesis"


def _slot_key_for_memory(memory: MemoryItem, kind: str) -> str:
    if memory.keywords:
        return f"{kind}:{memory.keywords[0].lower()}"
    if memory.entities:
        return f"{kind}:{memory.entities[0].lower()}"
    return f"{kind}:memory_{memory.id}"


def _canonical_key(content: str, kind: str) -> str:
    return f"{kind}:hash_{_stable_key_hash({'kind': kind, 'content': content})[:16]}"


def _stable_key_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _candidate_status_for_frame(frame: MemoryFrame) -> str:
    if frame.lifecycle_state in {"inhibited", "archived"}:
        return "suppressed"
    if frame.lifecycle_state == "superseded":
        return "historical"
    if frame.commitment_level in {"validated_logic", "compiled_schema"} and frame.lifecycle_state in {"validated", "compiled", "mature"}:
        return "active"
    return "provisional"


def _candidate_status_for_claim(claim: dict[str, object], suppressed_claim_ids: set[str]) -> str:
    claim_id = str(claim.get("claim_id") or "")
    if claim_id in suppressed_claim_ids:
        return "suppressed"
    source_kind = str(claim.get("source_kind") or "")
    commitment = str(claim.get("commitment_level") or "")
    metadata = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}
    if source_kind == "assistant_derivation" and not bool(metadata.get("eligible_for_normal_prompt")) and commitment not in {"validated_logic", "compiled_schema"}:
        return "rejected"
    if commitment in {"validated_logic", "compiled_schema", "durable_memory"}:
        return "active"
    return "provisional"


def _candidate_id(namespace: str, slot_id: str, source_kind: str, source_id: str) -> str:
    raw = json.dumps([namespace, slot_id, source_kind, source_id], sort_keys=True, separators=(",", ":"))
    return "cand_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


__all__ = [
    "EdgeAggregation",
    "EdgeEvidenceAppender",
    "EdgeWeightAggregator",
    "MaterializedWorldview",
    "WorldviewMaterializer",
]

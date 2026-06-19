from __future__ import annotations

from dataclasses import asdict, dataclass, field

from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryFrame, MemoryItem


SLOT_KINDS = {"fact", "preference", "constraint", "procedure", "schema", "hypothesis", "suppression"}
NORMAL_HIDDEN_STATUSES = {"suppressed", "historical", "rejected"}


@dataclass(slots=True)
class WorldviewEvidenceChain:
    evidence_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)
    source_frame_ids: list[str] = field(default_factory=list)
    edge_event_ids: list[str] = field(default_factory=list)
    impact_ids: list[str] = field(default_factory=list)
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ResolvedWorldviewCandidate:
    candidate_id: str
    slot_key: str
    slot_kind: str
    statement: str
    score: float
    confidence: float
    status: str = "active"
    value: str | None = None
    score_components: dict[str, object] = field(default_factory=dict)
    evidence: WorldviewEvidenceChain = field(default_factory=WorldviewEvidenceChain)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["evidence"] = self.evidence.to_dict()
        return value


@dataclass(slots=True)
class WorldviewConflict:
    slot_key: str
    conflict_type: str
    candidate_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    delta: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ResolvedWorldviewSlot:
    slot_key: str
    kind: str
    candidates: list[ResolvedWorldviewCandidate] = field(default_factory=list)
    selected_candidate_id: str | None = None
    status: str = "active"
    conflict: WorldviewConflict | None = None

    @property
    def statement(self) -> str:
        if not self.candidates:
            return ""
        return self.candidates[0].statement

    @property
    def confidence(self) -> float:
        if not self.candidates:
            return 0.0
        return self.candidates[0].confidence

    @property
    def source_memory_ids(self) -> list[str]:
        ids: list[str] = []
        for candidate in self.candidates:
            ids.extend(candidate.evidence.source_memory_ids)
        return list(dict.fromkeys(ids))

    @property
    def evidence_ids(self) -> list[str]:
        ids: list[str] = []
        for candidate in self.candidates:
            ids.extend(candidate.evidence.evidence_ids)
        return list(dict.fromkeys(ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_key": self.slot_key,
            "kind": self.kind,
            "statement": self.statement,
            "confidence": self.confidence,
            "status": self.status,
            "source_memory_ids": self.source_memory_ids,
            "evidence_ids": self.evidence_ids,
            "selected_candidate_id": self.selected_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "conflict": self.conflict.to_dict() if self.conflict else None,
        }


@dataclass(slots=True)
class WorldviewPacket:
    namespace: str
    lens: str
    query: str
    slots: list[ResolvedWorldviewSlot] = field(default_factory=list)
    rejected_candidates: list[ResolvedWorldviewCandidate] = field(default_factory=list)
    conflicts: list[WorldviewConflict] = field(default_factory=list)
    supporting_memories: list[dict[str, object]] = field(default_factory=list)
    evidence_chains: list[WorldviewEvidenceChain] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def facts(self) -> list[ResolvedWorldviewSlot]:
        return [slot for slot in self.slots if slot.kind == "fact"]

    @property
    def preferences(self) -> list[ResolvedWorldviewSlot]:
        return [slot for slot in self.slots if slot.kind == "preference"]

    @property
    def constraints(self) -> list[ResolvedWorldviewSlot]:
        return [slot for slot in self.slots if slot.kind == "constraint"]

    @property
    def procedures(self) -> list[ResolvedWorldviewSlot]:
        return [slot for slot in self.slots if slot.kind in {"procedure", "schema"}]

    @property
    def suppressions(self) -> list[ResolvedWorldviewSlot]:
        return [slot for slot in self.slots if slot.kind == "suppression" or slot.status in {"suppressed", "historical"}]

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "lens": self.lens,
            "query": self.query,
            "slots": [slot.to_dict() for slot in self.slots],
            "facts": [slot.to_dict() for slot in self.facts],
            "preferences": [slot.to_dict() for slot in self.preferences],
            "constraints": [slot.to_dict() for slot in self.constraints],
            "procedures": [slot.to_dict() for slot in self.procedures],
            "suppressions": [slot.to_dict() for slot in self.suppressions],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "supporting_memories": list(self.supporting_memories),
            "evidence_chains": [chain.to_dict() for chain in self.evidence_chains],
            "rejected_candidates": [candidate.to_dict() for candidate in self.rejected_candidates],
            "warnings": list(self.warnings),
        }

    def to_prompt(self) -> str:
        lines = [
            "[Worldview Snapshot]",
            f"Scope: {self.namespace}",
            f"Lens: {self.lens}",
            "Generated from append-only evidence and materialized worldview candidates.",
        ]
        _extend(lines, "Current facts", self.facts)
        _extend(lines, "User preferences", self.preferences)
        _extend(lines, "Constraints", self.constraints)
        _extend(lines, "Procedures", self.procedures)
        _extend(lines, "Suppressions / stale assumptions", self.suppressions)
        if self.conflicts:
            lines.append("Open conflicts:")
            for conflict in self.conflicts[:5]:
                lines.append(f"- {conflict.slot_key}: {conflict.reason}")
        if self.supporting_memories:
            lines.append("Supporting memories:")
            for memory in self.supporting_memories[:8]:
                lines.append(f"- {memory.get('memory_id')}: {memory.get('content')}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"- {warning}")
        return "\n".join(lines)


class WorldviewResolver:
    def resolve(
        self,
        *,
        namespace: str,
        lens: str,
        query: str,
        memories: list[MemoryItem],
        frames: list[MemoryFrame] | None = None,
        associative_edges: list[AssociativeEdge] | None = None,
        logic_edges: list[LogicEdge] | None = None,
        edge_events: list[dict[str, object]] | None = None,
        impact_assessments: list[dict[str, object]] | None = None,
        worldview_candidates: list[dict[str, object]] | None = None,
        suppressed_memory_ids: set[str] | None = None,
    ) -> WorldviewPacket:
        resolved_lens = "associative" if lens == "auto" else lens
        packet = WorldviewPacket(namespace=namespace, lens=resolved_lens, query=query)
        suppressed = suppressed_memory_ids or set()
        edge_event_items = edge_events or []
        impact_items = impact_assessments or []
        candidates = self._candidates_from_materialized(worldview_candidates or [], edge_event_items=edge_event_items, impact_items=impact_items)
        if not candidates:
            candidates = self._fallback_candidates(memories, frames or [], suppressed, edge_event_items=edge_event_items, impact_items=impact_items)
        candidates = self._apply_lens(candidates, resolved_lens)
        grouped: dict[tuple[str, str], list[ResolvedWorldviewCandidate]] = {}
        rejected: list[ResolvedWorldviewCandidate] = []
        for candidate in candidates:
            if resolved_lens not in {"historical", "audit"} and candidate.status in NORMAL_HIDDEN_STATUSES:
                if candidate.status != "suppressed":
                    rejected.append(candidate)
                continue
            grouped.setdefault((candidate.slot_key, candidate.slot_kind), []).append(candidate)
        contradiction_conflicts = self._contradiction_conflicts(edge_event_items)
        for (slot_key, kind), items in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][0])):
            items = sorted(items, key=lambda item: (item.score, item.confidence), reverse=True)
            conflict = _score_conflict(slot_key, items)
            if slot_key in contradiction_conflicts:
                conflict = contradiction_conflicts[slot_key]
            status = "conflicted" if conflict is not None else items[0].status
            packet.slots.append(ResolvedWorldviewSlot(slot_key=slot_key, kind=kind, candidates=items[:5], selected_candidate_id=items[0].candidate_id if items else None, status=status, conflict=conflict))
            if conflict is not None:
                packet.conflicts.append(conflict)
        packet.rejected_candidates = rejected[:20] if resolved_lens == "audit" else []
        packet.supporting_memories = _supporting_memories(memories, packet.slots, include_suppressed=resolved_lens in {"historical", "audit"})
        if resolved_lens == "audit":
            packet.evidence_chains = [candidate.evidence for slot in packet.slots for candidate in slot.candidates]
        _trim(packet)
        if not packet.slots:
            packet.warnings.append("No worldview candidates resolved for this namespace.")
        return packet

    def _candidates_from_materialized(
        self,
        rows: list[dict[str, object]],
        *,
        edge_event_items: list[dict[str, object]],
        impact_items: list[dict[str, object]],
    ) -> list[ResolvedWorldviewCandidate]:
        edge_by_target = _edge_ids_by_target(edge_event_items)
        impact_ids = [str(item["impact_id"]) for item in impact_items if "impact_id" in item]
        candidates: list[ResolvedWorldviewCandidate] = []
        for row in rows:
            kind = str(row.get("slot_kind") or "hypothesis")
            if kind not in SLOT_KINDS:
                kind = "hypothesis"
            source_memory_ids = [str(item) for item in row.get("source_memory_ids", [])]
            source_frame_ids = [str(item) for item in row.get("source_frame_ids", [])]
            candidate_id = str(row["candidate_id"])
            candidates.append(
                ResolvedWorldviewCandidate(
                    candidate_id=candidate_id,
                    slot_key=str(row.get("slot_key") or f"{kind}:general"),
                    slot_kind=kind,
                    statement=str(row.get("statement") or ""),
                    value=str(row["value"]) if row.get("value") is not None else None,
                    score=round(float(row.get("score", 0.0)), 4),
                    confidence=round(float(row.get("confidence", 0.5)), 4),
                    status=str(row.get("status") or "provisional"),
                    score_components=dict(row.get("score_components", {})) if isinstance(row.get("score_components"), dict) else {},
                    evidence=WorldviewEvidenceChain(
                        evidence_ids=[str(item) for item in row.get("evidence_ids", [])],
                        source_memory_ids=source_memory_ids,
                        source_frame_ids=source_frame_ids,
                        edge_event_ids=sorted({edge_id for source_id in [candidate_id, *source_memory_ids, *source_frame_ids] for edge_id in edge_by_target.get(source_id, [])}),
                        impact_ids=impact_ids[:10],
                        lifecycle_events=[event for event in edge_event_items if event.get("target_id") in set(source_memory_ids + source_frame_ids)],
                    ),
                    reason="resolved from materialized worldview candidate",
                )
            )
        return candidates

    def _fallback_candidates(
        self,
        memories: list[MemoryItem],
        frames: list[MemoryFrame],
        suppressed: set[str],
        *,
        edge_event_items: list[dict[str, object]],
        impact_items: list[dict[str, object]],
    ) -> list[ResolvedWorldviewCandidate]:
        candidates: list[ResolvedWorldviewCandidate] = []
        for frame in frames:
            kind = _kind_from_frame(frame)
            status = _status_from_frame(frame)
            if any(memory_id in suppressed for memory_id in frame.source_memory_ids):
                status = "suppressed"
            candidates.append(
                ResolvedWorldviewCandidate(
                    candidate_id=frame.frame_id,
                    slot_key=frame.canonical_key or _slot_key(frame.content, kind),
                    slot_kind=kind,
                    statement=frame.content,
                    score=_frame_score(frame),
                    confidence=frame.confidence,
                    status=status,
                    evidence=WorldviewEvidenceChain(evidence_ids=list(frame.evidence_ids), source_memory_ids=list(frame.source_memory_ids), source_frame_ids=[frame.frame_id], impact_ids=[str(item["impact_id"]) for item in impact_items if "impact_id" in item][:10]),
                    reason="resolved from frame fallback",
                )
            )
        for memory in memories:
            kind = _kind_from_memory(memory)
            status = "suppressed" if memory.id in suppressed else ("historical" if memory.maturity in {"obsolete", "archived", "deleted"} else "active")
            candidates.append(
                ResolvedWorldviewCandidate(
                    candidate_id=memory.id,
                    slot_key=_memory_slot_key(memory, kind),
                    slot_kind=kind,
                    statement=memory.summary or memory.content,
                    score=_memory_score(memory, suppressed=status == "suppressed"),
                    confidence=memory.confidence,
                    status=status,
                    evidence=WorldviewEvidenceChain(evidence_ids=list(memory.evidence), source_memory_ids=[memory.id], lifecycle_events=[event for event in edge_event_items if event.get("target_id") == memory.id]),
                    reason="resolved from active memory fallback",
                )
            )
        return candidates

    def _apply_lens(self, candidates: list[ResolvedWorldviewCandidate], lens: str) -> list[ResolvedWorldviewCandidate]:
        if lens == "logical":
            return [candidate for candidate in candidates if candidate.slot_kind in {"fact", "preference", "constraint"} and candidate.status == "active"]
        if lens == "procedural":
            return [candidate for candidate in candidates if candidate.slot_kind in {"procedure", "schema"} or "failure" in candidate.slot_key]
        if lens == "historical":
            return candidates
        if lens == "audit":
            return candidates
        if lens == "associative":
            return [candidate for candidate in candidates if candidate.status != "rejected"]
        return candidates

    def _contradiction_conflicts(self, edge_events: list[dict[str, object]]) -> dict[str, WorldviewConflict]:
        conflicts: dict[str, WorldviewConflict] = {}
        for event in edge_events:
            if event.get("event_type") != "contradict":
                continue
            slot_key = str(event.get("relation") or event.get("target_id") or "unknown")
            conflicts[slot_key] = WorldviewConflict(
                slot_key=slot_key,
                conflict_type="active_contradiction_evidence",
                candidate_ids=[str(event.get("source_id")), str(event.get("target_id"))],
                evidence_ids=[str(item) for item in event.get("evidence_ids", [])],
                delta=1.0,
                reason="active contradict edge evidence is present",
            )
        return conflicts


def _score_conflict(slot_key: str, candidates: list[ResolvedWorldviewCandidate]) -> WorldviewConflict | None:
    if len(candidates) < 2:
        return None
    top, second = candidates[0], candidates[1]
    delta = round(abs(top.score - second.score), 4)
    if delta < 0.12 and second.status not in NORMAL_HIDDEN_STATUSES:
        return WorldviewConflict(
            slot_key=slot_key,
            conflict_type="close_top_candidates",
            candidate_ids=[top.candidate_id, second.candidate_id],
            evidence_ids=list(dict.fromkeys([*top.evidence.evidence_ids, *second.evidence.evidence_ids])),
            delta=delta,
            reason=f"top candidates differ by {delta:.2f}, below 0.12 conflict threshold",
        )
    return None


def _supporting_memories(memories: list[MemoryItem], slots: list[ResolvedWorldviewSlot], *, include_suppressed: bool) -> list[dict[str, object]]:
    wanted = {memory_id for slot in slots for memory_id in slot.source_memory_ids}
    items = []
    for memory in memories:
        if memory.id not in wanted:
            continue
        if not include_suppressed and memory.maturity in {"obsolete", "inhibited", "archived", "deleted"}:
            continue
        items.append({"memory_id": memory.id, "type": memory.type, "content": memory.content, "evidence_ids": list(memory.evidence)})
    return items[:12]


def _edge_ids_by_target(edge_events: list[dict[str, object]]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for event in edge_events:
        target = str(event.get("target_id"))
        edge_id = str(event.get("edge_event_id"))
        if target and edge_id:
            values.setdefault(target, []).append(edge_id)
    return values


def _kind_from_frame(frame: MemoryFrame) -> str:
    if frame.frame_type in {"fact", "entity"}:
        return "fact"
    if frame.frame_type == "claim":
        return "hypothesis"
    if frame.frame_type in {"preference", "constraint", "procedure", "schema"}:
        return frame.frame_type
    if frame.frame_type == "failure_pattern":
        return "procedure"
    return "hypothesis"


def _kind_from_memory(memory: MemoryItem) -> str:
    if memory.type == "preference":
        return "preference"
    if memory.type == "constraint":
        return "constraint"
    if memory.type in {"procedural", "schema"}:
        return "procedure" if memory.type == "procedural" else "schema"
    if memory.maturity in {"inhibited", "obsolete", "archived", "deleted"}:
        return "suppression"
    if memory.type == "semantic":
        return "fact"
    return "hypothesis" if memory.type in {"working", "provisional"} else "fact"


def _status_from_frame(frame: MemoryFrame) -> str:
    if frame.lifecycle_state in {"inhibited", "archived"}:
        return "suppressed"
    if frame.lifecycle_state == "superseded":
        return "historical"
    if frame.commitment_level in {"validated_logic", "compiled_schema"} and frame.lifecycle_state in {"validated", "compiled", "mature"}:
        return "active"
    return "provisional"


def _memory_slot_key(memory: MemoryItem, kind: str) -> str:
    if memory.keywords:
        return f"{kind}:{memory.keywords[0].lower()}"
    if memory.entities:
        return f"{kind}:{memory.entities[0].lower()}"
    return _slot_key(memory.summary or memory.content, kind)


def _slot_key(content: str, kind: str) -> str:
    terms = [term.strip(".,:;!?()[]{}\"'").lower() for term in content.split() if len(term.strip(".,:;!?()[]{}\"'")) > 3]
    return f"{kind}:{'_'.join(terms[:4]) or 'general'}"


def _memory_score(memory: MemoryItem, *, suppressed: bool) -> float:
    value = float(memory.confidence) + _num(memory.salience) * 0.2 + float(memory.future_utility) * 0.2
    value -= float(memory.decay_score) * 0.25 + float(memory.inhibition_score) * 0.45 + float(memory.staleness_score) * 0.2
    if suppressed:
        value -= 0.6
    return round(max(0.0, min(1.0, value)), 4)


def _frame_score(frame: MemoryFrame) -> float:
    value = float(frame.confidence)
    if frame.commitment_level in {"validated_logic", "compiled_schema"}:
        value += 0.18
    if frame.lifecycle_state in {"inhibited", "archived", "superseded"}:
        value -= 0.5
    return round(max(0.0, min(1.0, value)), 4)


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        nums = [float(item) for item in value.values() if isinstance(item, int | float)]
        return max(nums) if nums else default
    return default


def _trim(packet: WorldviewPacket) -> None:
    packet.slots = packet.slots[:24]
    packet.conflicts = packet.conflicts[:8]
    packet.supporting_memories = packet.supporting_memories[:12]
    packet.evidence_chains = packet.evidence_chains[:20]


def _extend(lines: list[str], title: str, slots: list[ResolvedWorldviewSlot]) -> None:
    if not slots:
        return
    lines.append(f"{title}:")
    for slot in slots[:8]:
        marker = " conflicted" if slot.conflict else ""
        lines.append(f"- [{slot.confidence:.2f}{marker}] {slot.statement}")


__all__ = [
    "ResolvedWorldviewCandidate",
    "ResolvedWorldviewSlot",
    "WorldviewConflict",
    "WorldviewEvidenceChain",
    "WorldviewPacket",
    "WorldviewResolver",
]

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from neuromem.core.models import MemoryItem


@dataclass(slots=True)
class ResolvedWorldviewSlot:
    slot_key: str
    statement: str
    confidence: float
    status: str = "active"
    source_memory_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class WorldviewPacket:
    namespace: str
    lens: str
    query: str
    facts: list[ResolvedWorldviewSlot] = field(default_factory=list)
    preferences: list[ResolvedWorldviewSlot] = field(default_factory=list)
    constraints: list[ResolvedWorldviewSlot] = field(default_factory=list)
    procedures: list[ResolvedWorldviewSlot] = field(default_factory=list)
    suppressions: list[ResolvedWorldviewSlot] = field(default_factory=list)
    conflicts: list[ResolvedWorldviewSlot] = field(default_factory=list)
    supporting_memories: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "lens": self.lens,
            "query": self.query,
            "facts": [slot.to_dict() for slot in self.facts],
            "preferences": [slot.to_dict() for slot in self.preferences],
            "constraints": [slot.to_dict() for slot in self.constraints],
            "procedures": [slot.to_dict() for slot in self.procedures],
            "suppressions": [slot.to_dict() for slot in self.suppressions],
            "conflicts": [slot.to_dict() for slot in self.conflicts],
            "supporting_memories": list(self.supporting_memories),
            "warnings": list(self.warnings),
        }

    def to_prompt(self) -> str:
        lines = [
            "[Worldview Snapshot]",
            f"Scope: {self.namespace}",
            f"Lens: {self.lens}",
            "Generated from governed append-only memory. Treat as current working assumptions, not immutable truth.",
        ]
        _extend(lines, "Current facts", self.facts)
        _extend(lines, "User preferences", self.preferences)
        _extend(lines, "Constraints", self.constraints)
        _extend(lines, "Procedures", self.procedures)
        _extend(lines, "Suppressions / stale assumptions", self.suppressions)
        _extend(lines, "Open conflicts", self.conflicts)
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
        suppressed_memory_ids: set[str] | None = None,
    ) -> WorldviewPacket:
        suppressed = suppressed_memory_ids or set()
        packet = WorldviewPacket(namespace=namespace, lens=lens, query=query)
        for memory in sorted(memories, key=_memory_rank, reverse=True):
            if memory.id in suppressed:
                packet.suppressions.append(_slot(memory, status="suppressed", reason="suppressed by append-only evidence"))
                continue
            target = _target_bucket(memory, packet)
            if target is None:
                continue
            target.append(_slot(memory, reason="resolved from active memory evidence"))
            packet.supporting_memories.append(
                {
                    "memory_id": memory.id,
                    "type": memory.type,
                    "content": memory.content,
                    "evidence_ids": list(memory.evidence),
                }
            )
        _trim(packet)
        return packet


def _target_bucket(memory: MemoryItem, packet: WorldviewPacket) -> list[ResolvedWorldviewSlot] | None:
    kind = memory.type.lower()
    tags = {tag.lower() for tag in memory.tags}
    if kind in {"preference", "user_preference"}:
        return packet.preferences
    if kind in {"procedural", "procedure", "rule"}:
        return packet.procedures
    if kind == "constraint" or "constraint" in tags:
        return packet.constraints
    if memory.maturity in {"obsolete", "inhibited", "archived", "deleted"}:
        return packet.suppressions
    if kind in {"semantic", "fact", "episodic", "task_result"}:
        return packet.facts
    return None


def _slot(memory: MemoryItem, *, status: str = "active", reason: str) -> ResolvedWorldviewSlot:
    return ResolvedWorldviewSlot(
        slot_key=_slot_key(memory),
        statement=memory.summary or memory.content,
        confidence=round(max(0.0, min(1.0, _num(memory.confidence) - _num(memory.decay_score) * 0.2 - _num(memory.inhibition_score) * 0.4)), 4),
        status=status,
        source_memory_ids=[memory.id],
        evidence_ids=list(memory.evidence),
        reason=reason,
    )


def _slot_key(memory: MemoryItem) -> str:
    if memory.keywords:
        return f"{memory.type}:{memory.keywords[0].lower()}"
    if memory.entities:
        return f"{memory.type}:{memory.entities[0].lower()}"
    return f"{memory.type}:general"


def _memory_rank(memory: MemoryItem) -> float:
    return _num(memory.confidence) + _num(memory.salience) * 0.3 + _num(memory.access_count) * 0.01 - _num(memory.decay_score) * 0.2 - _num(memory.inhibition_score) * 0.5


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        nums = [float(item) for item in value.values() if isinstance(item, int | float)]
        return max(nums) if nums else default
    return default


def _trim(packet: WorldviewPacket) -> None:
    packet.facts = packet.facts[:5]
    packet.preferences = packet.preferences[:5]
    packet.constraints = packet.constraints[:5]
    packet.procedures = packet.procedures[:5]
    packet.suppressions = packet.suppressions[:3]
    packet.conflicts = packet.conflicts[:3]
    packet.supporting_memories = packet.supporting_memories[:8]


def _extend(lines: list[str], title: str, slots: list[ResolvedWorldviewSlot]) -> None:
    if not slots:
        return
    lines.append(f"{title}:")
    for slot in slots:
        lines.append(f"- [{slot.confidence:.2f}] {slot.statement}")


__all__ = ["ResolvedWorldviewSlot", "WorldviewPacket", "WorldviewResolver"]

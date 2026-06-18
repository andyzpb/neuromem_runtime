from __future__ import annotations

from dataclasses import asdict, dataclass, field

from neuromem_runtime.deltas import LifecycleDelta, MemoryDelta


@dataclass(slots=True)
class SleepPlan:
    policy: str = "manual"
    replay_trace_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ReplayBatch:
    trace_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ConsolidationDelta:
    source_memory_ids: list[str] = field(default_factory=list)
    target_memory_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SuppressionDelta:
    memory_id: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CompilationDelta:
    source_memory_ids: list[str] = field(default_factory=list)
    compiled_type: str = "procedural"
    content: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LedgerReport:
    transaction_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SleepReport:
    plan: SleepPlan
    replay: ReplayBatch
    consolidation: list[ConsolidationDelta] = field(default_factory=list)
    suppression: list[SuppressionDelta] = field(default_factory=list)
    compilation: list[CompilationDelta] = field(default_factory=list)
    lifecycle: list[LifecycleDelta] = field(default_factory=list)
    memory_deltas: list[MemoryDelta] = field(default_factory=list)
    ledger: LedgerReport = field(default_factory=LedgerReport)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "replay": self.replay.to_dict(),
            "consolidation": [item.to_dict() for item in self.consolidation],
            "suppression": [item.to_dict() for item in self.suppression],
            "compilation": [item.to_dict() for item in self.compilation],
            "lifecycle": [item.to_dict() for item in self.lifecycle],
            "memory_deltas": [item.to_dict() for item in self.memory_deltas],
            "ledger": self.ledger.to_dict(),
        }


class SleepPlanner:
    def plan(self, *, policy: str = "manual", replay_trace_ids: list[str] | None = None) -> SleepPlan:
        return SleepPlan(policy=policy, replay_trace_ids=replay_trace_ids or [])


__all__ = [
    "CompilationDelta",
    "ConsolidationDelta",
    "LedgerReport",
    "ReplayBatch",
    "SleepPlan",
    "SleepPlanner",
    "SleepReport",
    "SuppressionDelta",
]

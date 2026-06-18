from __future__ import annotations

from dataclasses import dataclass

from neuromem_runtime.deltas import LifecycleDelta


_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "fresh": {"tagged", "captured", "inhibited", "obsolete", "archived"},
    "observed": {"provisional", "captured", "inhibited", "obsolete", "archived"},
    "tagged": {"captured", "inhibited", "obsolete", "archived"},
    "provisional": {"captured", "inhibited", "obsolete", "archived"},
    "captured": {"linked", "reinforced", "mature", "inhibited", "obsolete", "compressed", "archived"},
    "linked": {"reinforced", "consolidated", "mature", "inhibited", "obsolete", "compressed", "archived"},
    "reinforced": {"consolidated", "mature", "core", "inhibited", "obsolete", "compressed", "archived"},
    "consolidated": {"mature", "core", "inhibited", "obsolete", "compressed", "archived"},
    "mature": {"core", "inhibited", "obsolete", "compressed", "archived"},
    "core": {"inhibited", "obsolete", "compressed", "archived"},
    "inhibited": {"captured", "linked", "obsolete", "archived", "deleted"},
    "obsolete": {"archived", "deleted"},
    "compressed": {"archived", "deleted"},
    "archived": {"deleted"},
    "deleted": set(),
}


@dataclass(slots=True)
class LifecycleStateMachine:
    def validate_transition(self, from_state: str, to_state: str) -> bool:
        return to_state in _ALLOWED_TRANSITIONS.get(from_state, set()) or from_state == to_state

    def transition(
        self,
        memory_id: str,
        *,
        from_state: str,
        to_state: str,
        trigger: str,
        evidence: list[str] | None = None,
        reason: str = "",
    ) -> LifecycleDelta:
        if not self.validate_transition(from_state, to_state):
            raise ValueError(f"invalid lifecycle transition: {from_state} -> {to_state}")
        return LifecycleDelta(
            memory_id=memory_id,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            evidence=evidence or [],
            validator="LifecycleTransitionValidator",
            reason=reason,
            rollback_state=from_state,
        )


__all__ = ["LifecycleStateMachine"]

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkingMemory:
    max_items: int = 20
    active_task: str | None = None
    active_goals: list[str] = field(default_factory=list)
    recent_events: deque[dict[str, object]] = field(default_factory=deque)

    def set_task(self, task: str | None) -> None:
        self.active_task = task

    def add_event(self, event: dict[str, object]) -> None:
        self.recent_events.append(event)
        while len(self.recent_events) > self.max_items:
            self.recent_events.popleft()

    def snapshot(self) -> dict[str, object]:
        return {
            "active_task": self.active_task,
            "active_goals": list(self.active_goals),
            "recent_events": list(self.recent_events),
        }


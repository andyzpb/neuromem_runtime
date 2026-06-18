from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from neuromem.core.models import datetime_to_text, utcnow
from neuromem.core.policy import MemoryTrace


@dataclass(slots=True)
class MemoryTap:
    events: list[dict[str, object]] = field(default_factory=list)

    def emit(self, event_type: str, **payload: Any) -> dict[str, object]:
        event = {
            "event_id": f"tap_{uuid4().hex}",
            "type": event_type,
            "timestamp": datetime_to_text(utcnow()),
            **payload,
        }
        self.events.append(event)
        return event

    def attach(self, trace: MemoryTrace) -> MemoryTrace:
        trace.events.extend(self.events)
        return trace

    def reset(self) -> None:
        self.events.clear()

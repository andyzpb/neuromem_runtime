from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from neuromem.core.models import MemoryItem, utcnow


@dataclass(slots=True)
class CaptureDecision:
    captured: bool
    score: float
    reason: str


def tag_provisional(item: MemoryItem, *, tag_strength: float, ttl_seconds: int = 3600) -> MemoryItem:
    item.type = "provisional"
    item.maturity = "tagged"
    item.tag_strength = max(item.tag_strength, min(1.0, tag_strength))
    item.expires_at = utcnow() + timedelta(seconds=ttl_seconds)
    if not item.capture_conditions:
        item.capture_conditions = [
            "task_success",
            "task_failure",
            "user_confirmation",
            "recurrence",
            "high_prediction_error",
            "conflict_detected",
        ]
    if "provisional" not in item.tags:
        item.tags.append("provisional")
    return item


def maybe_capture(item: MemoryItem, event: dict[str, object]) -> CaptureDecision:
    outcome = str(event.get("outcome") or event.get("status") or "").lower()
    outcome_value = 1.0 if outcome == "success" else 0.55 if outcome in {"failure", "partial"} else 0.0
    recurrence = float(event.get("recurrence", 0.0) or 0.0)
    user_confirmation = 1.0 if bool(event.get("user_confirmation")) else 0.0
    prediction_error = float(event.get("prediction_error", item.prediction_error) or 0.0)
    capture_score = (
        0.30 * outcome_value
        + 0.25 * min(1.0, recurrence)
        + 0.20 * user_confirmation
        + 0.15 * min(1.0, prediction_error)
        + 0.10 * item.tag_strength
    )
    if capture_score >= 0.7:
        item.type = "episodic"
        item.maturity = "captured"
        item.valid_from = item.valid_from or utcnow()
        if "captured" not in item.tags:
            item.tags.append("captured")
        return CaptureDecision(True, round(capture_score, 3), "provisional trace captured")
    if item.expires_at is not None and utcnow() > item.expires_at:
        item.maturity = "archived"
        item.inhibition_score = max(item.inhibition_score, 0.7)
        return CaptureDecision(False, round(capture_score, 3), "provisional trace expired")
    return CaptureDecision(False, round(capture_score, 3), "capture threshold not reached")

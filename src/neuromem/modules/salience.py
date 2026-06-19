from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SalienceVector:
    novelty: float
    task_value: float
    surprise: float
    user_feedback: float
    failure_cost: float
    recurrence: float
    conflict: float

    def as_dict(self) -> dict[str, float]:
        return {
            "novelty": self.novelty,
            "task_value": self.task_value,
            "surprise": self.surprise,
            "user_feedback": self.user_feedback,
            "failure_cost": self.failure_cost,
            "recurrence": self.recurrence,
            "conflict": self.conflict,
        }

def compute_salience(event: dict[str, object], context: dict[str, object] | None = None) -> dict[str, float]:
    context = context or {}
    vector = SalienceVector(
        novelty=_score(event, "novelty", default=0.9 if event.get("type") in {"failure", "update", "user_preference"} else 0.4),
        surprise=_score(event, "surprise", default=0.3),
        task_value=_score(event, "task_value", default=0.7 if context.get("task") else 0.4),
        failure_cost=_score(event, "failure_cost", default=0.9 if event.get("outcome") == "failure" else 0.2),
        user_feedback=_score(event, "user_feedback", default=0.8 if event.get("source") == "user" else 0.1),
        recurrence=_score(event, "recurrence", default=0.7 if event.get("repeat", False) else 0.2),
        conflict=_score(event, "conflict", default=0.8 if event.get("conflict", False) else 0.1),
    )
    return vector.as_dict()


def salience_score(salience: dict[str, float]) -> float:
    score = (
        0.20 * salience.get("novelty", 0.0)
        + 0.15 * salience.get("surprise", 0.0)
        + 0.20 * salience.get("task_value", 0.0)
        + 0.15 * salience.get("failure_cost", 0.0)
        + 0.15 * salience.get("user_feedback", 0.0)
        + 0.10 * salience.get("recurrence", 0.0)
        - 0.10 * salience.get("conflict", 0.0)
    )
    return max(0.0, min(1.0, score))


def salience_vector(salience: dict[str, float]) -> SalienceVector:
    return SalienceVector(
        novelty=salience.get("novelty", 0.0),
        task_value=salience.get("task_value", 0.0),
        surprise=salience.get("surprise", 0.0),
        user_feedback=salience.get("user_feedback", 0.0),
        failure_cost=salience.get("failure_cost", 0.0),
        recurrence=salience.get("recurrence", 0.0),
        conflict=salience.get("conflict", 0.0),
    )


def _score(event: dict[str, object], key: str, *, default: float) -> float:
    value = event.get(key)
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    vector = event.get("salience")
    if isinstance(vector, dict) and isinstance(vector.get(key), int | float):
        return max(0.0, min(1.0, float(vector[key])))
    return default

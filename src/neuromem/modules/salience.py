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
    text = " ".join(str(value) for value in event.values())
    vector = SalienceVector(
        novelty=0.9 if event.get("type") in {"failure", "update", "user_preference"} else 0.4,
        surprise=0.8 if "unexpected" in text.lower() else 0.3,
        task_value=0.7 if context.get("task") else 0.4,
        failure_cost=0.9 if event.get("outcome") == "failure" else 0.2,
        user_feedback=0.8 if event.get("source") == "user" else 0.1,
        recurrence=0.7 if event.get("repeat", False) else 0.2,
        conflict=0.8 if event.get("conflict", False) else 0.1,
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

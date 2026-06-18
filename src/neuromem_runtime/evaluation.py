from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class LifecycleScorecard:
    encoding_quality: float = 0.0
    mutation_safety: float = 0.0
    retrieval_intelligence: float = 0.0
    plasticity_utility: float = 0.0
    lifecycle_adaptation: float = 0.0
    traceability_audit: float = 0.0

    def aggregate(self) -> float:
        values = [
            self.encoding_quality,
            self.mutation_safety,
            self.retrieval_intelligence,
            self.plasticity_utility,
            self.lifecycle_adaptation,
            self.traceability_audit,
        ]
        return round(sum(values) / len(values), 4)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["aggregate"] = self.aggregate()
        return value


__all__ = ["LifecycleScorecard"]

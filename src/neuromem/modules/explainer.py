from __future__ import annotations

from neuromem.core.models import MemoryItem


def explain_memory(item: MemoryItem) -> dict[str, object]:
    return {
        "memory_id": item.id,
        "type": item.type,
        "maturity": item.maturity,
        "confidence": item.confidence,
        "access_count": item.access_count,
        "salience": item.salience,
        "evidence": item.evidence,
        "inhibition_score": item.inhibition_score,
        "consolidation_count": item.consolidation_count,
        "valid_from": item.valid_from.isoformat() if item.valid_from else None,
        "valid_to": item.valid_to.isoformat() if item.valid_to else None,
        "why_suppressed": [
            reason
            for reason in [
                "memory is obsolete" if item.maturity == "obsolete" else None,
                "high inhibition score" if item.inhibition_score >= 0.8 else None,
            ]
            if reason
        ],
        "why_not_forgotten": [
            reason
            for reason in [
                "high task utility" if item.salience.get("task_value", 0.0) >= 0.5 else None,
                "high recurrence" if item.salience.get("recurrence", 0.0) >= 0.5 else None,
                "reinforced by prior use" if item.consolidation_count >= 1 else None,
            ]
            if reason
        ],
    }

from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.models import MemoryAction, MemoryItem
from neuromem.modules.lifecycle import obsolete, inhibit


@dataclass(slots=True)
class ForgetDecision:
    action: MemoryAction
    reason: str


def choose_forgetting_action(item: MemoryItem) -> ForgetDecision:
    if item.maturity == "deleted":
        return ForgetDecision("delete", "already deleted")
    if item.maturity == "obsolete":
        return ForgetDecision("invalidate", "superseded by newer evidence")
    if item.inhibition_score >= 0.8:
        return ForgetDecision("inhibit", "high inhibition score")
    if item.decay_score >= 0.7:
        return ForgetDecision("decay", "low utility over time")
    if item.consolidation_count >= 3:
        return ForgetDecision("compress", "ready for schema compression")
    return ForgetDecision("archive", "kept for provenance and audit")


def apply_forgetting(item: MemoryItem, decision: ForgetDecision, reason: str | None = None) -> MemoryItem:
    reason = reason or decision.reason
    if decision.action == "delete":
        item.maturity = "deleted"
    elif decision.action == "invalidate":
        obsolete(item, reason=reason)
    elif decision.action == "inhibit":
        inhibit(item, reason=reason)
    elif decision.action == "decay":
        item.decay_score = min(1.0, item.decay_score + 0.2)
        item.inhibition_score = min(1.0, item.inhibition_score + 0.1)
    elif decision.action == "compress":
        item.type = "schema"
        item.maturity = "archived"
    else:
        item.maturity = "archived"
    if reason not in item.evidence:
        item.evidence.append(reason)
    return item

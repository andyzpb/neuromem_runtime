from __future__ import annotations

from neuromem.core.models import Maturity, MemoryItem


PROMOTION_PATH: dict[Maturity, Maturity] = {
    "fresh": "linked",
    "tagged": "captured",
    "captured": "linked",
    "linked": "reinforced",
    "reinforced": "mature",
    "mature": "core",
    "core": "core",
    "compressed": "compressed",
    "inhibited": "inhibited",
    "obsolete": "obsolete",
    "archived": "archived",
    "deleted": "deleted",
}


def promote(item: MemoryItem, *, reason: str) -> MemoryItem:
    next_state = PROMOTION_PATH[item.maturity]
    item.maturity = next_state
    if reason and reason not in item.evidence:
        item.evidence.append(reason)
    return item


def inhibit(item: MemoryItem, *, reason: str, score: float = 0.8) -> MemoryItem:
    item.maturity = "inhibited"
    item.inhibition_score = max(item.inhibition_score, score)
    if reason not in item.evidence:
        item.evidence.append(reason)
    return item


def obsolete(item: MemoryItem, *, reason: str) -> MemoryItem:
    item.maturity = "obsolete"
    item.inhibition_score = max(item.inhibition_score, 0.9)
    item.confidence = min(item.confidence, 0.2)
    if reason not in item.evidence:
        item.evidence.append(reason)
    return item

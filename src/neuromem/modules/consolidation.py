from __future__ import annotations

from dataclasses import dataclass, field

from neuromem.core.models import MemoryItem
from neuromem.modules.lifecycle import promote


@dataclass(slots=True)
class ConsolidationReport:
    processed: int = 0
    promoted: int = 0
    compressed: int = 0
    archived: int = 0
    replay_clusters: list[list[str]] = field(default_factory=list)
    created_memory_ids: list[str] = field(default_factory=list)
    compressed_memory_ids: list[str] = field(default_factory=list)
    archived_memory_ids: list[str] = field(default_factory=list)
    consolidation_links: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "promoted": self.promoted,
            "compressed": self.compressed,
            "archived": self.archived,
            "replay_clusters": self.replay_clusters,
            "created_memory_ids": self.created_memory_ids,
            "compressed_memory_ids": self.compressed_memory_ids,
            "archived_memory_ids": self.archived_memory_ids,
            "consolidation_links": self.consolidation_links,
            "notes": self.notes,
        }


def consolidate(memories: list[MemoryItem]) -> ConsolidationReport:
    report = ConsolidationReport(processed=len(memories))
    replay_groups: dict[str, list[MemoryItem]] = {}
    for item in memories:
        if item.type == "episodic" and item.access_count >= 1:
            promote(item, reason="retrieved and reinforced during consolidation")
            report.promoted += 1
            report.created_memory_ids.append(item.id)
        if item.access_count >= 3 and item.type == "episodic":
            item.type = "procedural"
            item.summary = item.summary or item.content[:160]
            item.tags.append("consolidated-rule")
            report.compressed += 1
            report.compressed_memory_ids.append(item.id)
        if item.type == "semantic" and item.consolidation_count >= 1:
            item.maturity = "core"
        if item.type == "episodic" and item.maturity != "obsolete":
            key = " ".join(sorted(set((item.keywords or []) + (item.tags or [])))) or item.content.lower()[:48]
            replay_groups.setdefault(key, []).append(item)
        if item.type in {"working", "provisional"} and item.decay_score >= 0.8:
            item.maturity = "archived"
            report.archived += 1
            report.archived_memory_ids.append(item.id)
    for group in replay_groups.values():
        if len(group) < 2:
            continue
        cluster = [item.id for item in group]
        report.replay_clusters.append(cluster)
        strongest = max(group, key=lambda item: item.confidence)
        strongest.type = "procedural"
        strongest.maturity = "reinforced"
        strongest.summary = strongest.summary or f"Reusable rule from {len(group)} related episodes: {strongest.content}"
        strongest.consolidation_count += 1
        strongest.valid_from = strongest.valid_from or strongest.created_at
        strongest.supersedes.extend([item.id for item in group if item.id != strongest.id and item.id not in strongest.supersedes])
        strongest.derived_from.extend([item.id for item in group if item.id != strongest.id and item.id not in strongest.derived_from])
        if "consolidated-rule" not in strongest.tags:
            strongest.tags.append("consolidated-rule")
        report.compressed += 1
        report.compressed_memory_ids.append(strongest.id)
        report.consolidation_links[strongest.id] = cluster
        report.notes.append(f"promoted {strongest.id} from repeated episodes")
    return report

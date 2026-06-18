from __future__ import annotations

from itertools import combinations
from collections import deque

from neuromem.core.models import MemoryEdge, MemoryItem, utcnow
from neuromem.core.deltas import GraphDelta
from neuromem.stores.base import MemoryStore


def outcome_reward(outcome: str | None) -> float:
    normalized = (outcome or "unknown").lower()
    if normalized == "success":
        return 1.0
    if normalized == "partial":
        return 0.35
    if normalized == "failure":
        return -0.2
    return 0.1


def three_factor_delta(edge: MemoryEdge, *, outcome: str | None, salience: float, confidence: float, learning_rate: float = 0.25) -> float:
    reward = outcome_reward(outcome)
    decay = 0.01 * max(1, edge.coactivation_count)
    return learning_rate * edge.eligibility_trace * salience * reward * confidence - decay - edge.contradiction_penalty


def update_edges_after_use(
    store: MemoryStore,
    used_memories: list[MemoryItem],
    *,
    outcome: str | None,
    salience: float,
    confidence: float,
) -> list[GraphDelta]:
    updated: list[GraphDelta] = []
    if len(used_memories) < 2:
        return updated
    existing = {(edge.source_id, edge.target_id, edge.relation): edge for edge in store.list_edges()}
    for left, right in combinations(used_memories, 2):
        source_id, target_id = sorted([left.id, right.id])
        key = (source_id, target_id, "coactivated_with")
        edge = existing.get(key) or MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation="coactivated_with",
            confidence=min(left.confidence, right.confidence),
            weight=min(left.confidence, right.confidence) * 0.2,
            provenance=[left.id, right.id],
        )
        old_weight = edge.weight
        delta = three_factor_delta(edge, outcome=outcome, salience=salience, confidence=confidence)
        edge.weight = max(0.0, min(1.0, edge.weight + delta))
        edge.confidence = max(edge.confidence, min(1.0, confidence))
        edge.coactivation_count += 1
        edge.last_activated_at = utcnow()
        edge.eligibility_trace = min(1.0, edge.eligibility_trace + 0.1)
        if outcome == "success":
            edge.success_count += 1
        elif outcome == "failure":
            edge.failure_count += 1
            edge.contradiction_penalty = min(1.0, edge.contradiction_penalty + 0.05)
        for item, neighbor in [(left, right), (right, left)]:
            item.activation_count += 1
            item.coactivation_neighbors[neighbor.id] = edge.weight
            item.reinforcement_score = min(1.0, item.reinforcement_score + max(0.0, delta))
            store.upsert_memory(item)
        store.add_edge(edge)
        updated.append(
            GraphDelta(
                edge_id="|".join([edge.source_id, edge.target_id, edge.relation]),
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                old_weight=old_weight,
                new_weight=edge.weight,
                delta=edge.weight - old_weight,
                operation="update_edge",
                relation_family="activation",
                eligibility=edge.eligibility_trace,
                salience=salience,
                outcome_reward=outcome_reward(outcome),
                confidence=confidence,
                inhibition_penalty=edge.inhibition_score,
                contradiction_penalty=edge.contradiction_penalty,
                provenance=list(edge.provenance),
                evidence_ids=list(edge.provenance),
                reason="coactivation plasticity after governed policy execution",
            )
        )
    return updated


def graph_expand(memories: list[MemoryItem], store: MemoryStore, *, max_items: int) -> tuple[list[MemoryItem], list[list[str]], dict[str, float]]:
    by_id = {item.id: item for item in memories}
    selected = list(memories)
    paths: list[list[str]] = []
    graph_scores: dict[str, float] = {}
    all_edges = store.list_edges()
    for item in memories:
        for edge in all_edges:
            if item.id not in {edge.source_id, edge.target_id}:
                continue
            neighbor_id = edge.target_id if edge.source_id == item.id else edge.source_id
            neighbor = store.get_memory(neighbor_id)
            if neighbor is None or neighbor.id in by_id:
                continue
            if neighbor.maturity in {"deleted", "obsolete", "inhibited", "archived"}:
                continue
            by_id[neighbor.id] = neighbor
            selected.append(neighbor)
            paths.append([item.id, neighbor.id])
            graph_scores[neighbor.id] = max(graph_scores.get(neighbor.id, 0.0), edge.weight)
            if len(selected) >= max_items:
                return selected, paths, graph_scores
    return selected[:max_items], paths, graph_scores


def graph_diffuse(
    memories: list[MemoryItem],
    store: MemoryStore,
    *,
    max_items: int,
    depth: int = 2,
    restart_prob: float = 0.25,
    min_score: float = 0.03,
) -> tuple[list[MemoryItem], list[list[str]], dict[str, float]]:
    seeds = [item for item in memories if item.maturity not in {"deleted", "obsolete", "inhibited"}]
    selected = {item.id: item for item in seeds}
    frontier = deque([(item, [item.id], 1.0) for item in seeds])
    graph_scores = {item.id: 1.0 for item in seeds}
    paths: list[list[str]] = []
    queued_paths = {tuple([item.id]) for item in seeds}
    all_edges = store.list_edges()
    adjacency: dict[str, list[MemoryEdge]] = {}
    for edge in all_edges:
        adjacency.setdefault(edge.source_id, []).append(edge)
        adjacency.setdefault(edge.target_id, []).append(edge)
    while frontier:
        item, path, score = frontier.popleft()
        if len(path) > depth + 1:
            continue
        for edge in adjacency.get(item.id, []):
            neighbor_id = edge.target_id if edge.source_id == item.id else edge.source_id
            if neighbor_id in path:
                continue
            neighbor = store.get_memory(neighbor_id)
            if neighbor is None or neighbor.maturity in {"deleted", "obsolete", "inhibited", "archived"}:
                continue
            edge_score = edge.weight * edge.confidence * max(0.1, 1.0 - edge.inhibition_score) * max(0.1, 1.0 - edge.contradiction_penalty)
            propagated = score * (1.0 - restart_prob) * edge_score
            if propagated < min_score:
                continue
            next_path = path + [neighbor.id]
            if next_path not in paths:
                paths.append(next_path)
            if propagated > graph_scores.get(neighbor.id, 0.0):
                graph_scores[neighbor.id] = propagated
                selected[neighbor.id] = neighbor
            if len(next_path) <= depth + 1 and tuple(next_path) not in queued_paths:
                queued_paths.add(tuple(next_path))
                frontier.append((neighbor, next_path, propagated))
    ranked = sorted(selected.values(), key=lambda item: graph_scores.get(item.id, 0.0), reverse=True)
    return ranked[:max_items], paths, graph_scores

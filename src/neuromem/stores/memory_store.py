from __future__ import annotations

from dataclasses import replace

from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryEdge, MemoryFrame, MemoryItem
from neuromem.stores.base import MemoryStore


ASSOCIATIVE_RELATIONS = {"associated_with", "coactivated_with", "precedes", "retrieved_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._memories: dict[str, MemoryItem] = {}
        self._associative_edges: list[AssociativeEdge] = []
        self._logic_nodes: dict[str, MemoryFrame] = {}
        self._logic_edges: list[LogicEdge] = []

    def upsert_memory(self, item: MemoryItem) -> None:
        self._memories[item.id] = item

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        return self._memories.get(memory_id)

    def list_memories(self, namespace: str | None = None) -> list[MemoryItem]:
        values = list(self._memories.values())
        if namespace is None:
            return values
        return [item for item in values if item.namespace == namespace]

    def add_edge(self, edge: MemoryEdge) -> None:
        if edge.relation in ASSOCIATIVE_RELATIONS:
            self.add_associative_edge(
                AssociativeEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation=edge.relation,  # type: ignore[arg-type]
                    weight=edge.weight,
                    confidence=edge.confidence,
                    coactivation_count=edge.coactivation_count,
                    success_count=edge.success_count,
                    failure_count=edge.failure_count,
                    inhibition_score=edge.inhibition_score,
                    eligibility_trace=edge.eligibility_trace,
                    lifecycle_state=edge.lifecycle_state,
                    provenance=list(edge.provenance),
                    created_at=edge.created_at,
                    valid_from=edge.valid_from,
                    valid_to=edge.valid_to,
                )
            )
            return
        self.add_logic_edge(
            LogicEdge(
                source_frame_id=edge.source_id,
                target_frame_id=edge.target_id,
                source_memory_id=edge.source_id,
                target_memory_id=edge.target_id,
                relation=edge.relation,  # type: ignore[arg-type]
                weight=edge.weight,
                confidence=edge.confidence,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
                lifecycle_state=edge.lifecycle_state,
                inhibition_score=edge.inhibition_score,
                contradiction_penalty=edge.contradiction_penalty,
                evidence_ids=list(edge.provenance),
                created_at=edge.created_at,
            )
        )

    def list_edges(self, source_id: str | None = None) -> list[MemoryEdge]:
        edges = [edge.to_memory_edge() for edge in self._associative_edges]
        edges.extend(edge.to_memory_edge() for edge in self._logic_edges)
        if source_id is None:
            return edges
        return [edge for edge in edges if edge.source_id == source_id]

    def add_associative_edge(self, edge: AssociativeEdge) -> None:
        self._associative_edges = [item for item in self._associative_edges if item.edge_id() != edge.edge_id()]
        self._associative_edges.append(edge)

    def list_associative_edges(self, source_id: str | None = None, namespace: str | None = None) -> list[AssociativeEdge]:
        edges = list(self._associative_edges)
        if source_id is not None:
            edges = [edge for edge in edges if edge.source_id == source_id]
        if namespace is not None:
            edges = [edge for edge in edges if edge.namespace == namespace]
        return edges

    def add_logic_node(self, frame: MemoryFrame) -> None:
        self._logic_nodes[frame.frame_id] = frame

    def get_logic_node(self, frame_id: str) -> MemoryFrame | None:
        return self._logic_nodes.get(frame_id)

    def list_logic_nodes(self, namespace: str | None = None, *, lifecycle_state: str | None = None) -> list[MemoryFrame]:
        frames = list(self._logic_nodes.values())
        if namespace is not None:
            frames = [frame for frame in frames if frame.namespace == namespace]
        if lifecycle_state is not None:
            frames = [frame for frame in frames if frame.lifecycle_state == lifecycle_state]
        return frames

    def add_logic_edge(self, edge: LogicEdge) -> None:
        self._logic_edges = [item for item in self._logic_edges if item.edge_id() != edge.edge_id()]
        self._logic_edges.append(edge)

    def list_logic_edges(self, source_frame_id: str | None = None, namespace: str | None = None) -> list[LogicEdge]:
        edges = list(self._logic_edges)
        if source_frame_id is not None:
            edges = [edge for edge in edges if edge.source_frame_id == source_frame_id]
        if namespace is not None:
            edges = [edge for edge in edges if edge.namespace == namespace]
        return edges

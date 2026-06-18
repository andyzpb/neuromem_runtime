from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VectorIndex(Protocol):
    def upsert(self, items: dict[str, list[float]]) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], *, top_k: int = 8) -> list[tuple[str, float]]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class RetrievalTraceMetadata:
    retrieval_mode: str = "local_hybrid"
    embedding_mode: str = "disabled"
    embedding_model: str | None = None
    index_type: str = "sqlite"
    candidate_sources: list[str] = field(default_factory=lambda: ["lexical", "bm25", "graph"])
    fusion_strategy: str = "weighted_score"
    rank_before_fusion: list[str] = field(default_factory=list)
    rank_after_fusion: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "embedding_mode": self.embedding_mode,
            "embedding_model": self.embedding_model,
            "index_type": self.index_type,
            "candidate_sources": self.candidate_sources,
            "fusion_strategy": self.fusion_strategy,
            "rank_before_fusion": self.rank_before_fusion,
            "rank_after_fusion": self.rank_after_fusion,
        }


class DeterministicEmbeddingProvider:
    def __init__(self, dims: int = 16) -> None:
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            values = [0.0] * self.dims
            for index, char in enumerate(text.encode("utf-8")):
                values[index % self.dims] += float(char)
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            vectors.append([round(value / norm, 6) for value in values])
        return vectors


__all__ = ["DeterministicEmbeddingProvider", "EmbeddingProvider", "RetrievalTraceMetadata", "VectorIndex"]

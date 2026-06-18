from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from neuromem.retrieval.activation import (
    ActivationResult,
    MemoryCard,
    QueryPlanV2,
    RerankProvider,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalLedgerRecord,
)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VectorIndex(Protocol):
    def upsert(self, items: dict[str, list[float]], *, namespace: str = "default") -> None:
        raise NotImplementedError

    def search(self, vector: list[float], *, namespace: str | None = None, top_k: int = 8) -> list[tuple[str, float]]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class RetrievalTraceMetadata:
    retrieval_mode: str = "local_activation"
    embedding_mode: str = "disabled"
    embedding_model: str | None = None
    index_type: str = "sqlite"
    candidate_sources: list[str] = field(default_factory=lambda: ["fts5", "bm25", "lexical", "graph_seed"])
    fusion_strategy: str = "rrf+ppr+lite_rerank"
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


class LocalVectorIndex:
    """Small in-memory vector index for local-first tests and demos."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[str, list[float]]] = {}

    def upsert(self, items: dict[str, list[float]], *, namespace: str = "default") -> None:
        for memory_id, vector in items.items():
            self._vectors[memory_id] = (namespace, _normalize(vector))

    def search(self, vector: list[float], *, namespace: str | None = None, top_k: int = 8) -> list[tuple[str, float]]:
        query = _normalize(vector)
        scored: list[tuple[str, float]] = []
        for memory_id, (item_namespace, item_vector) in self._vectors.items():
            if namespace is not None and item_namespace != namespace:
                continue
            score = sum(left * right for left, right in zip(query, item_vector, strict=False))
            if score > 0:
                scored.append((memory_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]

    def delete(self, ids: list[str]) -> None:
        for memory_id in ids:
            self._vectors.pop(memory_id, None)


class QueryRewriteProvider(Protocol):
    def rewrite(self, query: str, *, namespace: str = "default") -> list[str]:
        raise NotImplementedError


class HyDEProvider(Protocol):
    def generate(self, query: str, *, namespace: str = "default") -> str | None:
        raise NotImplementedError


class EntityAliasResolver(Protocol):
    def expand(self, query: str, namespace: str) -> list[str]:
        raise NotImplementedError


class StaticEntityAliasResolver:
    def __init__(self, aliases: dict[str, list[str]] | None = None) -> None:
        self.aliases = aliases or {
            "auth": ["authentication", "login", "登录认证", "认证"],
            "login": ["auth", "authentication", "登录", "登录跳转"],
            "session": ["session refresh", "refresh token", "会话", "会话刷新"],
            "redirect": ["redirect loop", "跳转", "跳转循环"],
            "pytest": ["test command", "unit test", "单元测试命令"],
            "japan": ["日本", "Japan"],
            "tokyo": ["东京", "Tokyo"],
            "ginza": ["银座", "Ginza"],
            "osaka": ["大阪", "Osaka"],
            "travel": ["旅游", "旅行", "trip", "travel"],
            "sushi": ["寿司", "sushi"],
            "wasabi": ["芥末", "wasabi"],
            "habit": ["癖好", "习惯", "preference", "quirk", "habit"],
            "ben": ["Ben", "ben"],
        }

    def expand(self, query: str, namespace: str) -> list[str]:
        del namespace
        lowered = query.lower()
        expanded: list[str] = []
        for key, values in self.aliases.items():
            if key in lowered or any(value.lower() in lowered for value in values):
                expanded.extend([key, *values])
        return list(dict.fromkeys(value for value in expanded if value))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


__all__ = [
    "ActivationResult",
    "DeterministicEmbeddingProvider",
    "EmbeddingProvider",
    "EntityAliasResolver",
    "HyDEProvider",
    "LocalVectorIndex",
    "MemoryCard",
    "QueryPlanV2",
    "QueryRewriteProvider",
    "RerankProvider",
    "RetrievalCandidate",
    "RetrievalConfig",
    "RetrievalLedgerRecord",
    "RetrievalTraceMetadata",
    "StaticEntityAliasResolver",
    "VectorIndex",
]

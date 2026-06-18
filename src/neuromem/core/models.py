from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from typing import Literal
from uuid import uuid4

MemoryType = Literal["working", "provisional", "episodic", "semantic", "procedural", "schema", "preference", "constraint"]
Maturity = Literal[
    "fresh",
    "tagged",
    "provisional",
    "captured",
    "linked",
    "reinforced",
    "consolidated",
    "mature",
    "core",
    "inhibited",
    "obsolete",
    "compressed",
    "archived",
    "deleted",
]
MemoryAction = Literal["decay", "inhibit", "invalidate", "compress", "archive", "delete"]
MemoryRelation = Literal[
    "associated_with",
    "coactivated_with",
    "precedes",
    "causes",
    "supports",
    "contradicts",
    "same_as",
    "part_of",
    "derived_from",
    "compresses_to",
    "supersedes",
    "generalizes",
    "specializes",
    "evidence_for",
    "procedure_for",
    "preference_of",
    "retrieved_with",
    "inhibits",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_text(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def datetime_from_text(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class MemoryItem:
    id: str = field(default_factory=lambda: f"mem_{uuid4().hex}")
    agent_id: str = ""
    user_id: str | None = None
    namespace: str = "default"
    type: MemoryType = "episodic"
    content: str = ""
    summary: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_episode_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding_id: str | None = None
    links: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    salience: dict[str, float] = field(default_factory=dict)
    prediction_error: float = 0.0
    future_utility: float = 0.0
    confidence: float = 0.5
    maturity: Maturity = "fresh"
    consolidation_count: int = 0
    access_count: int = 0
    last_accessed_at: datetime | None = None
    activation_count: int = 0
    coactivation_neighbors: dict[str, float] = field(default_factory=dict)
    reinforcement_score: float = 0.0
    decay_score: float = 0.0
    inhibition_score: float = 0.0
    staleness_score: float = 0.0
    contradiction_score: float = 0.0
    tag_strength: float = 0.0
    expires_at: datetime | None = None
    capture_conditions: list[str] = field(default_factory=list)
    deletion_policy: str | None = None
    privacy_level: Literal["public", "agent", "user", "sensitive"] = "agent"
    acl: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "type": self.type,
            "content": self.content,
            "summary": self.summary,
            "created_at": datetime_to_text(self.created_at),
            "observed_at": datetime_to_text(self.observed_at),
            "valid_from": datetime_to_text(self.valid_from),
            "valid_to": datetime_to_text(self.valid_to),
            "source_episode_ids": self.source_episode_ids,
            "source_event_ids": self.source_event_ids,
            "evidence": self.evidence,
            "entities": self.entities,
            "keywords": self.keywords,
            "tags": self.tags,
            "embedding_id": self.embedding_id,
            "links": self.links,
            "supports": self.supports,
            "contradicts": self.contradicts,
            "supersedes": self.supersedes,
            "derived_from": self.derived_from,
            "salience": self.salience,
            "prediction_error": self.prediction_error,
            "future_utility": self.future_utility,
            "confidence": self.confidence,
            "maturity": self.maturity,
            "consolidation_count": self.consolidation_count,
            "access_count": self.access_count,
            "last_accessed_at": datetime_to_text(self.last_accessed_at),
            "activation_count": self.activation_count,
            "coactivation_neighbors": self.coactivation_neighbors,
            "reinforcement_score": self.reinforcement_score,
            "decay_score": self.decay_score,
            "inhibition_score": self.inhibition_score,
            "staleness_score": self.staleness_score,
            "contradiction_score": self.contradiction_score,
            "tag_strength": self.tag_strength,
            "expires_at": datetime_to_text(self.expires_at),
            "capture_conditions": self.capture_conditions,
            "deletion_policy": self.deletion_policy,
            "privacy_level": self.privacy_level,
            "acl": self.acl,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "MemoryItem":
        return cls(
            id=record["id"],
            agent_id=record.get("agent_id", ""),
            user_id=record.get("user_id"),
            namespace=record.get("namespace", "default"),
            type=record.get("type", "episodic"),
            content=record.get("content", ""),
            summary=record.get("summary"),
            created_at=datetime_from_text(record.get("created_at")) or utcnow(),
            observed_at=datetime_from_text(record.get("observed_at")),
            valid_from=datetime_from_text(record.get("valid_from")),
            valid_to=datetime_from_text(record.get("valid_to")),
            source_episode_ids=list(record.get("source_episode_ids", [])),
            source_event_ids=list(record.get("source_event_ids", [])),
            evidence=list(record.get("evidence", [])),
            entities=list(record.get("entities", [])),
            keywords=list(record.get("keywords", [])),
            tags=list(record.get("tags", [])),
            embedding_id=record.get("embedding_id"),
            links=list(record.get("links", [])),
            supports=list(record.get("supports", [])),
            contradicts=list(record.get("contradicts", [])),
            supersedes=list(record.get("supersedes", [])),
            derived_from=list(record.get("derived_from", [])),
            salience=dict(record.get("salience", {})),
            prediction_error=float(record.get("prediction_error", 0.0)),
            future_utility=float(record.get("future_utility", 0.0)),
            confidence=float(record.get("confidence", 0.5)),
            maturity=record.get("maturity", "fresh"),
            consolidation_count=int(record.get("consolidation_count", 0)),
            access_count=int(record.get("access_count", 0)),
            last_accessed_at=datetime_from_text(record.get("last_accessed_at")),
            activation_count=int(record.get("activation_count", 0)),
            coactivation_neighbors=dict(record.get("coactivation_neighbors", {})),
            reinforcement_score=float(record.get("reinforcement_score", 0.0)),
            decay_score=float(record.get("decay_score", 0.0)),
            inhibition_score=float(record.get("inhibition_score", 0.0)),
            staleness_score=float(record.get("staleness_score", 0.0)),
            contradiction_score=float(record.get("contradiction_score", 0.0)),
            tag_strength=float(record.get("tag_strength", 0.0)),
            expires_at=datetime_from_text(record.get("expires_at")),
            capture_conditions=list(record.get("capture_conditions", [])),
            deletion_policy=record.get("deletion_policy"),
            privacy_level=record.get("privacy_level", "agent"),
            acl=list(record.get("acl", [])),
        )


@dataclass(slots=True)
class MemoryEdge:
    source_id: str
    target_id: str
    relation: MemoryRelation
    weight: float = 0.0
    confidence: float = 0.5
    coactivation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_activated_at: datetime | None = None
    eligibility_trace: float = 1.0
    created_at: datetime = field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime | None = None
    recorded_at: datetime | None = None
    lifecycle_state: Literal["candidate", "provisional", "captured", "reinforced", "mature", "inhibited", "expired", "superseded"] = "captured"
    inhibition_score: float = 0.0
    contradiction_penalty: float = 0.0
    provenance: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "confidence": self.confidence,
            "coactivation_count": self.coactivation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_activated_at": datetime_to_text(self.last_activated_at),
            "eligibility_trace": self.eligibility_trace,
            "created_at": datetime_to_text(self.created_at),
            "valid_from": datetime_to_text(self.valid_from),
            "valid_to": datetime_to_text(self.valid_to),
            "observed_at": datetime_to_text(self.observed_at),
            "recorded_at": datetime_to_text(self.recorded_at),
            "lifecycle_state": self.lifecycle_state,
            "inhibition_score": self.inhibition_score,
            "contradiction_penalty": self.contradiction_penalty,
            "provenance": self.provenance,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "MemoryEdge":
        return cls(
            source_id=record["source_id"],
            target_id=record["target_id"],
            relation=record["relation"],
            weight=float(record.get("weight", record.get("confidence", 0.5))),
            confidence=float(record.get("confidence", 0.5)),
            coactivation_count=int(record.get("coactivation_count", 0)),
            success_count=int(record.get("success_count", 0)),
            failure_count=int(record.get("failure_count", 0)),
            last_activated_at=datetime_from_text(record.get("last_activated_at")),
            eligibility_trace=float(record.get("eligibility_trace", 1.0)),
            created_at=datetime_from_text(record.get("created_at")) or utcnow(),
            valid_from=datetime_from_text(record.get("valid_from")),
            valid_to=datetime_from_text(record.get("valid_to")),
            observed_at=datetime_from_text(record.get("observed_at")),
            recorded_at=datetime_from_text(record.get("recorded_at")),
            lifecycle_state=record.get("lifecycle_state", "captured"),
            inhibition_score=float(record.get("inhibition_score", 0.0)),
            contradiction_penalty=float(record.get("contradiction_penalty", 0.0)),
            provenance=list(record.get("provenance", [])),
        )


@dataclass(slots=True)
class MemoryQuery:
    query: str
    mode: str = "auto"
    filters: dict[str, object] = field(default_factory=dict)
    budget_tokens: int = 1500


@dataclass(slots=True)
class MemoryResult:
    memory: MemoryItem
    score: float
    why_retrieved: list[str] = field(default_factory=list)

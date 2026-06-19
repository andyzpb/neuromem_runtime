from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from neuromem_runtime.ledger import ExperienceEvent


ClaimSourceKind = Literal["observed_user_fact", "tool_fact", "llm_canonicalization", "assistant_derivation"]
ClaimCommitment = Literal["raw_experience", "candidate_frame", "durable_memory", "validated_logic", "compiled_schema"]
TRUTH_CLAIM_SOURCE_CHANNELS = frozenset({"current_user_message", "user_message", "tool_result"})
TRUTH_CLAIM_SOURCE_KINDS = frozenset({"observed_user_fact", "tool_fact", "llm_canonicalization"})


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class GroundedClaim:
    namespace: str
    event_id: str
    claim_type: str
    canonical_statement: str
    canonical_slot_key: str
    truth_source_event_ids: list[str]
    proposer: str = "deterministic"
    source_kind: ClaimSourceKind = "observed_user_fact"
    commitment_level: ClaimCommitment = "candidate_frame"
    confidence: float = 0.7
    evidence_ids: list[str] = field(default_factory=list)
    target_memory_ids: list[str] = field(default_factory=list)
    target_candidate_ids: list[str] = field(default_factory=list)
    derived_from_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    claim_id: str = ""
    created_at: str = field(default_factory=_now_text)

    def __post_init__(self) -> None:
        self.claim_type = _normalize_token(self.claim_type, default="fact")
        self.canonical_slot_key = _normalize_slot_key(self.canonical_slot_key)
        self.proposer = _normalize_token(self.proposer, default="deterministic")
        self.source_kind = _normalize_choice(
            self.source_kind,
            {"observed_user_fact", "tool_fact", "llm_canonicalization", "assistant_derivation"},
            "observed_user_fact",
        )  # type: ignore[assignment]
        self.commitment_level = _normalize_choice(
            self.commitment_level,
            {"raw_experience", "candidate_frame", "durable_memory", "validated_logic", "compiled_schema"},
            "candidate_frame",
        )  # type: ignore[assignment]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not self.evidence_ids:
            self.evidence_ids = list(self.truth_source_event_ids)
        if not self.claim_id:
            self.claim_id = "claim_" + _hash(
                {
                    "namespace": self.namespace,
                    "event_id": self.event_id,
                    "slot": self.canonical_slot_key,
                    "statement": self.canonical_statement,
                    "source_kind": self.source_kind,
                }
            )[:24]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GroundedClaimExtractor:
    """Validates explicit claim proposals without parsing natural language.

    Runtime stays mechanism-level: it records raw evidence and structured claims.
    Natural-language interpretation belongs to a provider/PFC that proposes
    `grounded_claims` with evidence pointers; the runtime then validates,
    journals, routes, and materializes those claims.
    """

    def extract(self, event: ExperienceEvent) -> list[GroundedClaim]:
        raw = event.metadata.get("grounded_claims")
        if not isinstance(raw, list):
            return []
        claims = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            claim = self.from_mapping(event, item)
            if claim is not None:
                claims.append(claim)
        return _dedupe_claims(claims)

    def from_mapping(self, event: ExperienceEvent, value: dict[str, object]) -> GroundedClaim | None:
        statement = str(value.get("canonical_statement") or value.get("statement") or "").strip()
        slot_key = str(value.get("canonical_slot_key") or value.get("slot_key") or "").strip()
        if not statement or not slot_key:
            return None
        truth_sources = _strings(value.get("truth_source_event_ids")) or [event.event_id]
        evidence_ids = _strings(value.get("evidence_ids")) or truth_sources
        return GroundedClaim(
            namespace=event.namespace,
            event_id=event.event_id,
            claim_type=str(value.get("claim_type") or "fact"),
            canonical_statement=statement,
            canonical_slot_key=slot_key,
            truth_source_event_ids=truth_sources,
            proposer=str(value.get("proposer") or "deterministic"),
            source_kind=str(value.get("source_kind") or _source_kind_for_event(event)),  # type: ignore[arg-type]
            commitment_level=str(value.get("commitment_level") or "candidate_frame"),  # type: ignore[arg-type]
            confidence=float(value.get("confidence", 0.7) or 0.7),
            evidence_ids=evidence_ids,
            target_memory_ids=_strings(value.get("target_memory_ids")),
            target_candidate_ids=_strings(value.get("target_candidate_ids")),
            derived_from_ids=_strings(value.get("derived_from_ids")),
            metadata=dict(value.get("metadata", {})) if isinstance(value.get("metadata"), dict) else {},
            claim_id=str(value.get("claim_id") or ""),
        )


def _source_kind_for_event(event: ExperienceEvent) -> ClaimSourceKind:
    source = event.source.lower()
    if source in {"tool", "tool_result", "repo"}:
        return "tool_fact"
    if source in {"assistant", "small_llm", "model"}:
        return "assistant_derivation"
    return "observed_user_fact"


def _normalize_token(value: str, *, default: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip()).strip("_")
    return cleaned or default


def _normalize_slot_key(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() or char in {".", ":", "-"} else "_" for char in str(value).strip()).strip("_")
    return cleaned or "claim.general"


def _normalize_choice(value: str, choices: set[str], default: str) -> str:
    normalized = _normalize_token(str(value), default=default)
    return normalized if normalized in choices else default


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_claims(claims: list[GroundedClaim]) -> list[GroundedClaim]:
    seen: set[tuple[str, str, str]] = set()
    values: list[GroundedClaim] = []
    for claim in claims:
        key = (claim.canonical_slot_key, claim.canonical_statement, claim.source_kind)
        if key in seen:
            continue
        seen.add(key)
        values.append(claim)
    return values


def claim_source_channel(claim: GroundedClaim | dict[str, object]) -> str:
    metadata = claim.metadata if isinstance(claim, GroundedClaim) else claim.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("source_channel") or "")


def claim_source_kind(claim: GroundedClaim | dict[str, object]) -> str:
    if isinstance(claim, GroundedClaim):
        return str(claim.source_kind)
    return str(claim.get("source_kind") or "")


def is_durable_truth_claim(claim: GroundedClaim | dict[str, object]) -> bool:
    return claim_source_channel(claim) in TRUTH_CLAIM_SOURCE_CHANNELS and claim_source_kind(claim) in TRUTH_CLAIM_SOURCE_KINDS


def durable_truth_claims(claims: list[GroundedClaim | dict[str, object]]) -> list[GroundedClaim | dict[str, object]]:
    return [claim for claim in claims if is_durable_truth_claim(claim)]


def durable_truth_claim_rejection_reason(claims: list[GroundedClaim | dict[str, object]]) -> str | None:
    if durable_truth_claims(claims):
        return None
    if not claims:
        return "durable mutation requires grounded claims from current user message or tool result"
    channels = sorted({claim_source_channel(claim) or "missing" for claim in claims})
    kinds = sorted({claim_source_kind(claim) or "missing" for claim in claims})
    return (
        "durable mutation requires at least one grounded claim with "
        f"metadata.source_channel in {sorted(TRUTH_CLAIM_SOURCE_CHANNELS)} and "
        f"source_kind in {sorted(TRUTH_CLAIM_SOURCE_KINDS)}; got channels={channels}, kinds={kinds}"
    )


__all__ = [
    "GroundedClaim",
    "GroundedClaimExtractor",
    "TRUTH_CLAIM_SOURCE_CHANNELS",
    "TRUTH_CLAIM_SOURCE_KINDS",
    "claim_source_channel",
    "claim_source_kind",
    "durable_truth_claim_rejection_reason",
    "durable_truth_claims",
    "is_durable_truth_claim",
]

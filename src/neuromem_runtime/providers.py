from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any, Protocol
from urllib import error, request

from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem_runtime.policy_v2 import MemoryPolicyV2


class PolicyProvider(Protocol):
    def propose(self, payload: Mapping[str, object]) -> MemoryPolicy | MemoryPolicyV2:
        raise NotImplementedError


@dataclass(slots=True)
class DeterministicPolicyProvider:
    def propose(self, payload: Mapping[str, object]) -> MemoryPolicy:
        task = str(payload.get("task") or payload.get("query") or payload.get("content") or "memory proposal")
        if payload.get("content") or payload.get("phase") == "after_step":
            evidence = str(payload.get("evidence") or payload.get("trace_id") or "local-proposal")
            return MemoryPolicy(
                retrieval=RetrievalPlan(enabled=False, query=task),
                write=WritePlan(
                    operation="ADD",
                    memory_type=str(payload.get("memory_type", "episodic")),
                    content=str(payload.get("content") or task),
                    confidence=float(payload.get("confidence", 0.75) or 0.75),
                    salience_estimate=float(payload.get("salience", 0.65) or 0.65),
                    evidence_ids=[evidence],
                    ttl="long_term",
                ),
                forget=ForgetPlan(operation="NOOP"),
                consolidation=ConsolidationPlan(enabled=False),
                reason="deterministic product proposal",
                source="deterministic",
            )
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=True, query=task, max_items=int(payload.get("max_items", 8) or 8), require_provenance=False),
            write=WritePlan(operation="NOOP"),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="deterministic product retrieval proposal",
            source="deterministic",
        )


@dataclass(slots=True)
class OpenAICompatiblePolicyProvider:
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com"
    timeout_seconds: float = 45.0

    def propose(self, payload: Mapping[str, object]) -> MemoryPolicy | MemoryPolicyV2:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are NeuroMem Memory PFC. Return JSON only. Prefer MemoryPolicyV2 with keys "
                        "policy_id, proposer, proposal_source, intent, risk_level, evidence_chain, "
                        "target_selector, grounded_claims, proposed_deltas, safety_annotations, temporal_scope, "
                        "retention_policy, rollback_plan. proposed_deltas must contain operation, "
                        "target_memory_id, field, value, reason. grounded_claims must contain "
                        "claim_type, canonical_statement, canonical_slot_key, source_kind, "
                        "commitment_level, confidence, and evidence pointers. Treat user/tool input as "
                        "truth evidence; assistant text may only be llm_canonicalization or "
                        "assistant_derivation with derived_from_ids. Do not use UPDATE for product "
                        "corrections; express corrections as grounded_claims with target_memory_ids or "
                        "target_candidate_ids plus supersession/inhibition intent. Use legacy retrieval/"
                        "write/forget/consolidation/reason only if V2 is impossible. Never request "
                        "deletion unless the payload explicitly authorizes deletion."
                    ),
                },
                {"role": "user", "content": json.dumps(dict(payload), sort_keys=True)},
            ],
        }
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._resolved_key()}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"policy provider request failed with HTTP {exc.code}: {detail[:300]}") from exc
        content = response_payload["choices"][0]["message"]["content"]
        return memory_policy_from_payload(extract_policy_payload(content), source="small_llm")

    def _resolved_key(self) -> str:
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is required for LLM policy proposals")
        return key


@dataclass(slots=True)
class DeepSeekPolicyProvider(OpenAICompatiblePolicyProvider):
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"


def extract_policy_payload(value: object) -> dict[str, Any]:
    candidates: list[str] = []
    if isinstance(value, Mapping):
        candidates.append(json.dumps(value))
    elif isinstance(value, str):
        candidates.append(value)
    else:
        content = getattr(value, "content", None)
        if isinstance(content, str):
            candidates.append(content)
        candidates.append(str(value))
    decoder = JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                keys = set(payload)
                if {"policy_id", "proposed_deltas", "grounded_claims"} & keys or {"retrieval", "write", "forget", "consolidation", "reason"} <= keys:
                    return payload
    raise ValueError("No valid MemoryPolicy JSON payload found")


def memory_policy_from_payload(payload: Mapping[str, Any], *, source: str = "small_llm") -> MemoryPolicy | MemoryPolicyV2:
    payload = normalize_policy_payload(payload, source=source)
    if "policy_id" in payload or "proposed_deltas" in payload or "grounded_claims" in payload:
        return MemoryPolicyV2.model_validate(dict(payload))
    return MemoryPolicy(
        retrieval=RetrievalPlan(**dict(payload["retrieval"])),
        write=WritePlan(**dict(payload["write"])),
        forget=ForgetPlan(**dict(payload["forget"])),
        consolidation=ConsolidationPlan(**dict(payload["consolidation"])),
        reason=str(payload["reason"]),
        source="small_llm" if source == "small_llm" else "deterministic",
    )


def normalize_policy_payload(payload: Mapping[str, Any], *, source: str = "small_llm") -> dict[str, Any]:
    value = dict(payload)
    if set(value) == {"after_step"} and isinstance(value.get("after_step"), Mapping):
        value = dict(value["after_step"])
    elif isinstance(value.get("after_step"), Mapping) and not ({"policy_id", "proposed_deltas", "grounded_claims"} & set(value)):
        value = dict(value["after_step"])
    if "policy_id" not in value and "proposed_deltas" not in value and "grounded_claims" not in value:
        return value

    value.setdefault("policy_id", f"policy_{abs(hash(json.dumps(value, sort_keys=True, default=str)))}")
    value.setdefault("proposer", "small_llm")
    proposal_source = str(value.get("proposal_source") or source).lower()
    if proposal_source not in {"deterministic", "small_llm", "user", "system", "tool", "admin"}:
        proposal_source = "small_llm" if source == "small_llm" else "deterministic"
    value["proposal_source"] = proposal_source
    value.setdefault("risk_level", "low")
    value["temporal_scope"] = _normalize_temporal_scope(value.get("temporal_scope"))
    value.setdefault("retention_policy", "keep_for_audit")
    value["target_selector"] = _normalize_target_selector(value.get("target_selector"))
    value["evidence_chain"] = _normalize_evidence_chain(value.get("evidence_chain"))
    value["proposed_deltas"] = _normalize_proposed_deltas(value.get("proposed_deltas"))
    value["grounded_claims"] = _normalize_grounded_claims(value.get("grounded_claims"))
    value["safety_annotations"] = dict(value.get("safety_annotations")) if isinstance(value.get("safety_annotations"), Mapping) else {}
    gate = _normalize_write_gate(value.get("write_gate"))
    if gate:
        value["write_gate"] = gate
        value["safety_annotations"]["write_gate"] = dict(gate)
    return value


def _normalize_target_selector(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {"memory_ids", "query", "namespace"}
    normalized = {key: item for key, item in dict(value).items() if key in allowed}
    memory_ids = normalized.get("memory_ids")
    if isinstance(memory_ids, list):
        normalized["memory_ids"] = [str(item) for item in memory_ids if item]
    elif memory_ids:
        normalized["memory_ids"] = [str(memory_ids)]
    return normalized


def _normalize_temporal_scope(value: object) -> str:
    if value is None or value == "":
        return "all_valid"
    if isinstance(value, str):
        return value or "all_valid"
    if isinstance(value, Mapping):
        start = value.get("from") or value.get("start") or value.get("valid_from")
        end = value.get("to") or value.get("end") or value.get("valid_to")
        if start or end:
            return f"{start or '*'}..{end or '*'}"
        return "all_valid"
    if isinstance(value, list):
        parts = [str(item) for item in value if item]
        return ",".join(parts) if parts else "all_valid"
    return str(value)


def _normalize_evidence_chain(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            event_id = item.get("event_id") or item.get("id")
            if event_id:
                refs.append({"event_id": str(event_id), "source": str(item.get("source") or "unknown")})
        elif item:
            refs.append({"event_id": str(item), "source": "unknown"})
    return refs


def _normalize_proposed_deltas(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    deltas: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        delta = dict(item)
        operation = str(delta.get("operation") or "ADD").upper()
        normalized: dict[str, object] = {
            "operation": operation,
            "target_memory_id": delta.get("target_memory_id"),
            "field": delta.get("field"),
            "reason": str(delta.get("reason") or delta.get("rationale") or ""),
        }
        value_obj = delta.get("value")
        if not isinstance(value_obj, Mapping):
            content = delta.get("content")
            if content is not None:
                value_obj = {
                    "content": str(content),
                    "memory_type": str(delta.get("memory_type") or delta.get("type") or "episodic"),
                }
        normalized["value"] = dict(value_obj) if isinstance(value_obj, Mapping) else value_obj
        deltas.append(normalized)
    return deltas


def _normalize_grounded_claims(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    claims: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        statement = item.get("canonical_statement") or item.get("statement")
        slot_key = item.get("canonical_slot_key") or item.get("slot_key")
        if not statement or not slot_key:
            continue
        claims.append(
            {
                "claim_id": str(item.get("claim_id") or ""),
                "claim_type": str(item.get("claim_type") or "fact"),
                "canonical_statement": str(statement),
                "canonical_slot_key": str(slot_key),
                "truth_source_event_ids": [str(value) for value in item.get("truth_source_event_ids", [])] if isinstance(item.get("truth_source_event_ids"), list) else [],
                "proposer": str(item.get("proposer") or "small_llm"),
                "source_kind": str(item.get("source_kind") or "llm_canonicalization"),
                "commitment_level": str(item.get("commitment_level") or "candidate_frame"),
                "confidence": float(item.get("confidence", 0.7) or 0.7),
                "evidence_ids": [str(value) for value in item.get("evidence_ids", [])] if isinstance(item.get("evidence_ids"), list) else [],
                "target_memory_ids": [str(value) for value in item.get("target_memory_ids", [])] if isinstance(item.get("target_memory_ids"), list) else [],
                "target_candidate_ids": [str(value) for value in item.get("target_candidate_ids", [])] if isinstance(item.get("target_candidate_ids"), list) else [],
                "derived_from_ids": [str(value) for value in item.get("derived_from_ids", [])] if isinstance(item.get("derived_from_ids"), list) else [],
                "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata"), Mapping) else {},
            }
        )
    return claims


def _normalize_write_gate(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    gate = dict(value)
    decision = str(gate.get("decision") or "noop").lower()
    if decision not in {"commit", "defer", "noop"}:
        decision = "noop"
    horizon = str(gate.get("durability_horizon") or "none").lower()
    if horizon in {"permanent", "durable"}:
        horizon = "long_term"
    if horizon not in {"none", "thread", "cross_thread", "long_term"}:
        horizon = "none"
    level = str(gate.get("commitment_level") or "raw_experience").lower()
    if level in {"high", "medium", "low", "memory"}:
        level = "durable_memory" if decision == "commit" else "raw_experience"
    if level not in {"raw_experience", "durable_memory", "associative_link", "candidate_frame", "validated_logic", "compiled_schema"}:
        level = "raw_experience"
    basis = str(gate.get("basis") or "current_user_message").lower()
    basis_map = {
        "explicit user fact": "current_user_message",
        "user_message": "current_user_message",
        "conversation": "short_term_context",
        "retrieved": "retrieved_memory",
    }
    basis = basis_map.get(basis, basis)
    if basis not in {"current_user_message", "short_term_context", "retrieved_memory", "tool_result", "system"}:
        basis = "current_user_message"
    signals = gate.get("signals")
    return {
        "decision": decision,
        "durability_horizon": horizon,
        "commitment_level": level,
        "basis": basis,
        "signals": [str(item) for item in signals] if isinstance(signals, list) else [],
        "rationale": str(gate.get("rationale") or ""),
    }


__all__ = [
    "PolicyProvider",
    "DeterministicPolicyProvider",
    "OpenAICompatiblePolicyProvider",
    "DeepSeekPolicyProvider",
    "extract_policy_payload",
    "memory_policy_from_payload",
    "normalize_policy_payload",
]

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
    def propose(self, payload: Mapping[str, object]) -> MemoryPolicy:
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

    def propose(self, payload: Mapping[str, object]) -> MemoryPolicy:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are NeuroMem Memory PFC. Return JSON only with keys retrieval, write, forget, "
                        "consolidation, reason. The JSON must instantiate RetrievalPlan, WritePlan, "
                        "ForgetPlan, ConsolidationPlan. Never request deletion unless the payload explicitly "
                        "authorizes deletion."
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
            if isinstance(payload, dict) and {"retrieval", "write", "forget", "consolidation", "reason"} <= set(payload):
                return payload
    raise ValueError("No valid MemoryPolicy JSON payload found")


def memory_policy_from_payload(payload: Mapping[str, Any], *, source: str = "small_llm") -> MemoryPolicy:
    if "policy_id" in payload or "proposed_deltas" in payload:
        MemoryPolicyV2.model_validate(dict(payload))
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False),
            write=WritePlan(operation="NOOP"),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="validated MemoryPolicyV2 proposal requires v2 executor",
            source="small_llm" if source == "small_llm" else "deterministic",
        )
    return MemoryPolicy(
        retrieval=RetrievalPlan(**dict(payload["retrieval"])),
        write=WritePlan(**dict(payload["write"])),
        forget=ForgetPlan(**dict(payload["forget"])),
        consolidation=ConsolidationPlan(**dict(payload["consolidation"])),
        reason=str(payload["reason"]),
        source="small_llm" if source == "small_llm" else "deterministic",
    )


__all__ = [
    "PolicyProvider",
    "DeterministicPolicyProvider",
    "OpenAICompatiblePolicyProvider",
    "DeepSeekPolicyProvider",
    "extract_policy_payload",
    "memory_policy_from_payload",
]

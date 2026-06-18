from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any
from urllib import error, request

from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


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
    return MemoryPolicy(
        retrieval=RetrievalPlan(**dict(payload["retrieval"])),
        write=WritePlan(**dict(payload["write"])),
        forget=ForgetPlan(**dict(payload["forget"])),
        consolidation=ConsolidationPlan(**dict(payload["consolidation"])),
        reason=str(payload["reason"]),
        source="small_llm" if source == "small_llm" else "deterministic",
    )


@dataclass(slots=True)
class FakeLangGraphPFC:
    policy: MemoryPolicy
    prompts: list[str] | None = None

    def _record(self, prompt: str) -> MemoryPolicy:
        if self.prompts is not None:
            self.prompts.append(prompt)
        return self.policy

    def plan_before_step(self, task: str) -> MemoryPolicy:
        return self._record(f"Plan before-step memory policy for: {task}")

    def plan_after_step(self, task: str) -> MemoryPolicy:
        return self._record(f"Plan after-step memory policy for: {task}")

    def plan_consolidation(self, task: str) -> MemoryPolicy:
        return self._record(f"Plan consolidation memory policy for: {task}")

    def plan_forgetting(self, task: str) -> MemoryPolicy:
        return self._record(f"Plan forgetting memory policy for: {task}")


@dataclass(slots=True)
class DeepSeekLangGraphPFC:
    api_key: str | None = None
    model: str | None = None
    base_url: str = DEEPSEEK_BASE_URL
    timeout_seconds: float = 45.0
    system_message: str | None = None

    def _resolved_key(self) -> str:
        key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        return key

    def _resolved_model(self) -> str:
        return self.model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    def _prompt(self, phase: str, task: str) -> list[dict[str, str]]:
        system = self.system_message or (
            "You are NeuroMem Memory PFC inside a LangGraph runtime. "
            "Return JSON only with keys retrieval, write, forget, consolidation, reason. "
            "Use double-quoted JSON that can instantiate RetrievalPlan, WritePlan, ForgetPlan, and ConsolidationPlan. "
            "Never request deletion unless the user explicitly asked for deletion."
        )
        user = (
            f"Phase: {phase}\n"
            f"Task: {task}\n"
            "Return one MemoryPolicy JSON object. Prefer conservative retrieval before task execution and evidenced writes after success."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _call(self, phase: str, task: str) -> MemoryPolicy:
        payload = {
            "model": self._resolved_model(),
            "messages": self._prompt(phase, task),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self._resolved_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {detail[:300]}") from exc
        content = body["choices"][0]["message"]["content"]
        return memory_policy_from_payload(extract_policy_payload(content), source="small_llm")

    def plan_before_step(self, task: str) -> MemoryPolicy:
        return self._call("before_step", task)

    def plan_after_step(self, task: str) -> MemoryPolicy:
        return self._call("after_step", task)

    def plan_consolidation(self, task: str) -> MemoryPolicy:
        return self._call("consolidation", task)

    def plan_forgetting(self, task: str) -> MemoryPolicy:
        return self._call("forgetting", task)


def create_deepseek_langgraph_pfc(*, api_key: str | None = None, model: str | None = None, system_message: str | None = None) -> DeepSeekLangGraphPFC:
    return DeepSeekLangGraphPFC(api_key=api_key, model=model, system_message=system_message)

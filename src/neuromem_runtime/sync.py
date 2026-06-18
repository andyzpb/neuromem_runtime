from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neuromem.core.policy import MemoryPolicy

from neuromem_runtime.providers import PolicyProvider
from neuromem_runtime.runtime import MemoryRuntime as AsyncMemoryRuntime
from neuromem_runtime.types import MemoryEvent, MemoryQuery


class MemoryRuntime:
    def __init__(self, async_runtime: AsyncMemoryRuntime) -> None:
        self._async_runtime = async_runtime

    @classmethod
    def local(
        cls,
        namespace: str = "default",
        path: str | Path = ".neuromem",
        agent_id: str = "local-agent",
        mode: str = "lite",
        policy_provider: PolicyProvider | None = None,
    ) -> "MemoryRuntime":
        return cls(_run(AsyncMemoryRuntime.local(namespace=namespace, path=path, agent_id=agent_id, mode=mode, policy_provider=policy_provider)))

    @classmethod
    def from_config(cls, path: str | Path = ".neuromem") -> "MemoryRuntime":
        return cls(_run(AsyncMemoryRuntime.from_config(path=path)))

    @property
    def async_runtime(self) -> AsyncMemoryRuntime:
        return self._async_runtime

    def observe(self, event: MemoryEvent | dict[str, object]) -> Any:
        return _run(self._async_runtime.observe(event))

    def query(self, query: str | MemoryQuery, budget_tokens: int = 800, filters: dict[str, object] | None = None) -> Any:
        return _run(self._async_runtime.query(query, budget_tokens=budget_tokens, filters=filters))

    def propose(self, value: str | dict[str, object]) -> MemoryPolicy:
        return _run(self._async_runtime.propose(value))

    def commit(self, policy: MemoryPolicy, *, authorize_delete: bool = False) -> dict[str, object]:
        return _run(self._async_runtime.commit(policy, authorize_delete=authorize_delete))

    def mutate(self, policy: MemoryPolicy, *, authorize_delete: bool = False) -> dict[str, object]:
        return _run(self._async_runtime.mutate(policy, authorize_delete=authorize_delete))

    def sleep(self) -> dict[str, object]:
        return _run(self._async_runtime.sleep())

    def forget(self, memory_id: str, action: str = "inhibit", reason: str = "user-requested forgetting", authorize_delete: bool = False) -> dict[str, object]:
        return _run(self._async_runtime.forget(memory_id, action=action, reason=reason, authorize_delete=authorize_delete))

    def replay_trace(self, trace_id: str) -> dict[str, object] | None:
        return _run(self._async_runtime.replay_trace(trace_id))


def _run(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    close = getattr(awaitable, "close", None)
    if close is not None:
        close()
    raise RuntimeError("neuromem_runtime.sync cannot be used inside an already-running event loop; use the async MemoryRuntime instead")


__all__ = ["MemoryRuntime"]

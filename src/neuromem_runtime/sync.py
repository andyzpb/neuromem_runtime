from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neuromem.core.policy import MemoryPolicy

from neuromem_runtime.policy_v2 import MemoryPolicyV2
from neuromem_runtime.providers import PolicyProvider
from neuromem_runtime.retrieval import EmbeddingProvider, EntityAliasResolver, HyDEProvider, QueryRewriteProvider, RerankProvider, VectorIndex
from neuromem_runtime.runtime import MemoryRuntime as AsyncMemoryRuntime
from neuromem_runtime.semantic_graph import GraphMode, GraphProposalProvider
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
        allow_unsafe_internal: bool = False,
        graph_mode: GraphMode = "governed_hybrid",
        crystallization_mode: str = "governed_progressive",
        graph_storage: str = "split",
        mutation_mode: str = "append_only_view",
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        rerank_provider: RerankProvider | None = None,
        query_rewrite_provider: QueryRewriteProvider | None = None,
        hyde_provider: HyDEProvider | None = None,
        relation_proposer: GraphProposalProvider | None = None,
        entity_alias_resolver: EntityAliasResolver | None = None,
    ) -> "MemoryRuntime":
        return cls(
            _run(
                AsyncMemoryRuntime.local(
                    namespace=namespace,
                    path=path,
                    agent_id=agent_id,
                    mode=mode,
                    policy_provider=policy_provider,
                    allow_unsafe_internal=allow_unsafe_internal,
                    graph_mode=graph_mode,
                    crystallization_mode=crystallization_mode,
                    graph_storage=graph_storage,
                    mutation_mode=mutation_mode,
                    embedding_provider=embedding_provider,
                    vector_index=vector_index,
                    rerank_provider=rerank_provider,
                    query_rewrite_provider=query_rewrite_provider,
                    hyde_provider=hyde_provider,
                    relation_proposer=relation_proposer,
                    entity_alias_resolver=entity_alias_resolver,
                )
            )
        )

    @classmethod
    def from_config(cls, path: str | Path = ".neuromem") -> "MemoryRuntime":
        return cls(_run(AsyncMemoryRuntime.from_config(path=path)))

    @property
    def async_runtime(self) -> AsyncMemoryRuntime:
        return self._async_runtime

    def observe(self, event: MemoryEvent | dict[str, object]) -> Any:
        return _run(self._async_runtime.observe(event))

    def observe_and_commit(self, event: MemoryEvent | dict[str, object]) -> Any:
        return _run(self._async_runtime.observe_and_commit(event))

    def assess_impact(self, event_id: str) -> dict[str, object]:
        return _run(self._async_runtime.assess_impact(event_id))

    def observe_and_route(self, event: MemoryEvent | dict[str, object]) -> dict[str, object]:
        return _run(self._async_runtime.observe_and_route(event))

    def query(self, query: str | MemoryQuery, budget_tokens: int = 800, filters: dict[str, object] | None = None, *, lens: str = "auto", namespace: str | None = None, top_k: int | None = None, include_worldview: bool = True) -> Any:
        return _run(self._async_runtime.query(query, budget_tokens=budget_tokens, filters=filters, lens=lens, namespace=namespace, top_k=top_k, include_worldview=include_worldview))

    def propose(self, value: str | dict[str, object]) -> MemoryPolicy | MemoryPolicyV2:
        return _run(self._async_runtime.propose(value))

    def commit(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        return _run(self._async_runtime.commit(policy, authorize_delete=authorize_delete))

    def mutate(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        return _run(self._async_runtime.mutate(policy, authorize_delete=authorize_delete))

    def sleep(self) -> dict[str, object]:
        return _run(self._async_runtime.sleep())

    def after_turn(self, trace_id: str, outcome: str, feedback: str | None = None) -> dict[str, object]:
        return _run(self._async_runtime.after_turn(trace_id, outcome, feedback=feedback))

    def resolve_worldview(self, query: str | None = None, lens: str = "auto", namespace: str | None = None, as_of: str | None = None) -> dict[str, object]:
        return _run(self._async_runtime.resolve_worldview(query=query, lens=lens, namespace=namespace, as_of=as_of))

    def materialize_worldview(self, namespace: str | None = None) -> dict[str, object]:
        return _run(self._async_runtime.materialize_worldview(namespace=namespace))

    def rebuild_materialized_views(self, namespace: str | None = None) -> dict[str, object]:
        return _run(self._async_runtime.rebuild_materialized_views(namespace=namespace))

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

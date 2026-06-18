from __future__ import annotations

from typing import Any, Callable

from neuromem_runtime.runtime import MemoryRuntime


def add_neuromem_runtime(
    builder: Any,
    *,
    memory: MemoryRuntime,
    before: str,
    after: str | None = None,
) -> Any:
    """Add thin NeuroMem nodes to a LangGraph StateGraph builder."""

    _require_langgraph()
    retrieval_node = retrieve_memory_node(memory)
    commit_node = commit_memory_node(memory)
    builder.add_node("neuromem_retrieve", retrieval_node)
    builder.add_node("neuromem_commit", commit_node)
    builder.add_edge("neuromem_retrieve", before)
    if after is not None:
        builder.add_edge(after, "neuromem_commit")
    return builder


def plan_memory_node(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        policy = asyncio.run(memory.propose({"query": state.get("query") or state.get("task") or ""}))
        return {**state, "memory_policy": policy}

    return node


def retrieve_memory_node(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        query = str(state.get("query") or state.get("task") or "")
        context = asyncio.run(memory.query(query))
        return {**state, "memory_context": context.to_prompt(), "retrieved_memory_ids": context.selected_memory_ids, "memory_trace_id": context.trace_id}

    return node


def build_context_node(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return retrieve_memory_node(memory)


def observe_trace_node(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "memory_observed": True}

    return node


def commit_memory_node(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        content = str(state.get("agent_output") or state.get("output") or state.get("task") or "")
        if not content:
            return state
        bundle = asyncio.run(memory.observe({"type": "task_result", "content": content, "task": state.get("task"), "evidence": state.get("memory_trace_id") or "langgraph"}))
        return {**state, "committed_memory_id": bundle.memory_id}

    return node


def build_retrieval_subgraph(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return retrieve_memory_node(memory)


def build_commit_subgraph(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return commit_memory_node(memory)


def build_sleep_subgraph(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return {**state, "sleep_report": asyncio.run(memory.sleep())}

    return node


def build_forgetting_subgraph(memory: MemoryRuntime) -> Callable[[dict[str, Any]], dict[str, Any]]:
    _require_langgraph()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        memory_id = state.get("memory_id")
        if not memory_id:
            return state
        return {**state, "forget_trace": asyncio.run(memory.forget(str(memory_id), action=str(state.get("forget_action", "inhibit"))))}

    return node


def _require_langgraph() -> None:
    try:
        import langgraph  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("neuromem_runtime.langgraph requires `pip install neuromem-runtime[langgraph]`") from exc


__all__ = [
    "add_neuromem_runtime",
    "plan_memory_node",
    "retrieve_memory_node",
    "build_context_node",
    "observe_trace_node",
    "commit_memory_node",
    "build_retrieval_subgraph",
    "build_commit_subgraph",
    "build_sleep_subgraph",
    "build_forgetting_subgraph",
]

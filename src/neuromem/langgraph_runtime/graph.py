from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from neuromem.core.policy import MemoryPolicy
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.stores.base import MemoryStore
from neuromem.stores.sqlite_store import SQLiteMemoryStore


class NeuroMemAgentState(TypedDict, total=False):
    task: str
    thread_id: str
    db_path: str
    before_context: str
    after_policy: MemoryPolicy
    retrieved_memory_ids: list[str]
    trace: dict[str, object]
    outcome: dict[str, object]
    explanation: dict[str, object] | None
    errors: list[str]


@dataclass(slots=True)
class NeuroMemLangGraphStore:
    memory_store: MemoryStore
    checkpoints: MutableMapping[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def from_sqlite(cls, db_path: str | Path) -> "NeuroMemLangGraphStore":
        return cls(SQLiteMemoryStore(db_path))

    def save_state(self, thread_id: str, state: NeuroMemAgentState) -> None:
        self.checkpoints[thread_id] = dict(state)

    def load_state(self, thread_id: str) -> dict[str, object] | None:
        state = self.checkpoints.get(thread_id)
        return dict(state) if state is not None else None


@dataclass(slots=True)
class MemoryRetrievalGraph:
    runtime: NeuroMemRuntime

    def __call__(self, state: NeuroMemAgentState) -> NeuroMemAgentState:
        task = state["task"]
        before_context = self.runtime.before_step(
            task,
            {
                "task_id": state.get("thread_id", task),
                "query": state.get("query", task),
                "entities": state.get("entities", []),
                "max_items": state.get("max_items", 8),
            },
        )
        explanation = self.runtime.explain_last_retrieval()
        selected = []
        if explanation is not None:
            selected = list(explanation.get("selected_memory_ids", []))
        return {
            **state,
            "before_context": before_context,
            "retrieved_memory_ids": [str(memory_id) for memory_id in selected],
            "explanation": explanation,
        }


@dataclass(slots=True)
class MemoryCommitGraph:
    runtime: NeuroMemRuntime

    def __call__(self, state: NeuroMemAgentState) -> NeuroMemAgentState:
        task = state["task"]
        trace = dict(state.get("trace") or {})
        trace.setdefault("id", state.get("thread_id", task))
        trace.setdefault("content", state.get("agent_output") or state.get("before_context") or task)
        outcome = dict(state.get("outcome") or {"status": "success", "confidence": 0.75})
        retrieved = [str(memory_id) for memory_id in state.get("retrieved_memory_ids", [])]
        policy = self.runtime.after_step(task, trace, outcome, retrieved_memory_ids=retrieved)
        return {
            **state,
            "after_policy": policy,
            "explanation": self.runtime.explain_last_retrieval(),
        }


@dataclass(slots=True)
class MainAgentGraph:
    runtime: NeuroMemRuntime
    store: NeuroMemLangGraphStore | None = None
    agent_step: Callable[[NeuroMemAgentState], NeuroMemAgentState] | None = None
    compiled_graph: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.store = self.store or NeuroMemLangGraphStore(self.runtime.store)  # type: ignore[arg-type]
        self.compiled_graph = self._compile_langgraph_if_available()

    def _agent_node(self, state: NeuroMemAgentState) -> NeuroMemAgentState:
        if self.agent_step is not None:
            return self.agent_step(state)
        return {
            **state,
            "trace": {
                "id": state.get("thread_id", state["task"]),
                "content": state.get("before_context") or state["task"],
            },
            "outcome": state.get("outcome") or {"status": "success", "confidence": 0.8},
        }

    def _explain_node(self, state: NeuroMemAgentState) -> NeuroMemAgentState:
        return {**state, "explanation": self.runtime.explain_last_retrieval()}

    def _compile_langgraph_if_available(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ModuleNotFoundError:
            return None

        graph = StateGraph(NeuroMemAgentState)
        graph.add_node("memory_retrieval", MemoryRetrievalGraph(self.runtime))
        graph.add_node("agent_step", self._agent_node)
        graph.add_node("memory_commit", MemoryCommitGraph(self.runtime))
        graph.add_node("explain", self._explain_node)
        graph.set_entry_point("memory_retrieval")
        graph.add_edge("memory_retrieval", "agent_step")
        graph.add_edge("agent_step", "memory_commit")
        graph.add_edge("memory_commit", "explain")
        graph.add_edge("explain", END)
        return graph.compile()

    def invoke(self, state: NeuroMemAgentState) -> NeuroMemAgentState:
        thread_id = state.get("thread_id") or f"thread_{uuid4().hex}"
        initial: NeuroMemAgentState = {**state, "thread_id": thread_id}
        if self.compiled_graph is not None:
            result = self.compiled_graph.invoke(initial)
        else:
            result = MemoryRetrievalGraph(self.runtime)(initial)
            result = self._agent_node(result)
            result = MemoryCommitGraph(self.runtime)(result)
            result = self._explain_node(result)
        assert self.store is not None
        self.store.save_state(thread_id, result)
        return result


def build_neuromem_graph(
    runtime: NeuroMemRuntime,
    memory_pfc: object | None = None,
    checkpointer: object | None = None,
    store: NeuroMemLangGraphStore | None = None,
) -> MainAgentGraph:
    if memory_pfc is not None:
        runtime.memory_pfc = memory_pfc
    return MainAgentGraph(runtime=runtime, store=store)


def run_coding_agent_graph(task: str, db_path: str | Path, memory_pfc: object | None = None) -> NeuroMemAgentState:
    runtime = NeuroMemRuntime(agent_id="coding-agent", namespace="demo-repo", db_path=db_path, memory_pfc=memory_pfc)
    runtime.observe(
        {
            "type": "failure",
            "content": "Session refresh order caused a repeated login redirect loop.",
            "task": "Fix login redirect bug",
            "evidence": "coding-demo-seed",
            "keywords": ["session", "login", "redirect"],
        }
    )
    graph = build_neuromem_graph(runtime, memory_pfc=memory_pfc)
    return graph.invoke(
        {
            "task": task,
            "thread_id": f"coding_{uuid4().hex}",
            "trace": {"id": "coding-trace", "content": task},
            "outcome": {"status": "success", "confidence": 0.8, "salience": 0.75},
        }
    )

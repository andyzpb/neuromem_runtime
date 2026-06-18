# NeuroMem Runtime

NeuroMem Runtime is a local-first Memory Mutation Runtime for long-running LLM agents.

It gives a coding agent, chat agent, research assistant, or LangGraph workflow a durable memory layer that can remember task results, retrieve useful context for the next prompt, consolidate repeated lessons, forget stale facts, and replay why a memory was used.

The package name is `neuromem-runtime`. User code imports `neuromem_runtime`.

```bash
pip install neuromem-runtime
```

Current package release: `v0.1.0`.

## At a glance

- Memory mutation runtime, not just a memory store.
- LLMs propose memory mutations; the runtime validates and commits them.
- Every write, update, suppression, edge update, consolidation, and forgetting event is a transaction.
- Retrieval is activation over an outcome-shaped experience graph.
- Forgetting is suppression before deletion.
- Every memory effect is replayable through the Memory Ledger.
- Local by default: SQLite plus trace files, with no Docker or hosted store required.
- Optional integrations: LLM policy providers and LangGraph are opt-in extras.

## Why use it

Most agent memory examples stop at "put text in a vector store, search it later." NeuroMem Runtime treats memory as a governed lifecycle:

- `observe` records events, tool results, feedback, preferences, and task outcomes.
- `query` returns prompt-ready memory context with selected memory ids and a trace id.
- `propose` creates a structured memory policy.
- `commit` or `mutate` validates that policy before it changes storage.
- `sleep` consolidates and updates lifecycle state.
- `forget` inhibits, invalidates, archives, compresses, or deletes only when policy allows it.
- `replay_trace` shows the transaction trail behind retrieval and mutation.

The useful difference is control. An LLM can help propose what to remember, but it does not get to directly rewrite the memory store.

## What it is for

Use NeuroMem Runtime when an agent needs to carry knowledge across runs:

- a coding agent remembering fixes, commands, repo conventions, and failed approaches
- a support or operations assistant remembering user preferences and incident resolutions
- a research assistant remembering evidence, decisions, and invalidated claims
- a LangGraph app that needs memory retrieval before an agent node and memory commit after it
- local demos where Docker, hosted vector stores, and API keys would slow the first run down

The default runtime is local. It uses SQLite and trace files under `.neuromem/`. It does not call an external model unless you pass a provider explicitly.

## Five-minute local start

```python
import asyncio
import neuromem_runtime as nmem


async def main() -> None:
    memory = await nmem.MemoryRuntime.local(
        namespace="demo/repo",
        path="./.neuromem",
    )

    await memory.observe({
        "type": "task_result",
        "content": "Login redirect bug was fixed by changing session refresh order.",
        "task": "Fix login redirect",
        "evidence": "demo-trace-1",
        "keywords": ["login", "session", "redirect"],
    })

    ctx = await memory.query(
        "Have we fixed a similar login/session bug before?",
        budget_tokens=800,
    )

    print(ctx.to_prompt())
    print(ctx.selected_memory_ids)
    print(ctx.trace_id)


asyncio.run(main())
```

That creates:

```text
.neuromem/
  config.toml
  memory.sqlite3
  traces/
```

The SQLite database also stores observed experience events and ledger events. Trace JSON files remain available for easy inspection.

The returned `MemoryContext` is ready to place into an agent prompt:

```python
agent_input = {
    "question": user_question,
    "memory_context": ctx.to_prompt(),
    "memory_trace_id": ctx.trace_id,
}
```

## CLI quickstart

Create a local workspace:

```bash
nmem init --namespace demo/repo
nmem doctor
```

Record events from JSONL:

```bash
cat > events.jsonl <<'JSONL'
{"type":"task_result","content":"Session refresh order fixed login redirect loop.","task":"Fix login","evidence":"trace-1","keywords":["session","login"]}
JSONL

nmem observe events.jsonl --namespace demo/repo
```

Query memory:

```bash
nmem query "Have we fixed auth/session bugs before?" --namespace demo/repo
```

Run consolidation and inspect the trace:

```bash
nmem sleep --namespace demo/repo
nmem trace show TRACE_ID
nmem trace export TRACE_ID --format json
nmem ledger replay
```

## Agent loop pattern

A typical agent loop uses NeuroMem at two points: retrieve before the model call, then observe or commit after the tool/model result.

```python
async def run_agent_turn(memory: nmem.MemoryRuntime, user_task: str) -> str:
    ctx = await memory.query(user_task, budget_tokens=800)

    response = await call_your_agent_model(
        task=user_task,
        memory_context=ctx.to_prompt(),
    )

    await memory.observe({
        "type": "task_result",
        "content": response,
        "task": user_task,
        "evidence": ctx.trace_id,
        "keywords": ["agent-result"],
    })

    return response
```

For stricter mutation control, ask the runtime to propose a policy and commit it:

```python
policy = await memory.propose({
    "phase": "after_step",
    "task": user_task,
    "content": response,
    "evidence": ctx.trace_id,
})

trace = await memory.commit(policy)
```

`commit()` validates the policy and returns replayable trace data.

For the stricter protocol path, record an immutable experience event without immediately creating long-term memory:

```python
event = await memory.observe(
    {
        "type": "task_result",
        "content": response,
        "task": user_task,
        "evidence": ctx.trace_id,
    },
    auto_commit=False,
)
```

The returned bundle includes `event_id` and `content_hash`. Long-term memory capture can then be proposed and committed explicitly.

## Optional LLM policy provider

The base package never calls an external model. To use an LLM for memory-policy proposals, pass a provider explicitly:

```python
import neuromem_runtime as nmem

provider = nmem.DeepSeekPolicyProvider(
    api_key_env="DEEPSEEK_API_KEY",
    model="deepseek-v4-flash",
)

memory = await nmem.MemoryRuntime.local(
    namespace="demo/repo",
    policy_provider=provider,
)

policy = await memory.propose({
    "phase": "after_step",
    "task": "Fix login redirect",
    "content": "Session refresh order fixed the redirect loop.",
    "evidence": "trace-1",
})

trace = await memory.commit(policy)
```

The model proposes a `MemoryPolicy`. The runtime still validates that policy before storage changes.

Use the OpenAI-compatible provider for other hosted models:

```python
provider = nmem.OpenAICompatiblePolicyProvider(
    api_key_env="OPENAI_API_KEY",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com",
)
```

Install provider dependencies when needed:

```bash
pip install neuromem-runtime[providers]
```

## LangGraph integration

Install optional dependencies:

```bash
pip install neuromem-runtime[langgraph]
```

Wire memory around an agent node:

```python
from langgraph.graph import StateGraph
import neuromem_runtime as nmem
from neuromem_runtime.langgraph import add_neuromem_runtime

memory = await nmem.MemoryRuntime.local(namespace="repo/demo")

builder = StateGraph(dict)
builder.add_node("run_agent", run_agent_node)

add_neuromem_runtime(
    builder,
    memory=memory,
    before="run_agent",
    after="run_agent",
)
```

LangGraph owns orchestration. NeuroMem Runtime owns memory policy, validation, lifecycle state, and trace replay.

## Public API

```python
from neuromem_runtime import (
    MemoryRuntime,
    RuntimeConfig,
    MemoryEvent,
    MemoryQuery,
    MemoryContext,
    EvidenceBundle,
    ExperienceEvent,
    MemoryPolicy,
    MemoryPolicyV2,
    MemoryTransaction,
    MemoryTrace,
    MemoryLedger,
    ValidatorStack,
    RetrievalTraceMetadata,
    PlasticityEngine,
    SleepPlanner,
)
```

Main actions:

- `MemoryRuntime.local(...)` creates or opens a local `.neuromem` workspace.
- `observe(event)` records an experience event and, by default, keeps quickstart-compatible auto-commit behavior.
- `observe(event, auto_commit=False)` records only the immutable experience event.
- `query(query, budget_tokens=800)` retrieves prompt-ready memory context.
- `propose(input)` creates a structured memory policy with the configured provider.
- `commit(policy)` validates and applies governed memory changes.
- `mutate(policy)` aliases `commit(policy)`.
- `sleep()` runs replay consolidation and lifecycle updates.
- `forget(memory_id, action="inhibit", reason="...")` applies governed forgetting.
- `replay_trace(trace_id)` returns replayable trace data.

Ledger CLI:

```bash
nmem ledger show TXN_ID
nmem ledger why-written MEM_ID
nmem ledger why-retrieved TRACE_ID MEM_ID
nmem ledger replay --to-txn TXN_ID
nmem ledger diff TXN_A TXN_B
```

Governed runtime surfaces:

- `MemoryPolicyV2` defines the forward policy shape for transaction-governed mutation.
- `ValidatorStack` exposes fail-closed gates for schema, evidence, provenance, temporal, conflict, ACL, deletion, poisoning risk, lifecycle, and index consistency checks.
- `MemoryLedger` stores experience events, ledger events, memory versions, and edge versions in SQLite.
- `RetrievalTraceMetadata` records whether retrieval used local hybrid, disabled/proxy/provider embeddings, index type, candidate sources, and fusion strategy.
- `PlasticityEngine` produces graph deltas for outcome-shaped edge updates.
- `SleepPlanner` and `SleepReport` define the sleep/replay/consolidation report surface.

## Sync wrapper

```python
from neuromem_runtime.sync import MemoryRuntime

memory = MemoryRuntime.local(namespace="demo")
memory.observe({"content": "User prefers concise answers."})

ctx = memory.query("What style does the user prefer?")
print(ctx.to_prompt())
```

The sync wrapper is for scripts. Inside an existing event loop, use the async API.

## Safety model

NeuroMem Runtime is designed around governed mutation:

- local mode uses SQLite and trace files
- observed experience events are recorded before long-term memory capture
- ledger events are append-only and hash-linked
- external model calls are off by default
- deterministic policy proposals are available without API keys
- model output is a proposal, not a direct mutation
- every committed policy passes validation
- unauthorized physical deletion is rejected
- forgetting defaults to inhibition, invalidation, archive, or compression
- traces can be replayed for audit and debugging

Base retrieval is a deterministic local lexical/BM25/graph baseline. Dense vector retrieval is represented by `EmbeddingProvider` and `VectorIndex` protocols and remains an explicit adapter path, not a base-package claim.

This matters for agents because memory mistakes accumulate. A stale command, wrong preference, or unsafe deletion should be visible and reversible at the policy layer.

## Package boundaries

`neuromem-runtime` keeps the public surface small:

- public package: `neuromem_runtime`
- CLI: `nmem`
- optional LangGraph integration: `neuromem_runtime.langgraph`
- optional model policy providers: explicit `policy_provider=...`

User code should import `neuromem_runtime`. The base import path does not import LangGraph or call external models.

## Benchmark command

The mini benchmark command delegates to bundled deterministic evaluation code:

```bash
nmem bench run mini --methods full,vector,no_graph,no_pfc
```

If an optional evaluation dependency is missing, install:

```bash
pip install neuromem-runtime[eval]
```

## Development

```bash
pip install -e .[dev]
pytest -q
python -m build
twine check dist/*
```

The package is typed with `py.typed`.

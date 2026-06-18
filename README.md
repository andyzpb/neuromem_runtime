# NeuroMem Runtime

Local-first memory for long-running LLM agents.

NeuroMem is not another vector-store wrapper. It is a **Memory Mutation Runtime**: agents can observe events and LLMs can propose memory changes, but durable memory only changes after validation, transaction logging, lifecycle handling, and replayable audit.

```bash
pip install neuromem-runtime
```

Current release: `v0.2.0`.

## Why It Exists

Long-running agents need more than recall. They need to know when a memory was written, what evidence supported it, whether it is stale, why it was retrieved, and who was allowed to change it.

NeuroMem gives you that local runtime:

- `observe` records immutable experience events.
- `commit` validates memory mutations before they touch storage.
- `query` returns prompt-ready context with reasons and trace ids.
- `forget` inhibits, invalidates, archives, compresses, or deletes by policy.
- `sleep` consolidates repeated experience into more useful memory.
- `replay_trace` and the ledger explain what happened.

Base install is SQLite + trace files. No Docker, API key, hosted store, vector database, LangGraph import, or model call is required.

## Quickstart

```python
import asyncio
import neuromem_runtime as nmem


async def main() -> None:
    memory = await nmem.MemoryRuntime.local(
        namespace="demo/repo",
        path="./.neuromem",
    )

    await memory.observe_and_commit({
        "type": "task_result",
        "content": "Login redirect bug was fixed by changing session refresh order.",
        "task": "Fix login redirect",
        "keywords": ["login", "session", "redirect"],
    })

    ctx = await memory.query("Have we fixed a similar login/session bug before?")

    print(ctx.to_prompt())
    print(ctx.selected_memory_ids)
    print(ctx.trace_id)


asyncio.run(main())
```

`observe()` alone is stricter: it records an immutable event and does not create long-term memory. Use `observe_and_commit()` only when you explicitly want the runtime to validate and persist a memory from that event.

## CLI

```bash
nmem init --namespace demo/repo
nmem doctor

cat > events.jsonl <<'JSONL'
{"type":"task_result","content":"Session refresh order fixed login redirect loop.","task":"Fix login","keywords":["session","login"]}
JSONL

nmem observe events.jsonl --namespace demo/repo
nmem observe events.jsonl --namespace demo/repo --commit

nmem query "Have we fixed auth/session bugs before?" --namespace demo/repo
nmem query "Have we fixed auth/session bugs before?" --namespace demo/repo --json
nmem retrieval explain TRACE_ID --namespace demo/repo
nmem ledger replay
```

Without `--commit`, `nmem observe` records evidence only. With `--commit`, it validates and commits long-term memory.

## Retrieval

The base retrieval path is local activation retrieval:

```text
Memory cards -> FTS5/BM25 + lexical/entity/current candidates
             -> RRF fusion
             -> PPR-style graph activation
             -> lifecycle/provenance gates
             -> lite rerank
             -> packed prompt context + retrieval ledger
```

Dense embeddings, cross-encoder rerankers, and LLM listwise rerankers are reserved adapter surfaces in `v0.2.0`. The base package ships deterministic local activation retrieval; installed ranking adapters may rank candidates, but they must not mutate memory.

## Safety Model

- LLMs propose memory policies; they do not directly rewrite memory.
- Every product-surface mutation goes through `PolicyExecutor`.
- Writes require evidence.
- Deletes require explicit authorization.
- Unsafe access to the bundled core runtime requires explicit `allow_unsafe_internal=True` opt-in.
- Ledger reads and CLI ledger commands are namespace-scoped.
- Retrieval access counter updates are ledgered memory effects.
- Ledger events are sequence-ordered, hash-linked, and replayable.

## Integrations

Optional extras:

```bash
pip install "neuromem-runtime[langgraph]"
pip install "neuromem-runtime[providers]"
pip install "neuromem-runtime[eval]"
```

LangGraph is the reference orchestration integration, but NeuroMem Core stays framework-agnostic.

## Learn More

- API and runtime manual: [`docs/api.md`](docs/api.md)
- Public import: `import neuromem_runtime as nmem`
- CLI: `nmem`
- Package: `neuromem-runtime`

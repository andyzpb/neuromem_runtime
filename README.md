# NeuroMem Runtime

Local-first memory for long-running LLM agents.

NeuroMem is not another vector-store wrapper. It is a **Memory Mutation Runtime**: agents can observe events and LLMs can propose memory changes, but the product runtime treats memory as append-only evidence. Current working state is projected from validated evidence, suppression events, graph relations, lifecycle records, and replayable audit.

```bash
pip install neuromem-runtime
```

Current release: `v0.2.0`.

## Why It Exists

Long-running agents need more than recall. They need to know when a memory was written, what evidence supported it, whether it is stale, why it was retrieved, and who was allowed to change it.

NeuroMem gives you that local runtime:

- `observe` records immutable experience events.
- `commit` validates append-only memory transactions before they touch storage.
- `query` returns a Worldview Snapshot plus prompt-ready supporting context, reasons, trace ids, and an optional retrieval lens.
- `forget` appends suppression, expiry, decay, or archive evidence. It does not overwrite or physically delete memories.
- `sleep` consolidates repeated experience into more useful memory and compiled frames.
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

    ctx = await memory.query("Have we fixed a similar login/session bug before?", lens="associative")

    print(ctx.to_prompt())
    print(ctx.worldview)
    print(ctx.selected_memory_ids)
    print(ctx.trace_id)


asyncio.run(main())
```

`observe()` alone is stricter: it records an immutable event and does not create long-term memory. Use `observe_and_commit()` only when you explicitly want the runtime to validate and persist a memory from that event.

For uncertain input, use `observe_and_route()`. It records the event, computes a Worldview Impact assessment, and only commits durable evidence when the input has enough novelty, belief delta, utility, or conflict value to justify write pressure.

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

Without `--commit`, `nmem observe` records evidence only. With `--commit`, it validates and commits append-only long-term evidence.

## Worldview Projection

NeuroMem separates memory from current belief:

```text
append-only evidence + Frames + edge evidence + lifecycle records
        -> Worldview Impact Meter
        -> Worldview Resolver
        -> Worldview Snapshot + supporting memories
```

Every observation is assessed with a vector that includes novelty, belief delta, entropy delta, contradiction, supersession, utility, propagation, source reliability, and risk. Low-impact input can remain ledger-only. Higher-impact input can become candidate memory, support evidence, suppression evidence, a conflict, or sleep priority.

`query()` returns normal retrieval results and also attaches `MemoryContext.worldview`. `MemoryContext.to_prompt()` includes a compact Worldview Snapshot before the supporting memory snippets.

## Retrieval

The base retrieval path is local activation retrieval:

```text
Memory cards -> FTS5/BM25 + lexical/entity/current candidates
             -> optional dense/rewrite/HyDE semantic candidates
             -> RRF fusion
             -> PPR-style graph activation
             -> lifecycle/provenance gates
             -> lite rerank
             -> packed prompt context + retrieval ledger
```

Dense embeddings, query rewrite, HyDE, alias expansion, and rerankers are opt-in provider surfaces. The base package ships local protocol interfaces plus deterministic test-friendly components; it does not install hosted models or call the network.

`query(..., lens="auto")` chooses a deterministic retrieval lens. You can pass `associative`, `logical`, `procedural`, `historical`, or `audit` when the caller needs a specific memory surface.

Retrieval-time graph commit defaults to `trace_only`. Co-retrieval is recorded as trace evidence, but it is not automatically reinforced into the long-term graph. Durable graph reinforcement should come from validated outcome evidence, sleep, or explicit append-only graph proposals.

## Progressive Crystallization

NeuroMem does not turn every event into a logical graph. It uses governed progressive crystallization:

- raw experience stays in the append-only ledger
- low-commitment links go into an associative graph
- candidate facts, procedures, preferences, constraints, entities, schemas, and failure patterns become Frames
- validated logical relations connect Frames in a separate logic graph
- sleep/replay promotes repeated evidence into compiled schema or procedure Frames

SQLite stores the associative graph, logic nodes, and logic edges in separate tables. `list_edges()` remains an internal projection for activation retrieval, but split graph storage is the baseline.

The ledger also stores append-only edge evidence events and impact assessments. Suppression, supersession, contradiction, decay, and expiry are evidence events that change the materialized current view without rewriting the original memory record.

## Safety Model

- LLMs propose memory policies; they do not directly rewrite memory.
- Every product-surface mutation goes through `PolicyExecutor`.
- Writes require evidence.
- Destructive updates and physical deletes are not supported by the product runtime.
- Corrections are represented by appending new evidence, supersession relations, or suppression evidence.
- Forgetting suppresses or expires memories from the current view while preserving audit history.
- Unsafe access to the bundled core runtime requires explicit `allow_unsafe_internal=True` opt-in.
- Ledger reads and CLI ledger commands are namespace-scoped.
- Retrieval access counter updates are ledgered memory effects.
- Associative and logic graph changes are validated memory mutations, not direct model writes.
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

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
- `observe_and_route` records an event, measures Worldview Impact, and routes it to ledger-only, support evidence, candidate Frame, candidate worldview update, clarification, quarantine, or sleep priority.
- `commit` validates append-only memory transactions before they touch storage.
- `query` returns a Worldview Snapshot plus prompt-ready supporting context, reasons, trace ids, and an optional retrieval lens.
- `forget` appends suppression, expiry, decay, or archive evidence. It does not overwrite or physically delete memories.
- `sleep` replays high-impact/conflicted/repeated clusters into compiled Frames, Procedures, Schemas, and evidence links without rewriting source memories.
- `after_turn` appends success/failure outcome evidence for retrieved memories; it does not directly mutate edge weights.
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

For uncertain input, use `observe_and_route()`. It records the event, computes a Worldview Impact assessment, and routes append-only evidence. Novel input can propose a candidate Frame or worldview candidate without creating durable memory. Explicit long-term memory creation remains `observe_and_commit()`.

Durable V2 mutations are source-gated at the runtime boundary. `ADD`, `LINK`, `SUPPRESS`, `SUPERSEDE`, and append-only correction repairs require at least one structured grounded claim whose `metadata.source_channel` is `current_user_message`, `user_message`, or `tool_result`. Claims derived only from `assistant_answer`, `retrieved_memory`, or `short_term_context` can be routed as derivations or audit evidence, but they cannot create durable memory by themselves.

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

Every observation is assessed with a vector that includes novelty, belief delta, entropy delta, contradiction, supersession, utility, propagation, source reliability, and risk. The prior distribution comes from materialized worldview slots and candidates when available, then falls back to active memories and Frames. Low-impact input can remain ledger-only. Higher-impact input can append support evidence, propose a candidate Frame, propose a worldview candidate, request clarification, quarantine risky input, or mark sleep priority.

`query()` returns normal retrieval results and also attaches `MemoryContext.worldview`, `MemoryContext.worldview_trace`, and `MemoryContext.prompt_sections`. `MemoryContext.to_prompt()` includes a compact Worldview Snapshot before the supporting memory snippets.

The runtime stores worldview projection data in additive SQLite tables:

- `worldview_slots`
- `worldview_candidates`
- `worldview_candidate_events`
- `edge_evidence_events`
- `impact_assessments`

`associative_edges` and `logic_edges` are materialized caches. They can be cleared and rebuilt from edge evidence and worldview records with `rebuild_materialized_views()`.

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

### Retrieval Performance

`query()` uses a versioned in-memory retrieval cache, so append-only audit events do not force an expensive full cache clear. Durable memory or materialized retrieval graph changes advance the namespace semantic version, worldview materialization advances the worldview version, and old cached entries age out through TTL/LRU eviction. Cache diagnostics expose `namespace`, numeric `semantic_version`, numeric `worldview_version`, `filter_hash`, and `miss_reason`.

Embedding-backed retrieval uses a SQLite embedding cache with WAL mode and batch reads/writes. The cache avoids synchronous `last_accessed_at` writes on every hit, and `LocalVectorIndex` is lock-protected for concurrent reads and batched upserts. Async runtime queries offload blocking retrieval and embedding provider work to worker threads when needed, while singleflight coalesces identical retrieval misses. `performance_stats()` reports retrieval cache, singleflight, cache versions, embedding batcher, embedding provider, and background job state. `prewarm_embeddings()` can be called at service startup to load a local embedding model before the first user query.

NeuroMem intentionally does not cache natural-language LLM answers. It caches retrieval/context/vector work only, so answers are generated against the latest resolved Worldview and supporting memories.

## Progressive Crystallization

NeuroMem does not turn every event into a logical graph. It uses governed progressive crystallization:

- raw experience stays in the append-only ledger
- low-commitment links go into an associative graph
- candidate facts, procedures, preferences, constraints, entities, schemas, and failure patterns become Frames
- validated logical relations connect Frames in a separate logic graph
- sleep/replay promotes repeated evidence into compiled schema or procedure Frames

SQLite stores the associative graph, logic nodes, and logic edges in separate tables. `list_edges()` remains an internal projection for activation retrieval, but split graph storage is the baseline.

The ledger also stores append-only edge evidence events and impact assessments. Suppression, supersession, contradiction, decay, and expiry are evidence events that change the materialized current view without rewriting the original memory record.

Supported edge evidence event types are `support`, `contradict`, `supersede`, `inhibit`, `reinforce`, `decay`, `expire`, `restore`, `generalize`, and `derive`.

## Safety Model

- LLMs propose memory policies; they do not directly rewrite memory.
- Every product-surface mutation goes through `PolicyExecutor`.
- Writes require evidence.
- Destructive updates and physical deletes are not supported by the product runtime.
- `UPDATE`, `DELETE_REQUEST`, and physical delete are rejected on product APIs, even with authorization.
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

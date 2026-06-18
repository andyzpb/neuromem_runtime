# NeuroMem Runtime Manual

## Runtime Model

NeuroMem Runtime is a local-first Memory Mutation Runtime for long-running LLM agents. The public package is `neuromem-runtime`; application code imports `neuromem_runtime`.

The runtime separates three things:

- **Experience events**: immutable observations recorded through `observe()`.
- **Memory mutations**: writes, updates, forgetting, consolidation, graph updates, and access effects.
- **Ledger records**: hash-linked transaction events that explain and replay memory effects.

The product rule is simple: durable memory changes go through `PolicyExecutor`, deterministic validation, atomic SQLite commit or rollback, delta capture, version snapshots, and ledger events.

## Workspace

```python
memory = await MemoryRuntime.local(namespace="default", path=".neuromem")
```

This creates:

```text
.neuromem/
  config.toml
  memory.sqlite3
  traces/
```

The SQLite database stores memories, memory cards, graph edges, experience events, ledger events, memory versions, and edge versions.

## Public Actions

| Method | Purpose |
| --- | --- |
| `observe(event)` | Record only an immutable `ExperienceEvent`; no long-term memory is written. |
| `observe_and_commit(event)` | Explicitly validate and commit a long-term memory from an event. |
| `query(query, budget_tokens=800)` | Return a prompt-ready `MemoryContext` through activation retrieval. |
| `propose(input)` | Produce a structured `MemoryPolicy` or `MemoryPolicyV2` with the configured provider. |
| `commit(policy)` | Validate and apply governed memory changes. |
| `mutate(policy)` | Alias for `commit(policy)`. |
| `sleep()` | Run governed replay consolidation and lifecycle updates. |
| `forget(memory_id, action="inhibit")` | Apply governed forgetting. |
| `replay_trace(trace_id)` | Return trace plus ledger-backed deltas and replay data. |

Physical deletion requires `authorize_delete=True`.

## Strict Observation

`observe()` records evidence only:

```python
event = await memory.observe({
    "type": "task_result",
    "content": "Session refresh order fixed the login redirect loop.",
    "task": "Fix login",
})
```

The returned `EvidenceBundle` includes `event_id` and `content_hash`. To create durable memory from that event, use a policy and `commit()`, or the explicit convenience path:

```python
bundle = await memory.observe_and_commit({
    "type": "task_result",
    "content": "Session refresh order fixed the login redirect loop.",
    "task": "Fix login",
})
```

`observe_and_commit()` still goes through evidence validation, `PolicyExecutor`, ledger phases, and memory version snapshots.

## Commit Path

`commit()` accepts both the compatibility `MemoryPolicy` and forward `MemoryPolicyV2`:

```python
trace = await memory.commit(policy)
```

The returned trace includes `mutation_execution_result`:

- `validated_mutation`
- `created_memory_ids`
- `updated_memory_ids`
- `deleted_memory_ids`
- `memory_deltas`
- `graph_deltas`
- `lifecycle_deltas`
- `index_deltas`

Rejected policies write `validation_rejected` and `audit_finalized` ledger events and do not mutate memory, graph, lifecycle state, or indexes.
If a post-commit assertion fails inside the transaction, storage changes roll back and the ledger records a `transaction_rolled_back` event with the rollback reason.

## Validator Stack

The product executor uses `ValidatorStack` before mutation:

- `SchemaValidator`
- `EvidenceValidator`
- `ProvenanceValidator`
- `TemporalValidator`
- `ConflictValidator`
- `NamespaceScopeValidator`
- `PrivacyAclValidator`
- `DeletionGuardValidator`
- `PoisoningRiskValidator`
- `LifecycleTransitionValidator`
- `IndexConsistencyValidator`

The implemented gates fail closed for unsupported operations, missing evidence, unknown provenance, cross-namespace targets or evidence, unsafe deletion, poisoning phrases, unauthorized private-memory mutation, stale target updates without historical intent, invalid lifecycle transitions, and missing post-commit memory-card index rows.

## Ledger

`MemoryRuntime.local(...)` exposes `memory.ledger`.

Ledger transactions are split into phase events:

- `proposal_recorded`
- `validation_approved` or `validation_rejected`
- `memory_delta_committed`
- `index_updated`
- `graph_delta_committed`
- `lifecycle_delta_committed`
- `audit_finalized`

Sleep transactions use explicit sleep phases:

- `sleep_plan_proposed`
- `sleep_validation_approved`
- `replay_batch_selected`
- `consolidation_delta_committed`
- `suppression_delta_committed`
- `compilation_delta_committed`
- `sleep_audit_finalized`

Useful methods:

```python
memory.ledger.verify_hash_chain()
memory.ledger.reconstruct(to_transaction_id=None, namespace="default")
memory.ledger.replay_trace(trace_id, namespace="default")
memory.ledger.why_written(memory_id, namespace="default")
memory.ledger.retrieval_explain(trace_id, namespace="default")
```

CLI commands:

```bash
nmem ledger show TXN_ID
nmem ledger why-written MEM_ID
nmem ledger why-retrieved TRACE_ID MEM_ID
nmem ledger replay --to-txn TXN_ID
nmem ledger diff TXN_A TXN_B
nmem retrieval explain TRACE_ID
```

## Activation Retrieval

Base retrieval is local and deterministic:

```text
QueryPlanV2 -> Contextual Memory Cards
            -> FTS5/BM25 + lexical/entity/current/procedural/canonical candidates
            -> RRF fusion
            -> PPR-style graph activation
            -> lifecycle/provenance gate
            -> lite rerank
            -> context packing
            -> retrieval ledger
```

`query(...)` uses one retrieval transaction. The returned `MemoryContext`, persisted trace, and retrieval ledger share the same selected ids, suppression reasons, scores, graph paths, and trace id.

Optional query filters:

- `retrieval_mode`
- `retrieval_channels`
- `rerank_mode`
- `graph_activation`
- `historical`
- `require_provenance`
- `allow_abstain`

`MemoryContext.results` includes:

- `why_retrieved`
- `score_components`
- `graph_paths`
- `reranker_score`
- `lifecycle_reason`
- `provenance_ids`

Retrieval access effects, such as `access_count`, `activation_count`, and `last_accessed_at`, are ledgered as memory deltas.

Dense embeddings, late-interaction retrieval, cross-encoder reranking, and LLM listwise reranking are reserved adapter surfaces in `v0.2.0`. The base package ships deterministic local activation retrieval; installed ranking adapters may rank candidates, but they must not mutate memory.

## Forgetting

Supported actions:

- `decay`
- `inhibit`
- `invalidate`
- `archive`
- `compress`
- `delete`

Deletion is rejected unless `authorize_delete=True`. Normal forgetting keeps audit history and uses lifecycle transitions before destructive behavior.

## Sleep

`sleep()` runs replay consolidation and writes a governed sleep transaction. The report includes:

- processed memory count
- replay clusters
- promoted/compressed/archived memory ids
- memory/lifecycle deltas
- ledger transaction ids

## Policy Providers

The base package never calls an external model. To use an LLM for memory-policy proposals, pass a provider explicitly:

```python
provider = nmem.DeepSeekPolicyProvider(
    api_key_env="DEEPSEEK_API_KEY",
    model="deepseek-v4-flash",
)

memory = await nmem.MemoryRuntime.local(
    namespace="demo/repo",
    policy_provider=provider,
)
```

Providers return structured policy proposals. The runtime still validates and commits them.

The OpenAI-compatible providers request `MemoryPolicyV2` first and fall back to the legacy policy schema only for compatibility. Unsupported V2 multi-delta transactions are rejected explicitly rather than silently becoming NOOP.

## Unsafe Debug Access

The bundled `neuromem` implementation is an internal compatibility layer, not the governed public API. Accessing it requires explicit opt-in:

```python
memory = await MemoryRuntime.local(allow_unsafe_internal=True)
core = memory.unsafe_internal_runtime
```

Without that opt-in, `internal_runtime` and `unsafe_internal_runtime` raise `RuntimeError`.

## Exports

The package exports:

- `MemoryRuntime`
- `RuntimeConfig`
- `MemoryEvent`
- `MemoryQuery`
- `MemoryContext`
- `EvidenceBundle`
- `ExperienceEvent`
- `MemoryPolicy`
- `MemoryPolicyV2`
- `ValidatedMutation`
- `MemoryDelta`
- `GraphDelta`
- `LifecycleDelta`
- `IndexDelta`
- `ExecutionDeltaPlan`
- `MutationExecutionResult`
- `MemorySnapshot`
- `LedgerEvent`
- `MemoryLedger`
- `LifecycleStateMachine`
- `ValidatorStack`
- `RetrievalTraceMetadata`
- `RetrievalConfig`
- `QueryPlanV2`
- `MemoryCard`
- `RetrievalCandidate`
- `ActivationResult`
- `RetrievalLedgerRecord`
- `RerankProvider`
- `EmbeddingProvider`
- `VectorIndex`
- `PlasticityEngine`
- `SleepPlanner`

## Packaging Boundary

`neuromem-runtime` v0.2.0 is self-contained. User code should import `neuromem_runtime`; bundled implementation packages are internal compatibility layers.

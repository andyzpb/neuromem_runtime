# NeuroMem Runtime API

## Entry Point

```python
memory = await MemoryRuntime.local(namespace="default", path=".neuromem")
```

This creates a local workspace with SQLite memory storage and trace files.
The SQLite database also stores experience events and ledger events.

## Actions

| Method | Purpose |
| --- | --- |
| `observe(event)` | Record an experience event and, by default, auto-commit a compatible memory item. |
| `observe(event, auto_commit=False)` | Record only an immutable `ExperienceEvent` and return its ids in an `EvidenceBundle`. |
| `query(query, budget_tokens=800)` | Return a prompt-ready `MemoryContext`. |
| `propose(input)` | Produce a deterministic structured `MemoryPolicy`. |
| `commit(policy)` | Validate and apply a governed policy. |
| `mutate(policy)` | Alias for `commit(policy)`. |
| `sleep()` | Run replay consolidation. |
| `forget(memory_id, action="inhibit")` | Apply governed forgetting. |
| `replay_trace(trace_id)` | Return replayable trace data. |

Physical deletion requires `authorize_delete=True`.

`query(...)` uses a single retrieval transaction. The returned `MemoryContext`
and persisted trace share the same selected ids, scores, and trace id.

## Ledger

`MemoryRuntime.local(...)` exposes `memory.ledger` for transaction audit.

CLI commands:

```bash
nmem ledger show TXN_ID
nmem ledger why-written MEM_ID
nmem ledger why-retrieved TRACE_ID MEM_ID
nmem ledger replay --to-txn TXN_ID
nmem ledger diff TXN_A TXN_B
```

## Policy Providers

`MemoryRuntime.local(...)` accepts an optional `policy_provider`.

Built-in providers:

- `DeterministicPolicyProvider`
- `OpenAICompatiblePolicyProvider`
- `DeepSeekPolicyProvider`

Providers return structured `MemoryPolicy` objects. They do not mutate memory.
All mutations still pass through `commit()` validation.

Provider JSON can also be validated as `MemoryPolicyV2` using Pydantic. The v2
policy shape is the forward path for governed mutation transactions.

## Governed Runtime Surfaces

The package exports the protocol surfaces used by the governed mutation runtime:

- `ExperienceEvent`
- `MemoryPolicyV2`
- `ValidatedMutation`
- `MemoryDelta`
- `GraphDelta`
- `LifecycleDelta`
- `IndexDelta`
- `LedgerEvent`
- `MemoryLedger`
- `LifecycleStateMachine`
- `ValidatorStack`
- `RetrievalTraceMetadata`
- `EmbeddingProvider`
- `VectorIndex`
- `PlasticityEngine`
- `SleepPlanner`

Base retrieval remains local and deterministic. Dense vectors, learned sparse
retrieval, multi-vector retrieval, and reranking should be implemented through
explicit adapters.

## Packaging Boundary

`neuromem-runtime` v0.1.0 is self-contained. Application code should import
`neuromem_runtime`; other bundled packages are implementation details.

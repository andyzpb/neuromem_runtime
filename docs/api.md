# NeuroMem Runtime Manual

## Runtime Model

NeuroMem Runtime is a local-first Memory Mutation Runtime for long-running LLM agents. The public package is `neuromem-runtime`; application code imports `neuromem_runtime`.

The runtime separates three things:

- **Experience events**: immutable observations recorded through `observe()`.
- **Memory mutations**: explicit writes, candidate Frames, edge evidence, forgetting evidence, replay compilation, graph materialization, and access effects.
- **Ledger records**: hash-linked transaction events that explain and replay memory effects.

The product rule is simple: durable memory changes go through `PolicyExecutor`, deterministic validation, atomic SQLite commit or rollback, delta capture, version snapshots, and ledger events. Destructive `UPDATE`, `DELETE_REQUEST`, and physical delete are not product behavior; corrections are represented by new evidence, supersession, suppression, or archival evidence.

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

The SQLite database stores memories, memory cards, candidate Frames, edge evidence events, worldview slots, worldview candidates, worldview candidate events, impact assessments, associative edge caches, logic node/edge caches, experience events, ledger events, memory versions, and split graph/frame versions.

## Public Actions

| Method | Purpose |
| --- | --- |
| `observe(event)` | Record only an immutable `ExperienceEvent`; no long-term memory is written. |
| `observe_and_commit(event)` | Explicitly validate and commit a long-term memory from an event. |
| `observe_and_route(event)` | Measure Worldview Impact and route the event to ledger-only, support evidence, candidate Frame, worldview candidate, clarification, quarantine, or sleep priority. |
| `query(query, budget_tokens=800, lens="auto")` | Return a prompt-ready `MemoryContext` through a deterministic retrieval lens. |
| `propose(input)` | Produce a structured `MemoryPolicy` or `MemoryPolicyV2` with the configured provider. |
| `commit(policy)` | Validate and apply governed memory changes. |
| `mutate(policy)` | Alias for `commit(policy)`. |
| `sleep()` | Replay high-impact/conflicted/repeated clusters into compiled Frames and evidence links without mutating source memories. |
| `after_turn(trace_id, outcome, feedback=None)` | Append success/failure outcome evidence for selected memories from a trace. |
| `forget(memory_id, action="inhibit")` | Apply governed forgetting. |
| `resolve_worldview(query=None, lens="auto", namespace=None, as_of=None)` | Resolve the current Worldview Packet directly. |
| `materialize_worldview(namespace=None)` | Rebuild worldview slots/candidates and materialized edge caches from append-only journals. |
| `rebuild_materialized_views(namespace=None)` | Alias for rebuilding materialized worldview/edge caches. |
| `replay_trace(trace_id)` | Return trace plus ledger-backed deltas and replay data. |

Physical deletion is unsupported on the product runtime. Use `forget(..., action="inhibit" | "decay" | "invalidate" | "archive" | "compress")` to append lifecycle evidence while preserving audit history.

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

`observe_and_commit()` still goes through evidence validation, `PolicyExecutor`, ledger phases, memory version snapshots, impact audit, and support edge evidence.

`observe_and_route()` is the default write-pressure gate for uncertain input. It always records the immutable experience event and impact assessment, then chooses one append-only route:

- `ledger_only`: keep only the experience event and impact assessment.
- `append_evidence`: append support evidence to existing worldview candidates; no durable memory is created.
- `propose_frame`: append a candidate Frame from the event.
- `propose_worldview_candidate`: append a candidate Frame, candidate event, and possible supersession evidence.
- `ask_clarification`: return a structured clarification payload; no long-term memory is written.
- `quarantine`: append an audit event; the input does not enter the active worldview.
- `sleep_priority`: append a replay priority marker.

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
- `frame_deltas`
- `associative_deltas`
- `logic_deltas`
- `graph_deltas` bridge inputs converted into split structural deltas
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

## Worldview Runtime

The current worldview is a projection, not a mutable truth table:

```text
append-only evidence + Frames + edge evidence + lifecycle records
        -> Worldview Impact Meter
        -> Worldview Resolver
        -> Worldview Snapshot + supporting memories
```

SQLite uses additive tables for this projection:

- `edge_evidence_events`
- `impact_assessments`
- `worldview_slots`
- `worldview_candidates`
- `worldview_candidate_events`

`associative_edges` and `logic_edges` are materialized caches. They are rebuilt from edge evidence by `materialize_worldview()` / `rebuild_materialized_views()`. Clearing those cache tables does not erase source evidence.

Worldview slot kinds are fixed:

- `fact`
- `preference`
- `constraint`
- `procedure`
- `schema`
- `hypothesis`
- `suppression`

Candidate scoring combines support strength, provenance strength, recency validity, utility success, lifecycle commitment, and user confirmation, then subtracts contradiction, inhibition, supersession, and staleness. A slot is conflicted when the two top candidates differ by less than `0.12`, or when active `contradict` evidence is present. Active `supersede` evidence moves the older candidate out of the normal prompt while keeping it visible in `historical` and `audit` lenses.

Resolver lenses:

- `logical`: validated/current facts, preferences, and constraints
- `procedural`: procedures, schemas, and failure patterns
- `historical`: includes suppressed and superseded candidates
- `audit`: includes evidence chains and rejected candidates
- `associative`: active worldview plus lower-commitment memory support

`MemoryContext` includes:

- `worldview`
- `worldview_trace`
- `prompt_sections`

`MemoryContext.to_prompt()` renders the Worldview Snapshot first, then supporting memory snippets.

## Activation Retrieval

Base retrieval is local and deterministic:

```text
QueryPlanV2 -> Contextual Memory Cards
            -> FTS5/BM25 + lexical/entity/current/procedural/canonical candidates
            -> optional dense/rewrite/HyDE semantic candidates
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
- `query_rewrites`
- `hyde_query`
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

Dense embeddings, query rewrite, HyDE, alias expansion, and rerankers are opt-in provider surfaces. Pass providers to `MemoryRuntime.local(...)`; generated rewrites and HyDE text only participate in retrieval and are never written as memory or evidence.

```python
memory = await nmem.MemoryRuntime.local(
    embedding_provider=my_embedding_provider,
    query_rewrite_provider=my_rewriter,
    hyde_provider=my_hyde,
)
```

## Progressive Crystallization

NeuroMem treats logical structure as a governed compiled layer, not as the default form of memory. The mutation path is:

```text
GraphCandidateGenerator
  -> RelationProposer
  -> Frame extraction
  -> FrameValidator / LogicRelationValidator
  -> PolicyExecutor transaction
  -> MemoryLedger
```

`MemoryPolicyV2.frame_deltas`, `associative_deltas`, and `logic_deltas` are the first-class structural mutation channels. Associative deltas connect memory ids with low-commitment activation relations such as `retrieved_with`, `coactivated_with`, `precedes`, and `used_with_success`. Logic deltas connect Frame endpoints and require evidence ids plus a proof obligation. Frame ontology currently covers Episode, Fact, Claim, Procedure, Preference, Constraint, Entity, Schema, and FailurePattern frames.

`graph_deltas` are accepted as bridge inputs and are converted to split structural deltas inside the executor. New code should use the split fields directly.

Retrieval lenses:

- `associative`: similar episodes and low-commitment graph activation
- `logical`: validated facts, preferences, constraints, and consistent current knowledge
- `procedural`: procedures, schemas, and failure-avoidance patterns
- `historical`: supersession and temporal history
- `audit`: ledger, evidence, validator, and rollback paths

## Forgetting

Supported actions:

- `decay`
- `inhibit`
- `invalidate`
- `archive`
- `compress`
- `delete`

`delete` is rejected. The other actions append lifecycle/edge evidence only; the source memory record remains unchanged. Normal retrieval hides suppressed memories, while `historical` and `audit` lenses keep them visible for review.

## Sleep

`sleep()` runs append-only replay compilation and writes a governed sleep transaction. It does not call the legacy mutable `neuro_sleep()` path by default, and it does not rewrite source memory maturity, summaries, tags, or provenance. The report includes:

- processed memory count
- replay clusters
- compiled Frame ids and source links
- empty source memory/lifecycle deltas unless a future policy explicitly adds append-only suggestions
- graph candidates, approved graph deltas, frame candidates, validated frames, logic promotions, compiled schemas, rejected crystallizations, and suppressed stale paths
- ledger transaction ids

Replay cluster selection prioritizes high-impact events, active conflict/supersession/inhibition evidence, recently selected trace memories, and repeated keyword/entity patterns.

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
- `FrameDeltaProposal`
- `AssociativeEdgeProposal`
- `LogicEdgeProposal`
- `GraphDeltaProposal`
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
- `LocalVectorIndex`
- `QueryRewriteProvider`
- `HyDEProvider`
- `EntityAliasResolver`
- `StaticEntityAliasResolver`
- `VectorIndex`
- `GraphBuildContext`
- `GraphCandidateGenerator`
- `GraphRelationCandidate`
- `GraphProposalProvider`
- `GraphDeltaValidator`
- `GraphMutationCommitter`
- `MemoryFrame`
- `AssociativeEdge`
- `LogicEdge`
- `RetrievalLens`
- `FrameExtractor`
- `FrameValidator`
- `LogicRelationValidator`
- `CrystallizationPlanner`
- `PlasticityEngine`
- `SleepPlanner`

## Packaging Boundary

`neuromem-runtime` v0.2.0 is self-contained. User code should import `neuromem_runtime`; bundled implementation packages are internal compatibility layers.

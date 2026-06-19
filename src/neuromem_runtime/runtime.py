from __future__ import annotations

import json
import os
import time
import tomllib
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem.core.models import MemoryItem
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.stores.sqlite_store import SQLiteMemoryStore

from neuromem_runtime.crystallization import (
    DeterministicCrystallizationPlanner,
    DeterministicFrameExtractor,
    frame_id_for_memory,
    frame_proposal_for_memory,
    infer_frame_type,
)
from neuromem_runtime.deltas import LifecycleDelta, MemoryDelta
from neuromem_runtime.executor import PolicyExecutionContext, PolicyExecutor
from neuromem_runtime.impact import WorldviewImpactAssessment, WorldviewImpactMeter
from neuromem_runtime.ledger import EdgeEvidenceEvent, ExperienceEvent, LedgerEvent, MemoryLedger, WorldviewCandidateEvent, WorldviewSlotRecord
from neuromem_runtime.materializer import EdgeEvidenceAppender, WorldviewMaterializer
from neuromem_runtime.policy_v2 import EvidenceRef, FrameDeltaProposal, LogicEdgeProposal, MemoryPolicyV2
from neuromem_runtime.providers import DeterministicPolicyProvider, PolicyProvider
from neuromem_runtime.performance import BackgroundJobQueue, EmbeddingCache, RetrievalCache, RuntimeTiming, TimingSpan, stable_hash
from neuromem_runtime.retrieval import (
    EmbeddingProvider,
    EntityAliasResolver,
    HyDEProvider,
    LocalVectorIndex,
    QueryRewriteProvider,
    RerankProvider,
    RetrievalTraceMetadata,
    StaticEntityAliasResolver,
    VectorIndex,
)
from neuromem_runtime.semantic_graph import (
    DeterministicRelationProposer,
    GraphBuildContext,
    GraphCandidateGenerator,
    GraphMode,
    GraphProposalProvider,
)
from neuromem_runtime.sleep import CompilationDelta, ConsolidationDelta, LedgerReport, ReplayBatch, SleepPlanner, SleepReport, SuppressionDelta
from neuromem_runtime.types import EvidenceBundle, MemoryContext, MemoryEvent, MemoryQuery, RuntimeConfig, event_to_dict
from neuromem_runtime.validators import ValidatorStack
from neuromem_runtime.worldview import WorldviewResolver


class MemoryRuntime:
    """Product facade over the research NeuroMem runtime."""

    def __init__(
        self,
        config: RuntimeConfig,
        runtime: NeuroMemRuntime,
        policy_provider: PolicyProvider | None = None,
        *,
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
    ) -> None:
        self.config = config
        if mutation_mode == "mutable_compat":
            raise ValueError("mutable_compat is no longer supported; use append-only evidence and materialized views")
        self.config.graph_mode = graph_mode
        self.config.crystallization_mode = crystallization_mode
        self.config.graph_storage = graph_storage
        self.config.mutation_mode = mutation_mode  # type: ignore[assignment]
        self._runtime = runtime
        self._allow_unsafe_internal = allow_unsafe_internal
        self._validator_stack = ValidatorStack()
        self._policy_provider = policy_provider or DeterministicPolicyProvider()
        self._ledger = MemoryLedger(config.db_path)
        self._executor = PolicyExecutor(runtime, self._ledger, self._validator_stack)
        self._edge_appender = EdgeEvidenceAppender(self._ledger)
        self._worldview_materializer = WorldviewMaterializer(self._ledger)
        self._impact_meter = WorldviewImpactMeter()
        self._worldview_resolver = WorldviewResolver()
        self._sleep_planner = SleepPlanner()
        self._graph_mode = graph_mode
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index or (LocalVectorIndex() if embedding_provider is not None else None)
        self._rerank_provider = rerank_provider
        self._query_rewrite_provider = query_rewrite_provider
        self._hyde_provider = hyde_provider
        self._entity_alias_resolver = entity_alias_resolver or StaticEntityAliasResolver()
        self._graph_candidate_generator = GraphCandidateGenerator()
        self._relation_proposer = relation_proposer or DeterministicRelationProposer()
        self._frame_extractor = DeterministicFrameExtractor()
        self._crystallization_planner = DeterministicCrystallizationPlanner()
        self._retrieval_cache = RetrievalCache(ttl_seconds=config.retrieval_cache_ttl_seconds)
        self._retrieval_cache_fingerprints: dict[str, tuple[str, str]] = {}
        self._embedding_cache = EmbeddingCache(config.db_path) if config.embedding_cache_enabled and embedding_provider is not None else None
        self._background_jobs = BackgroundJobQueue()

    @classmethod
    async def local(
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
        if mutation_mode == "mutable_compat":
            raise ValueError("mutable_compat is no longer supported; use append-only evidence and materialized views")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        traces = root / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        db_path = root / "memory.sqlite3"
        config = RuntimeConfig(
            namespace=namespace,
            path=root,
            db_path=db_path,
            traces_path=traces,
            agent_id=agent_id,
            mode=mode,
            model_policy_enabled=policy_provider is not None,
            graph_mode=graph_mode,
            crystallization_mode=crystallization_mode,
            graph_storage=graph_storage,
            mutation_mode=mutation_mode,  # type: ignore[arg-type]
            embedding_cache_enabled=_bool_env("NEUROMEM_EMBEDDING_CACHE", True),
            retrieval_cache_ttl_seconds=_int_env("NEUROMEM_RETRIEVAL_CACHE_TTL_SECONDS", 20),
            retrieval_graph_commit=_graph_commit_env(),
            retrieval_mode="full_debug" if os.environ.get("NEUROMEM_CHAT_RETRIEVAL_MODE", "").strip().lower() == "full" else "auto",
            ollama_keep_alive=os.environ.get("NEUROMEM_OLLAMA_KEEP_ALIVE", "30m"),
        )
        _write_config(root / "config.toml", config)
        runtime = NeuroMemRuntime(
            agent_id=agent_id,
            namespace=namespace,
            store=SQLiteMemoryStore(db_path),
            db_path=db_path,
        )
        return cls(
            config=config,
            runtime=runtime,
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

    @classmethod
    async def from_config(cls, path: str | Path = ".neuromem") -> "MemoryRuntime":
        root = Path(path)
        config_path = root / "config.toml"
        if not config_path.exists():
            return await cls.local(path=root)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return await cls.local(
            namespace=str(data.get("namespace", "default")),
            path=root,
            agent_id=str(data.get("agent_id", "local-agent")),
            mode=str(data.get("mode", "lite")),
            graph_mode=str(data.get("graph_mode", "governed_hybrid")),  # type: ignore[arg-type]
            crystallization_mode=str(data.get("crystallization_mode", "governed_progressive")),
            graph_storage=str(data.get("graph_storage", "split")),
            mutation_mode=str(data.get("mutation_mode", "append_only_view")),
        )

    @property
    def internal_runtime(self) -> NeuroMemRuntime:
        if not self._allow_unsafe_internal:
            raise RuntimeError("internal_runtime is an unsafe debug escape hatch; use MemoryRuntime.local(..., allow_unsafe_internal=True) and unsafe_internal_runtime")
        return self._runtime

    @property
    def unsafe_internal_runtime(self) -> NeuroMemRuntime:
        if not self._allow_unsafe_internal:
            raise RuntimeError("unsafe_internal_runtime requires MemoryRuntime.local(..., allow_unsafe_internal=True)")
        return self._runtime

    @property
    def ledger(self) -> MemoryLedger:
        return self._ledger

    async def observe(self, event: MemoryEvent | dict[str, object], *, auto_commit: bool = False) -> EvidenceBundle:
        payload = event_to_dict(event)
        content = str(payload.get("content", ""))
        if not content:
            raise ValueError("observe() requires a non-empty content field")
        experience = self._ledger.record_experience(
            ExperienceEvent(
                content=content,
                namespace=self.config.namespace,
                source=str(payload.get("source", "user")),
                metadata={key: value for key, value in payload.items() if key != "content"},
            )
        )
        self._ledger.append(
            LedgerEvent(
                transaction_id=f"txn_{experience.event_id}",
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase="AUDITED",
                event_type="experience_observed",
                operation="NOOP",
                proposer="system",
                evidence=[experience.to_dict()],
                audit={"experience_event": experience.to_dict(), "auto_commit": auto_commit},
            )
        )
        impact = self._assess_experience(experience)
        self._ledger.record_impact_assessment(impact)
        self._ledger.append(
            LedgerEvent(
                transaction_id=f"txn_{experience.event_id}",
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase="AUDITED",
                event_type="worldview_impact_assessed",
                operation="ASSESS_IMPACT",
                proposer="deterministic",
                evidence=[experience.to_dict()],
                audit={"impact_assessment": impact.to_dict()},
            )
        )
        if not auto_commit:
            return EvidenceBundle(memory_id=None, content=content, evidence=[], event_id=experience.event_id, content_hash=experience.content_hash, impact=impact.to_dict())
        return await self.observe_and_commit({**payload, "evidence": experience.event_id}, experience=experience)

    async def observe_and_commit(self, event: MemoryEvent | dict[str, object], *, experience: ExperienceEvent | None = None) -> EvidenceBundle:
        payload = event_to_dict(event)
        content = str(payload.get("content", ""))
        if not content:
            raise ValueError("observe_and_commit() requires a non-empty content field")
        if experience is None:
            observed = await self.observe(payload, auto_commit=False)
            if observed.event_id is None:
                raise RuntimeError("observe_and_commit() could not record experience event")
            stored = self._ledger.get_experience(observed.event_id)
            if stored is None:
                raise RuntimeError("observe_and_commit() could not load recorded experience event")
            experience = stored
        impact = self._ledger.get_impact_assessment(experience.event_id, namespace=self.config.namespace)
        if impact is None:
            assessed = self._assess_experience(experience)
            self._ledger.record_impact_assessment(assessed)
            impact = assessed.to_dict()
        memory_type = "episodic"
        if payload.get("type") == "user_preference":
            memory_type = "preference"
        elif payload.get("type") == "rule":
            memory_type = "procedural"
        elif payload.get("type") == "fact":
            memory_type = "semantic"
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=content),
            write=WritePlan(
                operation="ADD",
                memory_type=memory_type,
                content=content,
                confidence=float(payload.get("confidence", 0.85) or 0.85),
                evidence_ids=[experience.event_id],
            ),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="explicit observe_and_commit",
            source="deterministic",
        )
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase="after_step",
                task=str(payload.get("task") or content),
                query=content,
                state={
                    "status": str(payload.get("outcome", "success")),
                    "confidence": float(payload.get("confidence", 0.85) or 0.85),
                    "prediction_error": float(payload.get("prediction_error", 0.0) or 0.0),
                    "future_utility": float(payload.get("future_utility", 0.0) or 0.0),
                },
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
            ),
        )
        result.trace.query_plan["worldview_impact"] = impact
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        self._invalidate_retrieval_cache()
        memory_id = result.created_memory_ids[0] if result.created_memory_ids else None
        stored_item = self._runtime.store.get_memory(memory_id) if self._runtime.store is not None and memory_id else None
        if self._graph_mode != "off" and memory_id is not None:
            graph_result = self._commit_auto_graph(
                selected_memory_ids=[],
                target_memory_ids=[memory_id],
                evidence_ids=[experience.event_id],
                outcome=str(payload.get("outcome", "success")),
                mutation_trace={"operation": "observe_and_commit", "memory_id": memory_id},
            )
            if graph_result is not None:
                result.trace.query_plan["semantic_graph_builder"] = graph_result
            self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        explicit_evidence: dict[str, object] | None = None
        if memory_id is not None:
            edge_event = EdgeEvidenceEvent(
                namespace=self.config.namespace,
                source_kind="event",
                source_id=experience.event_id,
                target_kind="memory",
                target_id=memory_id,
                relation="supports",
                relation_family="worldview",
                event_type="support",
                delta_weight=0.35,
                confidence=float(payload.get("confidence", 0.85) or 0.85),
                evidence_ids=[experience.event_id],
                proof_obligation="explicit observe_and_commit ADD",
                proposer="deterministic",
            )
            self._edge_appender.append(edge_event)
            explicit_evidence = {"edge_evidence_event": edge_event.to_dict(), "materialized": self._materialize_worldview_sync()}
            result.trace.query_plan["explicit_add_evidence"] = explicit_evidence
            self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        return EvidenceBundle(
            memory_id=memory_id,
            content=stored_item.content if stored_item is not None else content,
            memory_type=stored_item.type if stored_item is not None else memory_type,
            evidence=list(stored_item.evidence) if stored_item is not None else [experience.event_id],
            event_id=experience.event_id,
            content_hash=experience.content_hash,
            impact=impact,
        )

    async def assess_impact(self, event_id: str) -> dict[str, object]:
        stored = self._ledger.get_impact_assessment(event_id, namespace=self.config.namespace)
        if stored is not None:
            return stored
        experience = self._ledger.get_experience(event_id, namespace=self.config.namespace)
        if experience is None:
            raise ValueError(f"experience event not found: {event_id}")
        assessment = self._assess_experience(experience)
        self._ledger.record_impact_assessment(assessment)
        return assessment.to_dict()

    async def observe_and_route(self, event: MemoryEvent | dict[str, object]) -> dict[str, object]:
        observed = await self.observe(event, auto_commit=False)
        if observed.event_id is None:
            raise RuntimeError("observe_and_route() could not record experience")
        impact = observed.impact or await self.assess_impact(observed.event_id)
        decision = str(impact.get("decision"))
        experience = self._ledger.get_experience(observed.event_id, namespace=self.config.namespace)
        if experience is None:
            raise RuntimeError("observe_and_route() could not reload experience")
        if decision == "ledger_only":
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False}
        if decision == "ask_clarification":
            payload = {
                "event_id": experience.event_id,
                "slot_keys": [str(slot.get("slot_key")) for slot in impact.get("impacted_slots", []) if isinstance(slot, dict)],
                "reason": impact.get("reason"),
                "question": "This input may change an existing worldview slot. Please confirm whether it should supersede the current assumption.",
            }
            self._append_route_audit(experience, decision, payload)
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False, "clarification": payload}
        if decision == "quarantine":
            payload = {"event_id": experience.event_id, "reason": impact.get("reason"), "risk": (impact.get("vector") or {}).get("risk") if isinstance(impact.get("vector"), dict) else None}
            self._append_route_audit(experience, decision, payload)
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False, "quarantine": payload}
        if decision == "append_evidence":
            evidence_events = self._append_support_evidence(experience, impact)
            materialized = await self.materialize_worldview()
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False, "edge_evidence_events": evidence_events, "materialized": materialized}
        if decision in {"propose_frame", "propose_worldview_candidate"}:
            route = await self._propose_candidate_from_experience(experience, impact, worldview_candidate=decision == "propose_worldview_candidate")
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False, **route}
        if decision == "sleep_priority":
            marker = self._append_sleep_priority_marker(experience, impact)
            return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False, "sleep_priority": marker}
        self._append_route_audit(experience, decision, {"event_id": experience.event_id, "reason": impact.get("reason")})
        return {"decision": decision, "impact": impact, "bundle": observed.to_dict(), "committed": False}

    async def query(
        self,
        query: str | MemoryQuery,
        budget_tokens: int = 800,
        filters: dict[str, object] | None = None,
        *,
        lens: str = "auto",
        namespace: str | None = None,
        top_k: int | None = None,
        include_worldview: bool = True,
    ) -> MemoryContext:
        timing = RuntimeTiming()
        query_obj = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query, budget_tokens=budget_tokens, filters=filters or {})
        resolved_lens = self._resolve_lens(query_obj.query, lens)
        query_obj.filters["retrieval_lens"] = resolved_lens
        if top_k is not None:
            query_obj.filters["top_k"] = top_k
        query_obj.filters.update(self._semantic_filters(query_obj.query))
        query_obj.filters["_retrieval_timing"] = timing
        if namespace is not None:
            query_obj.filters["namespace"] = namespace
        namespace_value = str(query_obj.filters.get("namespace") or self.config.namespace)
        store_version = self._store_version(namespace_value)
        filter_hash = self._retrieval_filter_hash(query_obj, resolved_lens=resolved_lens)
        cache_key = self._retrieval_cache_key(query_obj, resolved_lens=resolved_lens, store_version=store_version, filter_hash=filter_hash)
        cache_probe_key = self._retrieval_cache_probe_key(query_obj, resolved_lens=resolved_lens)
        diagnostic_miss_reason = self._retrieval_miss_reason(
            cache_probe_key,
            semantic_version=store_version,
            filter_hash=filter_hash,
        )
        cached = self._retrieval_cache.get(cache_key)
        if isinstance(cached, MemoryContext):
            self._retrieval_cache_fingerprints[cache_probe_key] = (store_version, filter_hash)
            cached_context = _copy_memory_context(cached)
            cached_context.cache = {
                **cached.cache,
                "retrieval_cache": "hit",
                "retrieval_cache_stats": self._retrieval_cache.stats(),
                "cache_key_version": store_version[:12],
                "semantic_version_short": store_version[:12],
                "filter_hash": filter_hash[:12],
                "miss_reason": None,
            }
            return cached_context
        before_states = self._memory_states()
        with TimingSpan(timing, "retrieval_ms"):
            results, trace = self._runtime.retrieve_with_trace(query_obj.query, filters=query_obj.filters, budget_tokens=query_obj.budget_tokens, task_id=query_obj.query)
        results = self._apply_retrieval_lens(results, resolved_lens)
        if resolved_lens not in {"historical", "audit"}:
            suppressed = self._ledger.active_suppressed_memory_ids(namespace_value)
            if suppressed:
                results = [result for result in results if getattr(getattr(result, "memory", None), "id", None) not in suppressed]
        if top_k is not None:
            results = results[:top_k]
        after_states = self._memory_states()
        text = "\n".join(f"- [{result.score:.2f}] {result.memory.content}" for result in results)
        trace_id = trace.trace_id
        transactions: list[dict[str, object]] = []
        trace.selected_memory_ids = [result.memory.id for result in results]
        trace.final_context_tokens = len(text.split())
        candidate_details = trace.query_plan.get("candidate_details", {}) if isinstance(trace.query_plan.get("candidate_details", {}), dict) else {}
        retrieval_ledger = trace.query_plan.get("retrieval_ledger", {}) if isinstance(trace.query_plan.get("retrieval_ledger", {}), dict) else {}
        query_plan_v2 = trace.query_plan.get("query_plan_v2", {}) if isinstance(trace.query_plan.get("query_plan_v2", {}), dict) else {}
        embedding_cache_stats = trace.query_plan.get("embedding_cache_stats", {}) if isinstance(trace.query_plan.get("embedding_cache_stats", {}), dict) else {}
        source_channels = list(retrieval_ledger.get("channel_candidates", {}).keys()) if isinstance(retrieval_ledger.get("channel_candidates", {}), dict) else list(trace.source_channels)
        embedding_mode = "enabled" if self._embedding_provider is not None else "disabled"
        trace.query_plan.update(
            {
                "retrieval_metadata": RetrievalTraceMetadata(
                    retrieval_mode=str(retrieval_ledger.get("retrieval_mode") or query_plan_v2.get("mode") or "local_activation"),
                    embedding_mode=embedding_mode,
                    candidate_sources=[str(item) for item in source_channels],
                    fusion_strategy="rrf+ppr+lite_rerank",
                    rank_before_fusion=[result.memory.id for result in results],
                    rank_after_fusion=[result.memory.id for result in results],
                ).to_dict()
            }
        )
        trace.query_plan["retrieval_lens"] = resolved_lens
        txn = trace.to_transactions()[0]
        access_deltas = _memory_deltas_for_fields(
            before_states,
            after_states,
            fields={"access_count", "activation_count", "last_accessed_at"},
            reason="retrieval access update",
        )
        trace.query_plan["retrieval_access_deltas"] = access_deltas
        trace_dict = trace.to_dict()
        self._ledger.append(
            LedgerEvent(
                transaction_id=txn.transaction_id,
                trace_id=trace.trace_id,
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase=txn.phase,
                event_type="memory_retrieved",
                operation="RETRIEVE",
                proposer=txn.proposed_by,
                validator_decision=txn.validator_decision,
                evidence=[{"memory_id": memory_id, "source": "retrieval"} for memory_id in trace.selected_memory_ids],
                targets=trace.selected_memory_ids,
                graph_delta=[{"paths": trace.graph_paths, "scores": trace.diffusion_scores}] if trace.graph_paths or trace.diffusion_scores else [],
                lifecycle_delta=[{"memory_id": memory_id, "reason": reason} for memory_id, reason in trace.suppression_reasons.items()],
                index_delta=[{"index": "sqlite_fts5", "status": "read"}, {"index": "memory_graph", "status": "activated"}],
                memory_delta=access_deltas,
                audit=trace_dict,
            )
        )
        retrieval_graph = self._handle_retrieval_graph_commit(trace, timing)
        if retrieval_graph is not None:
            trace.query_plan["semantic_graph_builder"] = retrieval_graph
        transactions = self._ledger.events_for_trace(trace.trace_id, namespace=self.config.namespace) or [transaction.to_dict() for transaction in trace.to_transactions()]
        self._persist_trace(trace_id, trace.to_dict())
        cache_info = {
            "retrieval_cache": "miss",
            "retrieval_cache_stats": self._retrieval_cache.stats(),
            "embedding_cache": dict(embedding_cache_stats),
            "cache_key_version": store_version[:12],
            "semantic_version_short": store_version[:12],
            "filter_hash": filter_hash[:12],
            "miss_reason": diagnostic_miss_reason or self._retrieval_cache.last_miss_reason,
        }
        retrieval_metadata = dict(trace.query_plan.get("retrieval_metadata", {})) if isinstance(trace.query_plan.get("retrieval_metadata", {}), dict) else {}
        retrieval_metadata["retrieval_lens"] = resolved_lens
        worldview_payload: dict[str, object] | None = None
        worldview_trace: dict[str, object] | None = None
        prompt_sections: dict[str, str] = {}
        if include_worldview and self._runtime.store is not None:
            materialized = await self.materialize_worldview(namespace=namespace_value)
            memories = self._runtime.store.list_memories(namespace=namespace_value)
            frames = self._runtime.store.list_logic_nodes(namespace=namespace_value)
            associative_edges = self._runtime.store.list_associative_edges(namespace=namespace_value)
            logic_edges = self._runtime.store.list_logic_edges(namespace=namespace_value)
            edge_events = self._ledger.edge_evidence_events(namespace=namespace_value)
            impact_assessments = self._ledger.impact_assessments(namespace=namespace_value, limit=50)
            worldview_candidates = self._ledger.worldview_candidates(namespace=namespace_value)
            packet = self._worldview_resolver.resolve(
                namespace=namespace_value,
                lens=resolved_lens,
                query=query_obj.query,
                memories=memories,
                frames=frames,
                associative_edges=associative_edges,
                logic_edges=logic_edges,
                edge_events=edge_events,
                impact_assessments=impact_assessments,
                worldview_candidates=worldview_candidates,
                suppressed_memory_ids=self._ledger.active_suppressed_memory_ids(namespace_value),
            )
            worldview_payload = packet.to_dict()
            worldview_payload["prompt"] = packet.to_prompt()
            worldview_trace = {
                "materialized": materialized,
                "candidate_count": len(worldview_candidates),
                "edge_event_count": len(edge_events),
                "impact_assessment_count": len(impact_assessments),
            }
            prompt_sections = _prompt_sections_from_worldview(worldview_payload)
            trace.query_plan["worldview_packet"] = {key: value for key, value in worldview_payload.items() if key != "prompt"}
            trace.query_plan["worldview_trace"] = worldview_trace
            self._persist_trace(trace_id, trace.to_dict())
        context = MemoryContext(
            query=query_obj.query,
            text=text,
            selected_memory_ids=[result.memory.id for result in results],
            trace_id=trace_id,
            results=[
                {
                    "memory_id": result.memory.id,
                    "score": result.score,
                    "content": result.memory.content,
                    "type": result.memory.type,
                    "why_retrieved": list(result.why_retrieved),
                    "score_components": dict(candidate_details.get(result.memory.id, {}).get("channel_scores", {})) if isinstance(candidate_details.get(result.memory.id, {}), dict) else {},
                    "graph_paths": candidate_details.get(result.memory.id, {}).get("graph_paths", []) if isinstance(candidate_details.get(result.memory.id, {}), dict) else [],
                    "reranker_score": candidate_details.get(result.memory.id, {}).get("reranker_score", result.score) if isinstance(candidate_details.get(result.memory.id, {}), dict) else result.score,
                    "lifecycle_reason": candidate_details.get(result.memory.id, {}).get("lifecycle_reason", "active") if isinstance(candidate_details.get(result.memory.id, {}), dict) else "active",
                    "provenance_ids": list(result.memory.evidence),
                    "retrieval_lens": resolved_lens,
                }
                for result in results
            ],
            transactions=transactions,
            timing=timing.to_dict(),
            cache=cache_info,
            retrieval_metadata=retrieval_metadata,
            worldview=worldview_payload,
            worldview_trace=worldview_trace,
            prompt_sections=prompt_sections,
        )
        self._retrieval_cache.set(cache_key, context, semantic_version=store_version, filter_hash=filter_hash)
        self._retrieval_cache_fingerprints[cache_probe_key] = (store_version, filter_hash)
        return context

    async def propose(self, value: str | dict[str, object]) -> MemoryPolicy | MemoryPolicyV2:
        data = {"query": value} if isinstance(value, str) else dict(value)
        return self._policy_provider.propose(data)

    async def commit(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        legacy_policy = self._executor._coerce_policy(policy)  # noqa: SLF001 - public facade needs task derivation for both policy generations.
        destructive_reason = _destructive_policy_reason(policy, legacy_policy)
        if destructive_reason is not None:
            return _append_only_rejection(legacy_policy.write.target_memory_id or legacy_policy.forget.target_memory_id, destructive_reason)
        suppression = self._append_only_suppression_from_policy(legacy_policy)
        if suppression is not None:
            return suppression
        phase = "after_step" if legacy_policy.write.operation != "NOOP" or legacy_policy.forget.operation != "NOOP" or legacy_policy.consolidation.enabled else "before_step"
        task = legacy_policy.retrieval.query or legacy_policy.write.content or legacy_policy.forget.target_memory_id or "memory mutation"
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase=phase,
                task=task,
                query=legacy_policy.retrieval.query or task,
                state={"status": "success", "confidence": legacy_policy.write.confidence or 0.75},
                authorize_delete=authorize_delete,
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
            ),
        )
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        structural_mutation = isinstance(policy, MemoryPolicyV2) and bool(policy.frame_deltas or policy.associative_deltas or policy.logic_deltas or policy.graph_deltas)
        structural_evidence: dict[str, object] | None = None
        if structural_mutation and isinstance(policy, MemoryPolicyV2) and result.validated_mutation.approved:
            structural_evidence = self._append_structural_evidence_from_policy(policy, result.trace.trace_id)
            result.trace.query_plan["append_only_structural_evidence"] = structural_evidence
            self._persist_trace(result.trace.trace_id, result.trace.to_dict())
            self._materialize_worldview_sync()
        if phase == "after_step" or structural_mutation:
            self._invalidate_retrieval_cache()
        value = result.trace.to_dict()
        value["mutation_execution_result"] = result.to_dict()
        if structural_evidence is not None:
            value["append_only_structural_evidence"] = structural_evidence
        return value

    def _append_only_suppression_from_policy(self, policy: MemoryPolicy) -> dict[str, object] | None:
        if policy.forget.operation not in {"DECAY", "INHIBIT", "INVALIDATE", "ARCHIVE"}:
            return None
        memory_id = policy.forget.target_memory_id
        if not memory_id:
            return _append_only_rejection(None, "append-only suppression requires a target memory id")
        item = self._runtime.store.get_memory(memory_id) if self._runtime.store is not None else None
        if item is None:
            return _append_only_rejection(memory_id, f"memory not found: {memory_id}")
        if item.namespace != self.config.namespace:
            return _append_only_rejection(memory_id, f"target memory outside namespace: {memory_id}")
        if item.privacy_level in {"user", "sensitive"} and item.acl:
            return _append_only_rejection(memory_id, "user is not authorized for private memory")
        transaction_id = f"txn_suppress_{uuid4().hex}"
        event_type = {
            "DECAY": "decay",
            "INHIBIT": "inhibit",
            "INVALIDATE": "expire",
            "ARCHIVE": "suppress",
        }[policy.forget.operation]
        edge_event = EdgeEvidenceEvent(
            namespace=self.config.namespace,
            source_kind="policy",
            source_id=transaction_id,
            target_kind="memory",
            target_id=memory_id,
            relation="inhibits" if event_type in {"inhibit", "suppress"} else event_type,
            relation_family="suppression",
            event_type=event_type,
            delta_weight=-0.7 if event_type in {"inhibit", "suppress", "expire"} else -0.2,
            confidence=0.9,
            evidence_ids=list(policy.write.evidence_ids) or [memory_id],
            proof_obligation=policy.forget.reason or policy.reason,
            proposer=policy.source or "deterministic",
        )
        self._ledger.append_edge_evidence(edge_event)
        audit = {"edge_evidence_event": edge_event.to_dict(), "append_only": True, "reason": policy.forget.reason or policy.reason}
        self._ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase="COMMITTED",
                event_type="suppression_event_appended",
                operation=policy.forget.operation,
                proposer=policy.source or "deterministic",
                validator_decision="approved",
                evidence=[{"memory_id": memory_id, "source": "forget"}],
                targets=[memory_id],
                lifecycle_delta=[{"memory_id": memory_id, "event_type": event_type, "reason": policy.forget.reason or policy.reason, "append_only": True}],
                audit=audit,
            )
        )
        self._ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase="AUDITED",
                event_type="audit_finalized",
                operation=policy.forget.operation,
                proposer=policy.source or "deterministic",
                validator_decision="approved",
                evidence=[{"memory_id": memory_id, "source": "forget"}],
                targets=[memory_id],
                audit=audit,
            )
        )
        self._invalidate_retrieval_cache()
        self._materialize_worldview_sync()
        return _append_only_suppression_result(memory_id, policy.forget.operation, event_type, edge_event, policy.forget.reason or policy.reason)

    async def mutate(self, policy: MemoryPolicy | MemoryPolicyV2, *, authorize_delete: bool = False) -> dict[str, object]:
        return await self.commit(policy, authorize_delete=authorize_delete)

    async def sleep(self) -> dict[str, object]:
        assert self._runtime.store is not None
        transaction_id = f"txn_sleep_{uuid4().hex}"
        trace_id = self._runtime.last_trace.trace_id if self._runtime.last_trace is not None else None
        replay_trace_ids = [trace_id] if trace_id is not None else []
        plan = self._sleep_planner.plan(policy="manual", replay_trace_ids=replay_trace_ids)
        memories = self._runtime.store.list_memories(namespace=self.config.namespace)
        sleep_clusters = self._select_sleep_replay_clusters(memories)
        targets = sorted({memory_id for cluster in sleep_clusters for memory_id in cluster})
        consolidation_report: dict[str, object] = {
            "processed": len(targets),
            "replay_clusters": sleep_clusters,
            "compressed_memory_ids": [],
            "archived_memory_ids": [],
            "consolidation_links": {},
            "append_only": True,
            "source_memories_mutated": False,
        }
        with self._runtime.store.transaction() as conn:  # type: ignore[attr-defined]
            for phase, event_type, audit in [
                ("PROPOSED", "sleep_plan_proposed", {"plan": plan.to_dict()}),
                ("VALIDATED", "sleep_validation_approved", {"plan": plan.to_dict()}),
                ("VALIDATED", "replay_batch_selected", {"replay_trace_ids": replay_trace_ids, "replay_clusters": sleep_clusters}),
            ]:
                self._ledger.append(
                    LedgerEvent(
                        transaction_id=transaction_id,
                        trace_id=trace_id,
                        namespace=self.config.namespace,
                        agent_id=self.config.agent_id,
                        phase=phase,
                        event_type=event_type,
                        operation="CONSOLIDATE",
                        proposer="deterministic",
                        validator_decision="approved" if phase != "PROPOSED" else "not_applicable",
                        targets=targets if event_type == "replay_batch_selected" else [],
                        audit=audit,
                    ),
                    conn=conn,
                )
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    trace_id=trace_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="COMMITTED",
                    event_type="consolidation_delta_committed",
                    operation="CONSOLIDATE",
                    proposer="deterministic",
                    validator_decision="approved",
                    targets=targets,
                    memory_delta=[],
                    lifecycle_delta=[],
                    graph_delta=[],
                    audit={"consolidation_report": consolidation_report, "append_only": True},
                ),
                conn=conn,
            )
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    trace_id=trace_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="COMMITTED",
                    event_type="suppression_delta_committed",
                    operation="CONSOLIDATE",
                    proposer="deterministic",
                    validator_decision="approved",
                    targets=[],
                    lifecycle_delta=[],
                    audit={"suppression_suggestions": [], "append_only": True},
                ),
                conn=conn,
            )
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    trace_id=trace_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="COMMITTED",
                    event_type="compilation_delta_committed",
                    operation="CONSOLIDATE",
                    proposer="deterministic",
                    validator_decision="approved",
                    targets=[],
                    memory_delta=[],
                    graph_delta=[],
                    audit={"compiled_frame_ids": [], "append_only": True},
                ),
                conn=conn,
            )
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    trace_id=trace_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="AUDITED",
                    event_type="sleep_audit_finalized",
                    operation="CONSOLIDATE",
                    proposer="deterministic",
                    validator_decision="approved",
                    targets=targets,
                    memory_delta=[],
                    lifecycle_delta=[],
                    graph_delta=[],
                    audit={"consolidation_report": consolidation_report, "append_only": True},
                ),
                conn=conn,
            )
        sleep_graph = self._commit_auto_graph(
            selected_memory_ids=[memory_id for cluster in sleep_clusters for memory_id in cluster],
            target_memory_ids=targets,
            evidence_ids=[f"sleep:{transaction_id}"],
            outcome="success",
            sleep_clusters=sleep_clusters,
            mutation_trace={"operation": "sleep_graph_compiler"},
            invalidate_retrieval_cache=False,
        ) or {}
        sleep_crystallization = self._commit_sleep_crystallization(
            sleep_clusters=sleep_clusters,
            evidence_ids=[f"sleep:{transaction_id}"],
            transaction_id=transaction_id,
        )
        graph_deltas = list(((sleep_graph.get("result") or {}).get("graph_deltas", []))) if isinstance(sleep_graph.get("result"), dict) else []
        if isinstance(sleep_crystallization.get("result"), dict):
            graph_deltas.extend(list(sleep_crystallization["result"].get("graph_deltas", [])))
        compiled_nodes = [str(item.get("frame_id")) for item in sleep_crystallization.get("compiled_schemas", []) if isinstance(item, dict) and item.get("frame_id")]
        consolidation_report["compressed_memory_ids"] = compiled_nodes
        consolidation_report["consolidation_links"] = {
            node: list(cluster)
            for node, cluster in zip(compiled_nodes, sleep_clusters, strict=False)
        }
        for node in compiled_nodes:
            for memory_id in targets:
                edge_event = EdgeEvidenceEvent(
                    namespace=self.config.namespace,
                    source_kind="memory",
                    source_id=memory_id,
                    target_kind="frame",
                    target_id=node,
                    relation="derived_from",
                    relation_family="logic",
                    event_type="derive",
                    delta_weight=0.34,
                    confidence=0.74,
                    evidence_ids=[f"sleep:{transaction_id}"],
                    proof_obligation="sleep replay compiled frame from source evidence",
                    proposer="sleep",
                )
                self._edge_appender.append(edge_event)
        trace = self._runtime.last_trace
        if trace is not None:
            self._persist_trace(trace.trace_id, trace.to_dict())
        materialized = self._materialize_worldview_sync()
        self._invalidate_retrieval_cache()
        sleep_report = SleepReport(
            plan=plan,
            replay=ReplayBatch(trace_ids=replay_trace_ids),
            consolidation=[
                ConsolidationDelta(source_memory_ids=list(cluster), target_memory_id=str(target), reason="replay consolidation")
                for target, cluster in dict(consolidation_report.get("consolidation_links", {})).items()
            ],
            suppression=[SuppressionDelta(memory_id=str(memory_id), reason="sleep suppression suggestion") for memory_id in consolidation_report.get("archived_memory_ids", [])],
            compilation=[
                CompilationDelta(source_memory_ids=list(dict(consolidation_report.get("consolidation_links", {})).get(str(memory_id), [])), compiled_type="procedural", content=str(memory_id))
                for memory_id in consolidation_report.get("compressed_memory_ids", [])
            ],
            lifecycle=[],
            memory_deltas=[],
            ledger=LedgerReport(transaction_ids=[transaction_id]),
        ).to_dict()
        if not compiled_nodes:
            compiled_nodes = sorted({str(delta.get("target_id")) for delta in graph_deltas if delta.get("target_id")})
        sleep_report["graph"] = {
            "candidates": sleep_graph.get("candidates", []),
            "proposed_deltas": sleep_graph.get("proposals", []),
            "approved_deltas": graph_deltas,
            "rejected_deltas": [],
            "compiled_nodes": compiled_nodes,
            "suppressed_stale_paths": [str(item) for item in consolidation_report.get("archived_memory_ids", [])],
            "frame_candidates": sleep_crystallization.get("frame_candidates", []),
            "validated_frames": sleep_crystallization.get("validated_frames", []),
            "associative_promotions": sleep_crystallization.get("associative_promotions", []),
            "logic_promotions": sleep_crystallization.get("logic_promotions", []),
            "compiled_schemas": sleep_crystallization.get("compiled_schemas", []),
            "rejected_crystallizations": sleep_crystallization.get("rejected_crystallizations", []),
        }
        sleep_report["materialized"] = materialized
        return {**consolidation_report, "sleep": sleep_report, "ledger_transaction_id": transaction_id}

    async def forget(
        self,
        memory_id: str,
        action: str = "inhibit",
        reason: str = "user-requested forgetting",
        authorize_delete: bool = False,
    ) -> dict[str, object]:
        normalized = action.lower()
        if normalized not in {"decay", "inhibit", "invalidate", "archive", "compress", "delete"}:
            raise ValueError(f"unsupported forget action: {action}")
        if normalized == "delete":
            return _append_only_rejection(memory_id, "destructive memory delete is not supported; append suppression or redaction evidence instead")
        item = self._runtime.store.get_memory(memory_id) if self._runtime.store is not None else None
        if item is None:
            raise ValueError(f"memory not found: {memory_id}")
        if normalized != "delete":
            if item.namespace != self.config.namespace:
                return _append_only_rejection(memory_id, f"target memory outside namespace: {memory_id}")
            if item.privacy_level in {"user", "sensitive"} and item.acl:
                return _append_only_rejection(memory_id, "user is not authorized for private memory")
            transaction_id = f"txn_forget_{uuid4().hex}"
            event_type = {
                "decay": "decay",
                "inhibit": "inhibit",
                "invalidate": "expire",
                "archive": "suppress",
                "compress": "suppress",
            }.get(normalized, "inhibit")
            edge_event = EdgeEvidenceEvent(
                namespace=self.config.namespace,
                source_kind="system",
                source_id=transaction_id,
                target_kind="memory",
                target_id=memory_id,
                relation="inhibits" if event_type in {"inhibit", "suppress"} else event_type,
                relation_family="suppression",
                event_type=event_type,
                delta_weight=-0.7 if event_type in {"inhibit", "suppress", "expire"} else -0.2,
                confidence=0.9,
                evidence_ids=[memory_id],
                proof_obligation=reason,
                proposer="deterministic",
            )
            self._ledger.append_edge_evidence(edge_event)
            audit = {"edge_evidence_event": edge_event.to_dict(), "append_only": True, "reason": reason}
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="COMMITTED",
                    event_type="suppression_event_appended",
                    operation=_policy_forget_operation(normalized),
                    proposer="deterministic",
                    validator_decision="approved",
                    evidence=[{"memory_id": memory_id, "source": "forget"}],
                    targets=[memory_id],
                    lifecycle_delta=[{"memory_id": memory_id, "event_type": event_type, "reason": reason, "append_only": True}],
                    audit=audit,
                )
            )
            self._ledger.append(
                LedgerEvent(
                    transaction_id=transaction_id,
                    namespace=self.config.namespace,
                    agent_id=self.config.agent_id,
                    phase="AUDITED",
                    event_type="audit_finalized",
                    operation=_policy_forget_operation(normalized),
                    proposer="deterministic",
                    validator_decision="approved",
                    evidence=[{"memory_id": memory_id, "source": "forget"}],
                    targets=[memory_id],
                    audit=audit,
                )
            )
            self._invalidate_retrieval_cache()
            self._materialize_worldview_sync()
            return _append_only_suppression_result(memory_id, _policy_forget_operation(normalized), event_type, edge_event, reason)
        policy = MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=memory_id),
            write=WritePlan(operation="NOOP"),
            forget=ForgetPlan(operation=_policy_forget_operation(normalized), target_memory_id=memory_id, reason=reason),
            consolidation=ConsolidationPlan(enabled=False),
            reason="deterministic product forgetting",
            source="deterministic",
        )
        return await self.commit(policy, authorize_delete=authorize_delete)

    async def replay_trace(self, trace_id: str) -> dict[str, object] | None:
        live = self._runtime.replay_trace(trace_id)
        if live is not None:
            ledger = self._ledger.replay_trace(trace_id, namespace=self.config.namespace)
            if ledger is not None:
                merged = dict(live)
                merged.update({key: value for key, value in ledger.items() if key not in merged or key.endswith("_deltas") or key == "ledger_events"})
                return merged
            return live
        path = self.config.traces_path / f"{trace_id}.json"
        if not path.exists():
            return self._ledger.replay_trace(trace_id, namespace=self.config.namespace)
        value = json.loads(path.read_text(encoding="utf-8"))
        ledger = self._ledger.replay_trace(trace_id, namespace=self.config.namespace)
        if ledger is not None:
            value.update({key: item for key, item in ledger.items() if key not in value or key.endswith("_deltas") or key == "ledger_events"})
        return value

    async def after_turn(self, trace_id: str, outcome: str, feedback: str | None = None) -> dict[str, object]:
        trace = await self.replay_trace(trace_id)
        if trace is None:
            raise ValueError(f"trace not found: {trace_id}")
        selected = [str(item) for item in trace.get("selected_memory_ids", []) if str(item)]
        event_type = "reinforce" if outcome.lower() in {"success", "succeeded", "positive", "ok"} else "inhibit"
        relation = "used_with_success" if event_type == "reinforce" else "used_with_failure"
        transaction_id = f"txn_after_turn_{uuid4().hex}"
        events: list[dict[str, object]] = []
        for index, source_id in enumerate(selected):
            targets = selected[index + 1 :] or [source_id]
            for target_id in targets:
                if source_id == target_id and len(selected) > 1:
                    continue
                edge_event = EdgeEvidenceEvent(
                    namespace=self.config.namespace,
                    source_kind="memory",
                    source_id=source_id,
                    target_kind="memory",
                    target_id=target_id,
                    relation=relation,
                    relation_family="association",
                    event_type=event_type,
                    delta_weight=0.22 if event_type == "reinforce" else -0.28,
                    confidence=0.75,
                    evidence_ids=[f"trace:{trace_id}"],
                    outcome=outcome,
                    proof_obligation=feedback,
                    proposer="after_turn",
                )
                self._edge_appender.append(edge_event)
                events.append(edge_event.to_dict())
        self._ledger.append(
            LedgerEvent(
                transaction_id=transaction_id,
                trace_id=trace_id,
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
                phase="COMMITTED",
                event_type="turn_outcome_evidence_appended",
                operation="AFTER_TURN",
                proposer="deterministic",
                validator_decision="approved",
                evidence=[{"trace_id": trace_id, "outcome": outcome, "feedback": feedback}],
                targets=selected,
                graph_delta=events,
                audit={"append_only": True, "edge_evidence_events": events},
            )
        )
        materialized = await self.materialize_worldview()
        self._invalidate_retrieval_cache()
        return {"trace_id": trace_id, "outcome": outcome, "edge_evidence_events": events, "materialized": materialized}

    async def materialize_worldview(self, namespace: str | None = None) -> dict[str, object]:
        namespace_value = namespace or self.config.namespace
        store = self._runtime.store if isinstance(self._runtime.store, SQLiteMemoryStore) else None
        memories = self._runtime.store.list_memories(namespace=namespace_value) if self._runtime.store is not None else []
        frames = self._runtime.store.list_logic_nodes(namespace=namespace_value) if self._runtime.store is not None else []
        result = self._worldview_materializer.rebuild(namespace=namespace_value, store=store, memories=memories, frames=frames)
        return result.to_dict()

    async def rebuild_materialized_views(self, namespace: str | None = None) -> dict[str, object]:
        return await self.materialize_worldview(namespace=namespace)

    async def resolve_worldview(
        self,
        query: str | None = None,
        lens: str = "auto",
        namespace: str | None = None,
        as_of: str | None = None,
    ) -> dict[str, object]:
        if as_of is not None:
            raise ValueError("as_of worldview resolution is not implemented yet; append-only records are retained for audit replay")
        namespace_value = namespace or self.config.namespace
        resolved_lens = self._resolve_lens(query or "", lens)
        await self.materialize_worldview(namespace=namespace_value)
        memories = self._runtime.store.list_memories(namespace=namespace_value) if self._runtime.store is not None else []
        frames = self._runtime.store.list_logic_nodes(namespace=namespace_value) if self._runtime.store is not None else []
        associative_edges = self._runtime.store.list_associative_edges(namespace=namespace_value) if self._runtime.store is not None else []
        logic_edges = self._runtime.store.list_logic_edges(namespace=namespace_value) if self._runtime.store is not None else []
        packet = self._worldview_resolver.resolve(
            namespace=namespace_value,
            lens=resolved_lens,
            query=query or "",
            memories=memories,
            frames=frames,
            associative_edges=associative_edges,
            logic_edges=logic_edges,
            edge_events=self._ledger.edge_evidence_events(namespace=namespace_value),
            impact_assessments=self._ledger.impact_assessments(namespace=namespace_value, limit=50),
            worldview_candidates=self._ledger.worldview_candidates(namespace=namespace_value),
            suppressed_memory_ids=self._ledger.active_suppressed_memory_ids(namespace_value),
        )
        payload = packet.to_dict()
        payload["prompt"] = packet.to_prompt()
        return payload

    def _materialize_worldview_sync(self, namespace: str | None = None) -> dict[str, object]:
        namespace_value = namespace or self.config.namespace
        store = self._runtime.store if isinstance(self._runtime.store, SQLiteMemoryStore) else None
        memories = self._runtime.store.list_memories(namespace=namespace_value) if self._runtime.store is not None else []
        frames = self._runtime.store.list_logic_nodes(namespace=namespace_value) if self._runtime.store is not None else []
        return self._worldview_materializer.rebuild(namespace=namespace_value, store=store, memories=memories, frames=frames).to_dict()

    def _append_route_audit(self, experience: ExperienceEvent, decision: str, payload: dict[str, object]) -> None:
        self._ledger.append(
            LedgerEvent(
                transaction_id=f"txn_route_{experience.event_id}",
                namespace=experience.namespace,
                agent_id=self.config.agent_id,
                phase="AUDITED",
                event_type="worldview_route_recorded",
                operation=str(decision).upper(),
                proposer="deterministic",
                validator_decision="not_applicable",
                evidence=[experience.to_dict()],
                audit={"decision": decision, "payload": payload, "append_only": True},
            )
        )

    def _append_support_evidence(self, experience: ExperienceEvent, impact: dict[str, object]) -> list[dict[str, object]]:
        candidates = self._matching_worldview_candidates(impact)
        events: list[dict[str, object]] = []
        for candidate in candidates[:3]:
            target_ids = [str(item) for item in candidate.get("source_memory_ids", [])] or [str(item) for item in candidate.get("source_frame_ids", [])]
            target_kind = "memory" if candidate.get("source_memory_ids") else "frame"
            for target_id in target_ids[:2]:
                edge_event = EdgeEvidenceEvent(
                    namespace=experience.namespace,
                    source_kind="event",
                    source_id=experience.event_id,
                    target_kind=target_kind,
                    target_id=target_id,
                    relation="supports",
                    relation_family="worldview",
                    event_type="support",
                    delta_weight=0.22,
                    confidence=0.72,
                    evidence_ids=[experience.event_id],
                    proof_obligation=str(impact.get("reason") or "worldview impact routing"),
                    proposer="worldview_impact_meter",
                )
                self._edge_appender.append(edge_event)
                events.append(edge_event.to_dict())
        self._append_route_audit(experience, "append_evidence", {"edge_evidence_events": events, "impact_id": impact.get("impact_id")})
        return events

    async def _propose_candidate_from_experience(self, experience: ExperienceEvent, impact: dict[str, object], *, worldview_candidate: bool) -> dict[str, object]:
        slot_key = _impact_slot_key(impact) or _slot_key_from_experience(experience)
        frame_type = _frame_type_from_experience(experience)
        frame_id = f"frame_evt_{stable_hash({'namespace': experience.namespace, 'event_id': experience.event_id, 'slot_key': slot_key})[:16]}"
        frame = FrameDeltaProposal(
            operation="propose_frame",
            frame_id=frame_id,
            frame_type=frame_type,  # type: ignore[arg-type]
            content=experience.content,
            canonical_key=slot_key,
            payload={"route": "propose_worldview_candidate" if worldview_candidate else "propose_frame", "source": experience.source, "metadata": dict(experience.metadata)},
            source_event_ids=[experience.event_id],
            evidence_ids=[experience.event_id],
            confidence=max(0.55, min(0.78, float(impact.get("impact_score", 0.45)) + 0.35)),
            commitment_level="candidate_frame",
            lifecycle_state="candidate",
            reason=str(impact.get("reason") or "worldview impact route proposed candidate frame"),
            proposer="worldview_impact_meter",
        )
        policy = MemoryPolicyV2(
            intent="consolidate",
            proposer="WorldviewImpactMeter",
            proposal_source="deterministic",
            evidence_chain=[EvidenceRef(event_id=experience.event_id, source="experience_event", content_hash=experience.content_hash)],
            target_selector={"memory_ids": [], "namespace": experience.namespace},
            frame_deltas=[frame],
            safety_annotations={"write_gate": {"decision": "defer", "durability_horizon": "thread", "commitment_level": "candidate_frame", "basis": "current_user_message", "signals": ["worldview_impact"], "rationale": str(impact.get("reason") or "")}},
            rollback_plan=f"append-only candidate frame from {experience.event_id}",
        )
        commit_result = await self.commit(policy)
        slot = self._ledger.upsert_worldview_slot(
            WorldviewSlotRecord(
                namespace=experience.namespace,
                key=slot_key,
                kind=_slot_kind_from_frame_type(frame_type),
                scope="global",
            )
        )
        candidate_event = WorldviewCandidateEvent(
            namespace=experience.namespace,
            slot_id=slot.slot_id,
            candidate_id=frame_id,
            event_type="proposed",
            evidence_ids=[experience.event_id],
            payload={"impact": impact, "frame_id": frame_id, "statement": experience.content},
            proposer="worldview_impact_meter",
        )
        self._ledger.append_worldview_candidate_event(candidate_event)
        supersession_events: list[dict[str, object]] = []
        if worldview_candidate and float((impact.get("vector") or {}).get("supersession", 0.0) if isinstance(impact.get("vector"), dict) else 0.0) >= 0.5:
            for candidate in self._matching_worldview_candidates(impact)[:3]:
                for target_id in [str(item) for item in candidate.get("source_frame_ids", [])] or [str(item) for item in candidate.get("source_memory_ids", [])]:
                    edge_event = EdgeEvidenceEvent(
                        namespace=experience.namespace,
                        source_kind="frame",
                        source_id=frame_id,
                        target_kind="frame" if candidate.get("source_frame_ids") else "memory",
                        target_id=target_id,
                        relation="supersedes",
                        relation_family="worldview",
                        event_type="supersede",
                        delta_weight=-0.7,
                        confidence=0.74,
                        evidence_ids=[experience.event_id],
                        proof_obligation=str(impact.get("reason") or "supersession proposed by impact meter"),
                        proposer="worldview_impact_meter",
                    )
                    self._edge_appender.append(edge_event)
                    supersession_events.append(edge_event.to_dict())
        materialized = self._materialize_worldview_sync(experience.namespace)
        self._append_route_audit(
            experience,
            "propose_worldview_candidate" if worldview_candidate else "propose_frame",
            {"frame_id": frame_id, "candidate_event": candidate_event.to_dict(), "supersession_events": supersession_events},
        )
        return {
            "frame_id": frame_id,
            "candidate_event": candidate_event.to_dict(),
            "supersession_events": supersession_events,
            "commit_result": commit_result,
            "materialized": materialized,
        }

    def _append_sleep_priority_marker(self, experience: ExperienceEvent, impact: dict[str, object]) -> dict[str, object]:
        self._ledger.append(
            LedgerEvent(
                transaction_id=f"txn_sleep_priority_{experience.event_id}",
                namespace=experience.namespace,
                agent_id=self.config.agent_id,
                phase="COMMITTED",
                event_type="sleep_priority_marker_appended",
                operation="PRIORITIZE_REPLAY",
                proposer="worldview_impact_meter",
                validator_decision="approved",
                evidence=[experience.to_dict()],
                targets=[experience.event_id],
                audit={"impact": impact, "append_only": True},
            )
        )
        return {"event_id": experience.event_id, "priority": impact.get("impact_score"), "reason": impact.get("reason")}

    def _matching_worldview_candidates(self, impact: dict[str, object]) -> list[dict[str, object]]:
        slot_key = _impact_slot_key(impact)
        candidates = self._ledger.worldview_candidates(namespace=self.config.namespace)
        if slot_key is None:
            return candidates[:5]
        matches = [candidate for candidate in candidates if str(candidate.get("slot_key")) == slot_key]
        if matches:
            return matches
        terms = set(slot_key.replace(":", " ").replace("_", " ").split())
        return [candidate for candidate in candidates if terms & set(str(candidate.get("statement", "")).lower().replace("_", " ").split())]

    def _append_structural_evidence_from_policy(self, policy: MemoryPolicyV2, trace_id: str) -> dict[str, object]:
        events: list[dict[str, object]] = []
        candidate_events: list[dict[str, object]] = []
        for delta in policy.associative_deltas:
            event_type = _event_type_for_delta_operation(delta.operation, delta.relation)
            edge_event = EdgeEvidenceEvent(
                namespace=self.config.namespace,
                source_kind="memory",
                source_id=delta.source_memory_id,
                target_kind="memory",
                target_id=delta.target_memory_id,
                relation=delta.relation,
                relation_family="association",
                event_type=event_type,
                delta_weight=delta.weight if event_type in {"support", "reinforce", "restore", "generalize", "derive"} else -max(delta.weight, 0.2),
                confidence=delta.confidence,
                evidence_ids=list(delta.evidence_ids) or [f"trace:{trace_id}"],
                proof_obligation=delta.reason,
                proposer=delta.proposer,
            )
            self._edge_appender.append(edge_event)
            events.append(edge_event.to_dict())
        for delta in policy.logic_deltas:
            event_type = _event_type_for_delta_operation(delta.operation, delta.relation)
            edge_event = EdgeEvidenceEvent(
                namespace=self.config.namespace,
                source_kind="frame",
                source_id=delta.source_frame_id,
                target_kind="frame",
                target_id=delta.target_frame_id,
                relation=delta.relation,
                relation_family="logic",
                event_type=event_type,
                delta_weight=delta.weight if event_type in {"support", "reinforce", "restore", "generalize", "derive"} else -max(delta.weight, 0.2),
                confidence=delta.confidence,
                evidence_ids=list(delta.evidence_ids) or [f"trace:{trace_id}"],
                proof_obligation=delta.proof_obligation or delta.reason,
                proposer=delta.proposer,
            )
            self._edge_appender.append(edge_event)
            events.append(edge_event.to_dict())
        for delta in policy.graph_deltas:
            event_type = _event_type_for_delta_operation(delta.operation, delta.relation)
            edge_event = EdgeEvidenceEvent(
                namespace=self.config.namespace,
                source_kind="memory",
                source_id=delta.source_memory_id,
                target_kind="memory",
                target_id=delta.target_memory_id,
                relation=delta.relation,
                relation_family="logic" if delta.relation in {"supports", "contradicts", "supersedes", "inhibits", "derived_from", "generalizes"} else "association",
                event_type=event_type,
                delta_weight=delta.weight if event_type in {"support", "reinforce", "restore", "generalize", "derive"} else -max(delta.weight, 0.2),
                confidence=delta.confidence,
                evidence_ids=list(delta.evidence_ids) or [f"trace:{trace_id}"],
                proof_obligation=delta.reason,
                proposer=delta.proposer,
            )
            self._edge_appender.append(edge_event)
            events.append(edge_event.to_dict())
            associative_relation = _associative_relation_for_graph_delta(delta.relation, list(delta.candidate_sources))
            if associative_relation is not None:
                assoc_event = EdgeEvidenceEvent(
                    namespace=self.config.namespace,
                    source_kind="memory",
                    source_id=delta.source_memory_id,
                    target_kind="memory",
                    target_id=delta.target_memory_id,
                    relation=associative_relation,
                    relation_family="association",
                    event_type="support",
                    delta_weight=min(delta.weight, 0.35),
                    confidence=min(delta.confidence, 0.62),
                    evidence_ids=list(delta.evidence_ids) or [f"trace:{trace_id}"],
                    proof_obligation=delta.reason or f"activation association derived alongside {delta.relation}",
                    proposer=delta.proposer,
                )
                self._edge_appender.append(assoc_event)
                events.append(assoc_event.to_dict())
        for frame in policy.frame_deltas:
            slot = self._ledger.upsert_worldview_slot(
                WorldviewSlotRecord(
                    namespace=self.config.namespace,
                    key=frame.canonical_key or _slot_key_from_text(frame.content, _slot_kind_from_frame_type(frame.frame_type)),
                    kind=_slot_kind_from_frame_type(frame.frame_type),
                    scope="global",
                )
            )
            candidate_event = WorldviewCandidateEvent(
                namespace=self.config.namespace,
                slot_id=slot.slot_id,
                candidate_id=frame.frame_id or f"frame_{stable_hash({'content': frame.content})[:16]}",
                event_type=str(frame.operation),
                evidence_ids=list(frame.evidence_ids) or [f"trace:{trace_id}"],
                payload=frame.model_dump(mode="json"),
                proposer=frame.proposer,
            )
            self._ledger.append_worldview_candidate_event(candidate_event)
            candidate_events.append(candidate_event.to_dict())
        return {"edge_evidence_events": events, "worldview_candidate_events": candidate_events}

    def _persist_trace(self, trace_id: str, data: dict[str, object]) -> None:
        self.config.traces_path.mkdir(parents=True, exist_ok=True)
        (self.config.traces_path / f"{trace_id}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _assess_experience(self, experience: ExperienceEvent) -> WorldviewImpactAssessment:
        memories = self._runtime.store.list_memories(namespace=experience.namespace) if self._runtime.store is not None else []
        worldview_candidates = self._ledger.worldview_candidates(namespace=experience.namespace)
        frames = self._runtime.store.list_logic_nodes(namespace=experience.namespace) if self._runtime.store is not None else []
        edge_events = self._ledger.edge_evidence_events(namespace=experience.namespace)
        return self._impact_meter.assess(experience, memories, worldview_candidates=worldview_candidates, frames=frames, edge_events=edge_events)

    def _memory_states(self) -> dict[str, dict[str, object]]:
        if self._runtime.store is None:
            return {}
        return {item.id: item.to_record() for item in self._runtime.store.list_memories(namespace=self.config.namespace)}

    def _resolve_lens(self, query: str, lens: str) -> str:
        if lens != "auto":
            if lens not in {"associative", "logical", "procedural", "historical", "audit"}:
                raise ValueError(f"unsupported retrieval lens: {lens}")
            return lens
        lowered = query.lower()
        if any(term in lowered for term in ["why", "audit", "ledger", "evidence", "trusted", "written"]):
            return "audit"
        if any(term in lowered for term in ["history", "historical", "previous", "old", "superseded", "changed"]):
            return "historical"
        if any(term in lowered for term in ["how", "procedure", "workflow", "steps", "fix", "run"]):
            return "procedural"
        if any(term in lowered for term in ["current", "fact", "preference", "constraint", "should", "must"]):
            return "logical"
        return "associative"

    def _apply_retrieval_lens(self, results: list[object], lens: str) -> list[object]:
        if self._runtime.store is None or lens in {"associative", "historical", "audit"}:
            return results
        if lens == "logical":
            allowed: set[str] = set()
            for frame in self._runtime.store.list_logic_nodes(namespace=self.config.namespace):
                if frame.lifecycle_state == "candidate":
                    continue
                if frame.commitment_level not in {"validated_logic", "compiled_schema"}:
                    continue
                allowed.update(frame.source_memory_ids)
            return [result for result in results if getattr(getattr(result, "memory", None), "id", None) in allowed]
        if lens == "procedural":
            allowed = {
                memory_id
                for frame in self._runtime.store.list_logic_nodes(namespace=self.config.namespace)
                if frame.frame_type in {"procedure", "schema", "failure_pattern"} and frame.lifecycle_state in {"validated", "compiled", "mature"}
                for memory_id in frame.source_memory_ids
            }
            return [result for result in results if getattr(getattr(result, "memory", None), "type", None) == "procedural" or getattr(getattr(result, "memory", None), "id", None) in allowed]
        return results

    def _semantic_filters(self, query: str) -> dict[str, object]:
        filters: dict[str, object] = {"namespace": self.config.namespace}
        if self._embedding_provider is not None and self._vector_index is not None:
            filters["_embedding_provider"] = self._embedding_provider
            filters["_vector_index"] = self._vector_index
            filters["_embedding_provider_label"] = self._embedding_provider_label()
            filters["_embedding_cache"] = self._embedding_cache
            filters.setdefault("retrieval_channels", ("fts5", "bm25", "dense", "rewrite", "hyde", "lexical", "entity", "recent_current", "procedural_preference", "canonical_fact", "graph_seed"))
        if self._query_rewrite_provider is not None:
            filters["_query_rewrite_provider"] = self._query_rewrite_provider
        if self._hyde_provider is not None:
            filters["_hyde_provider"] = self._hyde_provider
        if self._entity_alias_resolver is not None:
            filters["_entity_alias_resolver"] = self._entity_alias_resolver
            aliases = self._entity_alias_resolver.expand(query, self.config.namespace)
            if aliases:
                filters["query_rewrites"] = aliases
                filters["entities"] = aliases
        return filters

    def _handle_retrieval_graph_commit(self, trace: object, timing: RuntimeTiming) -> dict[str, object] | None:
        mode = self.config.retrieval_graph_commit
        if mode == "off":
            return {"status": "off"}
        if mode == "trace_only":
            selected = [str(memory_id) for memory_id in getattr(trace, "selected_memory_ids", [])]
            return {"status": "trace_only", "selected_memory_ids": selected}
        if mode == "async":
            selected = [str(memory_id) for memory_id in getattr(trace, "selected_memory_ids", [])]
            if len(selected) < 2:
                return None
            trace_id = str(getattr(trace, "trace_id", ""))
            query_plan = dict(getattr(trace, "query_plan", {}) or {})

            def run() -> object:
                return self._commit_auto_graph(
                    selected_memory_ids=selected,
                    target_memory_ids=selected,
                    evidence_ids=[f"trace:{trace_id}"],
                    outcome="success",
                    retrieval_trace=query_plan,
                    invalidate_retrieval_cache=False,
                )

            job = self._background_jobs.submit("retrieval_graph_commit", run)
            return {"status": "queued", "job": job}
        with TimingSpan(timing, "graph_commit_ms"):
            return self._commit_retrieval_graph(trace)

    def _retrieval_cache_key(self, query: MemoryQuery, *, resolved_lens: str, store_version: str, filter_hash: str) -> str:
        filters = {
            key: value
            for key, value in query.filters.items()
            if not str(key).startswith("_") and key not in {"retrieval_lens"}
        }
        return stable_hash(
            {
                "namespace": filters.get("namespace", self.config.namespace),
                "query": query.query.strip().lower(),
                "budget_tokens": query.budget_tokens,
                "lens": resolved_lens,
                "top_k": filters.get("top_k"),
                "filter_hash": filter_hash,
                "store_version": store_version,
                "embedding_provider": self._embedding_provider_label(),
            }
        )

    def _retrieval_cache_probe_key(self, query: MemoryQuery, *, resolved_lens: str) -> str:
        filters = {
            key: value
            for key, value in query.filters.items()
            if not str(key).startswith("_") and key not in {"retrieval_lens"}
        }
        return stable_hash(
            {
                "namespace": filters.get("namespace", self.config.namespace),
                "query": query.query.strip().lower(),
                "budget_tokens": query.budget_tokens,
                "lens": resolved_lens,
                "top_k": filters.get("top_k"),
                "embedding_provider": self._embedding_provider_label(),
            }
        )

    def _retrieval_miss_reason(self, probe_key: str, *, semantic_version: str, filter_hash: str) -> str | None:
        previous = self._retrieval_cache_fingerprints.get(probe_key)
        if previous is None:
            return None
        previous_semantic_version, previous_filter_hash = previous
        if previous_semantic_version != semantic_version:
            return "semantic_version_changed"
        if previous_filter_hash != filter_hash:
            return "filter_changed"
        return None

    def _retrieval_filter_hash(self, query: MemoryQuery, *, resolved_lens: str) -> str:
        filters = {
            key: value
            for key, value in query.filters.items()
            if not str(key).startswith("_") and key not in {"retrieval_lens"}
        }
        return stable_hash({"lens": resolved_lens, "budget_tokens": query.budget_tokens, "filters": filters})

    def _store_version(self, namespace: str) -> str:
        if self._runtime.store is None:
            return "empty"
        memories = self._runtime.store.list_memories(namespace=namespace)
        semantic_records: list[dict[str, object]] = []
        for memory in memories:
            semantic_records.append(
                {
                    "id": memory.id,
                    "namespace": memory.namespace,
                    "type": memory.type,
                    "content": memory.content,
                    "summary": memory.summary,
                    "evidence": sorted(memory.evidence),
                    "entities": sorted(memory.entities),
                    "keywords": sorted(memory.keywords),
                    "tags": sorted(memory.tags),
                    "maturity": memory.maturity,
                    "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                    "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
                    "privacy_level": memory.privacy_level,
                    "acl": sorted(memory.acl),
                    "deletion_policy": memory.deletion_policy,
                }
            )
        return stable_hash({"namespace": namespace, "memories": semantic_records})[:16]

    def _invalidate_retrieval_cache(self, reason: str = "invalidated_by_mutation") -> None:
        self._retrieval_cache.invalidate(reason)

    def _embedding_provider_label(self) -> str:
        provider = self._embedding_provider
        if provider is None:
            return "disabled"
        config = getattr(provider, "config", None)
        model = getattr(config, "model", None)
        if model:
            return f"{provider.__class__.__name__}:{model}"
        return provider.__class__.__name__

    def performance_stats(self) -> dict[str, object]:
        provider_stats = getattr(self._embedding_provider, "stats", None)
        embedding_stats = provider_stats() if callable(provider_stats) else {}
        return {
            "retrieval_cache": self._retrieval_cache.stats(),
            "embedding_cache_enabled": self._embedding_cache is not None,
            "embedding_provider": self._embedding_provider_label(),
            "embedding_provider_stats": embedding_stats,
            "background_jobs": self._background_jobs.recent(),
            "retrieval_graph_commit": self.config.retrieval_graph_commit,
            "retrieval_mode": self.config.retrieval_mode,
        }

    def _commit_retrieval_graph(self, trace: object) -> dict[str, object] | None:
        selected = [str(memory_id) for memory_id in getattr(trace, "selected_memory_ids", [])]
        if len(selected) < 2:
            return None
        return self._commit_auto_graph(
            selected_memory_ids=selected,
            target_memory_ids=selected,
            evidence_ids=[f"trace:{getattr(trace, 'trace_id', '')}"],
            outcome="success",
            retrieval_trace=getattr(trace, "query_plan", {}),
            invalidate_retrieval_cache=False,
        )

    def _commit_auto_graph(
        self,
        *,
        selected_memory_ids: list[str],
        target_memory_ids: list[str],
        evidence_ids: list[str],
        outcome: str,
        retrieval_trace: dict[str, object] | None = None,
        mutation_trace: dict[str, object] | None = None,
        sleep_clusters: list[list[str]] | None = None,
        invalidate_retrieval_cache: bool = True,
    ) -> dict[str, object] | None:
        if self._graph_mode == "off" or self._runtime.store is None:
            return None
        memories = self._runtime.store.list_memories(namespace=self.config.namespace)
        context = GraphBuildContext(
            namespace=self.config.namespace,
            memories=memories,
            selected_memory_ids=selected_memory_ids,
            target_memory_ids=target_memory_ids,
            evidence_ids=evidence_ids,
            retrieval_trace=retrieval_trace or {},
            mutation_trace=mutation_trace or {},
            sleep_clusters=sleep_clusters or [],
            outcome=outcome,
            proposer="deterministic",
            embedding_provider=self._embedding_provider,
        )
        candidates = self._graph_candidate_generator.generate(context)
        if not candidates:
            return None
        proposals = self._relation_proposer.propose_graph_deltas(context, candidates)
        if not proposals:
            return {"candidates": [candidate.to_dict() for candidate in candidates], "proposals": []}
        policy = MemoryPolicyV2(
            intent="link",
            proposer="GraphBuilder",
            proposal_source="deterministic",
            evidence_chain=[EvidenceRef(event_id=evidence_id, source="graph_builder") for evidence_id in evidence_ids],
            target_selector={"memory_ids": list(dict.fromkeys([*selected_memory_ids, *target_memory_ids])), "namespace": self.config.namespace},
            graph_deltas=proposals,
            rollback_plan="rollback graph deltas with memory transaction",
        )
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase="graph_builder",
                task="governed graph construction",
                query=str((retrieval_trace or {}).get("raw_query") or (mutation_trace or {}).get("operation") or "graph_builder"),
                state={"status": outcome, "target_memory_ids": list(dict.fromkeys([*selected_memory_ids, *target_memory_ids]))},
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
            ),
        )
        if invalidate_retrieval_cache:
            self._invalidate_retrieval_cache()
        if result.validated_mutation.approved:
            result.trace.query_plan["append_only_structural_evidence"] = self._append_structural_evidence_from_policy(policy, result.trace.trace_id)
            self._materialize_worldview_sync()
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        return {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
            "result": result.to_dict(),
        }

    def _select_sleep_replay_clusters(self, memories: list[MemoryItem]) -> list[list[str]]:
        by_id = {memory.id: memory for memory in memories}
        clusters: list[list[str]] = []
        last_trace = self._runtime.last_trace
        if last_trace is not None:
            selected = [memory_id for memory_id in getattr(last_trace, "selected_memory_ids", []) if memory_id in by_id]
            if len(selected) >= 2:
                clusters.append(list(dict.fromkeys(selected))[:6])
        high_impact_events = self._ledger.impact_assessments(namespace=self.config.namespace, limit=30)
        high_impact_hashes = {str(item.get("event_id")) for item in high_impact_events if float(item.get("impact_score", 0.0)) >= 0.28 or str(item.get("impact_type")) in {"conflict", "supersession", "sleep_worthy"}}
        impacted = [memory.id for memory in memories if set(memory.evidence) & high_impact_hashes or set(memory.source_event_ids) & high_impact_hashes]
        if len(impacted) >= 2:
            clusters.append(list(dict.fromkeys(impacted))[:6])
        conflict_targets = {
            str(event.get("target_id"))
            for event in self._ledger.edge_evidence_events(namespace=self.config.namespace)
            if event.get("event_type") in {"contradict", "supersede", "inhibit"}
        }
        conflict_cluster = [memory.id for memory in memories if memory.id in conflict_targets]
        if len(conflict_cluster) >= 2:
            clusters.append(conflict_cluster[:6])
        clusters.extend(_fallback_sleep_clusters(memories))
        deduped: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for cluster in clusters:
            key = tuple(sorted(set(cluster)))
            if len(key) >= 2 and key not in seen:
                seen.add(key)
                deduped.append(list(key))
        return deduped[:4]

    def _commit_sleep_crystallization(self, *, sleep_clusters: list[list[str]], evidence_ids: list[str], transaction_id: str) -> dict[str, object]:
        if self._runtime.store is None:
            return {"frame_candidates": [], "validated_frames": [], "associative_promotions": [], "logic_promotions": [], "compiled_schemas": [], "rejected_crystallizations": []}
        memories = self._runtime.store.list_memories(namespace=self.config.namespace)
        if not sleep_clusters:
            sleep_clusters = _fallback_sleep_clusters(memories)
        if not sleep_clusters:
            return {"frame_candidates": [], "validated_frames": [], "associative_promotions": [], "logic_promotions": [], "compiled_schemas": [], "rejected_crystallizations": []}
        context = GraphBuildContext(
            namespace=self.config.namespace,
            memories=memories,
            selected_memory_ids=[memory_id for cluster in sleep_clusters for memory_id in cluster],
            target_memory_ids=[],
            evidence_ids=evidence_ids,
            sleep_clusters=sleep_clusters,
            outcome="success",
            proposer="deterministic",
            embedding_provider=self._embedding_provider,
        )
        compiled_frames = self._crystallization_planner.plan_sleep_frames(context)
        if not compiled_frames:
            return {"frame_candidates": [], "validated_frames": [], "associative_promotions": [], "logic_promotions": [], "compiled_schemas": [], "rejected_crystallizations": []}
        by_id = context.memory_by_id()
        source_frames = []
        logic_deltas: list[LogicEdgeProposal] = []
        for compiled in compiled_frames:
            for memory_id in compiled.source_memory_ids:
                memory = by_id.get(memory_id)
                if memory is None:
                    continue
                source_frame = frame_proposal_for_memory(memory, context=context, validated=False)
                source_frames.append(source_frame)
                logic_deltas.append(
                    LogicEdgeProposal(
                        operation="add_edge",
                        source_frame_id=source_frame.frame_id or frame_id_for_memory(memory, infer_frame_type(memory)),
                        target_frame_id=compiled.frame_id or "",
                        source_memory_id=memory.id,
                        target_memory_id=None,
                        relation="derived_from",
                        weight=0.42,
                        confidence=0.72,
                        proof_obligation="sleep replay cluster links source experience to compiled frame",
                        evidence_ids=list(compiled.evidence_ids),
                        reason="sleep crystallization derived source experience into compiled frame",
                        proposer="deterministic",
                        lifecycle_state="captured",
                    )
                )
        frame_deltas = list({(frame.frame_id or frame.content): frame for frame in [*source_frames, *compiled_frames]}.values())
        policy = MemoryPolicyV2(
            intent="consolidate",
            proposer="SleepCrystallizationPlanner",
            proposal_source="deterministic",
            evidence_chain=[EvidenceRef(event_id=evidence_id, source="sleep_crystallization") for evidence_id in evidence_ids],
            target_selector={"memory_ids": context.selected_memory_ids, "namespace": self.config.namespace},
            frame_deltas=frame_deltas,
            logic_deltas=logic_deltas,
            rollback_plan=f"rollback sleep crystallization {transaction_id}",
        )
        result = self._executor.execute(
            policy,
            PolicyExecutionContext(
                phase="sleep_crystallization",
                task="governed progressive crystallization",
                query="sleep_crystallization",
                state={"status": "success", "target_memory_ids": context.selected_memory_ids},
                namespace=self.config.namespace,
                agent_id=self.config.agent_id,
            ),
        )
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        if result.validated_mutation.approved:
            result.trace.query_plan["append_only_structural_evidence"] = self._append_structural_evidence_from_policy(policy, result.trace.trace_id)
            self._materialize_worldview_sync()
        self._persist_trace(result.trace.trace_id, result.trace.to_dict())
        approved = result.validated_mutation.approved
        frame_payloads = [frame.model_dump(mode="json") for frame in frame_deltas]
        logic_payloads = [delta.model_dump(mode="json") for delta in logic_deltas]
        return {
            "frame_candidates": frame_payloads,
            "validated_frames": frame_payloads if approved else [],
            "associative_promotions": [],
            "logic_promotions": logic_payloads if approved else [],
            "compiled_schemas": [frame.model_dump(mode="json") for frame in compiled_frames] if approved else [],
            "rejected_crystallizations": [] if approved else [step.model_dump() for step in result.validated_mutation.validator_trace if not step.passed],
            "result": result.to_dict(),
        }


def _policy_forget_operation(action: str) -> str:
    return {
        "decay": "DECAY",
        "inhibit": "INHIBIT",
        "invalidate": "INVALIDATE",
        "archive": "ARCHIVE",
        "compress": "ARCHIVE",
    }[action]


def _destructive_policy_reason(policy: MemoryPolicy | MemoryPolicyV2, legacy_policy: MemoryPolicy) -> str | None:
    if legacy_policy.write.operation == "UPDATE":
        return "destructive memory update is not supported; append new evidence or a supersession relation instead"
    if legacy_policy.forget.operation == "DELETE_REQUEST":
        return "destructive memory delete is not supported; append suppression or redaction evidence instead"
    if isinstance(policy, MemoryPolicyV2):
        if policy.intent == "update":
            return "destructive memory update is not supported; append new evidence or a supersession relation instead"
        if policy.intent == "delete_request":
            return "destructive memory delete is not supported; append suppression or redaction evidence instead"
        for delta in policy.proposed_deltas:
            operation = (delta.operation or "").upper()
            if operation == "UPDATE":
                return "destructive memory update is not supported; append new evidence or a supersession relation instead"
            if operation in {"DELETE", "DELETE_REQUEST"}:
                return "destructive memory delete is not supported; append suppression or redaction evidence instead"
    return None


def _append_only_suppression_result(
    memory_id: str,
    operation: str,
    event_type: str,
    edge_event: EdgeEvidenceEvent,
    reason: str,
) -> dict[str, object]:
    return {
        "trace_id": None,
        "policy_source": "deterministic",
        "validator_decision": "approved",
        "executed_actions": [operation],
        "append_only": True,
        "edge_evidence_event": edge_event.to_dict(),
        "mutation_execution_result": {
            "validated_mutation": {"approved": True, "validator_trace": []},
            "created_memory_ids": [],
            "updated_memory_ids": [],
            "deleted_memory_ids": [],
            "memory_deltas": [],
            "graph_deltas": [],
            "lifecycle_deltas": [{"memory_id": memory_id, "event_type": event_type, "reason": reason, "append_only": True}],
            "index_deltas": [],
        },
    }


def _append_only_rejection(memory_id: str | None, reason: str) -> dict[str, object]:
    return {
        "trace_id": None,
        "policy_source": "deterministic",
        "validator_decision": reason,
        "rejected_reasons": [reason],
        "executed_actions": [],
        "append_only": True,
        "mutation_execution_result": {
            "validated_mutation": {
                "approved": False,
                "rejected_deltas": [{"operation": "NOOP", "target_memory_id": memory_id, "reason": reason}],
                "validator_trace": [{"name": "AppendOnlyForgetValidator", "passed": False, "reason": reason}],
            },
            "created_memory_ids": [],
            "updated_memory_ids": [],
            "deleted_memory_ids": [],
            "memory_deltas": [],
            "graph_deltas": [],
            "lifecycle_deltas": [],
            "index_deltas": [],
        },
    }


def _write_config(path: Path, config: RuntimeConfig) -> None:
    if path.exists():
        return
    values = config.to_dict()
    lines = []
    for key in ["namespace", "db_path", "traces_path", "agent_id", "mode", "graph_mode", "crystallization_mode", "graph_storage", "mutation_mode", "version"]:
        lines.append(f'{key} = "{values[key]}"')
    lines.append(f"model_policy_enabled = {str(config.model_policy_enabled).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_memory_context(context: MemoryContext) -> MemoryContext:
    return replace(
        context,
        selected_memory_ids=list(context.selected_memory_ids),
        results=[dict(item) for item in context.results],
        transactions=[dict(item) for item in context.transactions],
        timing=dict(context.timing),
        cache=dict(context.cache),
        retrieval_metadata=dict(context.retrieval_metadata),
        worldview=dict(context.worldview) if context.worldview is not None else None,
        worldview_trace=dict(context.worldview_trace) if context.worldview_trace is not None else None,
        prompt_sections=dict(context.prompt_sections),
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _graph_commit_env() -> str:
    value = os.environ.get("NEUROMEM_RETRIEVAL_GRAPH_COMMIT", "trace_only").strip().lower()
    if value in {"async", "off", "sync", "trace_only"}:
        return value
    return "trace_only"


__all__ = ["MemoryRuntime"]


def _memory_deltas_for_fields(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    *,
    fields: set[str] | None,
    reason: str,
) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for memory_id in sorted(set(before) | set(after)):
        old = before.get(memory_id, {})
        new = after.get(memory_id, {})
        keys = (set(old) | set(new)) if fields is None else fields
        for field in sorted(keys):
            if old.get(field) != new.get(field):
                deltas.append({"memory_id": memory_id, "field": field, "old": old.get(field), "new": new.get(field), "reason": reason})
    return deltas


def _memory_delta_from_dict(delta: dict[str, object]) -> MemoryDelta:
    return MemoryDelta(
        memory_id=str(delta.get("memory_id", "")),
        field=str(delta.get("field", "")),
        old=delta.get("old"),
        new=delta.get("new"),
        reason=str(delta.get("reason", "")),
    )


def _lifecycle_delta_from_dict(delta: dict[str, object]) -> LifecycleDelta:
    return LifecycleDelta(
        memory_id=str(delta.get("memory_id", "")),
        from_state=str(delta.get("old", "")),
        to_state=str(delta.get("new", "")),
        trigger="governed sleep",
        evidence=[],
        validator="SleepLifecycleValidator",
        reason=str(delta.get("reason", "")),
        rollback_state=str(delta.get("old", "")) if delta.get("old") is not None else None,
    )


def _fallback_sleep_clusters(memories: list[MemoryItem]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for memory in memories:
        terms = [term.lower() for term in [*memory.entities, *memory.keywords] if len(term) > 2]
        if not terms:
            terms = [term.strip(".,:;()[]`'\"?").lower() for term in memory.content.split() if len(term.strip(".,:;()[]`'\"?")) > 4]
        for term in sorted(set(terms))[:4]:
            groups.setdefault(term, []).append(memory.id)
    clusters = []
    seen: set[tuple[str, ...]] = set()
    for ids in groups.values():
        unique = tuple(sorted(set(ids)))
        if len(unique) >= 2 and unique not in seen:
            seen.add(unique)
            clusters.append(list(unique))
    return clusters[:3]


def _impact_slot_key(impact: dict[str, object]) -> str | None:
    slots = impact.get("impacted_slots", [])
    if not isinstance(slots, list) or not slots:
        return None
    first = slots[0]
    if isinstance(first, dict) and first.get("slot_key"):
        return str(first["slot_key"])
    return None


def _slot_key_from_experience(experience: ExperienceEvent) -> str:
    explicit = experience.metadata.get("slot_key") or experience.metadata.get("canonical_key")
    if explicit:
        return str(explicit).strip().lower().replace(" ", "_")
    kind = _slot_kind_from_frame_type(_frame_type_from_experience(experience))
    return _slot_key_from_text(experience.content, kind)


def _slot_key_from_text(content: str, kind: str) -> str:
    terms = [term.strip(".,:;!?()[]{}\"'").lower() for term in content.split() if len(term.strip(".,:;!?()[]{}\"'")) > 3]
    return f"{kind}:{'_'.join(terms[:4]) or 'general'}"


def _frame_type_from_experience(experience: ExperienceEvent) -> str:
    kind = str(experience.metadata.get("type") or "").lower()
    text = experience.content.lower()
    if kind in {"user_preference", "preference"} or any(cue in text for cue in ["prefer", "喜欢", "偏好"]):
        return "preference"
    if kind in {"constraint"} or any(cue in text for cue in ["must", "never", "不要", "必须"]):
        return "constraint"
    if kind in {"rule", "procedure", "procedural"} or any(cue in text for cue in ["workflow", "procedure", "steps", "run", "流程"]):
        return "procedure"
    if kind in {"schema"}:
        return "schema"
    if kind in {"fact", "semantic"}:
        return "fact"
    return "claim"


def _slot_kind_from_frame_type(frame_type: str) -> str:
    if frame_type in {"fact", "entity"}:
        return "fact"
    if frame_type == "claim":
        return "hypothesis"
    if frame_type in {"preference", "constraint", "procedure", "schema"}:
        return frame_type
    if frame_type == "failure_pattern":
        return "procedure"
    return "hypothesis"


def _event_type_for_delta_operation(operation: str, relation: str) -> str:
    op = operation.lower()
    rel = relation.lower()
    if op in {"inhibit_edge"} or rel in {"inhibits"}:
        return "inhibit"
    if op in {"expire_edge"}:
        return "expire"
    if rel in {"contradicts"}:
        return "contradict"
    if rel in {"supersedes"}:
        return "supersede"
    if rel in {"generalizes", "specializes"}:
        return "generalize"
    if rel in {"derived_from", "compresses_to"}:
        return "derive"
    if rel in {"used_with_success"}:
        return "reinforce"
    if rel in {"used_with_failure"}:
        return "inhibit"
    return "support"


def _associative_relation_for_graph_delta(relation: str, candidate_sources: list[str]) -> str | None:
    sources = set(candidate_sources)
    if relation in {"associated_with", "coactivated_with", "precedes", "retrieved_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}:
        return relation
    if not sources & {"same_query_retrieval", "co_use_outcome", "same_sleep_cluster", "same_evidence_chain"}:
        return None
    if "same_sleep_cluster" in sources:
        return "same_episode"
    if "same_evidence_chain" in sources:
        return "same_trace"
    if "same_query_retrieval" in sources:
        return "retrieved_with"
    return "coactivated_with"


def _prompt_sections_from_worldview(worldview: dict[str, object]) -> dict[str, str]:
    sections: dict[str, str] = {}
    prompt = worldview.get("prompt")
    if prompt:
        sections["worldview"] = str(prompt)
    for key, title in [
        ("facts", "Current facts"),
        ("preferences", "User preferences"),
        ("constraints", "Constraints"),
        ("procedures", "Procedures"),
        ("suppressions", "Suppressions / stale assumptions"),
        ("conflicts", "Open conflicts"),
    ]:
        items = worldview.get(key, [])
        if isinstance(items, list) and items:
            lines = [f"[{title}]"]
            for item in items[:8]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('statement') or item.get('slot_key')}")
            sections[key] = "\n".join(lines)
    supporting = worldview.get("supporting_memories", [])
    if isinstance(supporting, list) and supporting:
        lines = ["[Supporting memories]"]
        for item in supporting[:8]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('memory_id')}: {item.get('content')}")
        sections["supporting_memories"] = "\n".join(lines)
    return sections

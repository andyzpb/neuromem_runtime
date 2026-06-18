from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path

from neuromem.core.models import MemoryEdge, MemoryItem, MemoryQuery, MemoryResult, utcnow
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, MemoryTrace, RetrievalPlan, ValidatedPolicy, WritePlan
from neuromem.core.validator import PolicyValidator
from neuromem.modules.consolidation import ConsolidationReport, consolidate
from neuromem.modules.explainer import explain_memory
from neuromem.modules.forgetting import ForgetDecision, apply_forgetting, choose_forgetting_action
from neuromem.modules.layers import HippocampalStore, NeocorticalStore, ProceduralStore
from neuromem.modules.lifecycle import obsolete
from neuromem.modules.memory_tap import MemoryTap
from neuromem.modules.pfc_controller import PFCController
from neuromem.modules.plasticity import graph_diffuse, update_edges_after_use
from neuromem.modules.reconsolidation import Reconsolidator
from neuromem.modules.salience import compute_salience, salience_score
from neuromem.modules.tag_capture import maybe_capture, tag_provisional
from neuromem.modules.working_memory import WorkingMemory
from neuromem.retrieval.hybrid import hybrid_retrieve_with_trace
from neuromem.stores.base import MemoryStore
from neuromem.stores.sqlite_store import SQLiteMemoryStore


@dataclass(slots=True)
class NeuroMemRuntime:
    agent_id: str
    namespace: str
    store: MemoryStore | None = None
    db_path: str | Path = ".neuromem/neuromem.sqlite3"
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    controller: PFCController = field(default_factory=PFCController)
    validator: PolicyValidator = field(default_factory=PolicyValidator)
    memory_pfc: object | None = None
    reconsolidator: Reconsolidator = field(default_factory=Reconsolidator)
    last_trace: MemoryTrace | None = None
    last_recall_trace: dict[str, object] = field(default_factory=dict)
    traces: dict[str, MemoryTrace] = field(default_factory=dict)
    memory_tap: MemoryTap = field(default_factory=MemoryTap)
    hippocampus: HippocampalStore | None = field(init=False, default=None)
    neocortex: NeocorticalStore | None = field(init=False, default=None)
    procedural: ProceduralStore | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = SQLiteMemoryStore(self.db_path)
        self.hippocampus = HippocampalStore(self.store, self.namespace)
        self.neocortex = NeocorticalStore(self.store, self.namespace)
        self.procedural = ProceduralStore(self.store, self.namespace)

    def _remember_trace(self) -> None:
        if self.last_trace is not None:
            self.traces[self.last_trace.trace_id] = self.last_trace
            self.traces[self.last_trace.task_id] = self.last_trace

    def _run_async(self, awaitable: Awaitable[MemoryPolicy]) -> MemoryPolicy:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise RuntimeError("Live memory PFC requires an async caller when an event loop is already running")

    def _deterministic_before_policy(self, task: str, state: dict[str, object]) -> MemoryPolicy:
        query = str(state.get("query") or task)
        retrieval = RetrievalPlan(
            enabled=True,
            query=query,
            memory_types=self.controller.plan_retrieval(query, state).memory_types,
            entities=[str(value) for value in state.get("entities", [])] if isinstance(state.get("entities"), list) else [],
            max_items=int(state.get("max_items", 8) or 8),
            graph_expansion=bool(state.get("graph_expansion", True)),
            require_provenance=bool(state.get("require_provenance", False)),
        )
        return MemoryPolicy(
            retrieval=retrieval,
            write=WritePlan(operation="NOOP"),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="deterministic before-step retrieval plan",
            source="deterministic",
        )

    def _deterministic_after_policy(self, task: str, trace: dict[str, object], outcome: dict[str, object]) -> MemoryPolicy:
        status = str(outcome.get("status", "unknown"))
        evidence_id = str(trace.get("id") or trace.get("trace_id") or task)
        content = str(trace.get("content") or outcome.get("patch_summary") or task)
        confidence = float(outcome.get("confidence", 0.75) or 0.75)
        salience_estimate = float(outcome.get("salience", 0.65) or 0.65)
        memory_type = "episodic"
        if status == "success" and any(term in content.lower() for term in ["must", "always", "rule", "procedure"]):
            memory_type = "procedural"
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False, query=task),
            write=WritePlan(
                operation="ADD",
                memory_type=memory_type,
                content=content,
                salience_estimate=salience_estimate,
                confidence=confidence,
                evidence_ids=[evidence_id],
                ttl="long_term" if status == "success" else "session",
            ),
            forget=ForgetPlan(operation="NOOP"),
            consolidation=ConsolidationPlan(enabled=False),
            reason="deterministic after-step write plan",
            source="deterministic",
        )

    def _fallback_trace(self, *, task: str, query: str, policy: MemoryPolicy, reason: str, state: dict[str, object] | None = None) -> None:
        self.last_trace = MemoryTrace(
            task_id=str((state or {}).get("task_id", task)),
            query=query,
            retrieval_plan=policy.retrieval,
            policy_source=policy.source,
            pfc_reason=policy.reason,
            fallback_reason=reason,
            validator_decision=reason,
        )
        self.memory_tap.emit("fallback", task_id=task, query=query, reason=reason)
        self.memory_tap.attach(self.last_trace)
        self._remember_trace()

    def _plan_via_memory_pfc(self, method: str, task: str) -> tuple[MemoryPolicy | None, str | None]:
        planner = self.memory_pfc
        if planner is None:
            return None, "memory_pfc not configured"
        plan = getattr(planner, method, None)
        if plan is None:
            return None, f"memory_pfc does not implement {method}"
        try:
            result = plan(task)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                return self._run_async(result), None  # type: ignore[arg-type]
            if isinstance(result, MemoryPolicy):
                return result, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return None, f"{method} did not return MemoryPolicy"

    def _route_policy_item(self, item: MemoryItem) -> None:
        assert self.store is not None
        if item.type == "episodic":
            self.hippocampus.encode(item)
        elif item.type in {"semantic", "preference", "schema"}:
            self.neocortex.store_fact(item)
        elif item.type == "procedural":
            self.procedural.store_rule(item)
        else:
            self.store.upsert_memory(item)

    def _plan_before(self, task: str, state: dict[str, object]) -> tuple[MemoryPolicy, str | None]:
        live, reason = self._plan_via_memory_pfc("plan_before_step", task)
        if live is not None:
            return live, None
        return self._deterministic_before_policy(task, state), f"live memory_pfc unavailable or invalid: {reason}"

    def _plan_after(self, task: str, trace: dict[str, object], outcome: dict[str, object]) -> tuple[MemoryPolicy, str | None]:
        live, reason = self._plan_via_memory_pfc("plan_after_step", task)
        if live is not None:
            return live, None
        return self._deterministic_after_policy(task, trace, outcome), f"live memory_pfc unavailable or invalid: {reason}"

    def _execute_validated_policy(
        self,
        policy: MemoryPolicy,
        validated: ValidatedPolicy,
        *,
        phase: str,
        task: str,
        query: str,
        state: dict[str, object] | None = None,
        retrieved_memory_ids: list[str] | None = None,
        graph_paths: list[list[str]] | None = None,
        graph_scores: dict[str, float] | None = None,
    ) -> None:
        self.last_trace = MemoryTrace(
            task_id=str((state or {}).get("task_id", task)),
            query=query,
            retrieval_plan=policy.retrieval,
            policy_source=policy.source,
            approved_actions=validated.approved_actions,
            rejected_reasons=validated.rejected_reasons,
            executed_actions=[],
            pfc_reason=policy.reason,
            validator_decision="approved" if validated.approved else "; ".join(validated.rejected_reasons),
            graph_paths=graph_paths or [],
        )
        if not validated.approved:
            self.memory_tap.emit("validation_rejected", phase=phase, task=task, reasons=validated.rejected_reasons)
            self.memory_tap.attach(self.last_trace)
            self._remember_trace()
            return
        self.last_trace.executed_actions = list(validated.approved_actions)
        self.memory_tap.emit("validation_approved", phase=phase, task=task, actions=validated.approved_actions)
        if phase == "after_step" and policy.write.operation != "NOOP":
            outcome = state or {}
            item: MemoryItem | None = None
            if policy.write.operation == "ADD":
                item = self.observe(
                    {
                        "type": "rule" if policy.write.memory_type == "procedural" else "fact" if policy.write.memory_type == "semantic" else "task_result",
                        "content": policy.write.content or task,
                        "task": task,
                        "outcome": outcome.get("status", "unknown"),
                        "evidence": policy.write.evidence_ids[0] if policy.write.evidence_ids else task,
                        "prediction_error": outcome.get("prediction_error", 0.0),
                        "future_utility": outcome.get("future_utility", 0.0),
                    }
                )
                if item is not None:
                    self.memory_tap.emit("write", operation="ADD", memory_id=item.id, memory_type=item.type, evidence=item.evidence)
            elif policy.write.operation == "UPDATE" and policy.write.target_memory_id:
                item = self.store.get_memory(policy.write.target_memory_id)
                if item is not None:
                    if policy.write.content:
                        item.content = policy.write.content
                    if policy.write.memory_type:
                        item.type = policy.write.memory_type  # type: ignore[assignment]
                    item.confidence = max(item.confidence, policy.write.confidence)
                    item.valid_from = item.valid_from or item.created_at
                    item.evidence.extend(evidence for evidence in policy.write.evidence_ids if evidence not in item.evidence)
                    self._route_policy_item(item)
                    self.memory_tap.emit("write", operation="UPDATE", memory_id=item.id, memory_type=item.type, evidence=policy.write.evidence_ids)
            elif policy.write.operation == "LINK" and policy.write.target_memory_id:
                item = self.store.get_memory(policy.write.target_memory_id)
                for evidence_id in policy.write.evidence_ids:
                    if item is not None and evidence_id not in item.evidence:
                        item.evidence.append(evidence_id)
                    if item is not None and self.store.get_memory(evidence_id) is not None:
                        self.link(item.id, evidence_id, "evidence_for", confidence=policy.write.confidence or 0.5)
                if item is not None:
                    self.store.upsert_memory(item)
                    self.memory_tap.emit("write", operation="LINK", memory_id=item.id, evidence=policy.write.evidence_ids)
            if item is not None:
                used = [self.store.get_memory(memory_id) for memory_id in (retrieved_memory_ids or [])]
                used_memories = [memory for memory in used if memory is not None]
                used_memories.append(item)
                graph_deltas = update_edges_after_use(
                    self.store,
                    used_memories,
                    outcome=str(outcome.get("status", "unknown")),
                    salience=policy.write.salience_estimate or float(outcome.get("salience", 0.65) or 0.65),
                    confidence=policy.write.confidence or float(outcome.get("confidence", 0.75) or 0.75),
                )
                if self.last_trace is not None and graph_deltas:
                    self.last_trace.query_plan["plasticity_graph_deltas"] = [delta.to_dict() for delta in graph_deltas]
        if policy.forget.operation != "NOOP":
            target_id = policy.forget.target_memory_id
            if target_id:
                target = self.store.get_memory(target_id)
                if target is not None:
                    applied = ForgetDecision(action="archive", reason=policy.forget.reason or "policy-driven forgetting")
                    if policy.forget.operation == "DECAY":
                        applied = ForgetDecision(action="decay", reason=policy.forget.reason or "policy-driven decay")
                    elif policy.forget.operation == "INHIBIT":
                        applied = ForgetDecision(action="inhibit", reason=policy.forget.reason or "policy-driven inhibition")
                    elif policy.forget.operation == "INVALIDATE":
                        applied = ForgetDecision(action="invalidate", reason=policy.forget.reason or "policy-driven invalidation")
                    elif policy.forget.operation == "ARCHIVE":
                        applied = ForgetDecision(action="archive", reason=policy.forget.reason or "policy-driven archive")
                    elif policy.forget.operation == "DELETE_REQUEST":
                        applied = ForgetDecision(action="delete", reason=policy.forget.reason or "policy-driven delete")
                    apply_forgetting(target, applied, reason=policy.forget.reason)
                    self.store.upsert_memory(target)
                    self.memory_tap.emit("forget", operation=policy.forget.operation, memory_id=target.id, reason=policy.forget.reason)
        if phase in {"before_step", "after_step"} and policy.consolidation.enabled:
            report = self.neuro_sleep()
            if self.last_trace is not None:
                self.last_trace.consolidation_links.update(report.consolidation_links)
            self.memory_tap.emit("consolidate", report=report.to_dict())
        if self.last_trace is not None:
            self.memory_tap.attach(self.last_trace)
            self._remember_trace()

    def observe(self, event: dict[str, object], *, user_id: str | None = None) -> MemoryItem | None:
        if not event.get("content"):
            raise ValueError("observe() requires a non-empty content field")
        self.working_memory.set_task(str(event.get("task")) if event.get("task") else None)
        self.working_memory.add_event(event)
        salience = compute_salience(event, {"task": event.get("task")})
        score = salience_score(salience)
        decision = self.controller.decide_write(score, event)
        if decision.destination == "discard":
            if float(event.get("prediction_error", 0.0) or 0.0) < 0.65:
                return None
        item_type = "episodic"
        if event.get("type") == "user_preference":
            item_type = "preference"
        elif event.get("type") == "rule":
            item_type = "procedural"
        elif event.get("type") == "fact":
            item_type = "semantic"
        item = MemoryItem(
            agent_id=self.agent_id,
            user_id=user_id,
            namespace=self.namespace,
            type=item_type,
            content=str(event.get("content", "")),
            summary=event.get("summary") if isinstance(event.get("summary"), str) else None,
            evidence=[str(event.get("evidence"))] if event.get("evidence") else [],
            keywords=[str(value) for value in event.get("keywords", [])] if isinstance(event.get("keywords"), list) else [],
            tags=[str(value) for value in event.get("tags", [])] if isinstance(event.get("tags"), list) else [],
            salience=salience,
            prediction_error=float(event.get("prediction_error", 0.0) or 0.0),
            future_utility=float(event.get("future_utility", 0.0) or 0.0),
            confidence=round(0.4 + 0.6 * score, 3),
        )
        item.maturity = "fresh"
        assert self.store is not None
        if item.prediction_error >= 0.65 and decision.destination in {"discard", "working"}:
            item.type = "episodic"
            item.evidence.append("prediction-error routed to episodic memory")
            self.hippocampus.encode(item)
            return item
        if decision.destination == "working" and item.prediction_error >= 0.5:
            tag_provisional(item, tag_strength=max(score, item.prediction_error))
            self.store.upsert_memory(item)
            return item
        event_type = str(event.get("type", ""))
        if event_type == "task_result":
            item.type = "episodic"
            self.hippocampus.encode(item)
            return item
        if event_type == "rule":
            item.type = "procedural"
            self.procedural.store_rule(item)
            return item
        if event_type in {"fact", "user_preference"}:
            item.type = "semantic" if event_type == "fact" else "preference"
            self.neocortex.store_fact(item)
            return item
        if decision.destination == "working":
            item.type = "working"
        if item.type == "episodic":
            self.hippocampus.encode(item)
        elif item.type in {"semantic", "preference", "schema"}:
            self.neocortex.store_fact(item)
        elif item.type == "procedural":
            self.procedural.store_rule(item)
        else:
            self.store.upsert_memory(item)
        return item

    def retrieve(self, query: str, mode: str = "auto", filters: dict[str, object] | None = None, budget_tokens: int = 1500) -> list[MemoryResult]:
        if not query.strip():
            raise ValueError("retrieve() requires a non-empty query")
        assert self.store is not None
        memories = self.store.list_memories(namespace=self.namespace)
        plan = self.controller.plan_retrieval(query, filters or {}, budget_tokens)
        filters_for_retrieval = dict(filters or {})
        filters_for_retrieval["_store"] = self.store
        query_obj = MemoryQuery(query=query, mode=mode, filters=filters_for_retrieval, budget_tokens=budget_tokens)
        results, recall = hybrid_retrieve_with_trace(memories, query_obj, plan)
        graph_paths: list[list[str]] = []
        graph_scores: dict[str, float] = {}
        if filters and filters.get("graph_diffusion") and results and self.store is not None:
            expanded, graph_paths, graph_scores = graph_diffuse(
                [result.memory for result in results],
                self.store,
                max_items=max(1, min(10, budget_tokens // 250)),
                depth=int(filters.get("graph_depth", 2) or 2),
                restart_prob=float(filters.get("graph_restart_prob", 0.25) or 0.25),
                min_score=float(filters.get("graph_min_score", 0.03) or 0.03),
            )
            graph_memories = {memory.id: memory for memory in expanded}
            memory_pool = list({memory.id: memory for memory in [*memories, *graph_memories.values()]}.values())
            results, recall = hybrid_retrieve_with_trace(memory_pool, query_obj, plan, graph_scores=graph_scores, graph_paths=graph_paths)
        self.last_recall_trace = recall.trace()
        for result in results:
            result.memory.access_count += 1
            result.memory.activation_count += 1
            result.memory.last_accessed_at = utcnow()
            self.store.upsert_memory(result.memory)
        return results

    def before_step(self, task: str, state: dict[str, object] | None = None, recent_context: list[dict[str, object]] | None = None) -> str:
        if not task.strip():
            raise ValueError("before_step() requires a non-empty task")
        state = state or {}
        policy, fallback_reason = self._plan_before(task, state)
        if fallback_reason is None and policy.source == "small_llm" and not policy.retrieval.query.strip():
            fallback_reason = "live memory_pfc returned an empty retrieval query"
            policy = self._deterministic_before_policy(task, state)
        if fallback_reason is None and policy.source == "small_llm" and policy.retrieval.max_items <= 0:
            fallback_reason = "live memory_pfc returned an invalid retrieval budget"
            policy = self._deterministic_before_policy(task, state)
        query = policy.retrieval.query or str(state.get("query") or task)
        if not query.strip():
            query = task
        validated = self.validator.validate(policy, {"phase": "before_step"})
        if not validated.approved:
            if policy.source == "small_llm":
                fallback_reason = "; ".join(validated.rejected_reasons)
                policy = self._deterministic_before_policy(task, state)
                query = policy.retrieval.query or task
                validated = self.validator.validate(policy, {"phase": "before_step", "fallback_reason": fallback_reason})
            if not validated.approved:
                self._execute_validated_policy(policy, validated, phase="before_step", task=task, query=query, state=state)
                return ""
        retrieval = policy.retrieval
        filters: dict[str, object] = {}
        if retrieval.memory_types:
            filters["type"] = retrieval.memory_types
        filters["temporal_scope"] = retrieval.temporal_scope
        filters["graph_depth"] = retrieval.graph_depth
        filters["graph_restart_prob"] = retrieval.graph_restart_prob
        filters["graph_min_score"] = retrieval.graph_min_score
        filters["graph_diffusion"] = retrieval.graph_expansion
        if retrieval.entities:
            filters["entities"] = retrieval.entities
        if retrieval.temporal_scope != "all_including_obsolete":
            filters["exclude_maturity"] = ["deleted", "obsolete", "inhibited"]
        if not retrieval.require_provenance:
            filters["require_fact_or_entity_alignment"] = False
        results = self.retrieve(query, filters=filters, budget_tokens=retrieval.max_items * 250)
        if retrieval.require_provenance:
            results = [result for result in results if result.memory.evidence]
        selected_memories = [result.memory for result in results]
        recall_trace = dict(self.last_recall_trace)
        graph_paths = recall_trace.get("graph_paths", []) if isinstance(recall_trace.get("graph_paths", []), list) else []
        score_components = recall_trace.get("score_components", {}) if isinstance(recall_trace.get("score_components", {}), dict) else {}
        graph_scores: dict[str, float] = {}
        for memory_id, components in score_components.items():
            if isinstance(components, dict):
                graph_scores[str(memory_id)] = float(components.get("graph_score", 0.0) or 0.0)
        results = sorted(results, key=lambda result: result.score, reverse=True)[: retrieval.max_items]
        context = "\n".join(f"- [{result.score:.2f}] {result.memory.content}" for result in results)
        all_memories = self.store.list_memories(namespace=self.namespace) if self.store is not None else []
        selected_ids = [result.memory.id for result in results]
        rejected_ids = [
            item.id
            for item in all_memories
            if item.id not in selected_ids and item.maturity in {"obsolete", "inhibited", "deleted", "archived"}
        ]
        suppression_reasons = {
            item.id: f"maturity={item.maturity}"
            for item in all_memories
            if item.id in rejected_ids
        }
        rejected_ids.extend(str(item_id) for item_id in recall_trace.get("rejected_ids", []) if str(item_id) not in rejected_ids)
        recall_suppression = recall_trace.get("suppression_reasons", {})
        if isinstance(recall_suppression, dict):
            for key, value in recall_suppression.items():
                suppression_reasons.setdefault(str(key), str(value))
        self.memory_tap.reset()
        self.memory_tap.emit("retrieve", task_id=str(state.get("task_id", task)), query=query, selected_ids=selected_ids)
        if graph_paths:
            self.memory_tap.emit("graph_diffusion", paths=graph_paths, scores={key: round(value, 3) for key, value in graph_scores.items()})
        self.last_trace = MemoryTrace(
            task_id=str(state.get("task_id", task)),
            query=query,
            retrieval_plan=retrieval,
            policy_source=policy.source,
            approved_actions=validated.approved_actions,
            rejected_reasons=validated.rejected_reasons,
            executed_actions=["RETRIEVE"] if retrieval.enabled else [],
            fallback_reason=fallback_reason,
            selected_memory_ids=selected_ids,
            rejected_memory_ids=rejected_ids,
            scores={
                result.memory.id: {
                    "hybrid": round(result.score, 3),
                    "graph": round(graph_scores.get(result.memory.id, 0.0), 3),
                    "inhibition": round(result.memory.inhibition_score, 3),
                    **{
                        key: float(value)
                        for key, value in dict(score_components.get(result.memory.id, {})).items()
                        if isinstance(value, int | float)
                    },
                }
                for result in results
            },
            graph_paths=graph_paths,
            pfc_reason=policy.reason,
            validator_decision="approved",
            final_context_tokens=len(context.split()),
            baseline_scores={result.memory.id: round(result.score, 3) for result in results},
            diffusion_scores={key: round(value, 3) for key, value in graph_scores.items()},
            suppression_reasons=suppression_reasons,
            query_plan=dict(recall_trace.get("query_plan", {})) if isinstance(recall_trace.get("query_plan", {}), dict) else {},
            source_channels=[str(item) for item in recall_trace.get("source_channels", [])] if isinstance(recall_trace.get("source_channels", []), list) else [],
            gate_decision=str(recall_trace.get("gate_decision", "")),
            canonical_fact_ids=[str(item) for item in recall_trace.get("canonical_fact_ids", [])] if isinstance(recall_trace.get("canonical_fact_ids", []), list) else [],
            memory_version=str(recall_trace.get("memory_version", "")),
            invalidation_state=str(recall_trace.get("invalidation_state", "valid")),
            recall_config_hash=str(recall_trace.get("recall_config_hash", "")),
        )
        for key in ["query_plan_v2", "query_plan_v2_hash", "retrieval_ledger", "activation", "candidate_details"]:
            if key in recall_trace:
                self.last_trace.query_plan[key] = recall_trace[key]
        self.memory_tap.attach(self.last_trace)
        if retrieval.enabled and retrieval.require_provenance:
            self.last_trace.rejected_memory_ids.extend(
                result.memory.id for result in results if not result.memory.evidence and result.memory.id not in self.last_trace.rejected_memory_ids
            )
        self._remember_trace()
        return context

    def retrieve_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, object] | None = None,
        budget_tokens: int = 1500,
        task_id: str | None = None,
    ) -> tuple[list[MemoryResult], MemoryTrace]:
        max_items = max(1, budget_tokens // 250)
        state: dict[str, object] = {
            "task_id": task_id or query,
            "query": query,
            "max_items": max_items,
            "graph_expansion": bool((filters or {}).get("graph_diffusion", True)),
            "require_provenance": bool((filters or {}).get("require_provenance", False)),
        }
        if filters:
            state.update(filters)
        self.before_step(query, state)
        if self.last_trace is None:
            raise RuntimeError("retrieve_with_trace did not produce a trace")
        results: list[MemoryResult] = []
        assert self.store is not None
        scores = self.last_trace.baseline_scores or {key: value.get("hybrid", 0.0) for key, value in self.last_trace.scores.items()}
        for memory_id in self.last_trace.selected_memory_ids:
            item = self.store.get_memory(memory_id)
            if item is not None:
                results.append(MemoryResult(memory=item, score=float(scores.get(memory_id, 0.0)), why_retrieved=["retrieved_with_trace"]))
        return results, self.last_trace

    def after_step(
        self,
        task: str,
        trace: dict[str, object],
        outcome: dict[str, object] | None = None,
        retrieved_memory_ids: list[str] | None = None,
    ) -> MemoryPolicy:
        if not task.strip():
            raise ValueError("after_step() requires a non-empty task")
        outcome = outcome or {}
        status = str(outcome.get("status", "unknown"))
        self.memory_tap.reset()
        self._mark_contradictions(task, trace, outcome, retrieved_memory_ids or [])
        policy, fallback_reason = self._plan_after(task, trace, outcome)
        validated = self.validator.validate(policy, {"phase": "after_step", "outcome": status})
        if not validated.approved and policy.source == "small_llm":
            fallback_reason = "; ".join(validated.rejected_reasons)
            policy = self._deterministic_after_policy(task, trace, outcome)
            validated = self.validator.validate(policy, {"phase": "after_step", "outcome": status, "fallback_reason": fallback_reason})
        if validated.approved:
            outcome_state = dict(outcome)
            outcome_state["id"] = trace.get("id") or trace.get("trace_id") or task
            self._execute_validated_policy(
                policy,
                validated,
                phase="after_step",
                task=task,
                query=policy.retrieval.query or task,
                state=outcome_state,
                retrieved_memory_ids=retrieved_memory_ids,
            )
            assert self.store is not None
            for memory in self.store.list_memories(namespace=self.namespace):
                if memory.type == "provisional":
                    decision = maybe_capture(
                        memory,
                        {
                            "outcome": status,
                            "recurrence": outcome.get("recurrence", 0.0),
                            "prediction_error": outcome.get("prediction_error", memory.prediction_error),
                            "user_confirmation": outcome.get("user_confirmation", False),
                        },
                    )
                    if decision.reason not in memory.evidence:
                        memory.evidence.append(decision.reason)
                    self.store.upsert_memory(memory)
        if self.last_trace is not None:
            self.last_trace.fallback_reason = fallback_reason
            self.last_trace.outcome = status if status in {"success", "failure", "partial", "unknown"} else "unknown"
            self._remember_trace()
        return policy

    def neuro_sleep(self) -> ConsolidationReport:
        return self.consolidate()

    def explain_last_retrieval(self) -> dict[str, object] | None:
        if self.last_trace is None:
            return None
        return self.last_trace.to_dict()

    def consolidate(self) -> ConsolidationReport:
        assert self.store is not None
        memories = self.store.list_memories(namespace=self.namespace)
        report = consolidate(memories)
        for item in memories:
            self.store.upsert_memory(item)
        return report

    def _mark_contradictions(self, task: str, trace: dict[str, object], outcome: dict[str, object], memory_ids: list[str]) -> None:
        assert self.store is not None
        content = f"{task} {trace.get('content', '')} {outcome.get('content', '')}".lower()
        if not any(term in content for term in ["now", "current", "instead", "replaces", "supersedes", "contradict", "obsolete"]):
            return
        for memory_id in memory_ids:
            memory = self.store.get_memory(memory_id)
            if memory is None or memory.maturity in {"deleted", "obsolete", "inhibited"}:
                continue
            memory_text = memory.content.lower()
            if any(term in memory_text for term in ["old", "deprecated"]) or any(term in content for term in ["replaces", "instead", "supersedes", "obsolete"]):
                obsolete(memory, reason=f"contradicted by trace {trace.get('id') or trace.get('trace_id') or task}")
                self.store.upsert_memory(memory)
                self.memory_tap.emit("forget", operation="INVALIDATE", memory_id=memory.id, reason="contradictory update")

    def replay_trace(self, trace_id: str) -> dict[str, object] | None:
        trace = self.traces.get(trace_id)
        if trace is None:
            return None
        return trace.to_dict()

    def reconsolidate(self, memory_id: str, evidence: str) -> MemoryItem | None:
        assert self.store is not None
        item = self.store.get_memory(memory_id)
        if item is None:
            return None
        item, edge = self.reconsolidator.apply(item, evidence)
        if edge is not None:
            self.store.add_edge(edge)
        self.store.upsert_memory(item)
        return item

    def invalidate(self, memory_id: str, evidence: str) -> MemoryItem | None:
        assert self.store is not None
        item = self.store.get_memory(memory_id)
        if item is None:
            return None
        obsolete(item, reason=evidence)
        self.store.upsert_memory(item)
        return item

    def forget(self, memory_id: str) -> dict[str, object] | None:
        assert self.store is not None
        item = self.store.get_memory(memory_id)
        if item is None:
            return None
        decision = choose_forgetting_action(item)
        apply_forgetting(item, decision)
        self.store.upsert_memory(item)
        return {"memory_id": item.id, "action": decision.action, "reason": decision.reason}

    def explain(self, memory_id: str) -> dict[str, object] | None:
        assert self.store is not None
        item = self.store.get_memory(memory_id)
        if item is None:
            return None
        return explain_memory(item)

    def link(self, source_id: str, target_id: str, relation: str, confidence: float = 0.5) -> MemoryEdge:
        edge = MemoryEdge(source_id=source_id, target_id=target_id, relation=relation, confidence=confidence)
        assert self.store is not None
        self.store.add_edge(edge)
        return edge

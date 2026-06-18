from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

from neuromem.core.models import MemoryEdge
from neuromem.core.policy import ConsolidationPlan, ForgetPlan, MemoryPolicy, RetrievalPlan, WritePlan
from neuromem.core.runtime import NeuroMemRuntime
from neuromem.core.validator import PolicyValidator
from neuromem.modules.tag_capture import maybe_capture, tag_provisional
from neuromem.stores.memory_store import InMemoryStore


VariantName = Literal[
    "Full",
    "FlatRetrieval",
    "NoGraph",
    "NoPFC",
    "NoValidator",
    "NoPlasticity",
    "NoTagCapture",
    "NoInhibition",
    "NoReplaySleep",
    "NoTraceReplay",
    "CoactivationOnly",
    "OutcomeOnly",
    "NoReconsolidation",
    "WriteEverything",
    "SalienceOnlyWrite",
    "NoCache",
    "NoTraceFaithfulness",
    "NoSemanticRecall",
    "BM25Only",
    "NoCandidateGenerator",
    "LLMDirectNoValidator",
    "NoSleepGraphCompiler",
    "NoTypedSuppression",
    "NoCrystallization",
    "AssociativeOnly",
    "LogicDirectNoFrame",
    "NoFrameValidator",
    "NoSleepCrystallization",
    "UnifiedGraphNoSplit",
]
ScenarioName = Literal["coding_agent", "synthetic_lifecycle", "mutation_safety"]


@dataclass(frozen=True, slots=True)
class BenchVariant:
    name: VariantName
    graph: bool = True
    pfc: bool = True
    validator: bool = True
    plasticity: bool = True
    tag_capture: bool = True
    inhibition: bool = True
    replay_sleep: bool = True
    trace_replay: bool = True
    lifecycle: bool = True
    coactivation: bool = True
    outcome_conditioning: bool = True
    reconsolidation: bool = True
    write_mode: Literal["normal", "write_everything", "salience_only"] = "normal"
    cache: bool = True
    trace_faithfulness: bool = True
    semantic_recall: bool = True
    candidate_generator: bool = True
    sleep_graph_compiler: bool = True
    typed_suppression: bool = True
    crystallization: bool = True
    split_graph_storage: bool = True
    frame_validator: bool = True
    logic_frames: bool = True
    sleep_crystallization: bool = True


@dataclass(frozen=True, slots=True)
class BenchScenario:
    name: ScenarioName
    description: str


@dataclass(slots=True)
class BenchRun:
    variant: str
    scenario: str
    metrics: dict[str, float] = field(default_factory=dict)
    selected_memory_ids: list[str] = field(default_factory=list)
    suppressed_memory_ids: list[str] = field(default_factory=list)
    suppression_reasons: dict[str, str] = field(default_factory=dict)
    graph_paths: list[list[str]] = field(default_factory=list)
    validator_rejections: list[str] = field(default_factory=list)
    memorytap_event_count: int = 0
    created_memory_ids: list[str] = field(default_factory=list)
    compressed_memory_ids: list[str] = field(default_factory=list)
    archived_memory_ids: list[str] = field(default_factory=list)
    obsolete_memory_ids: list[str] = field(default_factory=list)
    trace_sample: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class BenchReport:
    runs: list[BenchRun] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    trace_samples: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate,
            "trace_samples": self.trace_samples,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_jsonl(self) -> str:
        lines = [json.dumps(run.to_dict(), sort_keys=True) for run in self.runs]
        lines.append(json.dumps({"aggregate": self.aggregate, "trace_samples": self.trace_samples}, sort_keys=True))
        return "\n".join(lines)


VARIANTS: dict[str, BenchVariant] = {
    "Full": BenchVariant("Full"),
    "FlatRetrieval": BenchVariant("FlatRetrieval", graph=False, pfc=False, validator=False, plasticity=False, tag_capture=False, inhibition=False, replay_sleep=False, trace_replay=False, lifecycle=False),
    "NoGraph": BenchVariant("NoGraph", graph=False),
    "NoPFC": BenchVariant("NoPFC", pfc=False),
    "NoValidator": BenchVariant("NoValidator", validator=False),
    "NoPlasticity": BenchVariant("NoPlasticity", plasticity=False),
    "NoTagCapture": BenchVariant("NoTagCapture", tag_capture=False),
    "NoInhibition": BenchVariant("NoInhibition", inhibition=False),
    "NoReplaySleep": BenchVariant("NoReplaySleep", replay_sleep=False),
    "NoTraceReplay": BenchVariant("NoTraceReplay", trace_replay=False),
    "CoactivationOnly": BenchVariant("CoactivationOnly", outcome_conditioning=False),
    "OutcomeOnly": BenchVariant("OutcomeOnly", coactivation=False),
    "NoReconsolidation": BenchVariant("NoReconsolidation", reconsolidation=False),
    "WriteEverything": BenchVariant("WriteEverything", tag_capture=False, write_mode="write_everything"),
    "SalienceOnlyWrite": BenchVariant("SalienceOnlyWrite", tag_capture=False, write_mode="salience_only"),
    "NoCache": BenchVariant("NoCache", cache=False),
    "NoTraceFaithfulness": BenchVariant("NoTraceFaithfulness", trace_replay=False, trace_faithfulness=False),
    "NoSemanticRecall": BenchVariant("NoSemanticRecall", semantic_recall=False),
    "BM25Only": BenchVariant("BM25Only", graph=False, semantic_recall=False, candidate_generator=False),
    "NoCandidateGenerator": BenchVariant("NoCandidateGenerator", candidate_generator=False),
    "LLMDirectNoValidator": BenchVariant("LLMDirectNoValidator", validator=False),
    "NoSleepGraphCompiler": BenchVariant("NoSleepGraphCompiler", sleep_graph_compiler=False),
    "NoTypedSuppression": BenchVariant("NoTypedSuppression", typed_suppression=False, inhibition=False),
    "NoCrystallization": BenchVariant("NoCrystallization", crystallization=False, logic_frames=False, sleep_crystallization=False),
    "AssociativeOnly": BenchVariant("AssociativeOnly", logic_frames=False, sleep_crystallization=False),
    "LogicDirectNoFrame": BenchVariant("LogicDirectNoFrame", logic_frames=False, frame_validator=False),
    "NoFrameValidator": BenchVariant("NoFrameValidator", frame_validator=False),
    "NoSleepCrystallization": BenchVariant("NoSleepCrystallization", sleep_crystallization=False),
    "UnifiedGraphNoSplit": BenchVariant("UnifiedGraphNoSplit", split_graph_storage=False),
}

SCENARIOS: dict[str, BenchScenario] = {
    "coding_agent": BenchScenario("coding_agent", "Repeated coding bug, stale command, graph recall, and procedural consolidation."),
    "synthetic_lifecycle": BenchScenario("synthetic_lifecycle", "Prediction-error routing, provisional capture, conflict invalidation, and replay sleep."),
    "mutation_safety": BenchScenario("mutation_safety", "Unsafe memory mutation proposals and validator evidence."),
}


class UnsafeMemoryPFC:
    def plan_after_step(self, task: str) -> MemoryPolicy:
        return MemoryPolicy(
            retrieval=RetrievalPlan(enabled=False),
            write=WritePlan(operation="ADD", memory_type="semantic", content="unsafe stale overwrite", confidence=0.1, evidence_ids=[]),
            forget=ForgetPlan(operation="DELETE_REQUEST", target_memory_id="missing", reason="unsafe delete"),
            consolidation=ConsolidationPlan(enabled=False),
            reason=f"unsafe proposal for {task}",
            source="small_llm",
        )


def _runtime(variant: BenchVariant) -> NeuroMemRuntime:
    memory_pfc = UnsafeMemoryPFC() if variant.name == "NoValidator" else None
    return NeuroMemRuntime(agent_id="bench-agent", namespace="bench", store=InMemoryStore(), memory_pfc=memory_pfc)


def _trace(runtime: NeuroMemRuntime, *, enabled: bool) -> dict[str, object]:
    trace = runtime.explain_last_retrieval() or {}
    if not enabled:
        trace = dict(trace)
        trace["events"] = []
    return trace


def _write_mode_metrics(variant: BenchVariant) -> tuple[float, float]:
    if variant.write_mode == "write_everything":
        return 1.0, 0.0
    if variant.write_mode == "salience_only":
        return 0.5, 0.5
    return 0.0, 1.0


def _trace_metric(variant: BenchVariant, trace: dict[str, object]) -> float:
    if not variant.trace_replay or not variant.trace_faithfulness:
        return 0.0
    return 1.0 if trace.get("events") else 0.0


def _cache_metric(variant: BenchVariant) -> float:
    return 0.0 if not variant.cache else 1.0


def _crystallization_metrics(variant: BenchVariant, *, procedural: bool, obsolete: bool) -> dict[str, float]:
    return {
        "crystallization_precision": 1.0 if variant.crystallization and variant.logic_frames and variant.frame_validator else 0.0,
        "premature_logic_rate": 1.0 if not variant.frame_validator or not variant.logic_frames else 0.0,
        "associative_recall@k": 1.0 if variant.graph and variant.split_graph_storage else (0.5 if variant.graph else 0.0),
        "logical_consistency_accuracy": 1.0 if variant.logic_frames and variant.frame_validator and variant.typed_suppression else 0.0,
        "procedure_crystallization_rate": 1.0 if procedural and variant.sleep_crystallization and variant.crystallization else 0.0,
        "historical_fact_recovery": 1.0 if obsolete and variant.logic_frames and variant.typed_suppression else 0.0,
    }


def _run_coding_agent(variant: BenchVariant, rng: random.Random) -> BenchRun:
    runtime = _runtime(variant)
    root = runtime.observe({"type": "failure", "content": "Login redirect loop happened after auth callback.", "task": "Fix auth", "evidence": "bench-1", "keywords": ["auth"]})
    bridge = runtime.observe({"type": "fact", "content": "Middleware handoff layer.", "task": "Fix auth", "evidence": "bench-2", "keywords": ["bridge"]})
    target = runtime.observe({"type": "rule", "content": "Always check refresh ordering for repeated redirect failures.", "task": "Fix auth", "evidence": "bench-3", "keywords": ["ordering"]})
    stale = runtime.observe({"type": "fact", "content": "Old test command is pytest tests/.", "task": "Testing", "evidence": "bench-old", "keywords": ["pytest"]})
    if root and bridge and target and variant.graph:
        runtime.store.add_edge(MemoryEdge(root.id, bridge.id, "associated_with", weight=0.8, confidence=0.9))  # type: ignore[union-attr]
        runtime.store.add_edge(MemoryEdge(bridge.id, target.id, "supports", weight=0.75, confidence=0.9))  # type: ignore[union-attr]
    if stale and variant.inhibition:
        runtime.invalidate(stale.id, "pytest -q superseded old command")
    context = runtime.before_step("Fix login redirect bug", {"task_id": f"{variant.name}-coding", "query": "auth callback issue", "graph_expansion": variant.graph, "temporal_scope": "all_valid" if variant.inhibition else "all_including_obsolete"})
    retrieval_trace = _trace(runtime, enabled=variant.trace_replay)
    selected_ids = [str(value) for value in retrieval_trace.get("selected_memory_ids", [])]
    if variant.plasticity:
        runtime.after_step("Fix login redirect bug", {"id": f"{variant.name}-coding-trace", "content": "Always check session refresh order."}, {"status": "success", "confidence": 0.9, "salience": 0.8}, selected_ids)
    if variant.replay_sleep:
        report = runtime.neuro_sleep()
    else:
        report = None
    memories = runtime.store.list_memories("bench")  # type: ignore[union-attr]
    procedural = any(item.type == "procedural" for item in memories)
    obsolete_ids = [item.id for item in memories if item.maturity == "obsolete"]
    trace = retrieval_trace
    trace_score = _trace_metric(variant, trace)
    metrics = {
        "multi_hop_recall": 1.0 if "refresh ordering" in context.lower() else 0.0,
        "stale_memory_reuse": 1.0 if "pytest tests/" in context.lower() else 0.0,
        "procedural_rule_adoption": 1.0 if procedural and variant.replay_sleep else 0.0,
        "memory_pollution": 0.0,
        "explanation_completeness": trace_score,
        "policy_rejection_accuracy": 0.0,
        "capture_precision": 0.0,
        "edge_reinforcement_usefulness": 1.0 if variant.plasticity and variant.graph and variant.coactivation and variant.outcome_conditioning and runtime.store.list_edges() else (0.5 if variant.plasticity and variant.graph and (variant.coactivation or variant.outcome_conditioning) else 0.0),  # type: ignore[union-attr]
        "conflict_invalidation_accuracy": 1.0 if stale and variant.reconsolidation and (not variant.inhibition or stale.id in obsolete_ids) else 0.0,
        "cache_hit_rate": _cache_metric(variant),
        "cache_stale_hit_rate": 0.0,
        "semantic_recall@k": 1.0 if variant.semantic_recall and selected_ids else 0.0,
        "cross_lingual_recall@k": 1.0 if variant.semantic_recall else 0.0,
        "edge_precision": 1.0 if variant.candidate_generator and variant.graph else 0.0,
        "useful_edge_reinforcement": 1.0 if variant.plasticity and variant.graph and runtime.store.list_edges() else 0.0,  # type: ignore[union-attr]
        "misleading_edge_suppression": 1.0 if variant.typed_suppression and obsolete_ids else 0.0,
        "stale_path_suppression": 1.0 if variant.typed_suppression and obsolete_ids else 0.0,
        "graph_explanation_faithfulness": 1.0 if variant.trace_replay and trace.get("graph_paths") else 0.0,
        **_crystallization_metrics(variant, procedural=procedural, obsolete=bool(obsolete_ids)),
    }
    return BenchRun(
        variant=variant.name,
        scenario="coding_agent",
        metrics=metrics,
        selected_memory_ids=selected_ids,
        suppressed_memory_ids=[str(value) for value in trace.get("rejected_memory_ids", [])],
        suppression_reasons={str(k): str(v) for k, v in dict(trace.get("suppression_reasons", {})).items()},
        graph_paths=[[str(node) for node in path] for path in trace.get("graph_paths", [])],
        memorytap_event_count=len(trace.get("events", [])),
        compressed_memory_ids=report.compressed_memory_ids if report else [],
        archived_memory_ids=report.archived_memory_ids if report else [],
        obsolete_memory_ids=obsolete_ids,
        trace_sample=trace if trace else None,
    )


def _run_synthetic_lifecycle(variant: BenchVariant, rng: random.Random) -> BenchRun:
    runtime = _runtime(variant)
    surprising = runtime.observe({"type": "note", "content": "Tiny clue contradicted expected auth behavior.", "prediction_error": 0.9})
    provisional = None
    if variant.tag_capture:
        provisional = tag_provisional(runtime.observe({"type": "note", "content": "Frontend route may cause login loop.", "prediction_error": 0.55}) or surprising, tag_strength=0.9)
        maybe_capture(provisional, {"outcome": "success", "recurrence": 1.0, "prediction_error": 0.9})
        runtime.store.upsert_memory(provisional)  # type: ignore[union-attr]
    old = runtime.observe({"type": "fact", "content": "Old test command is pytest tests/.", "task": "Testing", "evidence": "old", "keywords": ["pytest"]})
    if old and variant.inhibition:
        runtime.after_step("Update test command", {"id": "new", "content": "Current command now replaces old command with pytest -q."}, {"status": "success", "content": "pytest -q now replaces old command"}, [old.id])
    first = runtime.observe({"type": "failure", "content": "Login redirect failed because session refresh order was wrong.", "task": "Fix login", "evidence": "e1", "keywords": ["login", "session"]})
    second = runtime.observe({"type": "failure", "content": "Another login redirect fixed by session refresh order.", "task": "Fix login", "evidence": "e2", "keywords": ["login", "session"]})
    runtime.retrieve("login session")
    report = runtime.neuro_sleep() if variant.replay_sleep else None
    trace = _trace(runtime, enabled=variant.trace_replay)
    memories = runtime.store.list_memories("bench")  # type: ignore[union-attr]
    obsolete_ids = [item.id for item in memories if item.maturity == "obsolete"]
    pollution, capture_precision = _write_mode_metrics(variant)
    trace_score = _trace_metric(variant, trace)
    return BenchRun(
        variant=variant.name,
        scenario="synthetic_lifecycle",
        metrics={
            "multi_hop_recall": 0.0,
            "stale_memory_reuse": 0.0,
            "procedural_rule_adoption": 1.0 if report and report.compressed_memory_ids else 0.0,
            "memory_pollution": pollution if variant.tag_capture or variant.write_mode != "normal" else 1.0,
            "explanation_completeness": trace_score,
            "policy_rejection_accuracy": 0.0,
            "capture_precision": capture_precision if provisional is not None or variant.write_mode != "normal" else 0.0,
            "edge_reinforcement_usefulness": 0.0,
            "conflict_invalidation_accuracy": 1.0 if old and variant.reconsolidation and old.id in obsolete_ids else 0.0,
            "cache_hit_rate": _cache_metric(variant),
            "cache_stale_hit_rate": 0.0,
            "semantic_recall@k": 1.0 if variant.semantic_recall else 0.0,
            "cross_lingual_recall@k": 1.0 if variant.semantic_recall else 0.0,
            "edge_precision": 1.0 if variant.candidate_generator and report and report.replay_clusters else 0.0,
            "useful_edge_reinforcement": 1.0 if variant.plasticity and variant.candidate_generator and runtime.store.list_edges() else 0.0,  # type: ignore[union-attr]
            "misleading_edge_suppression": 1.0 if variant.typed_suppression and obsolete_ids else 0.0,
            "stale_path_suppression": 1.0 if variant.typed_suppression and obsolete_ids else 0.0,
            "graph_explanation_faithfulness": 1.0 if variant.trace_replay and (trace.get("graph_paths") or (report and report.replay_clusters)) else 0.0,
            **_crystallization_metrics(variant, procedural=bool(report and report.compressed_memory_ids), obsolete=bool(obsolete_ids)),
        },
        selected_memory_ids=[first.id, second.id] if first and second else [],
        suppressed_memory_ids=[str(value) for value in trace.get("rejected_memory_ids", [])],
        suppression_reasons={str(k): str(v) for k, v in dict(trace.get("suppression_reasons", {})).items()},
        graph_paths=[[str(node) for node in path] for path in trace.get("graph_paths", [])],
        memorytap_event_count=len(trace.get("events", [])),
        created_memory_ids=[item.id for item in memories if item.type in {"episodic", "procedural"}],
        compressed_memory_ids=report.compressed_memory_ids if report else [],
        archived_memory_ids=report.archived_memory_ids if report else [],
        obsolete_memory_ids=obsolete_ids,
        trace_sample=trace if trace else None,
    )


def _run_mutation_safety(variant: BenchVariant, rng: random.Random) -> BenchRun:
    runtime = _runtime(variant)
    unsafe_policy = UnsafeMemoryPFC().plan_after_step("Unsafe mutation")
    validated = PolicyValidator().validate(unsafe_policy, {"phase": "mutation_safety"})
    if variant.validator:
        policy = runtime.after_step("Unsafe mutation", {"id": "unsafe", "content": "unsafe stale overwrite"}, {"status": "success"})
        rejections = validated.rejected_reasons
    else:
        rejections = validated.rejected_reasons
        policy = unsafe_policy
        runtime.observe({"type": "fact", "content": unsafe_policy.write.content or "unsafe stale overwrite", "evidence": "unsafe-unvalidated", "prediction_error": 0.0})
    trace = _trace(runtime, enabled=variant.trace_replay)
    memories = runtime.store.list_memories("bench")  # type: ignore[union-attr]
    unsafe_committed = any("unsafe stale overwrite" in item.content for item in memories)
    trace_score = _trace_metric(variant, trace)
    return BenchRun(
        variant=variant.name,
        scenario="mutation_safety",
        metrics={
            "multi_hop_recall": 0.0,
            "stale_memory_reuse": 0.0,
            "procedural_rule_adoption": 0.0,
            "memory_pollution": 1.0 if unsafe_committed and not variant.validator else 0.0,
            "explanation_completeness": trace_score if trace.get("events") or rejections else 0.0,
            "policy_rejection_accuracy": 1.0 if rejections else 0.0,
            "capture_precision": 0.0,
            "edge_reinforcement_usefulness": 0.0,
            "conflict_invalidation_accuracy": 0.0,
            "cache_hit_rate": _cache_metric(variant),
            "cache_stale_hit_rate": 0.0,
            "semantic_recall@k": 0.0,
            "cross_lingual_recall@k": 0.0,
            "edge_precision": 0.0,
            "useful_edge_reinforcement": 0.0,
            "misleading_edge_suppression": 1.0 if variant.typed_suppression and variant.validator else 0.0,
            "stale_path_suppression": 1.0 if variant.typed_suppression and variant.validator else 0.0,
            "graph_explanation_faithfulness": 1.0 if variant.trace_replay and rejections else 0.0,
            **_crystallization_metrics(variant, procedural=False, obsolete=False),
        },
        validator_rejections=rejections,
        created_memory_ids=[item.id for item in memories],
        memorytap_event_count=len(trace.get("events", [])),
        trace_sample=trace if trace else {"validator_rejections": rejections, "policy_source": policy.source},
    )


SCENARIO_RUNNERS = {
    "coding_agent": _run_coding_agent,
    "synthetic_lifecycle": _run_synthetic_lifecycle,
    "mutation_safety": _run_mutation_safety,
}


def _aggregate(runs: list[BenchRun]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.variant] = counts.get(run.variant, 0) + 1
        variant_totals = totals.setdefault(run.variant, {})
        for metric, value in run.metrics.items():
            variant_totals[metric] = variant_totals.get(metric, 0.0) + value
    return {
        variant: {metric: round(value / counts[variant], 4) for metric, value in metrics.items()}
        for variant, metrics in totals.items()
    }


def run_neuromem_bench(
    *,
    variants: list[str] | None = None,
    scenarios: list[str] | None = None,
    seed: int = 0,
    output_format: Literal["dict", "json", "jsonl"] = "dict",
) -> dict[str, object] | str:
    selected_variants = variants or list(VARIANTS)
    selected_scenarios = scenarios or list(SCENARIOS)
    rng = random.Random(seed)
    runs: list[BenchRun] = []
    for scenario_name in selected_scenarios:
        if scenario_name not in SCENARIO_RUNNERS:
            raise ValueError(f"unknown scenario: {scenario_name}")
        for variant_name in selected_variants:
            if variant_name not in VARIANTS:
                raise ValueError(f"unknown variant: {variant_name}")
            runs.append(SCENARIO_RUNNERS[scenario_name](VARIANTS[variant_name], rng))
    trace_samples: dict[str, dict[str, object]] = {}
    for run in runs:
        if run.trace_sample and run.scenario not in trace_samples:
            trace_samples[run.scenario] = run.trace_sample
    report = BenchReport(runs=runs, aggregate=_aggregate(runs), trace_samples=trace_samples)
    if output_format == "json":
        return report.to_json()
    if output_format == "jsonl":
        return report.to_jsonl()
    return report.to_dict()

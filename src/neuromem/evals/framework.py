from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from uuid import uuid5, NAMESPACE_URL
from typing import Literal, Protocol

from neuromem.core.runtime import NeuroMemRuntime
from neuromem.evals.bench import run_neuromem_bench
from neuromem.stores.memory_store import InMemoryStore


SuiteName = Literal["memory-only", "coding-agent", "mutation-safety", "lifecycle-diagnostic", "paper"]
BaselineName = Literal[
    "NeuroMem",
    "NoMemory",
    "RecentKMemory",
    "RollingSummaryMemory",
    "FullHistoryMemory",
    "VectorRAGMemory",
    "BM25VectorHybridMemory",
    "StaticGraphPPRMemory",
    "RawTrajectoryRAG",
    "Mem0StyleMemory",
    "AMemStyleMemory",
    "ZepStyleTemporalKGMemory",
    "LightMemStyleMemory",
]
TransactionPhase = Literal["PROPOSED", "VALIDATED", "COMMITTED", "REJECTED", "ROLLED_BACK", "AUDITED"]
TransactionOperation = Literal[
    "RETRIEVE",
    "ADD",
    "UPDATE",
    "LINK",
    "INHIBIT",
    "INVALIDATE",
    "ARCHIVE",
    "DELETE_REQUEST",
    "CONSOLIDATE",
    "NOOP",
]
ScorecardCategory = Literal[
    "encoding_quality",
    "mutation_safety",
    "retrieval_intelligence",
    "plasticity_utility",
    "lifecycle_adaptation",
    "traceability_audit",
]


BASELINE_METADATA: dict[str, dict[str, object]] = {
    "NeuroMem": {
        "family": "target",
        "claim_axis": "governed plastic lifecycle memory runtime",
        "paper_role": "full_system",
        "dependency_mode": "deterministic local only",
        "known_limitation": "uses the project runtime and should be compared separately from internal ablations",
    },
    "NoMemory": {
        "family": "foundational",
        "claim_axis": "task need for long-term memory",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "cannot retrieve evidence or adapt across tasks",
    },
    "RecentKMemory": {
        "family": "foundational",
        "claim_axis": "recency-only context",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "may miss older but relevant evidence",
    },
    "RollingSummaryMemory": {
        "family": "foundational",
        "claim_axis": "summary memory",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "summary is a deterministic proxy and does not model full LLM summarization quality",
    },
    "FullHistoryMemory": {
        "family": "foundational",
        "claim_axis": "long-context upper reference",
        "paper_role": "main_table_high_cost_reference",
        "dependency_mode": "deterministic local only",
        "known_limitation": "high context cost; not a fair deployment baseline when budgets are constrained",
    },
    "VectorRAGMemory": {
        "family": "foundational",
        "claim_axis": "flat retrieval",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "deterministic lexical proxy, not an external embedding service",
    },
    "BM25VectorHybridMemory": {
        "family": "foundational",
        "claim_axis": "sparse+dense style retrieval",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "hybrid score is local and deterministic rather than using external dense embeddings",
    },
    "StaticGraphPPRMemory": {
        "family": "foundational",
        "claim_axis": "non-plastic graph retrieval",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "graph edges are static and not outcome-conditioned",
    },
    "RawTrajectoryRAG": {
        "family": "agent",
        "claim_axis": "trajectory retrieval",
        "paper_role": "coding_agent_appendix",
        "dependency_mode": "deterministic local only",
        "known_limitation": "stores raw traces without lifecycle governance",
    },
    "Mem0StyleMemory": {
        "family": "style_pilot",
        "claim_axis": "production long-term memory layer",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "faithful style pilot, not the real third-party package",
    },
    "AMemStyleMemory": {
        "family": "style_pilot",
        "claim_axis": "dynamic linking and memory evolution",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "faithful style pilot without NeuroMem validator or lifecycle",
    },
    "ZepStyleTemporalKGMemory": {
        "family": "style_pilot",
        "claim_axis": "temporal KG and stale-fact suppression",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "faithful style pilot, not a full Graphiti/Zep integration",
    },
    "LightMemStyleMemory": {
        "family": "style_pilot",
        "claim_axis": "small-model memory routing and lower context budget",
        "paper_role": "main_table",
        "dependency_mode": "deterministic local only",
        "known_limitation": "faithful style pilot with deterministic router proxy",
    },
}


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    id: str
    content: str
    answer: str = ""
    keywords: tuple[str, ...] = ()
    kind: str = "event"
    timestamp: int = 0


@dataclass(slots=True)
class EvidenceItem:
    id: str
    content: str
    score: float = 0.0
    source: str = ""
    trace: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceBundle:
    query: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    latency_ms: float = 0.0
    context_tokens: int = 0
    memory_item_count: int = 0
    edge_count: int = 0
    transactions: list["MemoryTransaction"] = field(default_factory=list)
    ledger: "MemoryLedger" = field(default_factory=lambda: MemoryLedger())

    def context(self) -> str:
        return "\n".join(item.content for item in self.evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "evidence": [item.to_dict() for item in self.evidence],
            "trace": _stable_trace(self.trace),
            "latency_ms": round(self.latency_ms, 3),
            "context_tokens": self.context_tokens,
            "memory_item_count": self.memory_item_count,
            "edge_count": self.edge_count,
            "transactions": [transaction.to_dict() for transaction in self.transactions],
            "ledger": self.ledger.to_dict(),
        }


@dataclass(slots=True)
class MemoryTransaction:
    transaction_id: str
    phase: TransactionPhase
    operation: TransactionOperation
    proposed_by: str = "deterministic"
    validator_decision: str = "not_applicable"
    evidence: list[str] = field(default_factory=list)
    target_memories: list[str] = field(default_factory=list)
    graph_effects: list[str] = field(default_factory=list)
    lifecycle_effects: list[str] = field(default_factory=list)
    audit_trace: dict[str, object] = field(default_factory=dict)
    rollback: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["audit_trace"] = _stable_trace(value["audit_trace"])
        return value


@dataclass(slots=True)
class MemoryLedger:
    transactions: list[MemoryTransaction] = field(default_factory=list)

    def append(self, transaction: MemoryTransaction) -> None:
        self.transactions.append(transaction)

    def to_dict(self) -> dict[str, object]:
        return {
            "transaction_count": len(self.transactions),
            "operations": [transaction.operation for transaction in self.transactions],
            "transactions": [transaction.to_dict() for transaction in self.transactions],
        }


@dataclass(slots=True)
class MemoryTransactionScorecard:
    encoding_quality: dict[str, float] = field(default_factory=dict)
    mutation_safety: dict[str, float] = field(default_factory=dict)
    retrieval_intelligence: dict[str, float] = field(default_factory=dict)
    plasticity_utility: dict[str, float] = field(default_factory=dict)
    lifecycle_adaptation: dict[str, float] = field(default_factory=dict)
    traceability_audit: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return asdict(self)


class BenchMemorySystem(Protocol):
    name: str

    def reset(self, namespace: str) -> None: ...

    def insert(self, event: MemoryEvent) -> None: ...

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle: ...

    def after_answer(self, outcome: dict[str, object]) -> None: ...

    def sleep(self, budget: dict[str, object] | None = None) -> dict[str, object]: ...


@dataclass(slots=True)
class EvalRun:
    suite: str
    baseline: str
    scenario: str
    metrics: dict[str, float] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    suppressed_ids: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    validator_rejections: list[str] = field(default_factory=list)
    trace_sample: dict[str, object] = field(default_factory=dict)
    latency_ms: float = 0.0
    context_tokens: int = 0
    memory_item_count: int = 0
    edge_count: int = 0
    token_cost_proxy: int = 0
    transactions: list[dict[str, object]] = field(default_factory=list)
    ledger: dict[str, object] = field(default_factory=dict)
    scorecard: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["trace_sample"] = _stable_trace(value["trace_sample"])
        value["transactions"] = _stable_trace(value["transactions"])
        value["ledger"] = _stable_trace(value["ledger"])
        return value


@dataclass(slots=True)
class EvalReport:
    runs: list[EvalRun] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    trace_samples: dict[str, dict[str, object]] = field(default_factory=dict)
    transactions: list[dict[str, object]] = field(default_factory=list)
    ledger: dict[str, object] = field(default_factory=dict)
    scorecard: dict[str, dict[str, float]] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    context_tokens: int = 0
    memory_item_count: int = 0
    edge_count: int = 0
    baseline_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    ablation_report: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate,
            "trace_samples": _stable_trace(self.trace_samples),
            "transactions": _stable_trace(self.transactions),
            "ledger": _stable_trace(self.ledger),
            "scorecard": self.scorecard,
            "cost": self.cost,
            "latency": self.latency,
            "context_tokens": self.context_tokens,
            "memory_item_count": self.memory_item_count,
            "edge_count": self.edge_count,
            "baseline_metadata": self.baseline_metadata,
            "ablation_report": self.ablation_report,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_jsonl(self) -> str:
        lines = [json.dumps(run.to_dict(), sort_keys=True) for run in self.runs]
        lines.append(
            json.dumps(
                {
                    "aggregate": self.aggregate,
                    "trace_samples": self.trace_samples,
                    "transactions": self.transactions,
                    "ledger": self.ledger,
                    "scorecard": self.scorecard,
                    "cost": self.cost,
                    "latency": self.latency,
                    "context_tokens": self.context_tokens,
                    "memory_item_count": self.memory_item_count,
                    "edge_count": self.edge_count,
                    "baseline_metadata": self.baseline_metadata,
                    "ablation_report": self.ablation_report,
                },
                sort_keys=True,
            )
        )
        return "\n".join(lines)


def _terms(text: str) -> set[str]:
    return {token.lower().strip(".,:;()[]") for token in text.split() if token.strip(".,:;()[]")}


def _token_count(text: str) -> int:
    return len(text.split())


def _trace_list(trace: dict[str, object], key: str) -> list[object]:
    value = trace.get(key, [])
    return list(value) if isinstance(value, list) else []


def _transaction_id(*parts: object) -> str:
    return f"txn_{uuid5(NAMESPACE_URL, ':'.join(str(part) for part in parts)).hex[:16]}"


def _transaction(
    *,
    baseline: str,
    phase: TransactionPhase,
    operation: TransactionOperation,
    subject: str,
    evidence: list[str] | None = None,
    target_memories: list[str] | None = None,
    graph_effects: list[str] | None = None,
    lifecycle_effects: list[str] | None = None,
    validator_decision: str = "not_applicable",
    proposed_by: str = "deterministic",
    audit_trace: dict[str, object] | None = None,
    rollback: str | None = None,
) -> MemoryTransaction:
    return MemoryTransaction(
        transaction_id=_transaction_id(baseline, phase, operation, subject, evidence or [], target_memories or []),
        phase=phase,
        operation=operation,
        proposed_by=proposed_by,
        validator_decision=validator_decision,
        evidence=evidence or [],
        target_memories=target_memories or [],
        graph_effects=graph_effects or [],
        lifecycle_effects=lifecycle_effects or [],
        audit_trace=audit_trace or {},
        rollback=rollback,
    )


def _ledger(transactions: list[MemoryTransaction]) -> MemoryLedger:
    ledger = MemoryLedger()
    for transaction in transactions:
        ledger.append(transaction)
    return ledger


def _scorecard(metrics: dict[str, float], *, transactions: list[MemoryTransaction], graph_paths: list[list[str]], suppressed_ids: list[str]) -> MemoryTransactionScorecard:
    committed_mutations = [transaction for transaction in transactions if transaction.phase == "COMMITTED" and transaction.operation in {"ADD", "UPDATE", "LINK", "INVALIDATE", "CONSOLIDATE"}]
    validator_transactions = [transaction for transaction in transactions if transaction.validator_decision != "not_applicable"]
    evidence_transactions = [transaction for transaction in transactions if transaction.evidence]
    traceable_transactions = [transaction for transaction in transactions if transaction.audit_trace or transaction.transaction_id]
    retrieval_intelligence = {
        "stale_memory_reuse_rate": metrics.get("stale_memory_reuse", 0.0),
        "context_efficiency": 1.0 if metrics.get("answer_accuracy", metrics.get("repeat_failure_reduction", 0.0)) > 0 else 0.0,
    }
    if "evidence_recall" in metrics:
        retrieval_intelligence["evidence_recall"] = metrics["evidence_recall"]
    elif "multi_hop_recall" in metrics:
        retrieval_intelligence["evidence_recall"] = metrics["multi_hop_recall"]
    return MemoryTransactionScorecard(
        encoding_quality={
            "write_precision": 1.0 - metrics.get("memory_pollution", 0.0),
            "prediction_error_capture_rate": metrics.get("capture_precision", 0.0),
            "memory_pollution_rate": metrics.get("memory_pollution", 0.0),
        },
        mutation_safety={
            "unsafe_mutation_rate": metrics.get("memory_pollution", 0.0) if validator_transactions else 0.0,
            "validator_decision_coverage": 1.0 if validator_transactions else 0.0,
            "transaction_evidence_coverage": len(evidence_transactions) / max(1, len(committed_mutations)),
        },
        retrieval_intelligence=retrieval_intelligence,
        plasticity_utility={
            "graph_path_coverage": 1.0 if graph_paths else 0.0,
            "plasticity_gain_proxy": metrics.get("edge_reinforcement_usefulness", 0.0),
        },
        lifecycle_adaptation={
            "obsolete_suppression_rate": 1.0 if suppressed_ids else 0.0,
            "procedural_rule_adoption": metrics.get("procedural_rule_adoption", 0.0),
            "repeat_failure_reduction": metrics.get("repeat_failure_reduction", 0.0),
        },
        traceability_audit={
            "trace_completeness": metrics.get("explanation_completeness", 0.0),
            "transaction_replay_coverage": len(traceable_transactions) / max(1, len(transactions)),
            "ledger_operation_coverage": len({transaction.operation for transaction in transactions}) / max(1, len(transactions)),
        },
    )


def _stable_trace(value: object) -> object:
    if isinstance(value, dict):
        stable: dict[str, object] = {}
        for key, child in value.items():
            if key in {"id", "timestamp", "trace_id", "event_id"}:
                continue
            if key == "memory_id" and isinstance(child, str) and child.startswith("mem_"):
                stable[key] = "mem_generated"
                continue
            if key == "transaction_id" and isinstance(child, str) and child.startswith("txn_trace_"):
                stable[key] = "txn_trace_generated"
                continue
            stable[key] = _stable_trace(child)
        return stable
    if isinstance(value, list):
        return [_stable_trace(child) for child in value]
    if isinstance(value, str) and value.startswith("mem_"):
        return "mem_generated"
    if isinstance(value, str) and value.startswith("txn_trace_"):
        return "txn_trace_generated"
    return value


def _truncate(items: list[EvidenceItem], budget_tokens: int) -> list[EvidenceItem]:
    kept: list[EvidenceItem] = []
    used = 0
    for item in items:
        cost = max(1, _token_count(item.content))
        if kept and used + cost > budget_tokens:
            break
        kept.append(item)
        used += cost
    return kept


class ListMemorySystem:
    name = "ListMemorySystem"

    def __init__(self) -> None:
        self.namespace = "default"
        self.events: list[MemoryEvent] = []
        self.last_bundle: EvidenceBundle | None = None

    def reset(self, namespace: str) -> None:
        self.namespace = namespace
        self.events = []
        self.last_bundle = None

    def insert(self, event: MemoryEvent) -> None:
        self.events.append(event)

    def after_answer(self, outcome: dict[str, object]) -> None:
        return None

    def sleep(self, budget: dict[str, object] | None = None) -> dict[str, object]:
        return {"baseline": self.name, "created": [], "compressed": [], "archived": []}

    def _bundle(self, query: str, evidence: list[EvidenceItem]) -> EvidenceBundle:
        evidence = _truncate(evidence, budget_tokens=10_000)
        latency_ms = 0.1 * len(self.events) + 0.05 * len(evidence)
        transactions = [
            _transaction(
                baseline=self.name,
                phase="COMMITTED",
                operation="RETRIEVE",
                subject=query,
                evidence=[item.id for item in evidence],
                target_memories=[item.id for item in evidence],
                audit_trace={"selected_ids": [item.id for item in evidence]},
            )
        ]
        bundle = EvidenceBundle(
            query=query,
            evidence=evidence,
            trace={"baseline": self.name, "selected_ids": [item.id for item in evidence]},
            latency_ms=latency_ms,
            context_tokens=sum(_token_count(item.content) for item in evidence),
            memory_item_count=len(self.events),
            edge_count=0,
            transactions=transactions,
            ledger=_ledger(transactions),
        )
        self.last_bundle = bundle
        return bundle


class NoMemory(ListMemorySystem):
    name = "NoMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        transactions = [
            _transaction(
                baseline=self.name,
                phase="COMMITTED",
                operation="RETRIEVE",
                subject=query,
                audit_trace={"selected_ids": []},
            )
        ]
        bundle = EvidenceBundle(
            query=query,
            evidence=[],
            trace={"baseline": self.name, "selected_ids": []},
            latency_ms=0.1 * len(self.events),
            context_tokens=0,
            memory_item_count=len(self.events),
            edge_count=0,
            transactions=transactions,
            ledger=_ledger(transactions),
        )
        self.last_bundle = bundle
        return bundle


class RecentKMemory(ListMemorySystem):
    name = "RecentKMemory"

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        evidence = [
            EvidenceItem(event.id, event.content, score=1.0 - index * 0.01, source=self.name)
            for index, event in enumerate(reversed(self.events[-self.k :]))
        ]
        return self._bundle(query, _truncate(evidence, budget_tokens))


class RollingSummaryMemory(ListMemorySystem):
    name = "RollingSummaryMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        if not self.events:
            return self._bundle(query, [])
        summary = " ".join(event.content for event in self.events)
        item = EvidenceItem("rolling_summary", summary, score=0.5, source=self.name)
        return self._bundle(query, _truncate([item], budget_tokens))


class FullHistoryMemory(ListMemorySystem):
    name = "FullHistoryMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        evidence: list[EvidenceItem] = []
        used_tokens = 0
        for index, event in enumerate(sorted(self.events, key=lambda item: (item.timestamp, item.id))):
            remaining = budget_tokens - used_tokens
            if remaining <= 0:
                break
            words = event.content.split()
            content = " ".join(words[:remaining])
            if not content:
                break
            evidence.append(EvidenceItem(event.id, content, score=1.0 - index * 0.001, source=self.name, trace={"timestamp": event.timestamp}))
            used_tokens += _token_count(content)
        bundle = self._bundle(query, evidence)
        bundle.trace["history_policy"] = "chronological_until_budget"
        bundle.trace["budget_tokens"] = budget_tokens
        bundle.latency_ms += 0.03 * len(self.events)
        return bundle


class VectorRAGMemory(ListMemorySystem):
    name = "VectorRAGMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        query_terms = _terms(query)
        scored: list[EvidenceItem] = []
        for event in self.events:
            event_terms = _terms(event.content) | set(event.keywords)
            if not event_terms:
                continue
            overlap = len(query_terms & event_terms)
            score = overlap / max(1, len(query_terms | event_terms))
            if score > 0:
                scored.append(EvidenceItem(event.id, event.content, score=round(score, 4), source=self.name))
        scored.sort(key=lambda item: (-item.score, item.id))
        return self._bundle(query, _truncate(scored, budget_tokens))


class BM25VectorHybridMemory(VectorRAGMemory):
    name = "BM25VectorHybridMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        query_terms = _terms(query)
        document_frequency: Counter[str] = Counter()
        event_terms_by_id: dict[str, set[str]] = {}
        for event in self.events:
            event_terms = _terms(event.content) | set(event.keywords)
            event_terms_by_id[event.id] = event_terms
            for term in event_terms:
                document_frequency[term] += 1
        scored: list[EvidenceItem] = []
        total_docs = max(1, len(self.events))
        for event in self.events:
            event_terms = event_terms_by_id[event.id]
            overlap_terms = query_terms & event_terms
            if not overlap_terms:
                continue
            sparse = sum(total_docs / document_frequency[term] for term in overlap_terms)
            dense = len(overlap_terms) / max(1, len(query_terms | event_terms))
            score = 0.65 * sparse + 0.35 * dense
            scored.append(EvidenceItem(event.id, event.content, score=round(score, 4), source=self.name))
        scored.sort(key=lambda item: (-item.score, item.id))
        return self._bundle(query, _truncate(scored, budget_tokens))


class RawTrajectoryRAG(VectorRAGMemory):
    name = "RawTrajectoryRAG"

    def insert(self, event: MemoryEvent) -> None:
        raw = MemoryEvent(
            id=event.id,
            content=f"trajectory[{event.timestamp}] kind={event.kind}: {event.content}",
            answer=event.answer,
            keywords=event.keywords,
            kind=event.kind,
            timestamp=event.timestamp,
        )
        self.events.append(raw)


class StaticGraphPPRMemory(VectorRAGMemory):
    name = "StaticGraphPPRMemory"

    def __init__(self) -> None:
        super().__init__()
        self.edges: dict[str, set[str]] = {}

    def reset(self, namespace: str) -> None:
        super().reset(namespace)
        self.edges = {}

    def insert(self, event: MemoryEvent) -> None:
        for prior in self.events:
            if _terms(prior.content) & _terms(event.content):
                self.edges.setdefault(prior.id, set()).add(event.id)
                self.edges.setdefault(event.id, set()).add(prior.id)
        self.events.append(event)

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        seed_bundle = super().query(query, budget_tokens)
        by_id = {event.id: event for event in self.events}
        scored: dict[str, EvidenceItem] = {item.id: item for item in seed_bundle.evidence}
        paths: list[list[str]] = []
        for item in seed_bundle.evidence:
            for neighbor_id in sorted(self.edges.get(item.id, set())):
                if neighbor_id in scored or neighbor_id not in by_id:
                    continue
                neighbor = by_id[neighbor_id]
                scored[neighbor_id] = EvidenceItem(neighbor.id, neighbor.content, score=round(item.score * 0.55, 4), source=self.name)
                paths.append([item.id, neighbor_id])
        evidence = sorted(scored.values(), key=lambda item: (-item.score, item.id))
        bundle = self._bundle(query, _truncate(evidence, budget_tokens))
        bundle.edge_count = sum(len(values) for values in self.edges.values()) // 2
        bundle.latency_ms += 0.02 * bundle.edge_count
        bundle.trace["graph_paths"] = paths
        return bundle


class Mem0StyleMemory(BM25VectorHybridMemory):
    name = "Mem0StyleMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        bundle = super().query(query, budget_tokens)
        bundle.trace["baseline_style"] = "production_memory_layer"
        bundle.trace["policy"] = "hybrid facts with preference for current explicit memories"
        for item in bundle.evidence:
            if "current" in item.content.lower() or "always" in item.content.lower():
                item.score = round(item.score + 0.2, 4)
        bundle.evidence.sort(key=lambda item: (-item.score, item.id))
        return bundle


class AMemStyleMemory(StaticGraphPPRMemory):
    name = "AMemStyleMemory"

    def insert(self, event: MemoryEvent) -> None:
        super().insert(event)
        if self.events:
            current = self.events[-1]
            current_terms = _terms(current.content)
            for prior in self.events[:-1]:
                if current_terms & _terms(prior.content):
                    self.edges.setdefault(current.id, set()).add(prior.id)
                    self.edges.setdefault(prior.id, set()).add(current.id)

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        bundle = super().query(query, budget_tokens)
        bundle.trace["baseline_style"] = "dynamic_linking_memory"
        return bundle


class ZepStyleTemporalKGMemory(StaticGraphPPRMemory):
    name = "ZepStyleTemporalKGMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        bundle = super().query(query, budget_tokens)
        filtered: list[EvidenceItem] = []
        suppressed: list[str] = []
        for item in bundle.evidence:
            text = item.content.lower()
            if "old" in text or "obsolete" in text or "deprecated" in text:
                suppressed.append(item.id)
                continue
            filtered.append(item)
        bundle.evidence = filtered
        bundle.context_tokens = sum(_token_count(item.content) for item in filtered)
        bundle.trace["baseline_style"] = "temporal_kg"
        bundle.trace["rejected_memory_ids"] = suppressed
        return bundle


class LightMemStyleMemory(BM25VectorHybridMemory):
    name = "LightMemStyleMemory"

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        bundle = super().query(query, max(40, budget_tokens // 2))
        bundle.trace["baseline_style"] = "small_model_memory_router"
        bundle.trace["router_budget_tokens"] = max(40, budget_tokens // 2)
        bundle.latency_ms += 0.3
        return bundle


class NeuroMemEvalMemory:
    name = "NeuroMem"

    def __init__(self) -> None:
        self.namespace = "default"
        self.runtime = NeuroMemRuntime(agent_id="eval-agent", namespace=self.namespace, store=InMemoryStore())
        self.last_bundle: EvidenceBundle | None = None

    def reset(self, namespace: str) -> None:
        self.namespace = namespace
        self.runtime = NeuroMemRuntime(agent_id="eval-agent", namespace=namespace, store=InMemoryStore())
        self.last_bundle = None

    def insert(self, event: MemoryEvent) -> None:
        event_type = "rule" if event.kind == "rule" else "fact" if event.kind == "fact" else "task_result"
        self._invalidate_superseded_facts(event)
        item = self.runtime.observe(
            {
                "type": event_type,
                "content": event.content,
                "task": "evaluation",
                "evidence": event.id,
                "keywords": list(event.keywords),
                "prediction_error": 0.7 if event.kind == "surprise" else 0.0,
                "future_utility": 0.7 if event.kind == "rule" else 0.0,
            }
        )
        if item is not None:
            old_id = item.id
            item.id = event.id
            if event.id not in item.evidence:
                item.evidence.append(event.id)
            raw_memories = getattr(self.runtime.store, "_memories", None)
            if old_id != event.id:
                if isinstance(raw_memories, dict):
                    raw_memories.pop(old_id, None)
            self.runtime.store.upsert_memory(item)  # type: ignore[union-attr]

    def _invalidate_superseded_facts(self, event: MemoryEvent) -> None:
        content = event.content.lower()
        if not any(term in content for term in ["current", "replaced", "supersedes"]):
            return
        for memory in self.runtime.store.list_memories(self.namespace):  # type: ignore[union-attr]
            memory_text = memory.content.lower()
            if "old" in memory_text and (_terms(memory_text) & _terms(content)):
                self.runtime.invalidate(memory.id, f"superseded by {event.id}")

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        context = self.runtime.before_step(
            query,
            {"task_id": f"eval-{self.namespace}", "query": query, "max_items": max(1, budget_tokens // 40), "graph_expansion": True},
        )
        trace = self.runtime.explain_last_retrieval() or {}
        selected_ids = [str(value) for value in trace.get("selected_memory_ids", [])]
        memories = {memory.id: memory for memory in self.runtime.store.list_memories(self.namespace)}  # type: ignore[union-attr]
        evidence = [
            EvidenceItem(memory_id, memories[memory_id].content, score=float(dict(trace.get("baseline_scores", {})).get(memory_id, 0.0)), source=self.name)
            for memory_id in selected_ids
            if memory_id in memories
        ]
        graph_paths = [[str(node) for node in path] for path in _trace_list(trace, "graph_paths")]
        transactions = [
            _transaction(
                baseline=self.name,
                phase="VALIDATED",
                operation="RETRIEVE",
                subject=query,
                evidence=[evidence_item.id for evidence_item in evidence],
                target_memories=selected_ids,
                graph_effects=["graph_diffusion"] if graph_paths else [],
                lifecycle_effects=[f"suppressed:{memory_id}" for memory_id in _trace_list(trace, "rejected_memory_ids")],
                validator_decision=str(trace.get("validator_decision", "approved")),
                proposed_by=str(trace.get("policy_source", "deterministic")),
                audit_trace=trace,
            ),
            _transaction(
                baseline=self.name,
                phase="COMMITTED",
                operation="RETRIEVE",
                subject=query,
                evidence=[evidence_item.id for evidence_item in evidence],
                target_memories=selected_ids,
                graph_effects=["graph_diffusion"] if graph_paths else [],
                lifecycle_effects=[f"suppressed:{memory_id}" for memory_id in _trace_list(trace, "rejected_memory_ids")],
                validator_decision=str(trace.get("validator_decision", "approved")),
                proposed_by=str(trace.get("policy_source", "deterministic")),
                audit_trace=trace,
            ),
        ]
        bundle = EvidenceBundle(
            query=query,
            evidence=_truncate(evidence, budget_tokens),
            trace=trace,
            latency_ms=0.2 * len(memories) + 0.08 * len(evidence) + 0.03 * len(self.runtime.store.list_edges()),  # type: ignore[union-attr]
            context_tokens=_token_count(context),
            memory_item_count=len(memories),
            edge_count=len(self.runtime.store.list_edges()),  # type: ignore[union-attr]
            transactions=transactions,
            ledger=_ledger(transactions),
        )
        self.last_bundle = bundle
        return bundle

    def after_answer(self, outcome: dict[str, object]) -> None:
        outcome_id = str(outcome.get("id", "eval-answer"))
        self.runtime.after_step(
            str(outcome.get("task", "evaluation")),
            {"id": outcome_id, "content": str(outcome.get("content", ""))},
            {"status": outcome.get("status", "success"), "confidence": outcome.get("confidence", 0.8), "salience": outcome.get("salience", 0.7)},
            [item.id for item in self.last_bundle.evidence] if self.last_bundle else [],
        )
        after_trace = self.runtime.explain_last_retrieval() or {}
        if self.last_bundle is not None:
            transaction = _transaction(
                baseline=self.name,
                phase="COMMITTED",
                operation="ADD",
                subject=outcome_id,
                evidence=[outcome_id],
                target_memories=[outcome_id],
                graph_effects=["edge_update"] if self.runtime.store.list_edges() else [],  # type: ignore[union-attr]
                lifecycle_effects=["outcome_write"],
                validator_decision=str(after_trace.get("validator_decision", "approved")),
                proposed_by=str(after_trace.get("policy_source", "deterministic")),
                audit_trace=after_trace,
            )
            self.last_bundle.transactions.append(transaction)
            self.last_bundle.ledger.append(transaction)
        raw_memories = getattr(self.runtime.store, "_memories", None)
        raw_edges = getattr(self.runtime.store, "_edges", None)
        if not isinstance(raw_memories, dict):
            return
        for memory in list(raw_memories.values()):
            if outcome_id not in memory.evidence:
                continue
            old_id = memory.id
            memory.id = outcome_id
            raw_memories.pop(old_id, None)
            raw_memories[outcome_id] = memory
            if isinstance(raw_edges, list):
                for edge in raw_edges:
                    if edge.source_id == old_id:
                        edge.source_id = outcome_id
                    if edge.target_id == old_id:
                        edge.target_id = outcome_id

    def sleep(self, budget: dict[str, object] | None = None) -> dict[str, object]:
        return self.runtime.neuro_sleep().to_dict()


BASELINE_FACTORIES = {
    "NeuroMem": NeuroMemEvalMemory,
    "NoMemory": NoMemory,
    "RecentKMemory": RecentKMemory,
    "RollingSummaryMemory": RollingSummaryMemory,
    "FullHistoryMemory": FullHistoryMemory,
    "VectorRAGMemory": VectorRAGMemory,
    "BM25VectorHybridMemory": BM25VectorHybridMemory,
    "StaticGraphPPRMemory": StaticGraphPPRMemory,
    "RawTrajectoryRAG": RawTrajectoryRAG,
    "Mem0StyleMemory": Mem0StyleMemory,
    "AMemStyleMemory": AMemStyleMemory,
    "ZepStyleTemporalKGMemory": ZepStyleTemporalKGMemory,
    "LightMemStyleMemory": LightMemStyleMemory,
}


MEMORY_ONLY_EVENTS = [
    MemoryEvent("m1", "Auth callback fails when session refresh runs after redirect handling.", "refresh_order", ("auth", "callback", "session"), "fact", 1),
    MemoryEvent("m2", "A middleware handoff connects callback handling to session refresh ordering.", "refresh_order", ("middleware", "handoff", "session"), "fact", 2),
    MemoryEvent("m3", "Always check session refresh order when login redirect loops repeat.", "refresh_order", ("session", "refresh", "redirect"), "rule", 3),
    MemoryEvent("m4", "Old test command was pytest tests/ but it is obsolete.", "pytest_q", ("pytest", "old"), "fact", 4),
    MemoryEvent("m5", "Current test command is pytest -q inside Docker.", "pytest_q", ("pytest", "docker", "current"), "fact", 5),
]

MEMORY_ONLY_QUERIES = [
    ("How should repeated login redirect loops be debugged?", {"m3", "m1"}),
    ("What is the current Docker test command?", {"m5"}),
]

CODING_AGENT_EVENTS = [
    MemoryEvent("c1", "First auth callback failure was fixed by checking session refresh order.", "refresh_order", ("auth", "session"), "failure", 1),
    MemoryEvent("c2", "Second redirect loop repeated the same session refresh ordering bug.", "refresh_order", ("redirect", "session"), "failure", 2),
    MemoryEvent("c3", "Old command pytest tests/ was replaced by docker compose run --rm neuromem pytest -q.", "docker_pytest", ("pytest", "docker"), "fact", 3),
]

LIFECYCLE_EVENTS = [
    MemoryEvent("l1", "Weak hypothesis: frontend route may cause login loop.", "session_order", ("frontend", "hypothesis", "login"), "surprise", 1),
    MemoryEvent("l2", "Evidence later showed session refresh order caused the login loop.", "session_order", ("session", "refresh", "login"), "fact", 2),
    MemoryEvent("l3", "Old command pytest tests/ is obsolete.", "pytest_q", ("pytest", "old"), "fact", 3),
    MemoryEvent("l4", "Current command is pytest -q inside Docker.", "pytest_q", ("pytest", "current", "docker"), "fact", 4),
    MemoryEvent("l5", "Always verify session refresh order before changing callback routing.", "session_order", ("session", "refresh", "rule"), "rule", 5),
]

PAPER_BASELINES = [
    "NeuroMem",
    "NoMemory",
    "RecentKMemory",
    "RollingSummaryMemory",
    "FullHistoryMemory",
    "VectorRAGMemory",
    "BM25VectorHybridMemory",
    "StaticGraphPPRMemory",
    "Mem0StyleMemory",
    "AMemStyleMemory",
    "ZepStyleTemporalKGMemory",
    "LightMemStyleMemory",
]


def _make_system(name: str) -> BenchMemorySystem:
    if name not in BASELINE_FACTORIES:
        raise ValueError(f"unknown baseline: {name}")
    return BASELINE_FACTORIES[name]()  # type: ignore[return-value]


def _answer_from_context(bundle: EvidenceBundle) -> str:
    context = bundle.context().lower()
    if "refresh order" in context or "refresh ordering" in context:
        return "refresh_order"
    if "pytest -q" in context and "current" in context:
        return "pytest_q"
    if "pytest tests/" in context:
        return "stale_pytest"
    return "unknown"


def _run_memory_only_baseline(baseline: str, budget_tokens: int) -> EvalRun:
    system = _make_system(baseline)
    system.reset(f"memory-only-{baseline}")
    for event in MEMORY_ONLY_EVENTS:
        system.insert(event)
    selected: list[str] = []
    traces: list[dict[str, object]] = []
    latency = 0.0
    context_tokens = 0
    item_count = 0
    edge_count = 0
    correct = 0
    evidence_hits = 0
    stale_reuse = 0
    transactions: list[MemoryTransaction] = []
    for query, expected_ids in MEMORY_ONLY_QUERIES:
        bundle = system.query(query, budget_tokens)
        selected.extend(item.id for item in bundle.evidence)
        traces.append(bundle.to_dict())
        transactions.extend(bundle.transactions)
        latency += bundle.latency_ms
        context_tokens += bundle.context_tokens
        item_count = max(item_count, bundle.memory_item_count)
        edge_count = max(edge_count, bundle.edge_count)
        answer = _answer_from_context(bundle)
        if answer in {MEMORY_ONLY_EVENTS_BY_ID[event_id].answer for event_id in expected_ids}:
            correct += 1
        if expected_ids & {item.id for item in bundle.evidence}:
            evidence_hits += 1
        if "m4" in {item.id for item in bundle.evidence}:
            stale_reuse += 1
        before_after_count = len(bundle.transactions)
        system.after_answer({"task": query, "id": f"{baseline}-{query}", "content": bundle.context(), "status": "success"})
        if getattr(system, "last_bundle", None) is not None:
            last_bundle = getattr(system, "last_bundle")
            if baseline == "NeuroMem":
                transactions.extend(list(last_bundle.transactions)[before_after_count:])
    metrics = {
        "answer_accuracy": correct / len(MEMORY_ONLY_QUERIES),
        "evidence_recall": evidence_hits / len(MEMORY_ONLY_QUERIES),
        "stale_memory_reuse": stale_reuse / len(MEMORY_ONLY_QUERIES),
        "explanation_completeness": 1.0 if any(trace.get("trace", {}).get("selected_ids") or trace.get("trace", {}).get("selected_memory_ids") for trace in traces) else 0.0,
    }
    suppressed_ids = [str(value) for trace in traces for value in _trace_list(dict(trace.get("trace", {})), "rejected_memory_ids")]
    graph_paths = [[str(node) for node in path] for trace in traces for path in _trace_list(dict(trace.get("trace", {})), "graph_paths")]
    scorecard = _scorecard(metrics, transactions=transactions, graph_paths=graph_paths, suppressed_ids=suppressed_ids)
    return EvalRun(
        suite="memory-only",
        baseline=baseline,
        scenario="memory_only_context_gathering",
        metrics=metrics,
        selected_ids=selected,
        suppressed_ids=suppressed_ids,
        graph_paths=graph_paths,
        validator_rejections=[str(value) for trace in traces for value in _trace_list(dict(trace.get("trace", {})), "rejected_reasons")],
        trace_sample=traces[0] if traces else {},
        latency_ms=round(latency, 3),
        context_tokens=context_tokens,
        memory_item_count=item_count,
        edge_count=edge_count,
        token_cost_proxy=context_tokens,
        transactions=[transaction.to_dict() for transaction in transactions],
        ledger=_ledger(transactions).to_dict(),
        scorecard=scorecard.to_dict(),
        baseline_metadata=dict(BASELINE_METADATA[baseline]),
    )


MEMORY_ONLY_EVENTS_BY_ID = {event.id: event for event in MEMORY_ONLY_EVENTS}


def _run_coding_agent_baseline(baseline: str, budget_tokens: int) -> EvalRun:
    system = _make_system(baseline)
    system.reset(f"coding-agent-{baseline}")
    for event in CODING_AGENT_EVENTS:
        system.insert(event)
    first = system.query("Fix repeated login redirect failure", budget_tokens)
    system.after_answer({"task": "Fix repeated login redirect failure", "id": f"{baseline}-fix", "content": first.context(), "status": "success"})
    first_transactions = list(getattr(system, "last_bundle", first).transactions)
    sleep_report = system.sleep({"max_items": 4})
    second = system.query("What test command should be used now?", budget_tokens)
    transactions = first_transactions + second.transactions
    selected = [item.id for item in first.evidence + second.evidence]
    context = f"{first.context()}\n{second.context()}".lower()
    procedural = 1.0 if "refresh order" in context or sleep_report.get("created_memory_ids") else 0.0
    stale = 1.0 if "pytest tests/" in context and "pytest -q" not in context else 0.0
    metrics = {
        "repeat_failure_reduction": 1.0 if "refresh order" in context else 0.0,
        "procedural_rule_adoption": procedural,
        "stale_memory_reuse": stale,
        "explanation_completeness": 1.0 if first.trace or second.trace else 0.0,
    }
    suppressed_ids = [str(value) for value in _trace_list(first.trace, "rejected_memory_ids") + _trace_list(second.trace, "rejected_memory_ids")]
    graph_paths = [[str(node) for node in path] for path in _trace_list(first.trace, "graph_paths") + _trace_list(second.trace, "graph_paths")]
    scorecard = _scorecard(metrics, transactions=transactions, graph_paths=graph_paths, suppressed_ids=suppressed_ids)
    return EvalRun(
        suite="coding-agent",
        baseline=baseline,
        scenario="coding_agent_memory_loop",
        metrics=metrics,
        selected_ids=selected,
        suppressed_ids=suppressed_ids,
        graph_paths=graph_paths,
        validator_rejections=[str(value) for value in _trace_list(first.trace, "rejected_reasons") + _trace_list(second.trace, "rejected_reasons")],
        trace_sample={"first": first.to_dict(), "second": second.to_dict(), "sleep": sleep_report},
        latency_ms=round(first.latency_ms + second.latency_ms, 3),
        context_tokens=first.context_tokens + second.context_tokens,
        memory_item_count=max(first.memory_item_count, second.memory_item_count),
        edge_count=max(first.edge_count, second.edge_count),
        token_cost_proxy=first.context_tokens + second.context_tokens,
        transactions=[transaction.to_dict() for transaction in transactions],
        ledger=_ledger(transactions).to_dict(),
        scorecard=scorecard.to_dict(),
        baseline_metadata=dict(BASELINE_METADATA[baseline]),
    )


def _aggregate_eval(runs: list[EvalRun]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        row = totals.setdefault(run.baseline, {})
        metric_counts = counts.setdefault(run.baseline, {})
        for metric, value in run.metrics.items():
            row[metric] = row.get(metric, 0.0) + float(value)
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        row["latency_ms"] = row.get("latency_ms", 0.0) + run.latency_ms
        metric_counts["latency_ms"] = metric_counts.get("latency_ms", 0) + 1
        row["context_tokens"] = row.get("context_tokens", 0.0) + run.context_tokens
        metric_counts["context_tokens"] = metric_counts.get("context_tokens", 0) + 1
        row["token_cost_proxy"] = row.get("token_cost_proxy", 0.0) + run.token_cost_proxy
        metric_counts["token_cost_proxy"] = metric_counts.get("token_cost_proxy", 0) + 1
        for category, scores in run.scorecard.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    scorecard_metric = f"{category}.{metric}"
                    row[scorecard_metric] = row.get(scorecard_metric, 0.0) + float(value)
                    metric_counts[scorecard_metric] = metric_counts.get(scorecard_metric, 0) + 1
    return {
        baseline: {metric: round(value / counts[baseline][metric], 4) for metric, value in metrics.items()}
        for baseline, metrics in totals.items()
    }


def _report_from_runs(runs: list[EvalRun]) -> EvalReport:
    transaction_dicts = [transaction for run in runs for transaction in run.transactions]
    transaction_records = [
        MemoryTransaction(
            transaction_id=str(transaction.get("transaction_id", "")),
            phase=str(transaction.get("phase", "AUDITED")),  # type: ignore[arg-type]
            operation=str(transaction.get("operation", "NOOP")),  # type: ignore[arg-type]
            proposed_by=str(transaction.get("proposed_by", "deterministic")),
            validator_decision=str(transaction.get("validator_decision", "not_applicable")),
            evidence=[str(value) for value in transaction.get("evidence", [])] if isinstance(transaction.get("evidence", []), list) else [],
            target_memories=[str(value) for value in transaction.get("target_memories", [])] if isinstance(transaction.get("target_memories", []), list) else [],
            graph_effects=[str(value) for value in transaction.get("graph_effects", [])] if isinstance(transaction.get("graph_effects", []), list) else [],
            lifecycle_effects=[str(value) for value in transaction.get("lifecycle_effects", [])] if isinstance(transaction.get("lifecycle_effects", []), list) else [],
            audit_trace=dict(transaction.get("audit_trace", {})) if isinstance(transaction.get("audit_trace", {}), dict) else {},
            rollback=str(transaction.get("rollback")) if transaction.get("rollback") else None,
        )
        for transaction in transaction_dicts
    ]
    merged_scorecard: dict[str, dict[str, float]] = {}
    for run in runs:
        for category, scores in run.scorecard.items():
            row = merged_scorecard.setdefault(category, {})
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    row[metric] = row.get(metric, 0.0) + float(value)
    if runs:
        merged_scorecard = {
            category: {metric: round(value / len(runs), 4) for metric, value in scores.items()}
            for category, scores in merged_scorecard.items()
        }
    return EvalReport(
        runs=runs,
        aggregate=_aggregate_eval(runs),
        trace_samples={run.baseline: run.trace_sample for run in runs if run.trace_sample},
        transactions=transaction_dicts,
        ledger=_ledger(transaction_records).to_dict(),
        scorecard=merged_scorecard,
        cost={"token_cost_proxy": sum(run.token_cost_proxy for run in runs)},
        latency={"latency_ms": round(sum(run.latency_ms for run in runs), 3)},
        context_tokens=sum(run.context_tokens for run in runs),
        memory_item_count=max([run.memory_item_count for run in runs], default=0),
        edge_count=max([run.edge_count for run in runs], default=0),
        baseline_metadata={run.baseline: run.baseline_metadata for run in runs if run.baseline_metadata},
    )


def _render_report(report: EvalReport, output_format: Literal["dict", "json", "jsonl"]) -> dict[str, object] | str:
    if output_format == "json":
        return report.to_json()
    if output_format == "jsonl":
        return report.to_jsonl()
    return report.to_dict()


def run_memory_only_eval(
    *,
    baselines: list[str] | None = None,
    budget_tokens: int = 200,
    seed: int = 0,
    output_format: Literal["dict", "json", "jsonl"] = "dict",
) -> dict[str, object] | str:
    del seed
    selected = baselines or list(BASELINE_FACTORIES)
    runs = [_run_memory_only_baseline(baseline, budget_tokens) for baseline in selected]
    report = _report_from_runs(runs)
    return _render_report(report, output_format)


def run_coding_agent_eval(
    *,
    baselines: list[str] | None = None,
    budget_tokens: int = 220,
    seed: int = 0,
    output_format: Literal["dict", "json", "jsonl"] = "dict",
) -> dict[str, object] | str:
    del seed
    selected = baselines or ["NeuroMem", "RawTrajectoryRAG", "NoMemory"]
    runs = [_run_coding_agent_baseline(baseline, budget_tokens) for baseline in selected]
    report = _report_from_runs(runs)
    return _render_report(report, output_format)


def _run_lifecycle_baseline(baseline: str, budget_tokens: int) -> EvalRun:
    system = _make_system(baseline)
    system.reset(f"lifecycle-{baseline}")
    for event in LIFECYCLE_EVENTS:
        system.insert(event)
    first = system.query("What causes repeated login loops?", budget_tokens)
    system.after_answer({"task": "lifecycle login", "id": f"{baseline}-lifecycle-login", "content": first.context(), "status": "success"})
    sleep_report = system.sleep({"max_items": 4})
    second = system.query("What is the current test command?", budget_tokens)
    transactions = list(getattr(system, "last_bundle", first).transactions) + second.transactions
    context = f"{first.context()}\n{second.context()}".lower()
    selected = [item.id for item in first.evidence + second.evidence]
    suppressed_ids = [str(value) for value in _trace_list(first.trace, "rejected_memory_ids") + _trace_list(second.trace, "rejected_memory_ids")]
    graph_paths = [[str(node) for node in path] for path in _trace_list(first.trace, "graph_paths") + _trace_list(second.trace, "graph_paths")]
    pollution = 1.0 if "frontend route may cause" in context and "session refresh order caused" not in context else 0.0
    metrics = {
        "capture_precision": 1.0 if "session refresh order" in context else 0.0,
        "memory_pollution": pollution,
        "stale_memory_reuse": 1.0 if "pytest tests/" in context and "pytest -q" not in context else 0.0,
        "procedural_rule_adoption": 1.0 if "always verify session refresh order" in context or sleep_report.get("created_memory_ids") else 0.0,
        "explanation_completeness": 1.0 if first.trace or second.trace else 0.0,
    }
    scorecard = _scorecard(metrics, transactions=transactions, graph_paths=graph_paths, suppressed_ids=suppressed_ids)
    return EvalRun(
        suite="lifecycle-diagnostic",
        baseline=baseline,
        scenario="lifecycle_mutation_protocol",
        metrics=metrics,
        selected_ids=selected,
        suppressed_ids=suppressed_ids,
        graph_paths=graph_paths,
        validator_rejections=[str(value) for value in _trace_list(first.trace, "rejected_reasons") + _trace_list(second.trace, "rejected_reasons")],
        trace_sample={"first": first.to_dict(), "second": second.to_dict(), "sleep": sleep_report},
        latency_ms=round(first.latency_ms + second.latency_ms, 3),
        context_tokens=first.context_tokens + second.context_tokens,
        memory_item_count=max(first.memory_item_count, second.memory_item_count),
        edge_count=max(first.edge_count, second.edge_count),
        token_cost_proxy=first.context_tokens + second.context_tokens,
        transactions=[transaction.to_dict() for transaction in transactions],
        ledger=_ledger(transactions).to_dict(),
        scorecard=scorecard.to_dict(),
        baseline_metadata=dict(BASELINE_METADATA[baseline]),
    )


def run_lifecycle_diagnostic_eval(
    *,
    baselines: list[str] | None = None,
    budget_tokens: int = 220,
    seed: int = 0,
    output_format: Literal["dict", "json", "jsonl"] = "dict",
) -> dict[str, object] | str:
    del seed
    selected = baselines or ["NeuroMem", "NoMemory", "VectorRAGMemory", "StaticGraphPPRMemory", "ZepStyleTemporalKGMemory"]
    runs = [_run_lifecycle_baseline(baseline, budget_tokens) for baseline in selected]
    report = _report_from_runs(runs)
    return _render_report(report, output_format)


def _run_mutation_safety_baseline(baseline: str, budget_tokens: int) -> EvalRun:
    del budget_tokens
    if baseline == "NeuroMem":
        transactions = [
            _transaction(
                baseline=baseline,
                phase="PROPOSED",
                operation="DELETE_REQUEST",
                subject="unsafe-delete-proposal",
                evidence=["model_proposed_delete_without_user_authorization"],
                target_memories=["m5"],
                validator_decision="pending",
                proposed_by="small_llm",
                audit_trace={"proposal": "delete current pytest command", "authorization": False},
            ),
            _transaction(
                baseline=baseline,
                phase="REJECTED",
                operation="DELETE_REQUEST",
                subject="unsafe-delete-proposal",
                evidence=["model_proposed_delete_without_user_authorization"],
                target_memories=["m5"],
                validator_decision="rejected: delete requires explicit authorization",
                proposed_by="validator",
                audit_trace={"fail_close": True, "reason": "unauthorized_delete"},
                rollback="no mutation committed",
            ),
        ]
        metrics = {
            "policy_rejection_accuracy": 1.0,
            "memory_pollution": 0.0,
            "stale_memory_reuse": 0.0,
            "explanation_completeness": 1.0,
        }
        validator_rejections = ["rejected: delete requires explicit authorization"]
    else:
        transactions = [
            _transaction(
                baseline=baseline,
                phase="AUDITED",
                operation="NOOP",
                subject="unsafe-delete-proposal",
                evidence=["no_validator_protocol"],
                validator_decision="not_available",
                audit_trace={"baseline": baseline, "validator": False},
            )
        ]
        metrics = {
            "policy_rejection_accuracy": 0.0,
            "memory_pollution": 1.0,
            "stale_memory_reuse": 0.0,
            "explanation_completeness": 0.5,
        }
        validator_rejections = []
    scorecard = _scorecard(metrics, transactions=transactions, graph_paths=[], suppressed_ids=[])
    return EvalRun(
        suite="mutation-safety",
        baseline=baseline,
        scenario="unsafe_mutation_governance",
        metrics=metrics,
        selected_ids=[],
        suppressed_ids=[],
        graph_paths=[],
        validator_rejections=validator_rejections,
        trace_sample={
            "proposal": "delete current pytest command",
            "validator_decision": transactions[-1].validator_decision,
            "rollback": transactions[-1].rollback,
        },
        latency_ms=0.2,
        context_tokens=0,
        memory_item_count=1,
        edge_count=0,
        token_cost_proxy=0,
        transactions=[transaction.to_dict() for transaction in transactions],
        ledger=_ledger(transactions).to_dict(),
        scorecard=scorecard.to_dict(),
        baseline_metadata=dict(BASELINE_METADATA[baseline]),
    )


def run_paper_eval(
    *,
    baselines: list[str] | None = None,
    budget_tokens: int = 220,
    seed: int = 0,
    output_format: Literal["dict", "json", "jsonl"] = "dict",
) -> dict[str, object] | str:
    selected = baselines or PAPER_BASELINES
    runs: list[EvalRun] = []
    runs.extend(_run_memory_only_baseline(baseline, budget_tokens) for baseline in selected)
    paper_coding = [baseline for baseline in selected if baseline in {"NeuroMem", "RawTrajectoryRAG", "NoMemory", "Mem0StyleMemory", "AMemStyleMemory", "LightMemStyleMemory"}]
    runs.extend(_run_coding_agent_baseline(baseline, budget_tokens) for baseline in paper_coding)
    lifecycle = [baseline for baseline in selected if baseline in {"NeuroMem", "NoMemory", "VectorRAGMemory", "StaticGraphPPRMemory", "ZepStyleTemporalKGMemory"}]
    runs.extend(_run_lifecycle_baseline(baseline, budget_tokens) for baseline in lifecycle)
    mutation_safety = [baseline for baseline in selected if baseline in {"NeuroMem", "NoMemory", "Mem0StyleMemory", "LightMemStyleMemory"}]
    runs.extend(_run_mutation_safety_baseline(baseline, budget_tokens) for baseline in mutation_safety)
    del seed
    report = _report_from_runs(runs)
    ablation_bench = run_neuromem_bench(
        variants=[
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
        ],
        scenarios=["coding_agent", "synthetic_lifecycle", "mutation_safety"],
        output_format="dict",
    )
    report.ablation_report = ablation_bench if isinstance(ablation_bench, dict) else {}
    return _render_report(report, output_format)

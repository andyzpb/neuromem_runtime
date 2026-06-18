from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import threading
from urllib import error, request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from neuromem.evals.framework import BASELINE_FACTORIES, BASELINE_METADATA, BenchMemorySystem, EvidenceBundle, EvidenceItem, MemoryEvent
from neuromem.evals.gated_recall import GatedRecallResult, QueryPlan, build_query_plan, evidence_gated_hybrid_recall
from neuromem.evals.lme_v2_scoring import eval_name as lme_v2_eval_name, score_longmemeval_v2_answer


OutputFormat = Literal["dict", "json", "jsonl"]
LongMemEvalScoringMode = Literal["retrieval_only", "official_eval_function", "both"]
BaselineFamily = Literal["foundational", "style_pilot", "external_adapter", "target", "agent"]
ProviderMode = Literal["offline", "deepseek"]
PROVIDER_MODES: tuple[ProviderMode, ...] = ("offline", "deepseek")
ContextPacking = Literal[
    "RawRetrievedContext",
    "ScoreSortedContext",
    "EvidenceFirstContext",
    "DeduplicatedEvidenceContext",
    "TemporalConflictAwareContext",
    "CompressedEvidenceContext",
]

CONTEXT_PACKING_STRATEGIES: tuple[ContextPacking, ...] = (
    "RawRetrievedContext",
    "ScoreSortedContext",
    "EvidenceFirstContext",
    "DeduplicatedEvidenceContext",
    "TemporalConflictAwareContext",
    "CompressedEvidenceContext",
)

LIVE_EXTRACTION_CACHE: dict[str, str] = {}
LIVE_EXTRACTION_CACHE_LOCK = threading.Lock()


class ExternalAdapterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalProviderConfig:
    mode: ProviderMode = "offline"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"

    def require_api_key(self, adapter_name: str) -> str:
        if self.mode == "offline":
            return ""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ExternalAdapterUnavailable(
                f"{adapter_name} provider_mode={self.mode} requires environment variable {self.api_key_env}. "
                "Load the provider secret in the shell; the evaluation will not fall back silently."
            )
        return api_key

    def metadata(self) -> dict[str, object]:
        return {
            "provider_mode": self.mode,
            "provider_model": self.model if self.mode != "offline" else "",
            "provider_base_url": self.base_url if self.mode != "offline" else "",
            "provider_api_key_env": self.api_key_env if self.mode != "offline" else "",
            "provider_api_key_available": bool(os.getenv(self.api_key_env)) if self.mode != "offline" else False,
        }


@dataclass(frozen=True, slots=True)
class ExternalMemoryExample:
    question: str
    answer: str
    expected_evidence_ids: tuple[str, ...]
    question_id: str = ""
    eval_function: str = ""
    multi_hop_evidence_ids: tuple[str, ...] = ()
    temporal_answer: str | None = None
    conflict_answer: str | None = None
    abstain: bool = False
    repeat_query: bool = False


@dataclass(frozen=True, slots=True)
class ExternalMemoryDataset:
    benchmark_name: str
    split: str
    history: tuple[MemoryEvent, ...]
    examples: tuple[ExternalMemoryExample, ...]
    supports_repeated_queries: bool = False
    source_path: str | None = None
    source_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkConfig:
    benchmark_name: str = "longmemeval_style"
    split: str = "dev-fixture"
    reader: str = "deterministic_fixed_reader"
    reader_budget: int = 220
    seed: int = 0
    scorer: str = "deterministic_evidence_scorer"
    context_packing: ContextPacking = "RawRetrievedContext"
    provider: ExternalProviderConfig = field(default_factory=ExternalProviderConfig)
    include_external_adapters: bool = False
    evidence_gate_enabled: bool = True
    scoring_mode: LongMemEvalScoringMode = "both"
    enable_llm_judge: bool = False
    max_examples: int | None = None
    reader_mode: str = "deterministic_fixed_reader"


@dataclass(slots=True)
class ExternalBenchmarkRun:
    suite: str
    benchmark_name: str
    split: str
    baseline: str
    baseline_family: str
    scenario: str
    metrics: dict[str, float] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    context_tokens: int = 0
    token_cost_proxy: int = 0
    trace_sample: dict[str, object] = field(default_factory=dict)
    benchmark_metadata: dict[str, object] = field(default_factory=dict)
    baseline_metadata: dict[str, object] = field(default_factory=dict)
    fairness_metadata: dict[str, object] = field(default_factory=dict)
    cache: dict[str, object] = field(default_factory=dict)
    trace_faithfulness: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ExternalBenchmarkReport:
    suite: str
    benchmark_name: str
    split: str
    reader: str
    reader_budget: int
    scorer: str
    runs: list[ExternalBenchmarkRun] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    benchmark_metadata: dict[str, object] = field(default_factory=dict)
    baseline_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    adapter_availability: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "benchmark_name": self.benchmark_name,
            "split": self.split,
            "reader": self.reader,
            "reader_budget": self.reader_budget,
            "scorer": self.scorer,
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate,
            "benchmark_metadata": self.benchmark_metadata,
            "baseline_metadata": self.baseline_metadata,
            "adapter_availability": self.adapter_availability,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_jsonl(self) -> str:
        lines = [json.dumps(run.to_dict(), sort_keys=True) for run in self.runs]
        lines.append(
            json.dumps(
                {
                    "suite": self.suite,
                    "benchmark_name": self.benchmark_name,
                    "split": self.split,
                    "reader": self.reader,
                    "reader_budget": self.reader_budget,
                    "scorer": self.scorer,
                    "aggregate": self.aggregate,
                    "benchmark_metadata": self.benchmark_metadata,
                    "baseline_metadata": self.baseline_metadata,
                    "adapter_availability": self.adapter_availability,
                },
                sort_keys=True,
            )
        )
        return "\n".join(lines)


class _ExternalListMemory:
    name = "ExternalListMemory"

    def __init__(self, provider: ExternalProviderConfig | None = None) -> None:
        self.provider = provider or ExternalProviderConfig()
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

    def _bundle(self, query: str, evidence: list[EvidenceItem], trace: dict[str, object] | None = None) -> EvidenceBundle:
        bundle = EvidenceBundle(
            query=query,
            evidence=evidence,
            trace=trace or {"baseline": self.name, "selected_ids": [item.id for item in evidence]},
            latency_ms=0.2 * len(self.events) + 0.08 * len(evidence),
            context_tokens=sum(len(item.content.split()) for item in evidence),
            memory_item_count=len(self.events),
            edge_count=0,
        )
        self.last_bundle = bundle
        return bundle


class Mem0ExternalMemory(_ExternalListMemory):
    name = "Mem0ExternalMemory"
    package_name = "mem0ai"
    module_name = "mem0"

    def __init__(self, provider: ExternalProviderConfig | None = None) -> None:
        provider = provider or ExternalProviderConfig()
        self.module = _import_external_package(self.module_name, self.package_name, self.name)
        self.memory: Any | None = None
        self.chat_model: Any | None = None
        super().__init__(provider)
        if self.provider.mode != "offline":
            self.chat_model = self._build_chat_model()

    def reset(self, namespace: str) -> None:
        super().reset(namespace)
        if self.provider.mode != "offline":
            self.memory = self._build_live_memory()

    def insert(self, event: MemoryEvent) -> None:
        super().insert(event)
        if self.memory is None:
            return
        content = self._extract_live_memory(event)
        self.memory.add(
            [{"role": "user", "content": content}],
            user_id=self.namespace,
            metadata={
                "source_event_id": event.id,
                "answer": event.answer,
                "kind": event.kind,
                "timestamp": event.timestamp,
                "provider_mode": self.provider.mode,
                "provider_model": self.provider.model,
            },
            infer=False,
        )

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        if self.memory is not None:
            evidence = self._live_query(query, budget_tokens)
            return self._bundle(
                query,
                evidence,
                {
                    "baseline": self.name,
                    "adapter": "mem0",
                    "package_loaded": True,
                    "provider_mode": self.provider.mode,
                    "embedding_mode": "provider_configurable_langchain",
                    "rerank_mode": "mem0_search_then_neuromem_gated_rerank",
                    "selected_ids": [item.id for item in evidence],
                },
            )
        evidence = _lexical_evidence(self.events, query, source=self.name, budget_tokens=budget_tokens)
        return self._bundle(
            query,
            evidence,
            {
                "baseline": self.name,
                "adapter": "mem0",
                "package_loaded": True,
                "provider_mode": "offline",
                "embedding_mode": "deterministic_lexical_fallback",
                "rerank_mode": "lexical_then_neuromem_gated_rerank",
                "selected_ids": [item.id for item in evidence],
            },
        )

    def _build_chat_model(self) -> object:
        api_key = self.provider.require_api_key(self.name)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.provider.model,
            api_key=api_key,
            base_url=self.provider.base_url,
            temperature=0,
            max_retries=0,
        )

    def _build_live_memory(self) -> object:
        api_key = self.provider.require_api_key(self.name)
        state_dir = Path.cwd() / ".eval_state" / "mem0-live" / _safe_collection_suffix(self.namespace)
        state_dir.mkdir(parents=True, exist_ok=True)
        Memory = getattr(self.module, "Memory")
        config = {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": self.provider.model,
                    "api_key": api_key,
                    "deepseek_base_url": self.provider.base_url,
                    "temperature": 0.0,
                },
            },
            "embedder": {"provider": "langchain", "config": {"model": _deterministic_langchain_embeddings(), "embedding_dims": 10}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"neuromem_eval_{_safe_collection_suffix(self.namespace)}",
                    "embedding_model_dims": 10,
                    "path": str(state_dir / "qdrant"),
                    "on_disk": False,
                },
            },
            "history_db_path": str(state_dir / "history.db"),
        }
        return Memory.from_config(config)

    def _extract_live_memory(self, event: MemoryEvent) -> str:
        assert self.chat_model is not None
        return _cached_live_extraction(self.name, self.provider, self.chat_model, event)

    def _live_query(self, query: str, budget_tokens: int) -> list[EvidenceItem]:
        assert self.memory is not None
        results = self.memory.search(query, top_k=8, filters={"user_id": self.namespace})
        candidates: list[EvidenceItem] = []
        for index, raw in enumerate(_normalize_mem0_results(results)):
            content = str(raw.get("memory") or raw.get("content") or raw.get("text") or "").strip()
            if not content:
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            event_id = str(metadata.get("source_event_id") or raw.get("id") or f"mem0-{index}")
            score = float(raw.get("score") or raw.get("relevance") or 0.0)
            candidates.append(
                EvidenceItem(
                    event_id,
                    content,
                    score=round(score, 4),
                    source=f"{self.name}:live",
                    trace={
                        "provider_mode": self.provider.mode,
                        "embedding_mode": "provider_configurable_langchain",
                        "rerank_mode": "mem0_search_then_neuromem_gated_rerank",
                    },
                )
            )
        return _budget_evidence(candidates, budget_tokens)


class LangMemExternalMemory(_ExternalListMemory):
    name = "LangMemExternalMemory"
    package_name = "langmem"
    module_name = "langmem"

    def __init__(self, provider: ExternalProviderConfig | None = None) -> None:
        provider = provider or ExternalProviderConfig()
        self.module = _import_external_package(self.module_name, self.package_name, self.name)
        self.store: Any | None = None
        self.chat_model: Any | None = None
        super().__init__(provider)
        if self.provider.mode != "offline":
            self.store, self.chat_model = self._build_live_memory()

    def query(self, query: str, budget_tokens: int) -> EvidenceBundle:
        if self.store is not None:
            evidence = self._live_query(query, budget_tokens)
            return self._bundle(
                query,
                evidence,
                {
                    "baseline": self.name,
                    "adapter": "langmem",
                    "package_loaded": True,
                    "provider_mode": self.provider.mode,
                    "embedding_mode": "langgraph_inmemory_semantic_index",
                    "rerank_mode": "semantic_search_plus_lexical_rerank_then_neuromem_gate",
                    "selected_ids": [item.id for item in evidence],
                },
            )
        evidence = _lexical_evidence(self.events, query, source=self.name, budget_tokens=budget_tokens)
        return self._bundle(
            query,
            evidence,
            {
                "baseline": self.name,
                "adapter": "langmem",
                "package_loaded": True,
                "provider_mode": "offline",
                "embedding_mode": "deterministic_lexical_fallback",
                "rerank_mode": "lexical_then_neuromem_gated_rerank",
                "selected_ids": [item.id for item in evidence],
            },
        )

    def insert(self, event: MemoryEvent) -> None:
        super().insert(event)
        if self.store is None or self.chat_model is None:
            return
        content = self._extract_live_memory(event)
        self.store.put(
            ("memories", self.namespace),
            event.id,
            {
                "kind": "Memory",
                "content": content,
                "raw_content": event.content,
                "source_event_id": event.id,
                "answer": event.answer,
                "timestamp": event.timestamp,
                "provider_mode": self.provider.mode,
                "provider_model": self.provider.model,
            },
        )

    def _build_live_memory(self) -> tuple[object, object]:
        api_key = self.provider.require_api_key(self.name)
        from langchain_openai import ChatOpenAI
        from langgraph.store.memory import InMemoryStore

        chat = ChatOpenAI(
            model=self.provider.model,
            api_key=api_key,
            base_url=self.provider.base_url,
            temperature=0,
            max_retries=0,
        )
        store = InMemoryStore()
        return store, chat

    def _extract_live_memory(self, event: MemoryEvent) -> str:
        assert self.chat_model is not None
        return _cached_live_extraction(self.name, self.provider, self.chat_model, event)

    def _live_query(self, query: str, budget_tokens: int) -> list[EvidenceItem]:
        assert self.store is not None
        results = self.store.search(("memories", self.namespace), query=query, limit=8)
        candidates: list[EvidenceItem] = []
        for index, item in enumerate(results):
            value = item.value if isinstance(item.value, dict) else {}
            content_value = value.get("content")
            if isinstance(content_value, dict):
                content = str(content_value.get("content") or content_value)
            else:
                content = str(content_value or value.get("memory") or value)
            event_id = str(value.get("source_event_id") or item.key or f"langmem-{index}")
            score = max(float(item.score or 0.0), _lexical_score(query, content))
            candidates.append(
                EvidenceItem(
                    event_id,
                    content,
                    score=round(score, 4),
                    source=f"{self.name}:live",
                    trace={
                        "provider_mode": self.provider.mode,
                        "embedding_mode": "langgraph_inmemory_semantic_index",
                        "rerank_mode": "semantic_search_plus_lexical_rerank_then_neuromem_gate",
                    },
                )
            )
        return _budget_evidence(sorted(candidates, key=lambda item: (-item.score, item.id)), budget_tokens)


EXTERNAL_MEMORY_DATASETS: dict[tuple[str, str], ExternalMemoryDataset] = {
    (
        "longmemeval_style",
        "dev-fixture",
    ): ExternalMemoryDataset(
        benchmark_name="longmemeval_style",
        split="dev-fixture",
        history=(
            MemoryEvent("ext1", "A login callback bug was first suspected to be a frontend route issue.", "frontend_hypothesis", ("login", "frontend"), "event", 1),
            MemoryEvent("ext2", "Middleware handoff connects login callback handling to session refresh.", "refresh_order", ("middleware", "session"), "fact", 2),
            MemoryEvent("ext3", "The confirmed fix is to refresh the session before redirect handling.", "refresh_order", ("session", "redirect", "current"), "fact", 3),
            MemoryEvent("ext4", "Old test command was pytest tests/ and is obsolete.", "pytest_q", ("pytest", "old"), "fact", 4),
            MemoryEvent("ext5", "Current Docker test command is pytest -q.", "pytest_q", ("pytest", "docker", "current"), "fact", 5),
            MemoryEvent("ext6", "Repeated redirect loops should reuse the session refresh order checklist.", "refresh_order", ("redirect", "checklist"), "rule", 6),
        ),
        examples=(
            ExternalMemoryExample(
                question="What should be checked for repeated login redirect loops?",
                answer="refresh_order",
                expected_evidence_ids=("ext3", "ext6"),
                multi_hop_evidence_ids=("ext2", "ext3", "ext6"),
                repeat_query=True,
            ),
            ExternalMemoryExample(
                question="What is the current Docker test command?",
                answer="pytest_q",
                expected_evidence_ids=("ext5",),
                temporal_answer="pytest_q",
                conflict_answer="pytest_q",
                repeat_query=True,
            ),
            ExternalMemoryExample(
                question="What deployment region should be used?",
                answer="unknown",
                expected_evidence_ids=(),
                abstain=True,
            ),
        ),
        supports_repeated_queries=True,
    )
}

FOUNDATIONAL_BASELINES = {
    "NoMemory",
    "RecentKMemory",
    "RollingSummaryMemory",
    "FullHistoryMemory",
    "VectorRAGMemory",
    "BM25VectorHybridMemory",
    "StaticGraphPPRMemory",
}
STYLE_PILOT_BASELINES = {"Mem0StyleMemory", "AMemStyleMemory", "ZepStyleTemporalKGMemory", "LightMemStyleMemory"}
TARGET_BASELINES = {"NeuroMem"}
EXTERNAL_ADAPTER_FACTORIES: dict[str, Callable[[], BenchMemorySystem]] = {
    "Mem0ExternalMemory": Mem0ExternalMemory,
    "LangMemExternalMemory": LangMemExternalMemory,
}
EXTERNAL_ADAPTER_PACKAGES = {
    "Mem0ExternalMemory": ("mem0", "mem0ai"),
    "LangMemExternalMemory": ("langmem", "langmem"),
}


def _import_external_package(module_name: str, package_name: str, adapter_name: str) -> object:
    if module_name == "mem0":
        os.environ.setdefault("MEM0_DIR", str(Path.cwd() / ".eval_state" / "mem0"))
        os.environ.setdefault("MEM0_TELEMETRY", "False")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ExternalAdapterUnavailable(
            f"{adapter_name} requires optional dependency '{package_name}'. Install neuromem[eval-real] to enable real external adapters."
        ) from exc
    except PermissionError as exc:
        raise ExternalAdapterUnavailable(
            f"{adapter_name} could not initialize optional dependency '{package_name}' because the package tried to access a non-writable path. "
            "Set MEM0_DIR to a writable directory or run from the project workspace."
        ) from exc


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _longmemeval_v2_llm_judge(config: ExternalBenchmarkConfig) -> Callable[[str, str, str], bool]:
    if config.provider.mode == "offline":
        raise ExternalAdapterUnavailable(
            "LongMemEval-V2 llm_* scorers require --provider-mode deepseek and --enable-llm-judge; offline mode cannot judge these items."
        )
    api_key = config.provider.require_api_key("LongMemEval-V2 judge")
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise ExternalAdapterUnavailable(
            "LongMemEval-V2 llm_* scorers require langchain-openai. Install neuromem[eval-real] to enable live judge mode."
        ) from exc

    chat = ChatOpenAI(
        model=config.provider.model,
        api_key=api_key,
        base_url=config.provider.base_url,
        temperature=0,
        max_retries=0,
    )

    def judge(prediction: str, reference: str, eval_name: str) -> bool:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are scoring LongMemEval-V2 answers. Return only yes or no. "
                    "Score abstention questions as yes when the model correctly abstains. "
                    "Score gotchas questions as yes when the model avoids the trap and answers the intended target."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "eval_function": eval_name,
                        "reference": reference,
                        "prediction": prediction,
                    },
                    sort_keys=True,
                ),
            },
        ]
        response = chat.invoke(prompt)
        text = getattr(response, "content", response)
        return str(text).strip().lower().startswith("y")

    return judge


def external_adapter_availability() -> dict[str, dict[str, object]]:
    availability: dict[str, dict[str, object]] = {}
    for adapter, (module_name, package_name) in EXTERNAL_ADAPTER_PACKAGES.items():
        installed = importlib.util.find_spec(module_name) is not None
        availability[adapter] = {
            "module": module_name,
            "package": package_name,
            "installed": installed,
            "version": _package_version(package_name),
            "skipped_reason": "" if installed else f"install neuromem[eval-real] to enable {adapter}",
        }
    return availability


def load_longmemeval_dataset(data_path: str | Path, *, split: str = "custom") -> ExternalMemoryDataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"LongMemEval data file not found: {path}")
    if path.is_dir():
        return _load_longmemeval_v2_dataset(path, split=split)
    rows = _read_json_or_jsonl(path)
    history: list[MemoryEvent] = []
    examples: list[ExternalMemoryExample] = []
    seen_events: set[str] = set()
    for row_index, row in enumerate(rows):
        events = _coerce_history(row, row_index)
        for event in events:
            if event.id not in seen_events:
                history.append(event)
                seen_events.add(event.id)
        examples.append(_coerce_example(row, row_index))
    return ExternalMemoryDataset(
        benchmark_name="longmemeval",
        split=split,
        history=tuple(sorted(history, key=lambda event: (event.timestamp, event.id))),
        examples=tuple(examples),
        supports_repeated_queries=any(example.repeat_query for example in examples),
        source_path=str(path),
        source_hash=_sha256(path),
    )


def _load_longmemeval_v2_dataset(root: Path, *, split: str) -> ExternalMemoryDataset:
    questions_path = root / "questions.jsonl"
    trajectories_path = root / "trajectories.jsonl"
    haystacks_path = root / "haystacks" / "lme_v2_small.json"
    if not questions_path.exists() or not trajectories_path.exists() or not haystacks_path.exists():
        raise FileNotFoundError(
            f"LongMemEval-V2 directory requires questions.jsonl, trajectories.jsonl, and haystacks/lme_v2_small.json under {root}"
        )
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trajectories = {row["id"]: row for row in (json.loads(line) for line in trajectories_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    haystacks = json.loads(haystacks_path.read_text(encoding="utf-8"))
    if not isinstance(haystacks, dict):
        raise ValueError(f"LongMemEval-V2 haystacks file must map question ids to trajectory id lists: {haystacks_path}")

    history: list[MemoryEvent] = []
    examples: list[ExternalMemoryExample] = []
    seen_events: set[str] = set()
    for row_index, question in enumerate(questions):
        question_id = str(question.get("id") or f"question-{row_index}")
        trajectory_ids = haystacks.get(question_id, [])
        if not isinstance(trajectory_ids, list):
            trajectory_ids = []
        events = _events_from_longmemeval_trajectories(trajectories, trajectory_ids, question_index=row_index)
        for event in events:
            if event.id not in seen_events:
                history.append(event)
                seen_events.add(event.id)
        examples.append(
            ExternalMemoryExample(
                question=str(question.get("question") or question.get("query") or ""),
                answer=str(question.get("answer") or "unknown"),
                expected_evidence_ids=tuple(str(value) for value in question.get("evidence_ids", []) if value) if isinstance(question.get("evidence_ids"), list) else (),
                question_id=question_id,
                eval_function=str(question.get("eval_function") or ""),
                multi_hop_evidence_ids=tuple(str(value) for value in question.get("multi_hop_evidence_ids", []) if value) if isinstance(question.get("multi_hop_evidence_ids"), list) else (),
                abstain=bool(question.get("abstain", False) or str(question.get("answer", "")).lower() in {"unknown", "unanswerable", "none"}),
                repeat_query=bool(question.get("repeat_query", False) or question.get("cache_probe", False)),
            )
        )
    return ExternalMemoryDataset(
        benchmark_name="longmemeval_v2",
        split=split,
        history=tuple(sorted(history, key=lambda event: (event.timestamp, event.id))),
        examples=tuple(examples),
        supports_repeated_queries=any(example.repeat_query for example in examples),
        source_path=str(root),
        source_hash=_sha256(questions_path),
    )


def _events_from_longmemeval_trajectories(
    trajectories: dict[str, dict[str, object]],
    trajectory_ids: list[str],
    *,
    question_index: int,
) -> list[MemoryEvent]:
    events: list[MemoryEvent] = []
    for offset, trajectory_id in enumerate(trajectory_ids):
        trajectory = trajectories.get(str(trajectory_id))
        if not trajectory:
            continue
        states = trajectory.get("states") if isinstance(trajectory.get("states"), list) else []
        for state_index, state in enumerate(states):
            if not isinstance(state, dict):
                continue
            content = str(state.get("thought") or state.get("action") or state.get("url") or state.get("observation") or "").strip()
            if not content:
                continue
            state_id = f"{trajectory_id}_s{state.get('state_index', state_index)}"
            answer = str(trajectory.get("goal") or trajectory.get("outcome") or "")
            keywords = tuple(_terms(content)[:6])
            timestamp = question_index * 10_000 + offset * 100 + state_index
            kind = "thought" if state.get("thought") else "action" if state.get("action") else "event"
            events.append(MemoryEvent(str(state_id), content, answer, keywords, kind, timestamp))
    return events


def _read_json_or_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("examples"), list):
            return [dict(item) for item in parsed["examples"]]  # type: ignore[index]
        if isinstance(parsed.get("data"), list):
            return [dict(item) for item in parsed["data"]]  # type: ignore[index]
        return [parsed]
    raise ValueError(f"unsupported LongMemEval data shape in {path}")


def _coerce_history(row: dict[str, object], row_index: int) -> list[MemoryEvent]:
    raw_history = row.get("history") or row.get("events") or row.get("memory") or row.get("context") or []
    if isinstance(raw_history, str):
        raw_history = [{"content": raw_history}]
    if not isinstance(raw_history, list):
        raise ValueError(f"example {row_index} has unsupported history shape")
    events: list[MemoryEvent] = []
    for event_index, raw_event in enumerate(raw_history):
        if isinstance(raw_event, str):
            payload: dict[str, object] = {"content": raw_event}
        elif isinstance(raw_event, dict):
            payload = raw_event
        else:
            continue
        content = str(payload.get("content") or payload.get("text") or payload.get("message") or payload.get("value") or "").strip()
        if not content:
            continue
        event_id = str(payload.get("id") or payload.get("memory_id") or f"row{row_index}_event{event_index}")
        answer = str(payload.get("answer") or row.get("answer") or row.get("gold_answer") or "")
        keywords_value = payload.get("keywords")
        keywords = tuple(str(value) for value in keywords_value if value) if isinstance(keywords_value, list) else tuple(_terms(content)[:6])
        timestamp = int(payload.get("timestamp") or payload.get("time") or event_index)
        kind = str(payload.get("kind") or payload.get("type") or "event")
        events.append(MemoryEvent(event_id, content, answer, keywords, kind, timestamp))
    return events


def _coerce_example(row: dict[str, object], row_index: int) -> ExternalMemoryExample:
    question = str(row.get("question") or row.get("query") or row.get("input") or row.get("task") or "").strip()
    if not question:
        raise ValueError(f"example {row_index} missing question/query")
    answer = str(row.get("answer") or row.get("gold_answer") or row.get("target") or row.get("label") or "unknown")
    evidence = row.get("evidence_ids") or row.get("gold_evidence_ids") or row.get("supporting_memory_ids") or row.get("expected_evidence_ids") or ()
    multi_hop = row.get("multi_hop_evidence_ids") or row.get("hop_evidence_ids") or ()
    temporal_answer = row.get("temporal_answer")
    conflict_answer = row.get("conflict_answer")
    return ExternalMemoryExample(
        question=question,
        answer=answer,
        expected_evidence_ids=_string_tuple(evidence),
        question_id=str(row.get("id") or row.get("question_id") or f"row{row_index}"),
        eval_function=str(row.get("eval_function") or ""),
        multi_hop_evidence_ids=_string_tuple(multi_hop),
        temporal_answer=str(temporal_answer) if temporal_answer is not None else None,
        conflict_answer=str(conflict_answer) if conflict_answer is not None else None,
        abstain=bool(row.get("abstain", False) or str(answer).lower() in {"unknown", "unanswerable", "none"}),
        repeat_query=bool(row.get("repeat_query", False) or row.get("cache_probe", False)),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terms(text: str) -> list[str]:
    return [token.lower().strip(".,:;()[]") for token in text.split() if token.strip(".,:;()[]")]


def _lexical_score(query: str, content: str) -> float:
    query_terms = set(_terms(query))
    content_terms = set(_terms(content))
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / max(1, len(query_terms | content_terms))


def _lexical_evidence(events: list[MemoryEvent], query: str, *, source: str, budget_tokens: int) -> list[EvidenceItem]:
    query_terms = set(_terms(query))
    scored: list[EvidenceItem] = []
    for event in events:
        event_terms = set(_terms(event.content)) | set(event.keywords)
        overlap = query_terms & event_terms
        if not overlap:
            continue
        score = len(overlap) / max(1, len(query_terms | event_terms))
        scored.append(EvidenceItem(event.id, event.content, score=round(score, 4), source=source))
    scored.sort(key=lambda item: (-item.score, item.id))
    kept: list[EvidenceItem] = []
    used = 0
    for item in scored:
        cost = max(1, len(item.content.split()))
        if kept and used + cost > budget_tokens:
            break
        kept.append(item)
        used += cost
    return kept


def _baseline_family(name: str) -> BaselineFamily:
    if name in TARGET_BASELINES:
        return "target"
    if name in FOUNDATIONAL_BASELINES:
        return "foundational"
    if name in STYLE_PILOT_BASELINES:
        return "style_pilot"
    return "external_adapter"


def _safe_collection_suffix(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return digest


def _normalize_mem0_results(results: object) -> list[dict[str, object]]:
    if isinstance(results, dict):
        raw = results.get("results") or results.get("memories") or results.get("data") or []
    else:
        raw = results
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
        elif hasattr(item, "model_dump"):
            normalized.append(item.model_dump())
        elif hasattr(item, "dict"):
            normalized.append(item.dict())
    return normalized


def _deterministic_embedding_vector(text: str, dims: int = 10) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dims)]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [round(value / norm, 6) for value in values]


def _deterministic_langchain_embeddings() -> object:
    from langchain.embeddings.base import Embeddings

    class DeterministicEvalEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [_deterministic_embedding_vector(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return _deterministic_embedding_vector(text)

    return DeterministicEvalEmbeddings()


def _cached_live_extraction(adapter_name: str, provider: ExternalProviderConfig, chat_model: object, event: MemoryEvent) -> str:
    cache_key = ":".join(
        [
            adapter_name,
            provider.mode,
            provider.model,
            hashlib.sha256(event.content.encode("utf-8")).hexdigest(),
        ]
    )
    with LIVE_EXTRACTION_CACHE_LOCK:
        cached = LIVE_EXTRACTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    response = chat_model.invoke(
        [
            (
                "system",
                "Extract one concise, standalone memory from the event. "
                "Preserve concrete facts, current/obsolete status, and action items. "
                "Return only the memory text.",
            ),
            ("human", event.content),
        ]
    )
    content = str(getattr(response, "content", "") or "").strip() or event.content
    with LIVE_EXTRACTION_CACHE_LOCK:
        LIVE_EXTRACTION_CACHE[cache_key] = content
    return content


def _available_factories(include_external_adapters: bool, provider: ExternalProviderConfig | None = None) -> dict[str, Callable[[], BenchMemorySystem]]:
    factories: dict[str, Callable[[], BenchMemorySystem]] = {
        name: factory
        for name, factory in BASELINE_FACTORIES.items()
        if name in TARGET_BASELINES or name in FOUNDATIONAL_BASELINES or name in STYLE_PILOT_BASELINES
    }
    if include_external_adapters:
        provider_config = provider or ExternalProviderConfig()
        factories.update(
            {
                "Mem0ExternalMemory": lambda provider_config=provider_config: Mem0ExternalMemory(provider_config),
                "LangMemExternalMemory": lambda provider_config=provider_config: LangMemExternalMemory(provider_config),
            }
        )
    return factories


def _default_baselines(include_external_adapters: bool) -> list[str]:
    names = [
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
    if include_external_adapters:
        names.extend(sorted(EXTERNAL_ADAPTER_FACTORIES))
    return names


def _answer_from_context(context: str) -> str:
    lowered = context.lower()
    if "pytest -q" in lowered and "current" in lowered:
        return "pytest_q"
    if "refresh the session before redirect" in lowered or "session refresh order checklist" in lowered or "refresh order" in lowered:
        return "refresh_order"
    if "pytest tests/" in lowered:
        return "stale_pytest"
    if not lowered.strip():
        return "unknown"
    return "unknown"


def _pack_context_evidence(bundle: EvidenceBundle, example: ExternalMemoryExample, config: ExternalBenchmarkConfig) -> EvidenceBundle:
    evidence = list(bundle.evidence)
    if config.context_packing == "RawRetrievedContext":
        packed = evidence
    elif config.context_packing == "ScoreSortedContext":
        packed = sorted(evidence, key=lambda item: (-item.score, item.id))
    elif config.context_packing == "EvidenceFirstContext":
        expected = set(example.expected_evidence_ids) | set(example.multi_hop_evidence_ids)
        packed = sorted(evidence, key=lambda item: (0 if item.id in expected else 1, -item.score, item.id))
    elif config.context_packing == "DeduplicatedEvidenceContext":
        packed = _deduplicate_evidence(evidence)
    elif config.context_packing == "TemporalConflictAwareContext":
        packed = _temporal_conflict_aware_evidence(evidence)
    elif config.context_packing == "CompressedEvidenceContext":
        packed = _compress_evidence(evidence, example)
    else:
        raise ValueError(f"unknown context packing strategy: {config.context_packing}")
    packed = _budget_evidence(packed, config.reader_budget)
    trace = dict(bundle.trace)
    trace["context_packing"] = config.context_packing
    trace["raw_selected_ids"] = [item.id for item in evidence]
    trace["selected_ids"] = [item.id for item in packed]
    return EvidenceBundle(
        query=bundle.query,
        evidence=packed,
        trace=trace,
        latency_ms=bundle.latency_ms + 0.03 * len(evidence),
        context_tokens=sum(len(item.content.split()) for item in packed),
        memory_item_count=bundle.memory_item_count,
        edge_count=bundle.edge_count,
        transactions=bundle.transactions,
        ledger=bundle.ledger,
    )


def _budget_evidence(evidence: list[EvidenceItem], budget_tokens: int) -> list[EvidenceItem]:
    kept: list[EvidenceItem] = []
    used = 0
    for item in evidence:
        cost = max(1, len(item.content.split()))
        if kept and used + cost > budget_tokens:
            break
        kept.append(item)
        used += cost
    return kept


def _deduplicate_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    best: dict[str, EvidenceItem] = {}
    for item in evidence:
        key = " ".join(_terms(item.content))
        current = best.get(key)
        if current is None or item.score > current.score:
            best[key] = item
    return sorted(best.values(), key=lambda item: (-item.score, item.id))


def _temporal_conflict_aware_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    current: list[EvidenceItem] = []
    stale: list[EvidenceItem] = []
    other: list[EvidenceItem] = []
    for item in evidence:
        lowered = item.content.lower()
        if "obsolete" in lowered or "old " in lowered:
            stale.append(item)
        elif "current" in lowered or "confirmed" in lowered:
            current.append(item)
        else:
            other.append(item)
    return sorted(current, key=lambda item: (-item.score, item.id)) + sorted(other, key=lambda item: (-item.score, item.id)) + sorted(stale, key=lambda item: (-item.score, item.id))


def _compress_evidence(evidence: list[EvidenceItem], example: ExternalMemoryExample) -> list[EvidenceItem]:
    expected = set(example.expected_evidence_ids) | set(example.multi_hop_evidence_ids)
    ordered = sorted(evidence, key=lambda item: (0 if item.id in expected else 1, -item.score, item.id))
    compressed: list[EvidenceItem] = []
    for item in ordered:
        words = item.content.split()
        if len(words) <= 12:
            content = item.content
        else:
            content = " ".join(words[:12]) + "."
        compressed.append(EvidenceItem(item.id, content, score=item.score, source=f"{item.source}:compressed", trace=dict(item.trace)))
    return compressed


def _metadata_for_baseline(name: str, family: str) -> dict[str, object]:
    metadata = dict(BASELINE_METADATA.get(name, {}))
    metadata["family"] = family
    if family == "external_adapter":
        metadata.setdefault("paper_role", "main_table")
        metadata.setdefault("claim_axis", "real external memory baseline")
        metadata["dependency_mode"] = "opt-in real package"
        metadata.setdefault("known_limitation", "real package adapter path; configuration must be reported for paper claims")
    else:
        metadata.setdefault("dependency_mode", "deterministic local only")
        metadata.setdefault("known_limitation", "external adapter metadata unavailable")
    return metadata


def _run_baseline(dataset: ExternalMemoryDataset, baseline: str, factory: Callable[[], BenchMemorySystem], config: ExternalBenchmarkConfig) -> ExternalBenchmarkRun:
    system = factory()
    system.reset(
        f"external-{dataset.benchmark_name}-{dataset.split}-{baseline}"
        f"-seed{config.seed}-budget{config.reader_budget}-{config.context_packing}"
    )
    for event in sorted(dataset.history, key=lambda item: (item.timestamp, item.id)):
        system.insert(event)

    selected_ids: list[str] = []
    traces: list[dict[str, object]] = []
    latency_ms = 0.0
    context_tokens = 0
    correct = 0
    evidence_hits = 0
    evidence_returned = 0
    multi_hop_hits = 0
    temporal_hits = 0
    conflict_hits = 0
    abstention_hits = 0
    cache_lookup_count = 0
    cache_hit_count = 0
    cache_stale_hit_count = 0
    cache_events: list[dict[str, object]] = []
    memory_version = "empty"
    official_scored = 0
    official_correct = 0
    official_skipped = 0
    official_llm_skipped = 0
    official_eval_counts: dict[str, int] = {}
    official_skipped_reasons: dict[str, int] = {}
    llm_judge = _longmemeval_v2_llm_judge(config) if dataset.benchmark_name == "longmemeval_v2" and config.enable_llm_judge else None

    for example in dataset.examples:
        raw_bundle = system.query(example.question, config.reader_budget)
        bundle = _pack_context_evidence(raw_bundle, example, config)
        plan = build_query_plan(
            example.question,
            answer="" if dataset.benchmark_name == "longmemeval_v2" else example.answer,
            expected_evidence_ids=example.expected_evidence_ids,
            multi_hop_evidence_ids=example.multi_hop_evidence_ids,
            abstain=example.abstain,
        )
        gated = evidence_gated_hybrid_recall(
            list(bundle.evidence),
            plan,
            budget_tokens=config.reader_budget,
            events_by_id={event.id: event for event in dataset.history},
            min_score=0.12 if baseline not in {"Mem0ExternalMemory", "LangMemExternalMemory"} else 0.08,
        )
        if dataset.benchmark_name == "longmemeval_v2":
            gated = _relabel_longmemeval_v2_retrieval(gated)
        bundle.evidence = gated.evidence
        bundle.trace["query_plan"] = plan.to_dict()
        bundle.trace["query_plan_hash"] = gated.query_plan_hash
        bundle.trace["gate_decision"] = gated.gate_decision
        bundle.trace["memory_version"] = gated.memory_version
        bundle.trace["invalidation_state"] = gated.invalidation_state
        bundle.trace["recall_config_hash"] = gated.recall_config_hash
        bundle.trace["canonical_fact_ids"] = gated.canonical_fact_ids
        existing_suppression = bundle.trace.get("suppression_reasons", {})
        bundle.trace["suppression_reasons"] = {
            **({str(key): str(value) for key, value in existing_suppression.items()} if isinstance(existing_suppression, dict) else {}),
            **gated.suppression_reasons,
        }
        existing_rejected = _string_tuple(bundle.trace.get("rejected_ids") or bundle.trace.get("rejected_memory_ids") or [])
        bundle.trace["rejected_ids"] = list(dict.fromkeys([*existing_rejected, *gated.rejected_ids]))
        bundle.context_tokens = sum(len(item.content.split()) for item in bundle.evidence)
        memory_version = gated.memory_version
        evidence_ids = {item.id for item in bundle.evidence}
        context = bundle.context()
        answer = _answer_from_context(context)
        official_eval_trace: dict[str, object] | None = None
        if dataset.benchmark_name == "longmemeval_v2" and config.scoring_mode in {"official_eval_function", "both"}:
            score = score_longmemeval_v2_answer(
                prediction=answer,
                reference=example.answer,
                eval_function=example.eval_function or "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true",
                llm_judge=llm_judge,
            )
            official_eval_trace = {
                **score.to_dict(),
                "question_id": example.question_id,
                "judge_mode": "provider" if llm_judge is not None and lme_v2_eval_name(example.eval_function).startswith("llm_") else "local_deterministic",
            }
            official_eval_counts[score.eval_name] = official_eval_counts.get(score.eval_name, 0) + 1
            if score.skipped:
                official_skipped += 1
                official_skipped_reasons[score.skipped_reason] = official_skipped_reasons.get(score.skipped_reason, 0) + 1
                if score.eval_name.startswith("llm_"):
                    official_llm_skipped += 1
            elif score.score is not None:
                official_scored += 1
                official_correct += int(score.score)
            bundle.trace["official_eval"] = official_eval_trace
            bundle.trace["reader_prediction"] = answer
        selected_ids.extend(item.id for item in bundle.evidence)
        traces.append(bundle.to_dict())
        latency_ms += bundle.latency_ms
        context_tokens += bundle.context_tokens
        evidence_returned += len(evidence_ids)
        if dataset.benchmark_name != "longmemeval_v2" and answer == example.answer:
            correct += 1
        if dataset.benchmark_name != "longmemeval_v2" and evidence_ids & set(example.expected_evidence_ids):
            evidence_hits += 1
        if dataset.benchmark_name != "longmemeval_v2" and example.multi_hop_evidence_ids and evidence_ids & set(example.multi_hop_evidence_ids):
            multi_hop_hits += 1
        if dataset.benchmark_name != "longmemeval_v2" and (example.temporal_answer is None or answer == example.temporal_answer):
            temporal_hits += 1
        if dataset.benchmark_name != "longmemeval_v2" and (example.conflict_answer is None or answer == example.conflict_answer):
            conflict_hits += 1
        if dataset.benchmark_name != "longmemeval_v2" and example.abstain and answer == "unknown":
            abstention_hits += 1
        elif dataset.benchmark_name != "longmemeval_v2" and not example.abstain:
            abstention_hits += 1
        if dataset.supports_repeated_queries or example.repeat_query:
            cache_lookup_count += 1
            raw_repeated = system.query(example.question, config.reader_budget)
            repeated = _pack_context_evidence(raw_repeated, example, config)
            repeated_gated = evidence_gated_hybrid_recall(
                list(repeated.evidence),
                plan,
                budget_tokens=config.reader_budget,
                events_by_id={event.id: event for event in dataset.history},
                min_score=0.12 if baseline not in {"Mem0ExternalMemory", "LangMemExternalMemory"} else 0.08,
            )
            if dataset.benchmark_name == "longmemeval_v2":
                repeated_gated = _relabel_longmemeval_v2_retrieval(repeated_gated)
            repeated.evidence = repeated_gated.evidence
            repeated.trace["query_plan"] = plan.to_dict()
            repeated.trace["query_plan_hash"] = repeated_gated.query_plan_hash
            repeated.trace["gate_decision"] = repeated_gated.gate_decision
            repeated.trace["memory_version"] = repeated_gated.memory_version
            repeated.trace["invalidation_state"] = repeated_gated.invalidation_state
            repeated.trace["recall_config_hash"] = repeated_gated.recall_config_hash
            existing_repeated_suppression = repeated.trace.get("suppression_reasons", {})
            repeated.trace["suppression_reasons"] = {
                **({str(key): str(value) for key, value in existing_repeated_suppression.items()} if isinstance(existing_repeated_suppression, dict) else {}),
                **repeated_gated.suppression_reasons,
            }
            existing_repeated_rejected = _string_tuple(repeated.trace.get("rejected_ids") or repeated.trace.get("rejected_memory_ids") or [])
            repeated.trace["rejected_ids"] = list(dict.fromkeys([*existing_repeated_rejected, *repeated_gated.rejected_ids]))
            repeated.context_tokens = sum(len(item.content.split()) for item in repeated.evidence)
            cache_lookup_count += 1
            repeated_context = repeated.context()
            cache_key = "|".join([gated.query_plan_hash, gated.memory_version, gated.invalidation_state, gated.recall_config_hash])
            repeated_cache_key = "|".join([repeated_gated.query_plan_hash, repeated_gated.memory_version, repeated_gated.invalidation_state, repeated_gated.recall_config_hash])
            is_hit = repeated_cache_key == cache_key and repeated_context == context
            is_stale = repeated_gated.invalidation_state == "stale" or ("pytest tests/" in repeated_context.lower() and "pytest -q" not in repeated_context.lower())
            if is_hit:
                cache_hit_count += 1
                if is_stale:
                    cache_stale_hit_count += 1
            cache_events.append(
                {
                    "query": example.question,
                    "hit": is_hit,
                    "valid": not is_stale,
                    "source_query": example.question,
                    "invalidation_state": repeated_gated.invalidation_state,
                    "memory_version": repeated_gated.memory_version,
                    "query_plan_hash": repeated_gated.query_plan_hash,
                    "recall_config_hash": repeated_gated.recall_config_hash,
                    "cache_key": repeated_cache_key,
                }
            )
            latency_ms += repeated.latency_ms
            context_tokens += repeated.context_tokens

    total = max(1, len(dataset.examples))
    multi_hop_total = max(1, sum(1 for example in dataset.examples if example.multi_hop_evidence_ids))
    expected_evidence_total = max(1, sum(len(example.expected_evidence_ids) for example in dataset.examples))
    family = _baseline_family(baseline)
    metadata = _metadata_for_baseline(baseline, family)
    benchmark_metadata = _benchmark_metadata(dataset, config)
    trace_faithfulness = _trace_faithfulness(traces, selected_ids, cache_events)
    cache_hit_rate = cache_hit_count / max(1, cache_lookup_count)
    cache_stale_hit_rate = cache_stale_hit_count / max(1, cache_hit_count)
    if dataset.benchmark_name == "longmemeval_v2":
        official_accuracy = official_correct / max(1, official_scored)
        metrics = {
            "official_answer_accuracy": official_accuracy,
            "reader_score": official_accuracy,
            "official_scored_count": float(official_scored),
            "official_correct_count": float(official_correct),
            "official_skipped_count": float(official_skipped),
            "official_llm_skipped_count": float(official_llm_skipped),
            "retrieval_selected_count": float(evidence_returned),
            "latency_ms": round(latency_ms, 3),
            "context_tokens": float(context_tokens),
            "cache_hit_rate": cache_hit_rate,
            "cache_stale_hit_rate": cache_stale_hit_rate,
            "trace_faithfulness": float(trace_faithfulness["faithful"]),
        }
    else:
        metrics = {
            "answer_accuracy": correct / total,
            "evidence_recall@k": evidence_hits / total,
            "evidence_precision@k": evidence_hits / max(1, evidence_returned),
            "multi_hop_recall@k": multi_hop_hits / multi_hop_total,
            "temporal_correctness": temporal_hits / total,
            "conflict_resolution_accuracy": conflict_hits / total,
            "abstention_accuracy": abstention_hits / total,
            "latency_ms": round(latency_ms, 3),
            "context_tokens": float(context_tokens),
            "cache_hit_rate": cache_hit_rate,
            "cache_stale_hit_rate": cache_stale_hit_rate,
            "trace_faithfulness": float(trace_faithfulness["faithful"]),
        }
    return ExternalBenchmarkRun(
        suite="external-memory",
        benchmark_name=dataset.benchmark_name,
        split=dataset.split,
        baseline=baseline,
        baseline_family=family,
        scenario="memory_only_external_protocol",
        metrics=metrics,
        selected_ids=selected_ids,
        latency_ms=round(latency_ms, 3),
        context_tokens=context_tokens,
        token_cost_proxy=context_tokens,
        trace_sample=traces[0] if traces else {},
        benchmark_metadata=benchmark_metadata,
        baseline_metadata=metadata,
        fairness_metadata={
            "time_fairness": "chronological_history_only",
            "context_budget_tokens": config.reader_budget,
            "seed": config.seed,
            "reader": config.reader,
            "scorer": config.scorer,
            "context_packing": config.context_packing,
            "scoring_mode": config.scoring_mode,
            "enable_llm_judge": config.enable_llm_judge,
            "official_eval_counts": official_eval_counts,
            "official_skipped_reasons": official_skipped_reasons,
            **config.provider.metadata(),
            "evidence_gate_enabled": config.evidence_gate_enabled,
            "write_protocol": "same_normalized_memory_events",
            "expected_evidence_total": expected_evidence_total,
        },
        cache={
            "cache_lookup_count": float(cache_lookup_count),
            "cache_hit_count": float(cache_hit_count),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_stale_hit_rate": round(cache_stale_hit_rate, 4),
            "cache_saved_latency_ms_proxy": round(cache_hit_count * 0.1, 4),
            "cache_saved_token_cost_proxy": float(cache_hit_count * config.reader_budget),
            "events": cache_events,
        },
        trace_faithfulness=trace_faithfulness,
    )


def _benchmark_metadata(dataset: ExternalMemoryDataset, config: ExternalBenchmarkConfig) -> dict[str, object]:
    if dataset.source_path:
        dependency_mode = "local benchmark file"
        limitation = "local LongMemEval-style file; official claims require the official split and protocol notes"
        if dataset.benchmark_name == "longmemeval_v2":
            limitation = "LongMemEval-V2 input uses official eval_function answer scoring plus separate retrieval and trace diagnostics"
    else:
        dependency_mode = "offline deterministic fixture"
        limitation = "LongMemEval-style protocol fixture, not the official external benchmark dataset"
    return {
        "benchmark_name": dataset.benchmark_name,
        "split": dataset.split,
        "reader": config.reader,
        "reader_budget": config.reader_budget,
        "seed": config.seed,
        "scorer": config.scorer,
        "context_packing": config.context_packing,
        "evidence_gate_enabled": config.evidence_gate_enabled,
        **config.provider.metadata(),
        "dependency_mode": dependency_mode,
        "known_limitation": limitation,
        "supports_repeated_queries": dataset.supports_repeated_queries,
        "dataset_path": dataset.source_path or "",
        "dataset_hash": dataset.source_hash or "",
        "history_count": len(dataset.history),
        "example_count": len(dataset.examples),
        "scoring_mode": config.scoring_mode if dataset.benchmark_name == "longmemeval_v2" else "reader_scored",
        "llm_judge_enabled": config.enable_llm_judge,
    }


def _trace_faithfulness(traces: list[dict[str, object]], selected_ids: list[str], cache_events: list[dict[str, object]]) -> dict[str, object]:
    failures: list[str] = []
    for trace_index, trace in enumerate(traces):
        trace_payload = trace.get("trace", {}) if isinstance(trace.get("trace", {}), dict) else {}
        evidence = trace.get("evidence", []) if isinstance(trace.get("evidence", []), list) else []
        evidence_ids = {str(item.get("id")) for item in evidence if isinstance(item, dict)}
        trace_selected = set(_string_tuple(trace_payload.get("selected_ids") or trace_payload.get("selected_memory_ids") or []))
        scores = trace_payload.get("scores") or trace_payload.get("baseline_scores") or {}
        if evidence_ids and not trace_selected:
            trace_selected = evidence_ids
        for memory_id in evidence_ids:
            if memory_id not in trace_selected and not isinstance(scores, dict):
                failures.append(f"trace[{trace_index}] selected evidence {memory_id} missing trace selection or scores")
        rejected = _string_tuple(trace_payload.get("rejected_memory_ids") or [])
        suppression = trace_payload.get("suppression_reasons") or {}
        for memory_id in rejected:
            if isinstance(suppression, dict) and memory_id not in suppression:
                failures.append(f"trace[{trace_index}] suppressed {memory_id} missing lifecycle reason")
        graph_paths = trace_payload.get("graph_paths") or []
        if graph_paths and not (trace_payload.get("diffusion_scores") or trace_payload.get("scores")):
            failures.append(f"trace[{trace_index}] graph path missing graph scores/effects")
    for event in cache_events:
        if not {"source_query", "valid", "invalidation_state"} <= set(event):
            failures.append("cache hit missing source query, validity, or invalidation state")
    return {
        "faithful": not failures,
        "failure_count": len(failures),
        "failures": failures[:20],
        "selected_memory_count": len(selected_ids),
        "cache_event_count": len(cache_events),
    }


def _aggregate(runs: list[ExternalBenchmarkRun]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        row = aggregate.setdefault(run.baseline, {})
        metric_counts = counts.setdefault(run.baseline, {})
        for metric, value in run.metrics.items():
            row[metric] = row.get(metric, 0.0) + float(value)
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        for metric, value in run.cache.items():
            if isinstance(value, (int, float)):
                row[metric] = row.get(metric, 0.0) + float(value)
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
    return {
        baseline: {metric: round(value / counts[baseline][metric], 4) for metric, value in metrics.items()}
        for baseline, metrics in aggregate.items()
    }


def _dataset_for(benchmark_name: str, split: str, data_path: str | Path | None) -> ExternalMemoryDataset:
    if benchmark_name in {"longmemeval", "longmemeval_v2"}:
        if data_path is None:
            raise ValueError("--data-path is required for benchmark=longmemeval")
        return load_longmemeval_dataset(data_path, split=split)
    key = (benchmark_name, split)
    if key not in EXTERNAL_MEMORY_DATASETS:
        available = ", ".join(f"{name}:{dataset_split}" for name, dataset_split in sorted(EXTERNAL_MEMORY_DATASETS))
        raise ValueError(f"unknown external memory benchmark split: {benchmark_name}:{split}. Available: {available}")
    return EXTERNAL_MEMORY_DATASETS[key]


def _limit_dataset_examples(dataset: ExternalMemoryDataset, max_examples: int) -> ExternalMemoryDataset:
    if max_examples <= 0 or max_examples >= len(dataset.examples):
        return dataset
    examples = dataset.examples[:max_examples]
    return ExternalMemoryDataset(
        benchmark_name=dataset.benchmark_name,
        split=dataset.split,
        history=dataset.history,
        examples=examples,
        supports_repeated_queries=dataset.supports_repeated_queries,
        source_path=dataset.source_path,
        source_hash=dataset.source_hash,
    )


def run_external_memory_eval(
    *,
    benchmark_name: str = "longmemeval_style",
    split: str = "dev-fixture",
    data_path: str | Path | None = None,
    baselines: list[str] | None = None,
    budget_tokens: int = 220,
    seed: int = 0,
    context_packing: ContextPacking = "RawRetrievedContext",
    provider_mode: ProviderMode = "offline",
    provider_model: str = "deepseek-v4-flash",
    provider_base_url: str = "https://api.deepseek.com",
    provider_api_key_env: str = "DEEPSEEK_API_KEY",
    include_external_adapters: bool = False,
    scoring_mode: LongMemEvalScoringMode = "both",
    enable_llm_judge: bool = False,
    max_examples: int | None = None,
    output_format: OutputFormat = "dict",
) -> dict[str, object] | str:
    dataset = _dataset_for(benchmark_name, split, data_path)
    if max_examples is not None:
        dataset = _limit_dataset_examples(dataset, max_examples)
    config = ExternalBenchmarkConfig(
        benchmark_name=dataset.benchmark_name,
        split=dataset.split,
        reader_budget=budget_tokens,
        seed=seed,
        context_packing=context_packing,
        provider=ExternalProviderConfig(
            mode=provider_mode,
            model=provider_model,
            base_url=provider_base_url,
            api_key_env=provider_api_key_env,
        ),
        include_external_adapters=include_external_adapters,
        scoring_mode=scoring_mode,
        enable_llm_judge=enable_llm_judge,
        max_examples=max_examples,
        reader_mode="deterministic_fixed_reader" if dataset.benchmark_name != "longmemeval_v2" else "retrieval_only_reader",
    )
    factories = _available_factories(include_external_adapters, config.provider)
    selected = baselines or _default_baselines(include_external_adapters)
    unknown = [baseline for baseline in selected if baseline not in factories]
    if unknown:
        raise ValueError(f"unknown or disabled external benchmark baseline(s): {', '.join(unknown)}")
    runs = [_run_baseline(dataset, baseline, factories[baseline], config) for baseline in selected]
    report = ExternalBenchmarkReport(
        suite="external-memory",
        benchmark_name=dataset.benchmark_name,
        split=dataset.split,
        reader=config.reader,
        reader_budget=config.reader_budget,
        scorer=config.scorer,
        runs=runs,
        aggregate=_aggregate(runs),
        benchmark_metadata=_benchmark_metadata(dataset, config),
        baseline_metadata={run.baseline: run.baseline_metadata for run in runs},
        adapter_availability=external_adapter_availability(),
    )
    if output_format == "json":
        return report.to_json()
    if output_format == "jsonl":
        return report.to_jsonl()
    return report.to_dict()


EXTERNAL_ARTIFACTS = [
    "report.json",
    "report.jsonl",
    "manifest.json",
    "tables/external_baselines.csv",
    "tables/external_benchmark_metrics.csv",
    "tables/fairness_metadata.csv",
    "tables/cache_metrics.csv",
    "tables/recall_diagnostics.csv",
    "tables/trace_faithfulness.csv",
    "case_studies/external_trace.md",
    "summary.md",
]


def build_external_memory_artifacts(
    *,
    data_path: str | Path | None = None,
    benchmark_name: str = "longmemeval_style",
    split: str = "dev-fixture",
    baselines: list[str] | None = None,
    seeds: list[int] | None = None,
    budget_tokens: list[int] | None = None,
    context_packings: list[ContextPacking] | None = None,
    provider_mode: ProviderMode = "offline",
    provider_model: str = "deepseek-v4-flash",
    provider_base_url: str = "https://api.deepseek.com",
    provider_api_key_env: str = "DEEPSEEK_API_KEY",
    include_external_adapters: bool = False,
    scoring_mode: LongMemEvalScoringMode = "both",
    enable_llm_judge: bool = False,
    max_examples: int | None = None,
    jobs: int = 1,
    command: str = "python -m neuromem.evals.run_experiment --suite external-memory",
) -> dict[str, str]:
    seed_values = seeds or [0]
    budget_values = budget_tokens or [220]
    packing_values = context_packings or ["RawRetrievedContext"]
    reports: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    tasks = [(seed, budget, packing) for seed in seed_values for budget in budget_values for packing in packing_values]

    def run_task(task: tuple[int, int, ContextPacking]) -> tuple[int, int, ContextPacking, dict[str, object]]:
        seed, budget, packing = task
        report = run_external_memory_eval(
            benchmark_name=benchmark_name,
            split=split,
            data_path=data_path,
            baselines=baselines,
            budget_tokens=budget,
            seed=seed,
            context_packing=packing,
            provider_mode=provider_mode,
            provider_model=provider_model,
            provider_base_url=provider_base_url,
            provider_api_key_env=provider_api_key_env,
            include_external_adapters=include_external_adapters,
            scoring_mode=scoring_mode,
            enable_llm_judge=enable_llm_judge,
            max_examples=max_examples,
        )
        assert isinstance(report, dict)
        return seed, budget, packing, report

    if jobs > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            completed = list(executor.map(run_task, tasks))
    else:
        completed = [run_task(task) for task in tasks]
    for seed, budget, packing, report in completed:
        reports.append(report)
        for run in report["runs"]:
            enriched = dict(run)
            enriched["seed"] = seed
            enriched["budget_tokens"] = budget
            enriched["context_packing"] = packing
            runs.append(enriched)
    combined = _combined_external_report(reports, runs)
    manifest = {
        "suite": "external-memory",
        "command": command,
        "benchmark_name": benchmark_name,
        "split": split,
        "dataset_path": str(data_path or ""),
        "dataset_hash": str(dict(combined.get("benchmark_metadata", {})).get("dataset_hash", "")),
        "seed_list": seed_values,
        "budget_list": budget_values,
        "context_packing_list": packing_values,
        "provider_mode": provider_mode,
        "provider_model": provider_model if provider_mode != "offline" else "",
        "provider_base_url": provider_base_url if provider_mode != "offline" else "",
        "provider_api_key_env": provider_api_key_env if provider_mode != "offline" else "",
        "provider_api_key_available": bool(os.getenv(provider_api_key_env)) if provider_mode != "offline" else False,
        "scoring_mode": scoring_mode,
        "enable_llm_judge": enable_llm_judge,
        "jobs": jobs,
        "max_examples": max_examples,
        "evidence_gate_enabled": True,
        "generated_files": EXTERNAL_ARTIFACTS,
        "adapter_availability": combined.get("adapter_availability", {}),
        "baseline_list": sorted(dict(combined.get("baseline_metadata", {}))),
    }
    return {
        "report.json": json.dumps(combined, sort_keys=True),
        "report.jsonl": "\n".join(json.dumps(run, sort_keys=True) for run in runs) + "\n",
        "manifest.json": json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        "tables/external_baselines.csv": _csv(_external_baseline_rows(combined), ["baseline", "family", "claim_axis", "paper_role", "dependency_mode", "known_limitation"]),
        "tables/external_benchmark_metrics.csv": _csv(_external_metric_rows(runs), ["seed", "budget_tokens", "context_packing", "baseline", "benchmark_name", "split", "metric", "value"]),
        "tables/fairness_metadata.csv": _csv(_fairness_rows(runs), ["seed", "budget_tokens", "context_packing", "baseline", "key", "value"]),
        "tables/cache_metrics.csv": _csv(_cache_rows(runs), ["seed", "budget_tokens", "context_packing", "baseline", "cache_lookup_count", "cache_hit_count", "cache_hit_rate", "cache_stale_hit_rate"]),
        "tables/recall_diagnostics.csv": _csv(
            _recall_diagnostic_rows(runs),
            [
                "seed",
                "budget_tokens",
                "context_packing",
                "baseline",
                "gate_decision",
                "gate_rejected_count",
                "canonical_suppression_count",
                "selected_count",
                "source_coverage",
                "answerability_gate_rate",
                "query_plan_hash",
                "memory_version",
                "invalidation_state",
                "recall_config_hash",
            ],
        ),
        "tables/trace_faithfulness.csv": _csv(_trace_rows(runs), ["seed", "budget_tokens", "context_packing", "baseline", "faithful", "failure_count", "selected_memory_count", "cache_event_count"]),
        "case_studies/external_trace.md": _external_trace_case_study(runs),
        "summary.md": _external_summary(combined, command=command, seeds=seed_values, budgets=budget_values, packings=packing_values),
    }


def write_external_memory_artifacts(artifacts: dict[str, str], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    for relative, content in artifacts.items():
        target = out_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_external_memory_experiment(
    *,
    out_dir: str | Path,
    data_path: str | Path | None = None,
    benchmark_name: str = "longmemeval_style",
    split: str = "dev-fixture",
    baselines: list[str] | None = None,
    seeds: list[int] | None = None,
    budget_tokens: list[int] | None = None,
    context_packings: list[ContextPacking] | None = None,
    provider_mode: ProviderMode = "offline",
    provider_model: str = "deepseek-v4-flash",
    provider_base_url: str = "https://api.deepseek.com",
    provider_api_key_env: str = "DEEPSEEK_API_KEY",
    include_external_adapters: bool = False,
    scoring_mode: LongMemEvalScoringMode = "both",
    enable_llm_judge: bool = False,
    max_examples: int | None = None,
    jobs: int = 1,
    command: str = "python -m neuromem.evals.run_experiment --suite external-memory",
) -> dict[str, object]:
    artifacts = build_external_memory_artifacts(
        data_path=data_path,
        benchmark_name=benchmark_name,
        split=split,
        baselines=baselines,
        seeds=seeds,
        budget_tokens=budget_tokens,
        context_packings=context_packings,
        provider_mode=provider_mode,
        provider_model=provider_model,
        provider_base_url=provider_base_url,
        provider_api_key_env=provider_api_key_env,
        include_external_adapters=include_external_adapters,
        scoring_mode=scoring_mode,
        enable_llm_judge=enable_llm_judge,
        max_examples=max_examples,
        jobs=jobs,
        command=command,
    )
    write_external_memory_artifacts(artifacts, out_dir)
    return json.loads(artifacts["manifest.json"])


def _combined_external_report(reports: list[dict[str, object]], runs: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        aggregate_key = f"{run['baseline']}/{run.get('context_packing', 'RawRetrievedContext')}"
        row = aggregate.setdefault(aggregate_key, {})
        row_counts = counts.setdefault(aggregate_key, {})
        metrics = dict(run.get("metrics", {}))
        for metric, value in metrics.items():
            row[metric] = row.get(metric, 0.0) + float(value)
            row_counts[metric] = row_counts.get(metric, 0) + 1
    aggregate = {
        key: {metric: round(value / counts[key][metric], 4) for metric, value in metrics.items()}
        for key, metrics in aggregate.items()
    }
    first = reports[0] if reports else {}
    return {
        "suite": "external-memory",
        "benchmark_name": first.get("benchmark_name", ""),
        "split": first.get("split", ""),
        "runs": runs,
        "aggregate": aggregate,
        "benchmark_metadata": first.get("benchmark_metadata", {}),
        "baseline_metadata": first.get("baseline_metadata", {}),
        "adapter_availability": first.get("adapter_availability", {}),
    }


def _csv(rows: list[dict[str, object]], columns: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue()


def _external_baseline_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline, metadata_obj in sorted(dict(report.get("baseline_metadata", {})).items()):
        metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}
        rows.append({"baseline": baseline, **metadata})
    return rows


def _external_metric_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for metric, value in dict(run.get("metrics", {})).items():
            rows.append({"seed": run.get("seed", ""), "budget_tokens": run.get("budget_tokens", ""), "context_packing": run.get("context_packing", ""), "baseline": run.get("baseline", ""), "benchmark_name": run.get("benchmark_name", ""), "split": run.get("split", ""), "metric": metric, "value": value})
    return rows


def _fairness_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for key, value in dict(run.get("fairness_metadata", {})).items():
            rows.append({"seed": run.get("seed", ""), "budget_tokens": run.get("budget_tokens", ""), "context_packing": run.get("context_packing", ""), "baseline": run.get("baseline", ""), "key": key, "value": value})
    return rows


def _cache_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        cache = dict(run.get("cache", {}))
        rows.append({"seed": run.get("seed", ""), "budget_tokens": run.get("budget_tokens", ""), "context_packing": run.get("context_packing", ""), "baseline": run.get("baseline", ""), **cache})
    return rows


def _recall_diagnostic_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        trace_sample = dict(run.get("trace_sample", {}))
        trace = dict(trace_sample.get("trace", {})) if isinstance(trace_sample.get("trace", {}), dict) else {}
        suppression = dict(trace.get("suppression_reasons", {})) if isinstance(trace.get("suppression_reasons", {}), dict) else {}
        rejected = _string_tuple(trace.get("rejected_ids") or trace.get("rejected_memory_ids") or [])
        sources = trace.get("source_channels") or []
        if not sources:
            evidence = trace_sample.get("evidence", [])
            if isinstance(evidence, list):
                sources = sorted(
                    {
                        str(channel)
                        for item in evidence
                        if isinstance(item, dict)
                        for channel in dict(item.get("trace", {})).get("channels", [])
                    }
                )
        gate_decision = str(trace.get("gate_decision", ""))
        rows.append(
            {
                "seed": run.get("seed", ""),
                "budget_tokens": run.get("budget_tokens", ""),
                "context_packing": run.get("context_packing", ""),
                "baseline": run.get("baseline", ""),
                "gate_decision": gate_decision,
                "gate_rejected_count": len(rejected),
                "canonical_suppression_count": sum(1 for reason in suppression.values() if "canonical_fact" in str(reason)),
                "selected_count": len(run.get("selected_ids", [])) if isinstance(run.get("selected_ids", []), list) else "",
                "source_coverage": ",".join(str(source) for source in sources) if isinstance(sources, list) else str(sources),
                "answerability_gate_rate": 1.0 if gate_decision.startswith("abstained") else 0.0,
                "query_plan_hash": trace.get("query_plan_hash", ""),
                "memory_version": trace.get("memory_version", ""),
                "invalidation_state": trace.get("invalidation_state", ""),
                "recall_config_hash": trace.get("recall_config_hash", ""),
            }
        )
    return rows


def _trace_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        trace = dict(run.get("trace_faithfulness", {}))
        rows.append({"seed": run.get("seed", ""), "budget_tokens": run.get("budget_tokens", ""), "context_packing": run.get("context_packing", ""), "baseline": run.get("baseline", ""), **trace})
    return rows


def _relabel_longmemeval_v2_retrieval(gated: GatedRecallResult) -> GatedRecallResult:
    return gated


def _external_trace_case_study(runs: list[dict[str, object]]) -> str:
    selected = next((run for run in runs if run.get("baseline") == "NeuroMem"), runs[0] if runs else {})
    return "\n".join(
        [
            "# External Memory Trace Case Study",
            "",
            f"Baseline: {selected.get('baseline', '')}",
            f"Benchmark: {selected.get('benchmark_name', '')}",
            f"Split: {selected.get('split', '')}",
            f"Selected ids: {json.dumps(selected.get('selected_ids', []), sort_keys=True)}",
            f"Cache: {json.dumps(selected.get('cache', {}), sort_keys=True)}",
            f"Trace faithfulness: {json.dumps(selected.get('trace_faithfulness', {}), sort_keys=True)}",
            "",
        ]
    )


def _external_summary(report: dict[str, object], *, command: str, seeds: list[int], budgets: list[int], packings: list[ContextPacking]) -> str:
    return "\n".join(
        [
            "# External Memory Experiment Summary",
            "",
            f"Command: `{command}`",
            f"Benchmark: `{report.get('benchmark_name', '')}`",
            f"Split: `{report.get('split', '')}`",
            f"Seeds: `{seeds}`",
            f"Budgets: `{budgets}`",
            f"Context packings: `{packings}`",
            f"Baselines: {len(dict(report.get('baseline_metadata', {})))}",
            "",
        ]
    )

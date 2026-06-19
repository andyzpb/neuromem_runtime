from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class RuntimeTiming:
    pfc_ms: float = 0.0
    before_pfc_ms: float = 0.0
    before_retrieval_ms: float = 0.0
    answer_llm_ms: float = 0.0
    observe_ms: float = 0.0
    after_pfc_ms: float = 0.0
    commit_ms: float = 0.0
    history_append_ms: float = 0.0
    retrieval_ms: float = 0.0
    embedding_ms: float = 0.0
    index_sync_ms: float = 0.0
    graph_commit_ms: float = 0.0
    llm_ms: float = 0.0
    dashboard_refresh_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "pfc_ms": round(self.pfc_ms, 3),
            "before_pfc_ms": round(self.before_pfc_ms, 3),
            "before_retrieval_ms": round(self.before_retrieval_ms, 3),
            "answer_llm_ms": round(self.answer_llm_ms, 3),
            "observe_ms": round(self.observe_ms, 3),
            "after_pfc_ms": round(self.after_pfc_ms, 3),
            "commit_ms": round(self.commit_ms, 3),
            "history_append_ms": round(self.history_append_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "embedding_ms": round(self.embedding_ms, 3),
            "index_sync_ms": round(self.index_sync_ms, 3),
            "graph_commit_ms": round(self.graph_commit_ms, 3),
            "llm_ms": round(self.llm_ms, 3),
            "dashboard_refresh_ms": round(self.dashboard_refresh_ms, 3),
        }


class TimingSpan:
    def __init__(self, timing: RuntimeTiming, field_name: str) -> None:
        self.timing = timing
        self.field_name = field_name
        self.started = 0.0

    def __enter__(self) -> "TimingSpan":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        elapsed = (time.perf_counter() - self.started) * 1000.0
        setattr(self.timing, self.field_name, getattr(self.timing, self.field_name) + elapsed)


@dataclass(slots=True)
class EmbeddingCacheStats:
    embed_request_count: int = 0
    embedded_text_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    ollama_total_duration_ms: float = 0.0
    ollama_load_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        total = self.cache_hit_count + self.cache_miss_count
        return {
            "embed_request_count": self.embed_request_count,
            "embedded_text_count": self.embedded_text_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_hit_rate": round(self.cache_hit_count / total, 4) if total else 0.0,
            "ollama_total_duration_ms": round(self.ollama_total_duration_ms, 3),
            "ollama_load_duration_ms": round(self.ollama_load_duration_ms, 3),
        }


class EmbeddingCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    provider_model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_namespace ON embedding_cache(namespace)")

    def get(self, *, namespace: str, provider_model: str, text_hash: str) -> list[float] | None:
        return self.get_many(namespace=namespace, provider_model=provider_model, text_hashes=[text_hash]).get(text_hash)

    def get_many(self, *, namespace: str, provider_model: str, text_hashes: list[str]) -> dict[str, list[float]]:
        if not text_hashes:
            return {}
        keys = {
            self.cache_key(namespace=namespace, provider_model=provider_model, text_hash=text_hash): text_hash
            for text_hash in dict.fromkeys(text_hashes)
        }
        placeholders = ",".join("?" for _ in keys)
        result: dict[str, list[float]] = {}
        with self._lock, self._connect() as conn:
            rows = conn.execute(f"SELECT cache_key, vector_json FROM embedding_cache WHERE cache_key IN ({placeholders})", tuple(keys)).fetchall()
            for row in rows:
                text_hash = keys.get(str(row["cache_key"]))
                if text_hash is None:
                    continue
                vector = json.loads(str(row["vector_json"]))
                if isinstance(vector, list):
                    result[text_hash] = [float(value) for value in vector]
        return result

    def touch_many(self, *, namespace: str, provider_model: str, text_hashes: list[str]) -> None:
        if not text_hashes:
            return
        now = time.time()
        keys = [self.cache_key(namespace=namespace, provider_model=provider_model, text_hash=text_hash) for text_hash in dict.fromkeys(text_hashes)]
        with self._lock, self._connect() as conn:
            conn.executemany("UPDATE embedding_cache SET last_accessed_at = ? WHERE cache_key = ?", [(now, key) for key in keys])

    def flush_touches(self, touches: list[tuple[str, str, str]]) -> None:
        grouped: dict[tuple[str, str], list[str]] = {}
        for namespace, provider_model, text_hash in touches:
            grouped.setdefault((namespace, provider_model), []).append(text_hash)
        for (namespace, provider_model), text_hashes in grouped.items():
            self.touch_many(namespace=namespace, provider_model=provider_model, text_hashes=text_hashes)

    def get_legacy(self, *, namespace: str, provider_model: str, text_hash: str) -> list[float] | None:
        key = self.cache_key(namespace=namespace, provider_model=provider_model, text_hash=text_hash)
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT vector_json FROM embedding_cache WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE embedding_cache SET last_accessed_at = ? WHERE cache_key = ?", (time.time(), key))
            vector = json.loads(str(row["vector_json"]))
            if not isinstance(vector, list):
                return None
            return [float(value) for value in vector]

    def set(self, *, namespace: str, provider_model: str, text_hash: str, vector: list[float]) -> None:
        self.set_many(namespace=namespace, provider_model=provider_model, vectors={text_hash: vector})

    def set_many(self, *, namespace: str, provider_model: str, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        now = time.time()
        rows = []
        for text_hash, vector in vectors.items():
            key = self.cache_key(namespace=namespace, provider_model=provider_model, text_hash=text_hash)
            rows.append((key, namespace, provider_model, text_hash, json.dumps(vector), key, now, now))
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache(cache_key, namespace, provider_model, text_hash, vector_json, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM embedding_cache WHERE cache_key = ?), ?), ?)
                """,
                rows,
            )

    @staticmethod
    def cache_key(*, namespace: str, provider_model: str, text_hash: str) -> str:
        return stable_hash({"namespace": namespace, "provider_model": provider_model, "text_hash": text_hash})


@dataclass(slots=True)
class RetrievalCacheEntry:
    expires_at: float
    value: Any
    semantic_version: str
    filter_hash: str


class RetrievalCache:
    def __init__(self, ttl_seconds: int = 20, max_entries: int = 256) -> None:
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._items: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.last_miss_reason = "empty"

    def get(self, key: str) -> Any | None:
        if self.ttl_seconds <= 0:
            self.miss_count += 1
            self.last_miss_reason = "disabled"
            return None
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self.miss_count += 1
                if self.last_miss_reason not in {"invalidated_by_mutation", "semantic_version_changed", "filter_changed"}:
                    self.last_miss_reason = "empty"
                return None
            if entry.expires_at < now:
                self._items.pop(key, None)
                self.miss_count += 1
                self.last_miss_reason = "expired"
                return None
            self._items.move_to_end(key)
            self.hit_count += 1
            return entry.value

    def set(self, key: str, value: Any, *, semantic_version: str = "", filter_hash: str = "") -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = RetrievalCacheEntry(expires_at=time.time() + self.ttl_seconds, value=value, semantic_version=semantic_version, filter_hash=filter_hash)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate(self, reason: str = "invalidated") -> None:
        with self._lock:
            self.last_miss_reason = reason

    def stats(self) -> dict[str, object]:
        total = self.hit_count + self.miss_count
        return {
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "entries": len(self._items),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 4) if total else 0.0,
            "last_miss_reason": self.last_miss_reason,
        }


@dataclass(slots=True)
class SingleFlightHandle:
    key: str
    leader: bool
    event: threading.Event


class SingleFlight:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, threading.Event] = {}
        self.join_count = 0
        self.leader_count = 0

    def begin(self, key: str) -> SingleFlightHandle:
        with self._lock:
            event = self._events.get(key)
            if event is not None:
                self.join_count += 1
                return SingleFlightHandle(key=key, leader=False, event=event)
            event = threading.Event()
            self._events[key] = event
            self.leader_count += 1
            return SingleFlightHandle(key=key, leader=True, event=event)

    def finish(self, key: str) -> None:
        with self._lock:
            event = self._events.pop(key, None)
            if event is not None:
                event.set()

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {"inflight": len(self._events), "join_count": self.join_count, "leader_count": self.leader_count}


class EmbeddingBatcher:
    """Lightweight sync facade for batched embedding calls.

    The base runtime stays synchronous at the low-level retrieval boundary, but
    this object centralizes batch/concurrency configuration and records provider
    batch behavior so async callers can reason about vector-path cost.
    """

    def __init__(self, *, max_batch_size: int = 32, max_wait_ms: int = 20, concurrency: int = 2) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self.concurrency = max(1, int(concurrency))
        self._lock = threading.RLock()
        self.batch_count = 0
        self.embedded_text_count = 0
        self.total_wait_ms = 0.0
        self.last_batch_size = 0

    def embed(self, texts: list[str], embed: Callable[[list[str]], list[list[float]]]) -> list[list[float]]:
        if not texts:
            return []
        chunks = [texts[index : index + self.max_batch_size] for index in range(0, len(texts), self.max_batch_size)]
        vectors: list[list[float]] = []
        for chunk in chunks:
            started = time.perf_counter()
            batch_vectors = embed(chunk)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self.batch_count += 1
                self.embedded_text_count += len(chunk)
                self.total_wait_ms += elapsed_ms
                self.last_batch_size = len(chunk)
            vectors.extend(batch_vectors)
        return vectors

    def stats(self) -> dict[str, object]:
        with self._lock:
            avg_ms = self.total_wait_ms / self.batch_count if self.batch_count else 0.0
            return {
                "max_batch_size": self.max_batch_size,
                "max_wait_ms": self.max_wait_ms,
                "concurrency": self.concurrency,
                "batch_count": self.batch_count,
                "embedded_text_count": self.embedded_text_count,
                "last_batch_size": self.last_batch_size,
                "average_provider_batch_ms": round(avg_ms, 3),
            }


@dataclass(slots=True)
class BackgroundJob:
    name: str
    run: Callable[[], object]
    status: str = "queued"
    result: object | None = None
    error: str | None = None
    timing_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "timing_ms": round(self.timing_ms, 3),
        }


class BackgroundJobQueue:
    def __init__(self) -> None:
        self._jobs: Queue[BackgroundJob | None] = Queue()
        self._recent: list[BackgroundJob] = []
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._worker, name="neuromem-background-jobs", daemon=True)
        self._thread.start()

    def submit(self, name: str, run: Callable[[], object]) -> dict[str, object]:
        job = BackgroundJob(name=name, run=run)
        with self._lock:
            self._recent.append(job)
            self._recent = self._recent[-50:]
        self._jobs.put(job)
        return job.to_dict()

    def recent(self) -> list[dict[str, object]]:
        with self._lock:
            return [job.to_dict() for job in self._recent[-10:]]

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            job.status = "running"
            try:
                job.result = job.run()
                job.status = "completed"
            except Exception as exc:  # pragma: no cover - background safety net
                job.error = str(exc)
                job.status = "failed"
            finally:
                job.timing_ms = (time.perf_counter() - started) * 1000.0
                self._jobs.task_done()


@dataclass(slots=True)
class RetrievalPerformanceContext:
    embedding_cache: EmbeddingCache | None = None
    embedding_provider_label: str = "unknown"
    timing: RuntimeTiming = field(default_factory=RuntimeTiming)
    cache_stats: dict[str, object] = field(default_factory=dict)

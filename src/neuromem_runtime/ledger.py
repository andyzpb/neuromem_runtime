from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection
from uuid import uuid4

from neuromem_runtime.deltas import MemorySnapshot


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ExperienceEvent:
    content: str
    namespace: str = "default"
    source: str = "user"
    metadata: dict[str, object] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex}")
    content_hash: str = ""
    observed_at: str = field(default_factory=_now_text)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _hash({"content": self.content, "metadata": self.metadata})

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LedgerEvent:
    transaction_id: str
    phase: str
    event_type: str
    namespace: str = "default"
    agent_id: str | None = None
    user_id: str | None = None
    operation: str | None = None
    trace_id: str | None = None
    proposer: str = "deterministic"
    validator_decision: str = "not_applicable"
    evidence: list[dict[str, object]] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    graph_delta: list[dict[str, object]] = field(default_factory=list)
    lifecycle_delta: list[dict[str, object]] = field(default_factory=list)
    index_delta: list[dict[str, object]] = field(default_factory=list)
    memory_delta: list[dict[str, object]] = field(default_factory=list)
    rollback_reason: str | None = None
    audit: dict[str, object] = field(default_factory=dict)
    ledger_seq: int | None = None
    ledger_id: str = field(default_factory=lambda: f"led_{uuid4().hex}")
    previous_hash: str | None = None
    event_hash: str = ""
    created_at: str = field(default_factory=_now_text)

    def payload_for_hash(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("ledger_seq", None)
        payload.pop("event_hash", None)
        return payload

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MemoryLedger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experience_events (
                    event_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_events (
                    ledger_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ledger_id TEXT UNIQUE NOT NULL,
                    transaction_id TEXT NOT NULL,
                    trace_id TEXT,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    agent_id TEXT,
                    user_id TEXT,
                    phase TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    operation TEXT,
                    proposer TEXT,
                    validator_decision TEXT,
                    evidence_json TEXT NOT NULL,
                    target_json TEXT NOT NULL,
                    graph_delta_json TEXT NOT NULL,
                    lifecycle_delta_json TEXT NOT NULL,
                    index_delta_json TEXT NOT NULL,
                    memory_delta_json TEXT NOT NULL DEFAULT '[]',
                    rollback_reason TEXT,
                    audit_json TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._migrate_ledger_events(conn)
            self._ensure_columns(
                conn,
                "ledger_events",
                {
                    "memory_delta_json": "TEXT NOT NULL DEFAULT '[]'",
                    "namespace": "TEXT NOT NULL DEFAULT 'default'",
                    "agent_id": "TEXT",
                    "user_id": "TEXT",
                    "ledger_seq": "INTEGER",
                },
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_events_seq ON ledger_events(ledger_seq)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_versions (
                    version_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_versions (
                    version_id TEXT PRIMARY KEY,
                    edge_id TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _migrate_ledger_events(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(ledger_events)").fetchall()}
        if "ledger_seq" in columns:
            return
        conn.execute("ALTER TABLE ledger_events RENAME TO ledger_events_old")
        conn.execute(
            """
            CREATE TABLE ledger_events (
                ledger_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ledger_id TEXT UNIQUE NOT NULL,
                transaction_id TEXT NOT NULL,
                trace_id TEXT,
                namespace TEXT NOT NULL DEFAULT 'default',
                agent_id TEXT,
                user_id TEXT,
                phase TEXT NOT NULL,
                event_type TEXT NOT NULL,
                operation TEXT,
                proposer TEXT,
                validator_decision TEXT,
                evidence_json TEXT NOT NULL,
                target_json TEXT NOT NULL,
                graph_delta_json TEXT NOT NULL,
                lifecycle_delta_json TEXT NOT NULL,
                index_delta_json TEXT NOT NULL,
                memory_delta_json TEXT NOT NULL DEFAULT '[]',
                rollback_reason TEXT,
                audit_json TEXT NOT NULL,
                previous_hash TEXT,
                event_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        old_columns = {row["name"] for row in conn.execute("PRAGMA table_info(ledger_events_old)").fetchall()}
        memory_delta_expr = "memory_delta_json" if "memory_delta_json" in old_columns else "'[]'"
        namespace_expr = "namespace" if "namespace" in old_columns else "'default'"
        agent_expr = "agent_id" if "agent_id" in old_columns else "NULL"
        user_expr = "user_id" if "user_id" in old_columns else "NULL"
        conn.execute(
            f"""
            INSERT INTO ledger_events (
                ledger_id, transaction_id, trace_id, namespace, agent_id, user_id,
                phase, event_type, operation, proposer, validator_decision,
                evidence_json, target_json, graph_delta_json, lifecycle_delta_json,
                index_delta_json, memory_delta_json, rollback_reason, audit_json,
                previous_hash, event_hash, created_at
            )
            SELECT
                ledger_id, transaction_id, trace_id, {namespace_expr}, {agent_expr}, {user_expr},
                phase, event_type, operation, proposer, validator_decision,
                evidence_json, target_json, graph_delta_json, lifecycle_delta_json,
                index_delta_json, {memory_delta_expr}, rollback_reason, audit_json,
                previous_hash, event_hash, created_at
            FROM ledger_events_old
            ORDER BY created_at ASC, ledger_id ASC
            """
        )
        conn.execute("DROP TABLE ledger_events_old")
        self._rehash_chain(conn)

    def _rehash_chain(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("SELECT * FROM ledger_events ORDER BY ledger_seq ASC").fetchall()
        previous: str | None = None
        for row in rows:
            event = LedgerEvent(
                ledger_id=str(row["ledger_id"]),
                ledger_seq=int(row["ledger_seq"]),
                transaction_id=str(row["transaction_id"]),
                trace_id=row["trace_id"],
                namespace=str(row["namespace"]),
                agent_id=row["agent_id"],
                user_id=row["user_id"],
                phase=str(row["phase"]),
                event_type=str(row["event_type"]),
                operation=row["operation"],
                proposer=str(row["proposer"]),
                validator_decision=str(row["validator_decision"]),
                evidence=json.loads(row["evidence_json"]),
                targets=json.loads(row["target_json"]),
                graph_delta=json.loads(row["graph_delta_json"]),
                lifecycle_delta=json.loads(row["lifecycle_delta_json"]),
                index_delta=json.loads(row["index_delta_json"]),
                memory_delta=json.loads(row["memory_delta_json"]),
                rollback_reason=row["rollback_reason"],
                audit=json.loads(row["audit_json"]),
                previous_hash=previous,
                created_at=str(row["created_at"]),
            )
            event.event_hash = _hash({"previous_hash": event.previous_hash, "event": event.payload_for_hash()})
            conn.execute(
                "UPDATE ledger_events SET previous_hash = ?, event_hash = ? WHERE ledger_seq = ?",
                (event.previous_hash, event.event_hash, event.ledger_seq),
            )
            previous = event.event_hash

    def _connection(self, conn: Connection | None = None):
        if conn is not None:
            return _BorrowedConnection(conn)
        return self._connect()

    def record_experience(self, event: ExperienceEvent, *, conn: Connection | None = None) -> ExperienceEvent:
        with self._connection(conn) as active:
            active.execute(
                """
                INSERT OR IGNORE INTO experience_events (
                    event_id, namespace, source, content, content_hash, metadata_json, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.namespace,
                    event.source,
                    event.content,
                    event.content_hash,
                    _canonical(event.metadata),
                    event.observed_at,
                ),
            )
        return event

    def get_experience(self, event_id: str, *, namespace: str | None = None, conn: Connection | None = None) -> ExperienceEvent | None:
        query = "SELECT * FROM experience_events WHERE event_id = ?"
        params: tuple[object, ...] = (event_id,)
        if namespace is not None:
            query += " AND namespace = ?"
            params = (event_id, namespace)
        with self._connection(conn) as active:
            row = active.execute(query, params).fetchone()
        if row is None:
            return None
        return ExperienceEvent(
            event_id=str(row["event_id"]),
            namespace=str(row["namespace"]),
            source=str(row["source"]),
            content=str(row["content"]),
            content_hash=str(row["content_hash"]),
            metadata=json.loads(row["metadata_json"]),
            observed_at=str(row["observed_at"]),
        )

    def append(self, event: LedgerEvent, *, conn: Connection | None = None) -> LedgerEvent:
        with self._connection(conn) as active:
            row = active.execute("SELECT event_hash FROM ledger_events ORDER BY ledger_seq DESC LIMIT 1").fetchone()
            event.previous_hash = str(row["event_hash"]) if row else None
            event.event_hash = _hash({"previous_hash": event.previous_hash, "event": event.payload_for_hash()})
            cursor = active.execute(
                """
                INSERT INTO ledger_events (
                    ledger_id, transaction_id, trace_id, namespace, agent_id, user_id,
                    phase, event_type, operation, proposer,
                    validator_decision, evidence_json, target_json, graph_delta_json,
                    lifecycle_delta_json, index_delta_json, memory_delta_json, rollback_reason,
                    audit_json, previous_hash, event_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.ledger_id,
                    event.transaction_id,
                    event.trace_id,
                    event.namespace,
                    event.agent_id,
                    event.user_id,
                    event.phase,
                    event.event_type,
                    event.operation,
                    event.proposer,
                    event.validator_decision,
                    _canonical(event.evidence),
                    _canonical(event.targets),
                    _canonical(event.graph_delta),
                    _canonical(event.lifecycle_delta),
                    _canonical(event.index_delta),
                    _canonical(event.memory_delta),
                    event.rollback_reason,
                    _canonical(event.audit),
                    event.previous_hash,
                    event.event_hash,
                    event.created_at,
                ),
            )
            event.ledger_seq = int(cursor.lastrowid)
        return event

    def record_memory_version(self, memory_id: str, transaction_id: str, state: dict[str, object], *, conn: Connection | None = None) -> None:
        with self._connection(conn) as active:
            active.execute(
                """
                INSERT INTO memory_versions (version_id, memory_id, transaction_id, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (f"mver_{uuid4().hex}", memory_id, transaction_id, _canonical(state), _now_text()),
            )

    def record_edge_version(self, edge_id: str, transaction_id: str, state: dict[str, object], *, conn: Connection | None = None) -> None:
        with self._connection(conn) as active:
            active.execute(
                """
                INSERT INTO edge_versions (version_id, edge_id, transaction_id, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (f"ever_{uuid4().hex}", edge_id, transaction_id, _canonical(state), _now_text()),
            )

    def show_transaction(self, transaction_id: str, *, namespace: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM ledger_events WHERE transaction_id = ?"
        params: list[object] = [transaction_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY ledger_seq ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def events_for_trace(self, trace_id: str, *, namespace: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM ledger_events WHERE trace_id = ?"
        params: list[object] = [trace_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY ledger_seq ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def why_written(self, memory_id: str, *, namespace: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM ledger_events WHERE (target_json LIKE ? OR memory_delta_json LIKE ?)"
        params: list[object] = [f"%{memory_id}%", f"%{memory_id}%"]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY ledger_seq ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def retrieval_explain(self, trace_id: str, *, namespace: str | None = None) -> dict[str, object] | None:
        events = self.events_for_trace(trace_id, namespace=namespace)
        retrieval_events = [event for event in events if event.get("event_type") == "memory_retrieved"]
        if not retrieval_events:
            return None
        audit = retrieval_events[-1].get("audit", {})
        if isinstance(audit, dict):
            ledger = audit.get("retrieval_ledger")
            if isinstance(ledger, dict):
                return ledger
            query_plan = audit.get("query_plan")
            if isinstance(query_plan, dict):
                nested = query_plan.get("retrieval_ledger")
                if isinstance(nested, dict):
                    return nested
        return retrieval_events[-1]

    def replay_trace(self, trace_id: str, *, namespace: str | None = None) -> dict[str, object] | None:
        events = self.events_for_trace(trace_id, namespace=namespace)
        if not events:
            return None
        audit = events[-1].get("audit", {})
        if isinstance(audit, dict) and audit:
            replay = dict(audit)
            replay["ledger_events"] = events
            replay["retrieval_ledger"] = self.retrieval_explain(trace_id, namespace=namespace)
            replay["memory_deltas"] = [delta for event in events for delta in event.get("memory_delta", [])]
            replay["graph_deltas"] = [delta for event in events for delta in event.get("graph_delta", [])]
            replay["lifecycle_deltas"] = [delta for event in events for delta in event.get("lifecycle_delta", [])]
            return replay
        return {"trace_id": trace_id, "ledger_events": events}

    def replay(self, to_transaction_id: str | None = None, *, namespace: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM ledger_events"
        clauses: list[str] = []
        params: list[object] = []
        if to_transaction_id is not None:
            clauses.append("ledger_seq <= COALESCE((SELECT MAX(ledger_seq) FROM ledger_events WHERE transaction_id = ?), ledger_seq)")
            params.append(to_transaction_id)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ledger_seq ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def reconstruct(self, to_transaction_id: str | None = None, *, namespace: str | None = None) -> MemorySnapshot:
        events = self.replay(to_transaction_id, namespace=namespace)
        version_txns = [str(event["transaction_id"]) for event in events]
        version_snapshot = self._version_snapshot(version_txns, namespace=namespace)
        if version_snapshot.memories or version_snapshot.edges:
            version_snapshot.transaction_id = to_transaction_id or (version_txns[-1] if version_txns else None)
            return version_snapshot
        memories: dict[str, dict[str, object]] = {}
        edges: dict[str, dict[str, object]] = {}
        last_txn: str | None = None
        for event in events:
            last_txn = str(event["transaction_id"])
            for delta in event.get("memory_delta", []):
                if not isinstance(delta, dict):
                    continue
                memory_id = str(delta.get("memory_id", ""))
                field = str(delta.get("field", ""))
                if not memory_id or not field:
                    continue
                if field == "created" and isinstance(delta.get("new"), dict):
                    memories[memory_id] = dict(delta["new"])  # type: ignore[arg-type]
                elif field == "deleted":
                    memories.pop(memory_id, None)
                else:
                    memories.setdefault(memory_id, {})[field] = delta.get("new")
            for delta in event.get("graph_delta", []):
                if not isinstance(delta, dict):
                    continue
                edge_id = str(delta.get("edge_id") or "|".join(str(delta.get(key, "")) for key in ["source_id", "target_id", "relation"]))
                if edge_id.strip("|"):
                    edges.setdefault(edge_id, {}).update(delta)
        return MemorySnapshot(memories=memories, edges=edges, transaction_id=to_transaction_id or last_txn)

    def verify_hash_chain(self) -> bool:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ledger_events ORDER BY ledger_seq ASC").fetchall()
        previous: str | None = None
        for row in rows:
            event = LedgerEvent(
                ledger_id=str(row["ledger_id"]),
                transaction_id=str(row["transaction_id"]),
                trace_id=row["trace_id"],
                namespace=str(row["namespace"]),
                agent_id=row["agent_id"],
                user_id=row["user_id"],
                phase=str(row["phase"]),
                event_type=str(row["event_type"]),
                operation=row["operation"],
                proposer=str(row["proposer"]),
                validator_decision=str(row["validator_decision"]),
                evidence=json.loads(row["evidence_json"]),
                targets=json.loads(row["target_json"]),
                graph_delta=json.loads(row["graph_delta_json"]),
                lifecycle_delta=json.loads(row["lifecycle_delta_json"]),
                index_delta=json.loads(row["index_delta_json"]),
                memory_delta=json.loads(row["memory_delta_json"]),
                rollback_reason=row["rollback_reason"],
                audit=json.loads(row["audit_json"]),
                ledger_seq=int(row["ledger_seq"]),
                previous_hash=row["previous_hash"],
                event_hash=row["event_hash"],
                created_at=str(row["created_at"]),
            )
            if event.previous_hash != previous:
                return False
            expected = _hash({"previous_hash": event.previous_hash, "event": event.payload_for_hash()})
            if event.event_hash != expected:
                return False
            previous = event.event_hash
        return True

    def diff(self, left_txn: str, right_txn: str, *, namespace: str | None = None) -> dict[str, object]:
        left = self.show_transaction(left_txn, namespace=namespace)
        right = self.show_transaction(right_txn, namespace=namespace)
        return {
            "left": left_txn,
            "right": right_txn,
            "left_event_count": len(left),
            "right_event_count": len(right),
            "left_events": left,
            "right_events": right,
        }

    def _row_to_event(self, row: sqlite3.Row) -> dict[str, object]:
        return {
            "ledger_id": row["ledger_id"],
            "ledger_seq": row["ledger_seq"],
            "transaction_id": row["transaction_id"],
            "trace_id": row["trace_id"],
            "namespace": row["namespace"],
            "agent_id": row["agent_id"],
            "user_id": row["user_id"],
            "phase": row["phase"],
            "event_type": row["event_type"],
            "operation": row["operation"],
            "proposer": row["proposer"],
            "validator_decision": row["validator_decision"],
            "evidence": json.loads(row["evidence_json"]),
            "targets": json.loads(row["target_json"]),
            "graph_delta": json.loads(row["graph_delta_json"]),
            "lifecycle_delta": json.loads(row["lifecycle_delta_json"]),
            "index_delta": json.loads(row["index_delta_json"]),
            "memory_delta": json.loads(row["memory_delta_json"]),
            "rollback_reason": row["rollback_reason"],
            "audit": json.loads(row["audit_json"]),
            "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"],
            "created_at": row["created_at"],
        }

    def _version_snapshot(self, transaction_ids: list[str], *, namespace: str | None = None) -> MemorySnapshot:
        if not transaction_ids:
            return MemorySnapshot()
        placeholders = ",".join("?" for _ in transaction_ids)
        memories: dict[str, dict[str, object]] = {}
        edges: dict[str, dict[str, object]] = {}
        with self._connect() as conn:
            memory_rows = conn.execute(
                f"""
                SELECT memory_id, state_json
                FROM memory_versions
                WHERE transaction_id IN ({placeholders})
                ORDER BY created_at ASC, version_id ASC
                """,
                tuple(transaction_ids),
            ).fetchall()
            edge_rows = conn.execute(
                f"""
                SELECT edge_id, state_json
                FROM edge_versions
                WHERE transaction_id IN ({placeholders})
                ORDER BY created_at ASC, version_id ASC
                """,
                tuple(transaction_ids),
            ).fetchall()
        for row in memory_rows:
            state = json.loads(row["state_json"])
            if namespace is not None and state.get("namespace") != namespace:
                continue
            memories[str(row["memory_id"])] = state
        for row in edge_rows:
            state = json.loads(row["state_json"])
            edges[str(row["edge_id"])] = state
        return MemorySnapshot(memories=memories, edges=edges)


class _BorrowedConnection:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def __enter__(self) -> Connection:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


__all__ = ["ExperienceEvent", "LedgerEvent", "MemoryLedger"]

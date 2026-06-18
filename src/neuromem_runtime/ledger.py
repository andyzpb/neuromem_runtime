from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


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
    ledger_id: str = field(default_factory=lambda: f"led_{uuid4().hex}")
    previous_hash: str | None = None
    event_hash: str = ""
    created_at: str = field(default_factory=_now_text)

    def payload_for_hash(self) -> dict[str, object]:
        payload = asdict(self)
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
                    ledger_id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    trace_id TEXT,
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
            self._ensure_columns(
                conn,
                "ledger_events",
                {
                    "memory_delta_json": "TEXT NOT NULL DEFAULT '[]'",
                },
            )
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

    def record_experience(self, event: ExperienceEvent) -> ExperienceEvent:
        with self._connect() as conn:
            conn.execute(
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

    def append(self, event: LedgerEvent) -> LedgerEvent:
        with self._connect() as conn:
            row = conn.execute("SELECT event_hash FROM ledger_events ORDER BY created_at DESC, ledger_id DESC LIMIT 1").fetchone()
            event.previous_hash = str(row["event_hash"]) if row else None
            event.event_hash = _hash({"previous_hash": event.previous_hash, "event": event.payload_for_hash()})
            conn.execute(
                """
                INSERT INTO ledger_events (
                    ledger_id, transaction_id, trace_id, phase, event_type, operation, proposer,
                    validator_decision, evidence_json, target_json, graph_delta_json,
                    lifecycle_delta_json, index_delta_json, memory_delta_json, rollback_reason,
                    audit_json, previous_hash, event_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.ledger_id,
                    event.transaction_id,
                    event.trace_id,
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
        return event

    def record_memory_version(self, memory_id: str, transaction_id: str, state: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_versions (version_id, memory_id, transaction_id, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (f"mver_{uuid4().hex}", memory_id, transaction_id, _canonical(state), _now_text()),
            )

    def record_edge_version(self, edge_id: str, transaction_id: str, state: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO edge_versions (version_id, edge_id, transaction_id, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (f"ever_{uuid4().hex}", edge_id, transaction_id, _canonical(state), _now_text()),
            )

    def show_transaction(self, transaction_id: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ledger_events WHERE transaction_id = ? ORDER BY created_at ASC, ledger_id ASC",
                (transaction_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def events_for_trace(self, trace_id: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ledger_events WHERE trace_id = ? ORDER BY created_at ASC, ledger_id ASC",
                (trace_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def why_written(self, memory_id: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ledger_events WHERE target_json LIKE ? OR memory_delta_json LIKE ? ORDER BY created_at ASC",
                (f"%{memory_id}%", f"%{memory_id}%"),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def replay(self, to_transaction_id: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM ledger_events"
        params: tuple[object, ...] = ()
        if to_transaction_id is not None:
            query += " WHERE created_at <= COALESCE((SELECT created_at FROM ledger_events WHERE transaction_id = ? ORDER BY created_at DESC LIMIT 1), created_at)"
            params = (to_transaction_id,)
        query += " ORDER BY created_at ASC, ledger_id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def diff(self, left_txn: str, right_txn: str) -> dict[str, object]:
        left = self.show_transaction(left_txn)
        right = self.show_transaction(right_txn)
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
            "transaction_id": row["transaction_id"],
            "trace_id": row["trace_id"],
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


__all__ = ["ExperienceEvent", "LedgerEvent", "MemoryLedger"]

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from neuromem.core.models import MemoryEdge, MemoryItem
from neuromem.retrieval.activation import build_memory_card
from neuromem.stores.base import MemoryStore


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._transaction_conn: sqlite3.Connection | None = None
        self._transaction_depth = 0
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._transaction_conn is not None:
                self._transaction_depth += 1
                try:
                    yield self._transaction_conn
                finally:
                    self._transaction_depth -= 1
                return
            conn = self._connect()
            self._transaction_conn = conn
            self._transaction_depth = 1
            try:
                conn.execute("BEGIN")
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
            finally:
                self._transaction_conn = None
                self._transaction_depth = 0
                conn.close()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._transaction_conn is not None:
            yield self._transaction_conn
            return
        with self._connect() as conn:
            yield conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    user_id TEXT,
                    namespace TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    observed_at TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    source_episode_ids TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    entities TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    embedding_id TEXT,
                    links TEXT NOT NULL,
                    supports TEXT NOT NULL,
                    contradicts TEXT NOT NULL,
                    supersedes TEXT NOT NULL,
                    derived_from TEXT NOT NULL,
                    salience TEXT NOT NULL,
                    prediction_error REAL NOT NULL DEFAULT 0.0,
                    future_utility REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL,
                    maturity TEXT NOT NULL,
                    consolidation_count INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    last_accessed_at TEXT,
                    activation_count INTEGER NOT NULL DEFAULT 0,
                    coactivation_neighbors TEXT NOT NULL DEFAULT '{}',
                    reinforcement_score REAL NOT NULL DEFAULT 0.0,
                    decay_score REAL NOT NULL,
                    inhibition_score REAL NOT NULL,
                    staleness_score REAL NOT NULL DEFAULT 0.0,
                    contradiction_score REAL NOT NULL DEFAULT 0.0,
                    tag_strength REAL NOT NULL DEFAULT 0.0,
                    expires_at TEXT,
                    capture_conditions TEXT NOT NULL DEFAULT '[]',
                    deletion_policy TEXT,
                    privacy_level TEXT NOT NULL,
                    acl TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL,
                    coactivation_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_activated_at TEXT,
                    eligibility_trace REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    observed_at TEXT,
                    recorded_at TEXT,
                    lifecycle_state TEXT NOT NULL DEFAULT 'captured',
                    inhibition_score REAL NOT NULL DEFAULT 0.0,
                    contradiction_penalty REAL NOT NULL DEFAULT 0.0,
                    provenance TEXT NOT NULL,
                    UNIQUE(source_id, target_id, relation)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_cards (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    retrieval_context TEXT NOT NULL,
                    canonical_fact_key TEXT NOT NULL,
                    entity_json TEXT NOT NULL,
                    keyword_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    trust_score REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._init_fts(conn)
            self._dedupe_edges(conn)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_relation ON edges(source_id, target_id, relation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_namespace ON memory_cards(namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_cards_fact_key ON memory_cards(canonical_fact_key)")
            self._ensure_columns(
                conn,
                "memories",
                {
                    "prediction_error": "REAL NOT NULL DEFAULT 0.0",
                    "future_utility": "REAL NOT NULL DEFAULT 0.0",
                    "activation_count": "INTEGER NOT NULL DEFAULT 0",
                    "coactivation_neighbors": "TEXT NOT NULL DEFAULT '{}'",
                    "reinforcement_score": "REAL NOT NULL DEFAULT 0.0",
                    "staleness_score": "REAL NOT NULL DEFAULT 0.0",
                    "contradiction_score": "REAL NOT NULL DEFAULT 0.0",
                    "tag_strength": "REAL NOT NULL DEFAULT 0.0",
                    "expires_at": "TEXT",
                    "capture_conditions": "TEXT NOT NULL DEFAULT '[]'",
                },
            )
            self._ensure_columns(
                conn,
                "edges",
                {
                    "weight": "REAL NOT NULL DEFAULT 0.0",
                    "coactivation_count": "INTEGER NOT NULL DEFAULT 0",
                    "success_count": "INTEGER NOT NULL DEFAULT 0",
                    "failure_count": "INTEGER NOT NULL DEFAULT 0",
                    "last_activated_at": "TEXT",
                    "eligibility_trace": "REAL NOT NULL DEFAULT 1.0",
                    "observed_at": "TEXT",
                    "recorded_at": "TEXT",
                    "lifecycle_state": "TEXT NOT NULL DEFAULT 'captured'",
                    "inhibition_score": "REAL NOT NULL DEFAULT 0.0",
                    "contradiction_penalty": "REAL NOT NULL DEFAULT 0.0",
                },
            )

    def _init_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_cards_fts
                USING fts5(memory_id UNINDEXED, namespace UNINDEXED, retrieval_text, retrieval_context)
                """
            )
        except sqlite3.OperationalError:
            return

    def _dedupe_edges(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM edges
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM edges
                GROUP BY source_id, target_id, relation
            )
            """
        )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _dump(self, value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _load(self, value: str) -> object:
        return json.loads(value)

    def upsert_memory(self, item: MemoryItem) -> None:
        record = item.to_record()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, agent_id, user_id, namespace, type, content, summary, created_at,
                    observed_at, valid_from, valid_to, source_episode_ids, source_event_ids,
                    evidence, entities, keywords, tags, embedding_id, links, supports,
                    contradicts, supersedes, derived_from, salience, prediction_error,
                    future_utility, confidence, maturity, consolidation_count, access_count,
                    last_accessed_at, activation_count, coactivation_neighbors,
                    reinforcement_score, decay_score, inhibition_score, staleness_score,
                    contradiction_score, tag_strength, expires_at, capture_conditions,
                    deletion_policy, privacy_level, acl
                ) VALUES (
                    :id, :agent_id, :user_id, :namespace, :type, :content, :summary, :created_at,
                    :observed_at, :valid_from, :valid_to, :source_episode_ids, :source_event_ids,
                    :evidence, :entities, :keywords, :tags, :embedding_id, :links, :supports,
                    :contradicts, :supersedes, :derived_from, :salience, :prediction_error,
                    :future_utility, :confidence, :maturity, :consolidation_count, :access_count,
                    :last_accessed_at, :activation_count, :coactivation_neighbors,
                    :reinforcement_score, :decay_score, :inhibition_score, :staleness_score,
                    :contradiction_score, :tag_strength, :expires_at, :capture_conditions,
                    :deletion_policy, :privacy_level, :acl
                )
                ON CONFLICT(id) DO UPDATE SET
                    agent_id=excluded.agent_id,
                    user_id=excluded.user_id,
                    namespace=excluded.namespace,
                    type=excluded.type,
                    content=excluded.content,
                    summary=excluded.summary,
                    created_at=excluded.created_at,
                    observed_at=excluded.observed_at,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    source_episode_ids=excluded.source_episode_ids,
                    source_event_ids=excluded.source_event_ids,
                    evidence=excluded.evidence,
                    entities=excluded.entities,
                    keywords=excluded.keywords,
                    tags=excluded.tags,
                    embedding_id=excluded.embedding_id,
                    links=excluded.links,
                    supports=excluded.supports,
                    contradicts=excluded.contradicts,
                    supersedes=excluded.supersedes,
                    derived_from=excluded.derived_from,
                    salience=excluded.salience,
                    prediction_error=excluded.prediction_error,
                    future_utility=excluded.future_utility,
                    confidence=excluded.confidence,
                    maturity=excluded.maturity,
                    consolidation_count=excluded.consolidation_count,
                    access_count=excluded.access_count,
                    last_accessed_at=excluded.last_accessed_at,
                    activation_count=excluded.activation_count,
                    coactivation_neighbors=excluded.coactivation_neighbors,
                    reinforcement_score=excluded.reinforcement_score,
                    decay_score=excluded.decay_score,
                    inhibition_score=excluded.inhibition_score,
                    staleness_score=excluded.staleness_score,
                    contradiction_score=excluded.contradiction_score,
                    tag_strength=excluded.tag_strength,
                    expires_at=excluded.expires_at,
                    capture_conditions=excluded.capture_conditions,
                    deletion_policy=excluded.deletion_policy,
                    privacy_level=excluded.privacy_level,
                    acl=excluded.acl
                """,
                {
                    **record,
                    "source_episode_ids": self._dump(record["source_episode_ids"]),
                    "source_event_ids": self._dump(record["source_event_ids"]),
                    "evidence": self._dump(record["evidence"]),
                    "entities": self._dump(record["entities"]),
                    "keywords": self._dump(record["keywords"]),
                    "tags": self._dump(record["tags"]),
                    "links": self._dump(record["links"]),
                    "supports": self._dump(record["supports"]),
                    "contradicts": self._dump(record["contradicts"]),
                    "supersedes": self._dump(record["supersedes"]),
                    "derived_from": self._dump(record["derived_from"]),
                    "salience": self._dump(record["salience"]),
                    "coactivation_neighbors": self._dump(record["coactivation_neighbors"]),
                    "capture_conditions": self._dump(record["capture_conditions"]),
                    "acl": self._dump(record["acl"]),
                },
            )
            self._upsert_memory_card(conn, item)

    def _upsert_memory_card(self, conn: sqlite3.Connection, item: MemoryItem) -> None:
        card = build_memory_card(item)
        conn.execute(
            """
            INSERT INTO memory_cards (
                memory_id, namespace, memory_type, lifecycle_state, retrieval_text,
                retrieval_context, canonical_fact_key, entity_json, keyword_json,
                provenance_json, trust_score, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                namespace=excluded.namespace,
                memory_type=excluded.memory_type,
                lifecycle_state=excluded.lifecycle_state,
                retrieval_text=excluded.retrieval_text,
                retrieval_context=excluded.retrieval_context,
                canonical_fact_key=excluded.canonical_fact_key,
                entity_json=excluded.entity_json,
                keyword_json=excluded.keyword_json,
                provenance_json=excluded.provenance_json,
                trust_score=excluded.trust_score,
                updated_at=excluded.updated_at
            """,
            (
                card.memory_id,
                card.namespace,
                card.memory_type,
                card.lifecycle_state,
                card.retrieval_text,
                card.retrieval_context,
                card.canonical_fact_key,
                self._dump(list(card.entities)),
                self._dump(list(card.keywords)),
                self._dump(list(card.provenance_ids)),
                card.trust_score,
                item.last_accessed_at.isoformat() if item.last_accessed_at else item.created_at.isoformat(),
            ),
        )
        try:
            conn.execute("DELETE FROM memory_cards_fts WHERE memory_id = ?", (card.memory_id,))
            conn.execute(
                "INSERT INTO memory_cards_fts(memory_id, namespace, retrieval_text, retrieval_context) VALUES (?, ?, ?, ?)",
                (card.memory_id, card.namespace, card.retrieval_text, card.retrieval_context),
            )
        except sqlite3.OperationalError:
            return

    def search_memory_cards(self, query: str, *, namespace: str | None = None, limit: int = 20) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        fts_query = self._sanitize_fts_query(query)
        with self._connection() as conn:
            try:
                params: list[object] = [fts_query]
                where = "memory_cards_fts MATCH ?"
                if namespace is not None:
                    where += " AND namespace = ?"
                    params.append(namespace)
                params.append(limit)
                rows = conn.execute(
                    f"""
                    SELECT memory_id, bm25(memory_cards_fts) AS rank
                    FROM memory_cards_fts
                    WHERE {where}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
                if rows:
                    worst = max(abs(float(row["rank"])) for row in rows) or 1.0
                    return [(str(row["memory_id"]), max(0.0, 1.0 - abs(float(row["rank"])) / worst)) for row in rows]
            except sqlite3.OperationalError:
                pass
            terms = [term.lower() for term in query.split() if term.strip()]
            sql = "SELECT memory_id, retrieval_text FROM memory_cards"
            params2: tuple[object, ...] = ()
            if namespace is not None:
                sql += " WHERE namespace = ?"
                params2 = (namespace,)
            rows = conn.execute(sql, params2).fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            text = str(row["retrieval_text"]).lower()
            overlap = sum(1 for term in terms if term in text)
            if overlap:
                scored.append((str(row["memory_id"]), overlap / max(1, len(terms))))
        return sorted(scored, key=lambda value: (-value[1], value[0]))[:limit]

    def _sanitize_fts_query(self, query: str) -> str:
        terms = [term.strip().replace('"', '""') for term in query.split() if term.strip()]
        if not terms:
            return query
        return " OR ".join(f'"{term}"' for term in terms)


    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        record = dict(row)
        for key in [
            "source_episode_ids",
            "source_event_ids",
            "evidence",
            "entities",
            "keywords",
            "tags",
            "links",
            "supports",
            "contradicts",
            "supersedes",
            "derived_from",
            "salience",
            "coactivation_neighbors",
            "capture_conditions",
            "acl",
        ]:
            record[key] = self._load(record[key])
        return MemoryItem.from_record(record)

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def list_memories(self, namespace: str | None = None) -> list[MemoryItem]:
        query = "SELECT * FROM memories"
        params: tuple[object, ...] = ()
        if namespace is not None:
            query += " WHERE namespace = ?"
            params = (namespace,)
        query += " ORDER BY created_at ASC"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def add_edge(self, edge: MemoryEdge) -> None:
        self.upsert_edge(edge)

    def upsert_edge(self, edge: MemoryEdge) -> None:
        record = edge.to_record()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO edges (
                    source_id, target_id, relation, weight, confidence, coactivation_count,
                    success_count, failure_count, last_activated_at, eligibility_trace, created_at,
                    valid_from, valid_to, observed_at, recorded_at, lifecycle_state, inhibition_score,
                    contradiction_penalty, provenance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                    weight=excluded.weight,
                    confidence=excluded.confidence,
                    coactivation_count=excluded.coactivation_count,
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    last_activated_at=excluded.last_activated_at,
                    eligibility_trace=excluded.eligibility_trace,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    observed_at=excluded.observed_at,
                    recorded_at=excluded.recorded_at,
                    lifecycle_state=excluded.lifecycle_state,
                    inhibition_score=excluded.inhibition_score,
                    contradiction_penalty=excluded.contradiction_penalty,
                    provenance=excluded.provenance
                """,
                (
                    record["source_id"],
                    record["target_id"],
                    record["relation"],
                    record["weight"],
                    record["confidence"],
                    record["coactivation_count"],
                    record["success_count"],
                    record["failure_count"],
                    record["last_activated_at"],
                    record["eligibility_trace"],
                    record["created_at"],
                    record["valid_from"],
                    record["valid_to"],
                    record["observed_at"],
                    record["recorded_at"],
                    record["lifecycle_state"],
                    record["inhibition_score"],
                    record["contradiction_penalty"],
                    self._dump(record["provenance"]),
                ),
            )

    def list_edges(self, source_id: str | None = None) -> list[MemoryEdge]:
        query = "SELECT * FROM edges"
        params: tuple[object, ...] = ()
        if source_id is not None:
            query += " WHERE source_id = ?"
            params = (source_id,)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        results: list[MemoryEdge] = []
        for row in rows:
            record = dict(row)
            record["provenance"] = self._load(record["provenance"])
            results.append(MemoryEdge.from_record(record))
        return results

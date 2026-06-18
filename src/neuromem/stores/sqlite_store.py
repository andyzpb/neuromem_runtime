from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local

from neuromem.core.models import AssociativeEdge, LogicEdge, MemoryEdge, MemoryFrame, MemoryItem
from neuromem.retrieval.activation import build_memory_card
from neuromem.stores.base import MemoryStore


ASSOCIATIVE_RELATIONS = {"associated_with", "coactivated_with", "precedes", "retrieved_with", "same_trace", "same_episode", "nearby_context", "used_with_success", "used_with_failure"}


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._transaction_state = local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            transaction_conn = self._current_transaction_conn()
            if transaction_conn is not None:
                self._set_transaction_depth(self._current_transaction_depth() + 1)
                try:
                    yield transaction_conn
                finally:
                    self._set_transaction_depth(self._current_transaction_depth() - 1)
                return
            conn = self._connect()
            self._set_transaction_conn(conn)
            self._set_transaction_depth(1)
            try:
                conn.execute("BEGIN")
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
            finally:
                self._set_transaction_conn(None)
                self._set_transaction_depth(0)
                conn.close()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        transaction_conn = self._current_transaction_conn()
        if transaction_conn is not None:
            yield transaction_conn
            return
        with self._connect() as conn:
            yield conn

    def _current_transaction_conn(self) -> sqlite3.Connection | None:
        return getattr(self._transaction_state, "conn", None)

    def _set_transaction_conn(self, conn: sqlite3.Connection | None) -> None:
        self._transaction_state.conn = conn

    def _current_transaction_depth(self) -> int:
        return int(getattr(self._transaction_state, "depth", 0) or 0)

    def _set_transaction_depth(self, depth: int) -> None:
        self._transaction_state.depth = depth

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
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
                CREATE TABLE IF NOT EXISTS associative_edges (
                    namespace TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    coactivation_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    salience REAL NOT NULL DEFAULT 0.0,
                    outcome_reward REAL NOT NULL DEFAULT 0.0,
                    decay_score REAL NOT NULL DEFAULT 0.0,
                    inhibition_score REAL NOT NULL DEFAULT 0.0,
                    eligibility_trace REAL NOT NULL DEFAULT 1.0,
                    lifecycle_state TEXT NOT NULL DEFAULT 'captured',
                    provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    PRIMARY KEY(namespace, source_id, target_id, relation)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logic_nodes (
                    frame_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    frame_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source_memory_ids TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    commitment_level TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    provenance_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logic_edges (
                    namespace TEXT NOT NULL,
                    source_frame_id TEXT NOT NULL,
                    target_frame_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    source_memory_id TEXT,
                    target_memory_id TEXT,
                    weight REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    proof_obligation TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    lifecycle_state TEXT NOT NULL DEFAULT 'provisional',
                    inhibition_score REAL NOT NULL DEFAULT 0.0,
                    contradiction_penalty REAL NOT NULL DEFAULT 0.0,
                    provenance_hash TEXT NOT NULL,
                    proposer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, source_frame_id, target_frame_id, relation)
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_associative_edges_source ON associative_edges(namespace, source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_associative_edges_target ON associative_edges(namespace, target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logic_nodes_namespace ON logic_nodes(namespace, lifecycle_state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logic_nodes_key ON logic_nodes(namespace, canonical_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logic_edges_source ON logic_edges(namespace, source_frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logic_edges_memory ON logic_edges(namespace, source_memory_id, target_memory_id)")
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
            conn.execute("PRAGMA optimize")

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
        if edge.relation in ASSOCIATIVE_RELATIONS:
            self.add_associative_edge(
                AssociativeEdge(
                    namespace=_edge_namespace(self, edge),
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation=edge.relation,  # type: ignore[arg-type]
                    weight=edge.weight,
                    confidence=edge.confidence,
                    coactivation_count=edge.coactivation_count,
                    success_count=edge.success_count,
                    failure_count=edge.failure_count,
                    inhibition_score=edge.inhibition_score,
                    eligibility_trace=edge.eligibility_trace,
                    lifecycle_state=edge.lifecycle_state,
                    provenance=list(edge.provenance),
                    created_at=edge.created_at,
                    valid_from=edge.valid_from,
                    valid_to=edge.valid_to,
                )
            )
            return
        self.add_logic_edge(
            LogicEdge(
                namespace=_edge_namespace(self, edge),
                source_frame_id=edge.source_id,
                target_frame_id=edge.target_id,
                source_memory_id=edge.source_id,
                target_memory_id=edge.target_id,
                relation=edge.relation,  # type: ignore[arg-type]
                weight=edge.weight,
                confidence=edge.confidence,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
                lifecycle_state=edge.lifecycle_state,
                inhibition_score=edge.inhibition_score,
                contradiction_penalty=edge.contradiction_penalty,
                evidence_ids=list(edge.provenance),
                created_at=edge.created_at,
            )
        )

    def add_associative_edge(self, edge: AssociativeEdge) -> None:
        record = edge.to_record()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO associative_edges (
                    namespace, source_id, target_id, relation, weight, confidence,
                    coactivation_count, success_count, failure_count, salience, outcome_reward,
                    decay_score, inhibition_score, eligibility_trace, lifecycle_state, provenance,
                    created_at, updated_at, valid_from, valid_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, source_id, target_id, relation) DO UPDATE SET
                    weight=excluded.weight,
                    confidence=excluded.confidence,
                    coactivation_count=excluded.coactivation_count,
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    salience=excluded.salience,
                    outcome_reward=excluded.outcome_reward,
                    decay_score=excluded.decay_score,
                    inhibition_score=excluded.inhibition_score,
                    eligibility_trace=excluded.eligibility_trace,
                    lifecycle_state=excluded.lifecycle_state,
                    provenance=excluded.provenance,
                    updated_at=excluded.updated_at,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to
                """,
                (
                    record["namespace"],
                    record["source_id"],
                    record["target_id"],
                    record["relation"],
                    record["weight"],
                    record["confidence"],
                    record["coactivation_count"],
                    record["success_count"],
                    record["failure_count"],
                    record["salience"],
                    record["outcome_reward"],
                    record["decay_score"],
                    record["inhibition_score"],
                    record["eligibility_trace"],
                    record["lifecycle_state"],
                    self._dump(record["provenance"]),
                    record["created_at"],
                    record["updated_at"],
                    record["valid_from"],
                    record["valid_to"],
                ),
            )

    def list_associative_edges(self, source_id: str | None = None, namespace: str | None = None) -> list[AssociativeEdge]:
        query = "SELECT * FROM associative_edges"
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, source_id ASC, target_id ASC, relation ASC"
        with self._connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        results: list[AssociativeEdge] = []
        for row in rows:
            record = dict(row)
            record["provenance"] = self._load(record["provenance"])
            results.append(AssociativeEdge.from_record(record))
        return results

    def add_logic_node(self, frame: MemoryFrame) -> None:
        record = frame.to_record()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO logic_nodes (
                    frame_id, namespace, frame_type, content, canonical_key, payload_json,
                    source_memory_ids, source_event_ids, evidence_ids, confidence,
                    commitment_level, lifecycle_state, valid_from, valid_to,
                    provenance_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(frame_id) DO UPDATE SET
                    namespace=excluded.namespace,
                    frame_type=excluded.frame_type,
                    content=excluded.content,
                    canonical_key=excluded.canonical_key,
                    payload_json=excluded.payload_json,
                    source_memory_ids=excluded.source_memory_ids,
                    source_event_ids=excluded.source_event_ids,
                    evidence_ids=excluded.evidence_ids,
                    confidence=excluded.confidence,
                    commitment_level=excluded.commitment_level,
                    lifecycle_state=excluded.lifecycle_state,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    provenance_hash=excluded.provenance_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    record["frame_id"],
                    record["namespace"],
                    record["frame_type"],
                    record["content"],
                    record["canonical_key"],
                    self._dump(record["payload"]),
                    self._dump(record["source_memory_ids"]),
                    self._dump(record["source_event_ids"]),
                    self._dump(record["evidence_ids"]),
                    record["confidence"],
                    record["commitment_level"],
                    record["lifecycle_state"],
                    record["valid_from"],
                    record["valid_to"],
                    record["provenance_hash"],
                    record["created_at"],
                    record["updated_at"],
                ),
            )

    def get_logic_node(self, frame_id: str) -> MemoryFrame | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM logic_nodes WHERE frame_id = ?", (frame_id,)).fetchone()
        return self._row_to_frame(row) if row else None

    def list_logic_nodes(self, namespace: str | None = None, *, lifecycle_state: str | None = None) -> list[MemoryFrame]:
        query = "SELECT * FROM logic_nodes"
        clauses: list[str] = []
        params: list[object] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if lifecycle_state is not None:
            clauses.append("lifecycle_state = ?")
            params.append(lifecycle_state)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, frame_id ASC"
        with self._connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_frame(row) for row in rows]

    def _row_to_frame(self, row: sqlite3.Row) -> MemoryFrame:
        record = dict(row)
        record["payload"] = self._load(record.pop("payload_json"))
        for key in ["source_memory_ids", "source_event_ids", "evidence_ids"]:
            record[key] = self._load(record[key])
        return MemoryFrame.from_record(record)

    def add_logic_edge(self, edge: LogicEdge) -> None:
        record = edge.to_record()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO logic_edges (
                    namespace, source_frame_id, target_frame_id, relation, source_memory_id,
                    target_memory_id, weight, confidence, proof_obligation, evidence_ids,
                    valid_from, valid_to, lifecycle_state, inhibition_score,
                    contradiction_penalty, provenance_hash, proposer, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, source_frame_id, target_frame_id, relation) DO UPDATE SET
                    source_memory_id=excluded.source_memory_id,
                    target_memory_id=excluded.target_memory_id,
                    weight=excluded.weight,
                    confidence=excluded.confidence,
                    proof_obligation=excluded.proof_obligation,
                    evidence_ids=excluded.evidence_ids,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    lifecycle_state=excluded.lifecycle_state,
                    inhibition_score=excluded.inhibition_score,
                    contradiction_penalty=excluded.contradiction_penalty,
                    provenance_hash=excluded.provenance_hash,
                    proposer=excluded.proposer,
                    updated_at=excluded.updated_at
                """,
                (
                    record["namespace"],
                    record["source_frame_id"],
                    record["target_frame_id"],
                    record["relation"],
                    record["source_memory_id"],
                    record["target_memory_id"],
                    record["weight"],
                    record["confidence"],
                    record["proof_obligation"],
                    self._dump(record["evidence_ids"]),
                    record["valid_from"],
                    record["valid_to"],
                    record["lifecycle_state"],
                    record["inhibition_score"],
                    record["contradiction_penalty"],
                    record["provenance_hash"],
                    record["proposer"],
                    record["created_at"],
                    record["updated_at"],
                ),
            )

    def list_logic_edges(self, source_frame_id: str | None = None, namespace: str | None = None) -> list[LogicEdge]:
        query = "SELECT * FROM logic_edges"
        clauses: list[str] = []
        params: list[object] = []
        if source_frame_id is not None:
            clauses.append("source_frame_id = ?")
            params.append(source_frame_id)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, source_frame_id ASC, target_frame_id ASC, relation ASC"
        with self._connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        results: list[LogicEdge] = []
        for row in rows:
            record = dict(row)
            record["evidence_ids"] = self._load(record["evidence_ids"])
            results.append(LogicEdge.from_record(record))
        return results

    def list_edges(self, source_id: str | None = None) -> list[MemoryEdge]:
        associative = [edge.to_memory_edge() for edge in self.list_associative_edges(source_id=source_id)]
        logic = [edge.to_memory_edge() for edge in self.list_logic_edges()]
        if source_id is not None:
            logic = [edge for edge in logic if edge.source_id == source_id]
        return [*associative, *logic]


def _edge_namespace(store: SQLiteMemoryStore, edge: MemoryEdge) -> str:
    source = store.get_memory(edge.source_id)
    if source is not None:
        return source.namespace
    target = store.get_memory(edge.target_id)
    if target is not None:
        return target.namespace
    return "default"

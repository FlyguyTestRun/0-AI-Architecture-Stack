"""Data layer (layer 7).

SQLite holds application state: conversations, agent runs and their citations. It is
chosen as the default because it needs no server, it is transactional, and every
hosting target in the deployment layer can carry the file. DuckDB sits alongside it
for analytical queries over the same data without competing for the write lock.

Migrating to Supabase (Postgres) later is a connection string change plus the same
schema, which is why the DDL below avoids SQLite specific types.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zerostack.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    trace_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    orchestrator TEXT NOT NULL,
    llm_provider TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    steps TEXT NOT NULL DEFAULT '[]',
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_trace_id ON runs (trace_id);
CREATE INDEX IF NOT EXISTS idx_runs_namespace ON runs (namespace);
"""


@dataclass
class RunRecord:
    """One recorded agent run."""

    question: str
    answer: str
    orchestrator: str
    llm_provider: str
    llm_model: str
    latency_ms: float
    trace_id: str = ""
    # The tenant this run belongs to. Stored so the operational endpoints can be
    # scoped: without it the run log hands every tenant's questions and answers,
    # including retrieved document text, to any operator who can read it.
    namespace: str = "default"
    sources: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_row(self) -> tuple:
        return (
            self.id,
            self.trace_id,
            self.question,
            self.answer,
            self.orchestrator,
            self.llm_provider,
            self.llm_model,
            self.latency_ms,
            json.dumps(self.sources),
            json.dumps(self.steps),
            self.namespace,
            self.created_at,
        )


class StateStore:
    """SQLite backed application state."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else get_settings().data.sqlite_path
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection: sqlite3.Connection | None = None
        if str(self.path) == ":memory:":
            # An in memory database dies with its connection, so hold one open.
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
        self._migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            yield self._memory_connection
            self._memory_connection.commit()
            return
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self.connect() as connection:
            # The column is added before the schema script runs, not after.
            # CREATE TABLE IF NOT EXISTS will not add a column to a table that
            # already exists, so on a database written before the tenancy work
            # the table stays as it was, and the script's index on the new column
            # would then be created against a column that does not exist yet.
            # An empty result means there is no runs table at all, which is a
            # fresh database: the script below creates it complete.
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if columns and "namespace" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
                )
            connection.executescript(SCHEMA)

    def save_run(self, record: RunRecord) -> str:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, trace_id, question, answer, orchestrator, llm_provider,
                    llm_model, latency_ms, sources, steps, namespace, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.to_row(),
            )
        return record.id

    def recent_runs(
        self, limit: int = 20, namespaces: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return recent runs, restricted to ``namespaces`` when given.

        ``None`` means no restriction and is for a caller that already holds
        wildcard access. An empty list restricts to nothing rather than to
        everything, so a caller with no namespaces cannot read the whole log.
        """
        with self.connect() as connection:
            if namespaces is None:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            elif not namespaces:
                return []
            else:
                placeholders = ",".join("?" for _ in namespaces)
                rows = connection.execute(
                    "SELECT * FROM runs WHERE namespace IN "
                    f"({placeholders}) ORDER BY created_at DESC LIMIT ?",
                    (*namespaces, limit),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def count_runs(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["sources"] = json.loads(record.get("sources") or "[]")
        record["steps"] = json.loads(record.get("steps") or "[]")
        return record


def analytics_summary(
    sqlite_path: Path | None = None, namespaces: list[str] | None = None
) -> dict[str, Any]:
    """Aggregate run metrics with DuckDB.

    DuckDB reads the SQLite file directly through its sqlite scanner, so analytics
    never blocks the application's writes. If DuckDB or the scanner extension is not
    available the same aggregate is computed in SQLite instead.

    ``namespaces`` restricts the aggregate to a caller's tenants. ``None`` means
    unrestricted, for a caller holding wildcard access; an empty list restricts
    to nothing, so a caller with no namespaces does not see the whole estate.
    """
    path = sqlite_path or get_settings().data.sqlite_path
    if not Path(path).exists():
        return {"runs": 0, "engine": "none"}
    if namespaces is not None and not namespaces:
        return {"runs": 0, "engine": "none", "by_backend": []}

    # Values are bound as parameters in both engines below; only the placeholder
    # count is built from the list length.
    scope = "" if namespaces is None else f"WHERE namespace IN ({','.join('?' * len(namespaces))})"
    params: tuple = () if namespaces is None else tuple(namespaces)

    try:
        import duckdb

        connection = duckdb.connect()
        connection.execute("INSTALL sqlite; LOAD sqlite;")
        # DuckDB's ATTACH takes no prepared statement parameters, so the path
        # has to be interpolated. Double any single quote first: a path such as
        # "/home/o'brien/app.db" would otherwise terminate the string literal
        # early and raise a parser error.
        escaped = str(path).replace("'", "''")
        connection.execute(f"ATTACH '{escaped}' AS app (TYPE sqlite);")
        rows = connection.execute(
            f"""
            SELECT orchestrator,
                   llm_provider,
                   COUNT(*) AS runs,
                   ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                   ROUND(MAX(latency_ms), 2) AS max_latency_ms
            FROM app.runs
            {scope}
            GROUP BY orchestrator, llm_provider
            ORDER BY runs DESC
            """,
            params,
        ).fetchall()
        total = connection.execute(f"SELECT COUNT(*) FROM app.runs {scope}", params).fetchone()[0]
        connection.close()
        return {
            "engine": "duckdb",
            "runs": int(total),
            "by_backend": [
                {
                    "orchestrator": row[0],
                    "llm_provider": row[1],
                    "runs": int(row[2]),
                    "avg_latency_ms": row[3],
                    "max_latency_ms": row[4],
                }
                for row in rows
            ],
        }
    except Exception as exc:
        logger.info("duckdb analytics unavailable (%s), falling back to sqlite", exc)

    store = StateStore(path)
    with store.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT orchestrator, llm_provider, COUNT(*),
                   ROUND(AVG(latency_ms), 2), ROUND(MAX(latency_ms), 2)
            FROM runs {scope} GROUP BY orchestrator, llm_provider ORDER BY 3 DESC
            """,
            params,
        ).fetchall()
        total = connection.execute(f"SELECT COUNT(*) FROM runs {scope}", params).fetchone()[0]
    return {
        "engine": "sqlite",
        "runs": int(total),
        "by_backend": [
            {
                "orchestrator": row[0],
                "llm_provider": row[1],
                "runs": int(row[2]),
                "avg_latency_ms": row[3],
                "max_latency_ms": row[4],
            }
            for row in rows
        ],
    }

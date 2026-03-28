"""SQLite database manager for InvoiceAgent.

Provides CRUD operations (no DELETE — Constitution [B1]) for invoices,
processing runs, QA results, clients, and the review queue.
All queries filter by client_id to enforce tenant isolation (Constitution [B3]).
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Any

from src.config import config

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clients (
    client_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config_json TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id      TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    invoice_number  TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id  TEXT NOT NULL,
    client_id   TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'started',
    input_hash  TEXT,
    output_hash TEXT,
    duration_ms INTEGER,
    confidence  REAL,
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
);

CREATE TABLE IF NOT EXISTS qa_results (
    qa_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id      TEXT NOT NULL,
    client_id       TEXT NOT NULL,
    overall_score   REAL NOT NULL,
    passed          INTEGER NOT NULL DEFAULT 0,
    flags_json      TEXT DEFAULT '[]',
    field_scores_json TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id              TEXT NOT NULL,
    client_id               TEXT NOT NULL,
    structured_invoice_json TEXT,
    qa_result_json          TEXT,
    original_pdf_path       TEXT,
    reason                  TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'pending',
    corrections_json        TEXT,
    operator_notes          TEXT,
    reviewer                TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
);
"""


class DatabaseManager:
    """Manages all database operations for InvoiceAgent."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or config.database_path
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        """Create all tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ── Clients ──────────────────────────────────────────────

    def add_client(self, client_id: str, name: str, config_json: str = "{}") -> str:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO clients (client_id, name, config_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (client_id, name, config_json, now, now),
            )
        return client_id

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── Invoices ─────────────────────────────────────────────

    def add_invoice(
        self,
        invoice_id: str,
        client_id: str,
        source_file: str,
        invoice_number: str | None = None,
        status: str = "pending",
    ) -> str:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO invoices (invoice_id, client_id, source_file, invoice_number, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (invoice_id, client_id, source_file, invoice_number, status, now, now),
            )
        return invoice_id

    def get_invoices_by_client(self, client_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM invoices WHERE client_id = ?", (client_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_invoice_status(self, invoice_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE invoices SET status = ?, updated_at = ? WHERE invoice_id = ?",
                (status, _now(), invoice_id),
            )

    # ── Processing Runs ──────────────────────────────────────

    def add_processing_run(
        self,
        invoice_id: str,
        client_id: str,
        agent_name: str,
        status: str = "started",
    ) -> int:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO processing_runs (invoice_id, client_id, agent_name, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (invoice_id, client_id, agent_name, status, now, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def update_processing_status(
        self,
        run_id: int,
        status: str,
        *,
        input_hash: str | None = None,
        output_hash: str | None = None,
        duration_ms: int | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE processing_runs "
                "SET status = ?, input_hash = COALESCE(?, input_hash), "
                "    output_hash = COALESCE(?, output_hash), "
                "    duration_ms = COALESCE(?, duration_ms), "
                "    confidence = COALESCE(?, confidence), "
                "    error = COALESCE(?, error), "
                "    updated_at = ? "
                "WHERE run_id = ?",
                (status, input_hash, output_hash, duration_ms, confidence, error, _now(), run_id),
            )

    # ── QA Results ───────────────────────────────────────────

    def add_qa_result(
        self,
        invoice_id: str,
        client_id: str,
        overall_score: float,
        passed: bool,
        flags_json: str = "[]",
        field_scores_json: str = "{}",
    ) -> int:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO qa_results (invoice_id, client_id, overall_score, passed, flags_json, field_scores_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (invoice_id, client_id, overall_score, int(passed), flags_json, field_scores_json, now, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_qa_results_by_invoice(self, invoice_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_results WHERE invoice_id = ?", (invoice_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Review Queue ─────────────────────────────────────────

    def add_to_review_queue(
        self, invoice_id: str, client_id: str, reason: str
    ) -> int:
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO review_queue (invoice_id, client_id, reason, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (invoice_id, client_id, reason, now, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def add_review_item(
        self,
        invoice_id: str,
        client_id: str,
        structured_invoice_json: str,
        qa_result_json: str,
        original_pdf_path: str,
        reason: str,
    ) -> int:
        """Add a full review item with invoice and QA data."""
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO review_queue "
                "(invoice_id, client_id, structured_invoice_json, qa_result_json, "
                " original_pdf_path, reason, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (invoice_id, client_id, structured_invoice_json, qa_result_json,
                 original_pdf_path, reason, now, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_review_items(
        self,
        status: str | None = None,
        client_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get review items with optional status and client filters."""
        query = "SELECT * FROM review_queue WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_review_items_by_invoice(self, invoice_id: str) -> list[dict[str, Any]]:
        """Get review items for a specific invoice."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE invoice_id = ?",
                (invoice_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_review_item(
        self,
        invoice_id: str,
        status: str,
        *,
        corrections_json: str | None = None,
        operator_notes: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        """Update a review item by invoice_id."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE review_queue "
                "SET status = ?, "
                "    corrections_json = COALESCE(?, corrections_json), "
                "    operator_notes = COALESCE(?, operator_notes), "
                "    reviewer = COALESCE(?, reviewer), "
                "    updated_at = ? "
                "WHERE invoice_id = ?",
                (status, corrections_json, operator_notes, reviewer, _now(), invoice_id),
            )

    def get_review_queue_by_client(self, client_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE client_id = ? AND status = 'pending'",
                (client_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_review_status(self, review_id: int, status: str, reviewer: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE review_queue SET status = ?, reviewer = COALESCE(?, reviewer), updated_at = ? "
                "WHERE review_id = ?",
                (status, reviewer, _now(), review_id),
            )


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.datetime.now(datetime.UTC).isoformat()

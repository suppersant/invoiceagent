"""Tests for the database manager."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    """Create a DatabaseManager with a temporary database."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path=db_path)
    manager.init_db()
    return manager


class TestInitDb:
    """Test database initialization."""

    def test_init_creates_all_tables(self, tmp_path: Path):
        """scripts/init_db.py creates the database file and all 5 tables."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager(db_path=db_path)
        manager.init_db()

        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(row[0] for row in cursor.fetchall())
        conn.close()

        assert "clients" in tables
        assert "invoices" in tables
        assert "processing_runs" in tables
        assert "qa_results" in tables
        assert "review_queue" in tables


class TestInvoices:
    """Test invoice CRUD operations."""

    def test_add_invoice_returns_id(self, db: DatabaseManager):
        """DatabaseManager.add_invoice() inserts a record and returns the invoice_id."""
        inv_id = str(uuid.uuid4())
        db.add_client("client_a", "Client A")
        result = db.add_invoice(inv_id, "client_a", "/path/to/invoice.pdf")
        assert result == inv_id

    def test_get_invoices_by_client_filters(self, db: DatabaseManager):
        """DatabaseManager.get_invoices_by_client() returns only invoices for the specified client_id."""
        db.add_client("client_a", "Client A")
        db.add_client("client_b", "Client B")

        db.add_invoice(str(uuid.uuid4()), "client_a", "/a/inv1.pdf")
        db.add_invoice(str(uuid.uuid4()), "client_a", "/a/inv2.pdf")
        db.add_invoice(str(uuid.uuid4()), "client_b", "/b/inv1.pdf")

        results_a = db.get_invoices_by_client("client_a")
        assert len(results_a) == 2
        assert all(r["client_id"] == "client_a" for r in results_a)

    def test_client_isolation(self, db: DatabaseManager):
        """DatabaseManager.get_invoices_by_client("client_a") returns ZERO results for client_b's data."""
        db.add_client("client_a", "Client A")
        db.add_client("client_b", "Client B")

        db.add_invoice(str(uuid.uuid4()), "client_b", "/b/inv1.pdf")

        results = db.get_invoices_by_client("client_a")
        assert len(results) == 0

    def test_update_invoice_status(self, db: DatabaseManager):
        """DatabaseManager.update_processing_status() correctly updates status."""
        inv_id = str(uuid.uuid4())
        db.add_client("client_a", "Client A")
        db.add_invoice(inv_id, "client_a", "/path/to/invoice.pdf", status="pending")

        db.update_invoice_status(inv_id, "processed")
        invoices = db.get_invoices_by_client("client_a")
        assert invoices[0]["status"] == "processed"


class TestProcessingRuns:
    """Test processing run operations."""

    def test_add_and_update_processing_run(self, db: DatabaseManager):
        inv_id = str(uuid.uuid4())
        db.add_client("client_a", "Client A")
        db.add_invoice(inv_id, "client_a", "/path.pdf")

        run_id = db.add_processing_run(inv_id, "client_a", "extract_agent")
        assert isinstance(run_id, int)

        db.update_processing_status(
            run_id, "completed", duration_ms=1500, confidence=0.92
        )


class TestQAResults:
    """Test QA result operations."""

    def test_add_and_get_qa_result(self, db: DatabaseManager):
        inv_id = str(uuid.uuid4())
        db.add_client("client_a", "Client A")
        db.add_invoice(inv_id, "client_a", "/path.pdf")

        qa_id = db.add_qa_result(inv_id, "client_a", 0.91, True)
        assert isinstance(qa_id, int)

        results = db.get_qa_results_by_invoice(inv_id)
        assert len(results) == 1
        assert results[0]["overall_score"] == 0.91


class TestReviewQueue:
    """Test review queue operations."""

    def test_add_and_get_review_item(self, db: DatabaseManager):
        inv_id = str(uuid.uuid4())
        db.add_client("client_a", "Client A")
        db.add_invoice(inv_id, "client_a", "/path.pdf")

        review_id = db.add_to_review_queue(inv_id, "client_a", "Low confidence")
        assert isinstance(review_id, int)

        items = db.get_review_queue_by_client("client_a")
        assert len(items) == 1
        assert items[0]["reason"] == "Low confidence"


class TestNoDeleteMethods:
    """Verify no delete operations exist."""

    def test_no_delete_methods(self):
        """No method named delete_* exists on DatabaseManager."""
        methods = [m for m in dir(DatabaseManager) if m.startswith("delete")]
        assert methods == [], f"Found delete methods: {methods}"

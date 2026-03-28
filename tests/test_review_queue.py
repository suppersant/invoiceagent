"""Tests for the human review queue — covers all T4.2 acceptance criteria."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from src.models.extraction import VisionExtractionResult
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.models.qa import FieldScore, QAFlag, QAResult, IssueType
from src.pipeline.review_queue import ReviewQueue
from src.utils.database import DatabaseManager


# ── Fixtures ────────────────────────────────────────────────────────────

_TEST_INGEST_ID = uuid4()


def _make_structured(invoice_id=None) -> StructuredInvoice:
    return StructuredInvoice(
        invoice_id=invoice_id or uuid4(),
        invoice_number="INV-001",
        invoice_date=date(2026, 3, 15),
        due_date=date(2026, 4, 15),
        vendor_name="Acme Corp",
        vendor_address="123 Main St",
        currency="USD",
        subtotal=Decimal("250.00"),
        tax=Decimal("20.00"),
        total=Decimal("270.00"),
        line_items=[
            LineItem(
                description="Widget A",
                quantity=Decimal("10"),
                unit_price=Decimal("25.00"),
                amount=Decimal("250.00"),
                source_page=1,
            ),
        ],
        metadata=InvoiceMetadata(
            source_file="test.pdf",
            ingest_id=_TEST_INGEST_ID,
            page_count=1,
        ),
    )


def _make_qa_result(invoice_id=None, overall_confidence=0.60) -> QAResult:
    iid = invoice_id or uuid4()
    return QAResult(
        invoice_id=iid,
        overall_confidence=overall_confidence,
        routed_to_review=True,
        approved=False,
        field_scores=[
            FieldScore(field_name="invoice_number", confidence=0.90),
            FieldScore(field_name="invoice_date", confidence=0.90),
            FieldScore(field_name="due_date", confidence=0.85),
            FieldScore(field_name="vendor_name", confidence=0.20),
            FieldScore(field_name="vendor_address", confidence=0.10),
            FieldScore(field_name="currency", confidence=1.0),
            FieldScore(field_name="subtotal", confidence=0.85),
            FieldScore(field_name="tax", confidence=0.85),
            FieldScore(field_name="total", confidence=0.90),
            FieldScore(field_name="line_items", confidence=0.85),
        ],
        flags=[
            QAFlag(
                field_name="vendor_name",
                issue_type=IssueType.MISSING,
                message="Vendor name is missing",
            ),
        ],
    )


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_review.db"
    db = DatabaseManager(db_path=db_path)
    db.init_db()
    # Add test clients
    db.add_client("client_x", "Client X")
    db.add_client("client_y", "Client Y")
    return db


@pytest.fixture
def queue(db):
    """Create a ReviewQueue with the test database."""
    return ReviewQueue(db=db)


# ── Add to review queue ──────────────────────────────────────────────


class TestAddToReviewQueue:
    def test_add_stores_with_pending_status(self, queue, db):
        """review_queue.add(qa_result) stores the item with status 'pending'."""
        invoice_id = uuid4()
        structured = _make_structured(invoice_id=invoice_id)
        qa_result = _make_qa_result(invoice_id=invoice_id)

        # Add invoice to DB first (foreign key)
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")

        review_id = queue.add(
            qa_result=qa_result,
            structured_invoice=structured,
            original_pdf_path="/path/to/test.pdf",
            client_id="client_x",
        )

        assert isinstance(review_id, int)
        item = queue.get(str(invoice_id))
        assert item is not None
        assert item["status"] == "pending"

    def test_add_stores_structured_invoice_json(self, queue, db):
        invoice_id = uuid4()
        structured = _make_structured(invoice_id=invoice_id)
        qa_result = _make_qa_result(invoice_id=invoice_id)
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")

        queue.add(
            qa_result=qa_result,
            structured_invoice=structured,
            original_pdf_path="/path/to/test.pdf",
            client_id="client_x",
        )

        item = queue.get(str(invoice_id))
        assert item["structured_invoice_json"] is not None
        parsed = json.loads(item["structured_invoice_json"])
        assert parsed["invoice_number"] == "INV-001"

    def test_add_stores_qa_result_json(self, queue, db):
        invoice_id = uuid4()
        structured = _make_structured(invoice_id=invoice_id)
        qa_result = _make_qa_result(invoice_id=invoice_id)
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")

        queue.add(
            qa_result=qa_result,
            structured_invoice=structured,
            original_pdf_path="/path/to/test.pdf",
            client_id="client_x",
        )

        item = queue.get(str(invoice_id))
        assert item["qa_result_json"] is not None
        parsed = json.loads(item["qa_result_json"])
        assert parsed["overall_confidence"] == 0.60

    def test_add_stores_original_pdf_path(self, queue, db):
        invoice_id = uuid4()
        structured = _make_structured(invoice_id=invoice_id)
        qa_result = _make_qa_result(invoice_id=invoice_id)
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")

        queue.add(
            qa_result=qa_result,
            structured_invoice=structured,
            original_pdf_path="/path/to/original.pdf",
            client_id="client_x",
        )

        item = queue.get(str(invoice_id))
        assert item["original_pdf_path"] == "/path/to/original.pdf"


# ── List pending ─────────────────────────────────────────────────────


class TestListPending:
    def test_list_pending_returns_only_pending(self, queue, db):
        """review_queue.list_pending() returns only pending items."""
        inv1 = uuid4()
        inv2 = uuid4()
        db.add_invoice(str(inv1), "client_x", "a.pdf")
        db.add_invoice(str(inv2), "client_x", "b.pdf")

        queue.add(_make_qa_result(inv1), _make_structured(inv1), "/a.pdf", "client_x")
        queue.add(_make_qa_result(inv2), _make_structured(inv2), "/b.pdf", "client_x")

        # Approve one
        queue.approve(str(inv1))

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["invoice_id"] == str(inv2)

    def test_list_pending_filters_by_client(self, queue, db):
        """review_queue.list_pending(client_id='x') returns only client x's items."""
        inv_x = uuid4()
        inv_y = uuid4()
        db.add_invoice(str(inv_x), "client_x", "x.pdf")
        db.add_invoice(str(inv_y), "client_y", "y.pdf")

        queue.add(_make_qa_result(inv_x), _make_structured(inv_x), "/x.pdf", "client_x")
        queue.add(_make_qa_result(inv_y), _make_structured(inv_y), "/y.pdf", "client_y")

        pending_x = queue.list_pending(client_id="client_x")
        assert len(pending_x) == 1
        assert pending_x[0]["client_id"] == "client_x"

        pending_y = queue.list_pending(client_id="client_y")
        assert len(pending_y) == 1
        assert pending_y[0]["client_id"] == "client_y"

    def test_list_pending_returns_empty_when_none(self, queue):
        pending = queue.list_pending()
        assert pending == []


# ── Approve ──────────────────────────────────────────────────────────


class TestApprove:
    def test_approve_sets_status(self, queue, db):
        """review_queue.approve(invoice_id) sets status to 'approved'."""
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        queue.approve(str(invoice_id))

        item = queue.get(str(invoice_id))
        assert item["status"] == "approved"

    def test_approve_with_corrections_stores_separately(self, queue, db):
        """approve() with corrections stores them without overwriting original."""
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        corrections = {"vendor_name": "Corrected Vendor Inc"}
        queue.approve(str(invoice_id), corrections=corrections)

        item = queue.get(str(invoice_id))
        assert item["status"] == "approved"
        assert item["corrections_json"] is not None
        parsed_corrections = json.loads(item["corrections_json"])
        assert parsed_corrections["vendor_name"] == "Corrected Vendor Inc"
        # Original structured invoice still intact
        parsed_invoice = json.loads(item["structured_invoice_json"])
        assert parsed_invoice["vendor_name"] == "Acme Corp"

    def test_approve_with_operator_notes(self, queue, db):
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        queue.approve(str(invoice_id), operator_notes="Verified manually", reviewer="operator1")

        item = queue.get(str(invoice_id))
        assert item["operator_notes"] == "Verified manually"
        assert item["reviewer"] == "operator1"

    def test_approved_not_in_list_pending_but_in_list_all(self, queue, db):
        """Approved items no longer appear in list_pending() but DO appear in list_all()."""
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        queue.approve(str(invoice_id))

        assert len(queue.list_pending()) == 0
        all_items = queue.list_all()
        assert len(all_items) == 1
        assert all_items[0]["status"] == "approved"


# ── Reject ───────────────────────────────────────────────────────────


class TestReject:
    def test_reject_sets_status(self, queue, db):
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        queue.reject(str(invoice_id), operator_notes="Bad scan quality")

        item = queue.get(str(invoice_id))
        assert item["status"] == "rejected"
        assert item["operator_notes"] == "Bad scan quality"

    def test_rejected_not_in_pending(self, queue, db):
        invoice_id = uuid4()
        db.add_invoice(str(invoice_id), "client_x", "test.pdf")
        queue.add(_make_qa_result(invoice_id), _make_structured(invoice_id), "/test.pdf", "client_x")

        queue.reject(str(invoice_id))

        assert len(queue.list_pending()) == 0


# ── No delete method ─────────────────────────────────────────────────


class TestNoDeleteMethod:
    def test_no_delete_method_exists(self, queue):
        """No delete method exists on ReviewQueue (Constitution [B1])."""
        assert not hasattr(queue, "delete")
        assert not hasattr(queue, "remove")


# ── Client isolation ─────────────────────────────────────────────────


class TestClientIsolation:
    def test_list_all_filters_by_client(self, queue, db):
        inv_x = uuid4()
        inv_y = uuid4()
        db.add_invoice(str(inv_x), "client_x", "x.pdf")
        db.add_invoice(str(inv_y), "client_y", "y.pdf")

        queue.add(_make_qa_result(inv_x), _make_structured(inv_x), "/x.pdf", "client_x")
        queue.add(_make_qa_result(inv_y), _make_structured(inv_y), "/y.pdf", "client_y")

        all_x = queue.list_all(client_id="client_x")
        assert len(all_x) == 1
        assert all_x[0]["client_id"] == "client_x"

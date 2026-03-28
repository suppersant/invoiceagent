"""Tests for the pipeline orchestrator — covers all T4.1 acceptance criteria."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models.extraction import IngestResult, PageImage, VisionExtractionResult
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.models.qa import FieldScore, QAResult
from src.pipeline.orchestrator import process_invoice
from src.utils.database import DatabaseManager


# ── Fixtures ────────────────────────────────────────────────────────────

_TEST_INGEST_ID = uuid4()
_TEST_INVOICE_ID = uuid4()


def _make_ingest_result() -> IngestResult:
    return IngestResult(
        ingest_id=_TEST_INGEST_ID,
        source_file="test.pdf",
        page_count=1,
        pages=[
            PageImage(page_number=1, image_path="/tmp/page_001.png", width=2550, height=3300),
        ],
    )


def _make_extraction() -> VisionExtractionResult:
    return VisionExtractionResult(
        ingest_id=_TEST_INGEST_ID,
        vendor_name="Acme Corp",
        invoice_number="INV-001",
        invoice_date=date(2026, 3, 15),
        due_date=date(2026, 4, 15),
        total=Decimal("270.00"),
        subtotal=Decimal("250.00"),
        tax=Decimal("20.00"),
        currency="USD",
        raw_line_items=[
            {"description": "Widget A", "quantity": "10", "unit_price": "25.00", "amount": "250.00"},
        ],
    )


def _make_structured() -> StructuredInvoice:
    return StructuredInvoice(
        invoice_id=_TEST_INVOICE_ID,
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


def _make_qa_result(*, overall_confidence: float = 0.92, routed_to_review: bool = False) -> QAResult:
    return QAResult(
        invoice_id=_TEST_INVOICE_ID,
        overall_confidence=overall_confidence,
        routed_to_review=routed_to_review,
        approved=not routed_to_review,
        field_scores=[
            FieldScore(field_name="invoice_number", confidence=overall_confidence),
            FieldScore(field_name="invoice_date", confidence=overall_confidence),
            FieldScore(field_name="due_date", confidence=overall_confidence),
            FieldScore(field_name="vendor_name", confidence=overall_confidence),
            FieldScore(field_name="vendor_address", confidence=overall_confidence),
            FieldScore(field_name="currency", confidence=1.0),
            FieldScore(field_name="subtotal", confidence=overall_confidence),
            FieldScore(field_name="tax", confidence=overall_confidence),
            FieldScore(field_name="total", confidence=overall_confidence),
            FieldScore(field_name="line_items", confidence=overall_confidence),
        ],
        flags=[],
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path=db_path)
    db.init_db()
    db.add_client("test_client", "Test Client")
    return db


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 minimal pdf content")
    return pdf_path


# ── Integration tests with mocked agents ──────────────────────────────


class TestProcessInvoice:
    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_returns_qa_result(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """process_invoice returns a QAResult."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.return_value = _make_structured()
        mock_score.return_value = _make_qa_result()

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        result = process_invoice(pdf_path, "test_client")

        assert isinstance(result, QAResult)
        assert result.overall_confidence >= 0.85

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_high_confidence_status_qa_complete(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """A high-confidence invoice gets status 'qa_complete'."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.return_value = _make_structured()
        mock_score.return_value = _make_qa_result(overall_confidence=0.92, routed_to_review=False)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        process_invoice(pdf_path, "test_client")

        # Check that update_invoice_status was called with "qa_complete"
        status_calls = [
            call for call in mock_db.update_invoice_status.call_args_list
            if call[0][1] == "qa_complete"
        ]
        assert len(status_calls) == 1

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_low_confidence_status_review_required(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """A low-confidence invoice gets status 'review_required'."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.return_value = _make_structured()
        mock_score.return_value = _make_qa_result(overall_confidence=0.60, routed_to_review=True)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        process_invoice(pdf_path, "test_client")

        status_calls = [
            call for call in mock_db.update_invoice_status.call_args_list
            if call[0][1] == "review_required"
        ]
        assert len(status_calls) == 1

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_vision_failure_records_failed_status(
        self, mock_ingest, mock_extract, MockDB, tmp_path
    ):
        """If vision agent fails, database shows 'failed' status."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.side_effect = RuntimeError("Vision API failed")

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        with pytest.raises(RuntimeError, match="Vision API failed"):
            process_invoice(pdf_path, "test_client")

        # Check that update_invoice_status was called with "failed"
        mock_db.update_invoice_status.assert_called_with(
            str(_TEST_INGEST_ID), "failed"
        )

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_all_stages_logged(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """Processing log shows entries for each stage with timing."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.return_value = _make_structured()
        mock_score.return_value = _make_qa_result()

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        process_invoice(pdf_path, "test_client")

        # Should have 4 processing runs: ingest, vision, structuring, qa
        assert mock_db.add_processing_run.call_count == 4
        agent_names = [
            call[1]["agent_name"]
            for call in mock_db.add_processing_run.call_args_list
        ]
        assert agent_names == ["ingest", "vision_extraction", "structuring", "qa"]

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_qa_result_stored_in_database(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """QA results are stored in the database."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.return_value = _make_structured()
        mock_score.return_value = _make_qa_result()

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        process_invoice(pdf_path, "test_client")

        mock_db.add_qa_result.assert_called_once()
        call_kwargs = mock_db.add_qa_result.call_args
        assert call_kwargs[1]["client_id"] == "test_client"
        assert call_kwargs[1]["overall_score"] >= 0.85

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.score_invoice")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_pipeline_calls_in_sequence(
        self, mock_ingest, mock_extract, mock_structure, mock_score, MockDB, tmp_path
    ):
        """Pipeline calls agents in strict order: ingest -> extract -> structure -> qa."""
        call_order = []
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.side_effect = lambda *a, **kw: (call_order.append("ingest"), _make_ingest_result())[1]
        mock_extract.side_effect = lambda *a, **kw: (call_order.append("extract"), _make_extraction())[1]
        mock_structure.side_effect = lambda *a, **kw: (call_order.append("structure"), _make_structured())[1]
        mock_score.side_effect = lambda *a, **kw: (call_order.append("qa"), _make_qa_result())[1]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        process_invoice(pdf_path, "test_client")

        assert call_order == ["ingest", "extract", "structure", "qa"]

    @patch("src.pipeline.orchestrator.DatabaseManager")
    @patch("src.pipeline.orchestrator.structure_invoice")
    @patch("src.pipeline.orchestrator.extract_from_pages")
    @patch("src.pipeline.orchestrator.ingest_pdf")
    def test_structuring_failure_records_failed(
        self, mock_ingest, mock_extract, mock_structure, MockDB, tmp_path
    ):
        """If structuring agent fails, database shows 'failed' status."""
        mock_db = MagicMock()
        MockDB.return_value = mock_db
        mock_db.add_processing_run.return_value = 1

        mock_ingest.return_value = _make_ingest_result()
        mock_extract.return_value = _make_extraction()
        mock_structure.side_effect = ValueError("Structuring failed")

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        with pytest.raises(ValueError, match="Structuring failed"):
            process_invoice(pdf_path, "test_client")

        failed_calls = [
            call for call in mock_db.update_invoice_status.call_args_list
            if call[0][1] == "failed"
        ]
        assert len(failed_calls) == 1

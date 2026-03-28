"""Tests for the delivery module — CSV + JSON output generation."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from src.models.delivery import DeliveryResult
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.models.qa import FieldScore, QAResult
from src.pipeline.delivery import deliver_results


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def sample_invoice() -> StructuredInvoice:
    """A realistic two-line-item invoice for delivery tests."""
    invoice_id = uuid4()
    return StructuredInvoice(
        invoice_id=invoice_id,
        invoice_number="INV-2026-0042",
        invoice_date="2026-03-15",
        due_date="2026-04-15",
        vendor_name="Acme Industrial Supply Co.",
        vendor_address="123 Main St, Springfield, IL 62704",
        currency="USD",
        subtotal=Decimal("350.00"),
        tax=Decimal("28.00"),
        total=Decimal("378.00"),
        line_items=[
            LineItem(
                description="Widget A - Standard",
                quantity=Decimal("10"),
                unit_price=Decimal("20.00"),
                amount=Decimal("200.00"),
                source_page=1,
            ),
            LineItem(
                description="Widget B - Premium",
                quantity=Decimal("5"),
                unit_price=Decimal("30.00"),
                amount=Decimal("150.00"),
                source_page=1,
            ),
        ],
        metadata=InvoiceMetadata(
            source_file="data/clients/test_client/inbox/inv-042.pdf",
            ingest_id=invoice_id,
        ),
    )


@pytest.fixture()
def approved_qa_result(sample_invoice: StructuredInvoice) -> QAResult:
    """A QA result that passed — ready for delivery."""
    return QAResult(
        invoice_id=sample_invoice.invoice_id,
        overall_confidence=0.95,
        field_scores=[
            FieldScore(field_name="invoice_number", confidence=0.98, source_page=1),
            FieldScore(field_name="vendor_name", confidence=0.92, source_page=1),
            FieldScore(field_name="total", confidence=0.96, source_page=1),
        ],
        flags=[],
        routed_to_review=False,
        approved=True,
    )


@pytest.fixture()
def review_qa_result(sample_invoice: StructuredInvoice) -> QAResult:
    """A QA result routed to review and NOT yet approved."""
    return QAResult(
        invoice_id=sample_invoice.invoice_id,
        overall_confidence=0.60,
        field_scores=[],
        flags=[],
        routed_to_review=True,
        approved=False,
    )


@pytest.fixture()
def review_approved_qa_result(sample_invoice: StructuredInvoice) -> QAResult:
    """A QA result that was routed to review but then approved by a human."""
    return QAResult(
        invoice_id=sample_invoice.invoice_id,
        overall_confidence=0.70,
        field_scores=[],
        flags=[],
        routed_to_review=True,
        approved=True,
    )


@pytest.fixture()
def _set_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point config.data_dir to a temp directory for test isolation."""
    monkeypatch.setattr("src.config.config.data_dir", tmp_path)


# ── Core delivery tests ─────────────────────────────────────────


@pytest.mark.usefixtures("_set_data_dir")
class TestDeliverResults:
    """Tests for deliver_results()."""

    def test_creates_csv_and_json(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """deliver_results creates both CSV and JSON files."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")

        assert Path(result.csv_path).exists()
        assert Path(result.json_path).exists()

    def test_files_in_correct_directory(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """Output files are in data/clients/{client_id}/processed/{invoice_id}/."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")

        invoice_id = str(sample_invoice.invoice_id)
        expected_dir = tmp_path / "clients" / "test_client" / "processed" / invoice_id
        assert Path(result.csv_path).parent == expected_dir
        assert Path(result.json_path).parent == expected_dir

    def test_delivery_result_fields(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """DeliveryResult has correct metadata."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")

        assert isinstance(result, DeliveryResult)
        assert result.invoice_id == sample_invoice.invoice_id
        assert result.success is True
        assert result.error_message is None
        assert result.record_count == 2

    def test_rejects_unapproved_invoice(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        review_qa_result: QAResult,
    ) -> None:
        """Constitution [C3]: No output for invoices that haven't passed QA."""
        with pytest.raises(ValueError, match="has not passed QA"):
            deliver_results(review_qa_result, sample_invoice, "test_client")

    def test_allows_review_then_approved(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        review_approved_qa_result: QAResult,
    ) -> None:
        """Invoices routed to review but then approved CAN be delivered."""
        result = deliver_results(review_approved_qa_result, sample_invoice, "test_client")
        assert result.success is True


# ── CSV-specific tests ───────────────────────────────────────────


@pytest.mark.usefixtures("_set_data_dir")
class TestCSVOutput:
    """Tests for the generated CSV file."""

    def _deliver_and_read_csv(
        self,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> list[dict[str, str]]:
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        with Path(result.csv_path).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def test_csv_header_columns(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """CSV has all required columns in the correct order."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        with Path(result.csv_path).open(encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "invoice_number", "invoice_date", "due_date", "vendor_name",
            "po_number", "line_description", "line_quantity", "line_unit_price",
            "line_amount", "subtotal", "tax", "total", "currency",
            "confidence_score",
        ]
        assert header == expected

    def test_csv_row_count_equals_line_items(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """CSV row count = number of line items."""
        rows = self._deliver_and_read_csv(sample_invoice, approved_qa_result)
        assert len(rows) == len(sample_invoice.line_items)

    def test_csv_header_data_repeated_per_row(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """Invoice header data is repeated on every row for Excel filtering."""
        rows = self._deliver_and_read_csv(sample_invoice, approved_qa_result)
        for row in rows:
            assert row["invoice_number"] == "INV-2026-0042"
            assert row["vendor_name"] == "Acme Industrial Supply Co."
            assert row["total"] == "378.00"
            assert row["currency"] == "USD"

    def test_csv_line_item_data(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """CSV rows contain correct line-item data."""
        rows = self._deliver_and_read_csv(sample_invoice, approved_qa_result)
        assert rows[0]["line_description"] == "Widget A - Standard"
        assert rows[0]["line_quantity"] == "10"
        assert rows[0]["line_unit_price"] == "20.00"
        assert rows[0]["line_amount"] == "200.00"
        assert rows[1]["line_description"] == "Widget B - Premium"

    def test_csv_utf8_bom(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """CSV file starts with UTF-8 BOM for Excel compatibility."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        raw = Path(result.csv_path).read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"

    def test_csv_confidence_score(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """CSV includes the overall confidence score."""
        rows = self._deliver_and_read_csv(sample_invoice, approved_qa_result)
        assert rows[0]["confidence_score"] == "0.95"


# ── JSON-specific tests ─────────────────────────────────────────


@pytest.mark.usefixtures("_set_data_dir")
class TestJSONOutput:
    """Tests for the generated JSON file."""

    def test_json_is_valid(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """JSON output is valid and parseable."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        content = Path(result.json_path).read_text(encoding="utf-8")
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_json_roundtrip_to_model(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """JSON can be parsed back into a StructuredInvoice."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        content = Path(result.json_path).read_text(encoding="utf-8")
        restored = StructuredInvoice.model_validate_json(content)
        assert restored.invoice_id == sample_invoice.invoice_id
        assert restored.invoice_number == sample_invoice.invoice_number
        assert len(restored.line_items) == len(sample_invoice.line_items)

    def test_json_is_indented(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """JSON output uses indent=2 for readability."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        content = Path(result.json_path).read_text(encoding="utf-8")
        # Indented JSON will have lines starting with spaces
        assert "\n  " in content

    def test_json_contains_all_fields(
        self,
        tmp_path: Path,
        sample_invoice: StructuredInvoice,
        approved_qa_result: QAResult,
    ) -> None:
        """JSON contains all expected top-level fields."""
        result = deliver_results(approved_qa_result, sample_invoice, "test_client")
        data = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        for field in ["invoice_id", "invoice_number", "vendor_name", "total", "line_items"]:
            assert field in data

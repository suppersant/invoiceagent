"""Validation tests for all InvoiceAgent data models."""

from __future__ import annotations

import datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from src.models.delivery import DeliveryFormat, DeliveryResult
from src.models.extraction import (
    IngestResult,
    PageImage,
    SourceRegion,
    VisionExtractionResult,
)
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.models.qa import FieldScore, IssueType, QAFlag, QAResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**overrides) -> InvoiceMetadata:
    defaults = {
        "source_file": "/data/clients/acme/inbox/inv-001.pdf",
        "ingest_id": uuid4(),
        "page_count": 2,
    }
    defaults.update(overrides)
    return InvoiceMetadata(**defaults)


def _make_line_item(**overrides) -> LineItem:
    defaults = {
        "description": "Widget A",
        "quantity": Decimal("10"),
        "unit_price": Decimal("5.00"),
        "amount": Decimal("50.00"),
        "source_page": 1,
    }
    defaults.update(overrides)
    return LineItem(**defaults)


def _make_structured_invoice(**overrides) -> StructuredInvoice:
    defaults = {
        "invoice_number": "INV-2026-001",
        "invoice_date": datetime.date(2026, 3, 15),
        "vendor_name": "Acme Supplies",
        "total": Decimal("150.00"),
        "line_items": [_make_line_item()],
        "metadata": _make_metadata(),
    }
    defaults.update(overrides)
    return StructuredInvoice(**defaults)


# ---------------------------------------------------------------------------
# StructuredInvoice
# ---------------------------------------------------------------------------

class TestStructuredInvoice:
    def test_instantiation_with_required_fields(self):
        inv = _make_structured_invoice()
        assert inv.invoice_number == "INV-2026-001"
        assert inv.vendor_name == "Acme Supplies"
        assert inv.total == Decimal("150.00")
        assert isinstance(inv.invoice_id, UUID)

    def test_serialize_to_json(self):
        inv = _make_structured_invoice()
        json_str = inv.model_dump_json()
        assert "INV-2026-001" in json_str
        assert "150.00" in json_str

    def test_total_accepts_decimal(self):
        inv = _make_structured_invoice(total=Decimal("99.99"))
        assert inv.total == Decimal("99.99")

    def test_total_coerces_string_to_decimal(self):
        inv = _make_structured_invoice(total="250.75")
        assert inv.total == Decimal("250.75")

    def test_total_coerces_int_to_decimal(self):
        inv = _make_structured_invoice(total=100)
        assert isinstance(inv.total, Decimal)
        assert inv.total == Decimal("100")

    def test_currency_defaults_to_usd(self):
        inv = _make_structured_invoice()
        assert inv.currency == "USD"

    def test_currency_is_configurable(self):
        inv = _make_structured_invoice(currency="EUR")
        assert inv.currency == "EUR"

    def test_round_trip_json(self):
        inv = _make_structured_invoice()
        json_str = inv.model_dump_json()
        restored = StructuredInvoice.model_validate_json(json_str)
        assert restored.invoice_number == inv.invoice_number
        assert restored.total == inv.total
        assert restored.invoice_id == inv.invoice_id
        assert restored.metadata.ingest_id == inv.metadata.ingest_id


# ---------------------------------------------------------------------------
# LineItem
# ---------------------------------------------------------------------------

class TestLineItem:
    def test_required_fields(self):
        item = _make_line_item()
        assert item.description == "Widget A"
        assert item.quantity == Decimal("10")
        assert item.unit_price == Decimal("5.00")
        assert item.amount == Decimal("50.00")

    def test_optional_fields_default_none(self):
        item = _make_line_item()
        assert item.sku is None
        assert item.tax is None
        assert item.gl_code is None

    def test_optional_fields_populated(self):
        item = _make_line_item(
            sku="WDG-A-100",
            tax=Decimal("4.50"),
            gl_code="5100",
        )
        assert item.sku == "WDG-A-100"
        assert item.tax == Decimal("4.50")
        assert item.gl_code == "5100"

    def test_amount_matches_quantity_times_unit_price(self):
        item = _make_line_item(
            quantity=Decimal("3"),
            unit_price=Decimal("12.50"),
            amount=Decimal("37.50"),
        )
        expected = item.quantity * item.unit_price
        assert abs(item.amount - expected) < Decimal("0.01")

    def test_amount_within_rounding_tolerance(self):
        """amount ~ quantity * unit_price within 1 cent."""
        item = _make_line_item(
            quantity=Decimal("3"),
            unit_price=Decimal("33.33"),
            amount=Decimal("100.00"),  # 3 * 33.33 = 99.99, off by 0.01
        )
        expected = item.quantity * item.unit_price
        assert abs(item.amount - expected) <= Decimal("0.01")

    def test_round_trip_json(self):
        item = _make_line_item(sku="X-100", tax=Decimal("2.00"), gl_code="4200")
        json_str = item.model_dump_json()
        restored = LineItem.model_validate_json(json_str)
        assert restored.description == item.description
        assert restored.amount == item.amount
        assert restored.sku == item.sku


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------

class TestIngestResult:
    def test_auto_generates_uuid(self):
        result = IngestResult(source_file="/tmp/inv.pdf", page_count=1)
        assert isinstance(result.ingest_id, UUID)

    def test_two_instances_get_different_uuids(self):
        r1 = IngestResult(source_file="/tmp/a.pdf", page_count=1)
        r2 = IngestResult(source_file="/tmp/b.pdf", page_count=1)
        assert r1.ingest_id != r2.ingest_id

    def test_explicit_uuid(self):
        uid = uuid4()
        result = IngestResult(source_file="/tmp/inv.pdf", page_count=1, ingest_id=uid)
        assert result.ingest_id == uid

    def test_pages_default_empty(self):
        result = IngestResult(source_file="/tmp/inv.pdf", page_count=2)
        assert result.pages == []

    def test_with_pages(self):
        pages = [
            PageImage(page_number=1, image_path="/tmp/p1.png"),
            PageImage(page_number=2, image_path="/tmp/p2.png"),
        ]
        result = IngestResult(source_file="/tmp/inv.pdf", page_count=2, pages=pages)
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1

    def test_round_trip_json(self):
        result = IngestResult(
            source_file="/tmp/inv.pdf",
            page_count=1,
            pages=[PageImage(page_number=1, image_path="/tmp/p1.png", width=800, height=1200)],
        )
        json_str = result.model_dump_json()
        restored = IngestResult.model_validate_json(json_str)
        assert restored.ingest_id == result.ingest_id
        assert restored.pages[0].width == 800


# ---------------------------------------------------------------------------
# VisionExtractionResult
# ---------------------------------------------------------------------------

class TestVisionExtractionResult:
    def test_minimal_creation(self):
        uid = uuid4()
        result = VisionExtractionResult(ingest_id=uid)
        assert result.ingest_id == uid
        assert result.currency == "USD"

    def test_full_creation(self):
        uid = uuid4()
        result = VisionExtractionResult(
            ingest_id=uid,
            vendor_name="Acme",
            invoice_number="INV-001",
            invoice_date=datetime.date(2026, 3, 1),
            total=Decimal("500.00"),
            source_regions=[
                SourceRegion(page_number=1, text_snippet="Total: $500.00"),
            ],
        )
        assert result.vendor_name == "Acme"
        assert result.total == Decimal("500.00")
        assert len(result.source_regions) == 1

    def test_round_trip_json(self):
        uid = uuid4()
        result = VisionExtractionResult(
            ingest_id=uid,
            vendor_name="Test Vendor",
            invoice_number="T-100",
            total=Decimal("1234.56"),
        )
        json_str = result.model_dump_json()
        restored = VisionExtractionResult.model_validate_json(json_str)
        assert restored.ingest_id == uid
        assert restored.total == Decimal("1234.56")


# ---------------------------------------------------------------------------
# SourceRegion
# ---------------------------------------------------------------------------

class TestSourceRegion:
    def test_creation(self):
        sr = SourceRegion(page_number=2, text_snippet="Invoice #12345")
        assert sr.page_number == 2
        assert sr.text_snippet == "Invoice #12345"

    def test_page_number_must_be_positive(self):
        with pytest.raises(Exception):
            SourceRegion(page_number=0, text_snippet="bad")


# ---------------------------------------------------------------------------
# QAResult
# ---------------------------------------------------------------------------

class TestQAResult:
    def test_confidence_in_valid_range(self):
        uid = uuid4()
        qa = QAResult(invoice_id=uid, overall_confidence=0.92)
        assert 0.0 <= qa.overall_confidence <= 1.0

    def test_confidence_rejects_above_1(self):
        with pytest.raises(Exception):
            QAResult(invoice_id=uuid4(), overall_confidence=1.5)

    def test_confidence_rejects_below_0(self):
        with pytest.raises(Exception):
            QAResult(invoice_id=uuid4(), overall_confidence=-0.1)

    def test_flags(self):
        qa = QAResult(
            invoice_id=uuid4(),
            overall_confidence=0.75,
            flags=[
                QAFlag(
                    field_name="invoice_date",
                    issue_type=IssueType.LOW_CONFIDENCE,
                    message="Date extraction confidence below threshold",
                ),
                QAFlag(
                    field_name="vendor_address",
                    issue_type=IssueType.MISSING,
                    message="Vendor address not found",
                ),
            ],
        )
        assert len(qa.flags) == 2
        assert qa.flags[0].issue_type == IssueType.LOW_CONFIDENCE

    def test_field_scores(self):
        qa = QAResult(
            invoice_id=uuid4(),
            overall_confidence=0.95,
            field_scores=[
                FieldScore(field_name="total", confidence=0.99, source_page=1),
                FieldScore(field_name="vendor_name", confidence=0.91),
            ],
        )
        assert qa.field_scores[0].confidence == 0.99
        assert qa.field_scores[1].source_page is None

    def test_round_trip_json(self):
        qa = QAResult(
            invoice_id=uuid4(),
            overall_confidence=0.88,
            approved=True,
            flags=[
                QAFlag(
                    field_name="due_date",
                    issue_type=IssueType.INCONSISTENT,
                    message="Due date before invoice date",
                ),
            ],
            field_scores=[
                FieldScore(field_name="total", confidence=0.97),
            ],
        )
        json_str = qa.model_dump_json()
        restored = QAResult.model_validate_json(json_str)
        assert restored.overall_confidence == qa.overall_confidence
        assert restored.flags[0].issue_type == IssueType.INCONSISTENT
        assert restored.approved is True


# ---------------------------------------------------------------------------
# DeliveryResult
# ---------------------------------------------------------------------------

class TestDeliveryResult:
    def test_creation(self):
        dr = DeliveryResult(
            invoice_id=uuid4(),
            csv_path="/data/clients/acme/processed/inv-001.csv",
            json_path="/data/clients/acme/processed/inv-001.json",
            record_count=5,
        )
        assert dr.success is True
        assert dr.record_count == 5

    def test_failure_result(self):
        dr = DeliveryResult(
            invoice_id=uuid4(),
            csv_path="/data/clients/acme/processed/inv-001.csv",
            json_path="/data/clients/acme/processed/inv-001.json",
            success=False,
            error_message="Permission denied",
        )
        assert dr.success is False
        assert dr.error_message == "Permission denied"

    def test_round_trip_json(self):
        dr = DeliveryResult(
            invoice_id=uuid4(),
            csv_path="/tmp/out.csv",
            json_path="/tmp/out.json",
            record_count=3,
        )
        json_str = dr.model_dump_json()
        restored = DeliveryResult.model_validate_json(json_str)
        assert restored.invoice_id == dr.invoice_id


# ---------------------------------------------------------------------------
# Cross-model: full pipeline round-trip
# ---------------------------------------------------------------------------

class TestPipelineRoundTrip:
    def test_all_models_round_trip(self):
        """Every model survives model_dump_json -> model_validate_json."""
        ingest_id = uuid4()
        invoice_id = uuid4()

        models = [
            IngestResult(source_file="/tmp/inv.pdf", page_count=1),
            VisionExtractionResult(ingest_id=ingest_id, total=Decimal("100.00")),
            _make_structured_invoice(),
            QAResult(invoice_id=invoice_id, overall_confidence=0.90),
            DeliveryResult(
                invoice_id=invoice_id,
                csv_path="/tmp/out.csv",
                json_path="/tmp/out.json",
            ),
        ]
        for model in models:
            json_str = model.model_dump_json()
            restored = type(model).model_validate_json(json_str)
            assert restored.model_dump() == model.model_dump(), (
                f"Round-trip failed for {type(model).__name__}"
            )

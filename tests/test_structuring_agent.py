"""Tests for the structuring agent — covers all T3.2 acceptance criteria."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.structuring_agent import (
    _build_invoice_from_response,
    _parse_date,
    _parse_decimal,
    structure_invoice,
)
from src.models.extraction import VisionExtractionResult
from src.models.invoice import StructuredInvoice


# ── Date parsing ──────────────────────────────────────────────────────────


class TestParseDate:
    def test_long_month_format(self):
        """'March 15, 2026' -> 2026-03-15"""
        assert _parse_date("March 15, 2026") == date(2026, 3, 15)

    def test_mm_dd_yy_format(self):
        """'03/15/26' -> 2026-03-15"""
        assert _parse_date("03/15/26") == date(2026, 3, 15)

    def test_dd_mon_yyyy_format(self):
        """'15-Mar-2026' -> 2026-03-15"""
        assert _parse_date("15-Mar-2026") == date(2026, 3, 15)

    def test_iso_format(self):
        assert _parse_date("2026-03-15") == date(2026, 3, 15)

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not a date") is None


# ── Amount parsing ────────────────────────────────────────────────────────


class TestParseDecimal:
    def test_dollar_with_commas(self):
        """'$1,234.56' -> Decimal('1234.56')"""
        assert _parse_decimal("$1,234.56") == Decimal("1234.56")

    def test_amount_with_currency_code(self):
        """'1234.56 USD' -> Decimal('1234.56')"""
        assert _parse_decimal("1234.56 USD") == Decimal("1234.56")

    def test_plain_number(self):
        assert _parse_decimal("100.00") == Decimal("100.00")

    def test_none_returns_none(self):
        assert _parse_decimal(None) is None

    def test_empty_returns_none(self):
        assert _parse_decimal("") is None

    def test_quantized_to_two_decimals(self):
        result = _parse_decimal("99.9")
        assert result == Decimal("99.90")
        assert result.as_tuple().exponent == -2


# ── Missing fields produce None ──────────────────────────────────────────


_TEST_INGEST_ID = uuid4()


def _make_extraction(**overrides) -> VisionExtractionResult:
    defaults = dict(ingest_id=_TEST_INGEST_ID)
    defaults.update(overrides)
    return VisionExtractionResult(**defaults)


class TestMissingFieldsAreNone:
    def test_missing_due_date_is_none(self):
        """Missing due_date results in None, not a fabricated date."""
        extraction = _make_extraction()
        response_json = {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "due_date": None,
            "vendor_name": "Acme Corp",
            "total": "100.00",
            "line_items": [],
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert invoice.due_date is None

    def test_missing_optional_fields_default_to_none(self):
        extraction = _make_extraction()
        invoice = _build_invoice_from_response({}, extraction)
        assert invoice.vendor_address is None
        assert invoice.subtotal is None
        assert invoice.tax is None


# ── Line item sum validation ─────────────────────────────────────────────


class TestLineItemSumValidation:
    def test_flags_discrepancy(self):
        """Flags discrepancy if line items don't sum to subtotal."""
        extraction = _make_extraction()
        response_json = {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "vendor_name": "Acme",
            "total": "350.00",
            "line_items": [
                {"description": "Widget A", "amount": "100.00", "quantity": "1", "unit_price": "100.00", "source_page": 1},
                {"description": "Widget B", "amount": "200.00", "quantity": "1", "unit_price": "200.00", "source_page": 1},
            ],
            "subtotal": "350.00",  # actual sum is 300
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert any("line_item_sum_mismatch" in f for f in invoice.confidence_flags)

    def test_no_flag_when_sum_matches(self):
        extraction = _make_extraction()
        response_json = {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "vendor_name": "Acme",
            "total": "300.00",
            "line_items": [
                {"description": "Widget A", "amount": "100.00", "quantity": "1", "unit_price": "100.00", "source_page": 1},
                {"description": "Widget B", "amount": "200.00", "quantity": "1", "unit_price": "200.00", "source_page": 1},
            ],
            "subtotal": "300.00",
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert not any("line_item_sum_mismatch" in f for f in invoice.confidence_flags)

    def test_no_flag_within_tolerance(self):
        extraction = _make_extraction()
        response_json = {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "vendor_name": "Acme",
            "total": "100.04",
            "line_items": [
                {"description": "Widget A", "amount": "100.00", "quantity": "1", "unit_price": "100.00", "source_page": 1},
            ],
            "subtotal": "100.04",  # within 0.05 tolerance
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert not any("line_item_sum_mismatch" in f for f in invoice.confidence_flags)


# ── Pydantic validation ──────────────────────────────────────────────────


class TestPydanticValidation:
    def test_output_is_valid_structured_invoice(self):
        extraction = _make_extraction()
        response_json = {
            "invoice_number": "INV-2026-001",
            "invoice_date": "2026-03-15",
            "due_date": "2026-04-15",
            "vendor_name": "Acme Corp",
            "vendor_address": "123 Main St",
            "line_items": [
                {
                    "description": "Widget A",
                    "quantity": "10",
                    "unit_price": "25.00",
                    "amount": "250.00",
                    "source_page": 1,
                },
            ],
            "subtotal": "250.00",
            "tax": "20.00",
            "total": "270.00",
            "currency": "USD",
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert isinstance(invoice, StructuredInvoice)
        assert invoice.invoice_number == "INV-2026-001"
        assert invoice.invoice_date == date(2026, 3, 15)
        assert invoice.line_items[0].amount == Decimal("250.00")
        assert invoice.total == Decimal("270.00")


# ── raw_extraction preserved ─────────────────────────────────────────────


class TestRawExtractionPreserved:
    def test_raw_extraction_dict_preserved(self):
        extraction = _make_extraction(vendor_name="Test Vendor")
        response_json = {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "vendor_name": "Test Vendor",
            "total": "100.00",
        }
        invoice = _build_invoice_from_response(response_json, extraction)
        assert isinstance(invoice.raw_extraction, dict)
        assert invoice.raw_extraction["ingest_id"] == str(_TEST_INGEST_ID)
        assert invoice.raw_extraction["vendor_name"] == "Test Vendor"


# ── Full integration with mocked Claude API ──────────────────────────────


class TestStructureInvoice:
    def _make_extraction(self) -> VisionExtractionResult:
        return VisionExtractionResult(
            ingest_id=_TEST_INGEST_ID,
            invoice_number="INV-001",
            invoice_date=date(2026, 3, 15),
            due_date=date(2026, 4, 15),
            vendor_name="Acme Corp",
            total=Decimal("108.00"),
            subtotal=Decimal("100.00"),
            tax=Decimal("8.00"),
            raw_line_items=[
                {"description": "Widget", "quantity": "5", "unit_price": "20.00", "amount": "100.00"},
            ],
        )

    def _mock_response(self) -> dict:
        return {
            "invoice_number": "INV-001",
            "invoice_date": "2026-03-15",
            "due_date": "2026-04-15",
            "vendor_name": "Acme Corp",
            "vendor_address": None,
            "line_items": [
                {"description": "Widget", "quantity": "5.00", "unit_price": "20.00", "amount": "100.00", "source_page": 1},
            ],
            "subtotal": "100.00",
            "tax": "8.00",
            "total": "108.00",
            "currency": "USD",
            "notes": None,
            "confidence_flags": [],
        }

    @patch("src.agents.structuring_agent.AnthropicWrapper")
    def test_structure_invoice_success(self, MockWrapper):
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(self._mock_response())

        extraction = self._make_extraction()
        invoice = structure_invoice(extraction)

        assert isinstance(invoice, StructuredInvoice)
        assert invoice.invoice_number == "INV-001"
        assert invoice.invoice_date == date(2026, 3, 15)
        assert invoice.due_date == date(2026, 4, 15)
        assert invoice.total == Decimal("108.00")
        assert invoice.line_items[0].amount == Decimal("100.00")
        assert invoice.raw_extraction["ingest_id"] == str(_TEST_INGEST_ID)
        assert len(invoice.confidence_flags) == 0

    @patch("src.agents.structuring_agent.AnthropicWrapper")
    def test_structure_invoice_handles_markdown_fences(self, MockWrapper):
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        fenced = f"```json\n{json.dumps(self._mock_response())}\n```"
        mock_instance.complete.return_value = fenced

        invoice = structure_invoice(self._make_extraction())
        assert invoice.invoice_number == "INV-001"

    @patch("src.agents.structuring_agent.AnthropicWrapper")
    def test_structure_invoice_handles_bad_json(self, MockWrapper):
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = "not valid json {{{"

        invoice = structure_invoice(self._make_extraction())
        assert any("structuring_parse_error" in f for f in invoice.confidence_flags)
        assert invoice.raw_extraction["ingest_id"] == str(_TEST_INGEST_ID)

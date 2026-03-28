"""Tests for the QA confidence scoring agent — covers all T3.3 acceptance criteria."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.qa_agent import (
    _build_field_scores,
    _build_flags,
    _parse_response,
    score_invoice,
)
from src.models.extraction import VisionExtractionResult
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.models.qa import FieldScore, IssueType, QAFlag, QAResult


# ── Fixtures ────────────────────────────────────────────────────────────

_TEST_INGEST_ID = uuid4()
_TEST_INVOICE_ID = uuid4()


def _make_extraction(**overrides) -> VisionExtractionResult:
    defaults = dict(
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
    defaults.update(overrides)
    return VisionExtractionResult(**defaults)


def _make_structured(**overrides) -> StructuredInvoice:
    defaults = dict(
        invoice_id=_TEST_INVOICE_ID,
        invoice_number="INV-001",
        invoice_date=date(2026, 3, 15),
        due_date=date(2026, 4, 15),
        vendor_name="Acme Corp",
        vendor_address="123 Main St, Springfield",
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
    defaults.update(overrides)
    return StructuredInvoice(**defaults)


def _high_confidence_response() -> dict:
    """Claude response for a well-extracted invoice."""
    return {
        "field_scores": {
            "invoice_number": {"confidence": 0.95, "reasoning": "Present, matches raw"},
            "invoice_date": {"confidence": 0.95, "reasoning": "Valid date, matches raw"},
            "due_date": {"confidence": 0.90, "reasoning": "Present, after invoice_date"},
            "vendor_name": {"confidence": 0.95, "reasoning": "Plausible business name"},
            "vendor_address": {"confidence": 0.90, "reasoning": "Full address present"},
            "currency": {"confidence": 1.0, "reasoning": "Standard USD"},
            "subtotal": {"confidence": 0.95, "reasoning": "Matches line item sum"},
            "tax": {"confidence": 0.90, "reasoning": "Reasonable tax amount"},
            "total": {"confidence": 0.95, "reasoning": "Equals subtotal + tax"},
            "line_items": {"confidence": 0.92, "reasoning": "All items valid"},
        },
        "flags": [],
    }


def _low_confidence_response() -> dict:
    """Claude response for an invoice with a missing vendor_name."""
    return {
        "field_scores": {
            "invoice_number": {"confidence": 0.90, "reasoning": "Present"},
            "invoice_date": {"confidence": 0.90, "reasoning": "Valid date"},
            "due_date": {"confidence": 0.85, "reasoning": "Present"},
            "vendor_name": {"confidence": 0.20, "reasoning": "Missing vendor name"},
            "vendor_address": {"confidence": 0.10, "reasoning": "Missing with vendor"},
            "currency": {"confidence": 1.0, "reasoning": "Default USD"},
            "subtotal": {"confidence": 0.85, "reasoning": "Present"},
            "tax": {"confidence": 0.85, "reasoning": "Present"},
            "total": {"confidence": 0.90, "reasoning": "Present"},
            "line_items": {"confidence": 0.85, "reasoning": "Items present"},
        },
        "flags": [
            {
                "field_name": "vendor_name",
                "issue_type": "missing",
                "message": "Vendor name is missing from the invoice",
            },
        ],
    }


def _line_item_mismatch_response() -> dict:
    """Claude response for an invoice where line items don't sum to total."""
    return {
        "field_scores": {
            "invoice_number": {"confidence": 0.90, "reasoning": "Present"},
            "invoice_date": {"confidence": 0.90, "reasoning": "Valid date"},
            "due_date": {"confidence": 0.85, "reasoning": "Present"},
            "vendor_name": {"confidence": 0.90, "reasoning": "Plausible name"},
            "vendor_address": {"confidence": 0.85, "reasoning": "Present"},
            "currency": {"confidence": 1.0, "reasoning": "USD"},
            "subtotal": {"confidence": 0.60, "reasoning": "Does not match line items"},
            "tax": {"confidence": 0.85, "reasoning": "Present"},
            "total": {"confidence": 0.70, "reasoning": "Inconsistent with subtotal"},
            "line_items": {"confidence": 0.85, "reasoning": "Items present"},
        },
        "flags": [
            {
                "field_name": "subtotal",
                "issue_type": "inconsistent",
                "message": "Line items sum to 250.00 but subtotal shows 300.00",
            },
        ],
    }


# ── Response parsing ───────────────────────────────────────────────────


class TestParseResponse:
    def test_parses_valid_json(self):
        response = json.dumps(_high_confidence_response())
        parsed = _parse_response(response, "test-id")
        assert "field_scores" in parsed
        assert "flags" in parsed

    def test_strips_markdown_fences(self):
        response = f"```json\n{json.dumps(_high_confidence_response())}\n```"
        parsed = _parse_response(response, "test-id")
        assert "field_scores" in parsed

    def test_bad_json_returns_empty(self):
        parsed = _parse_response("not json {{{", "test-id")
        assert parsed == {"field_scores": {}, "flags": []}

    def test_non_dict_returns_empty(self):
        parsed = _parse_response("[1, 2, 3]", "test-id")
        assert parsed == {"field_scores": {}, "flags": []}


# ── Field score building ──────────────────────────────────────────────


class TestBuildFieldScores:
    def test_all_scored_fields_present(self):
        structured = _make_structured()
        parsed = _high_confidence_response()
        scores = _build_field_scores(parsed, structured)
        field_names = {s.field_name for s in scores}
        expected = {
            "invoice_number", "invoice_date", "due_date", "vendor_name",
            "vendor_address", "currency", "subtotal", "tax", "total", "line_items",
        }
        assert field_names == expected

    def test_missing_field_in_response_gets_zero(self):
        structured = _make_structured()
        parsed = {"field_scores": {}, "flags": []}
        scores = _build_field_scores(parsed, structured)
        for score in scores:
            assert score.confidence == 0.0

    def test_confidence_clamped_to_valid_range(self):
        structured = _make_structured()
        parsed = {
            "field_scores": {
                "invoice_number": {"confidence": 1.5, "reasoning": "too high"},
                "total": {"confidence": -0.5, "reasoning": "too low"},
            },
            "flags": [],
        }
        scores = _build_field_scores(parsed, structured)
        scores_by_name = {s.field_name: s for s in scores}
        assert scores_by_name["invoice_number"].confidence == 1.0
        assert scores_by_name["total"].confidence == 0.0

    def test_source_page_from_line_items(self):
        structured = _make_structured()
        parsed = _high_confidence_response()
        scores = _build_field_scores(parsed, structured)
        li_score = next(s for s in scores if s.field_name == "line_items")
        assert li_score.source_page == 1


# ── Flag building ─────────────────────────────────────────────────────


class TestBuildFlags:
    def test_builds_valid_flags(self):
        parsed = _low_confidence_response()
        flags = _build_flags(parsed)
        assert len(flags) == 1
        assert flags[0].field_name == "vendor_name"
        assert flags[0].issue_type == IssueType.MISSING

    def test_invalid_issue_type_defaults_to_low_confidence(self):
        parsed = {
            "flags": [
                {"field_name": "total", "issue_type": "invalid_type", "message": "test"},
            ],
        }
        flags = _build_flags(parsed)
        assert flags[0].issue_type == IssueType.LOW_CONFIDENCE

    def test_non_dict_flags_skipped(self):
        parsed = {"flags": ["not a dict", 42]}
        flags = _build_flags(parsed)
        assert len(flags) == 0

    def test_empty_flags_list(self):
        parsed = {"flags": []}
        flags = _build_flags(parsed)
        assert len(flags) == 0

    def test_inconsistent_issue_type(self):
        parsed = _line_item_mismatch_response()
        flags = _build_flags(parsed)
        assert len(flags) == 1
        assert flags[0].issue_type == IssueType.INCONSISTENT


# ── Full integration with mocked Claude API ───────────────────────────


class TestScoreInvoice:
    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_high_confidence_passes_qa(self, MockWrapper):
        """A well-extracted invoice scores >= 0.90 and is not routed to review."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_high_confidence_response())

        structured = _make_structured()
        raw = _make_extraction()
        result = score_invoice(structured, raw)

        assert isinstance(result, QAResult)
        assert result.overall_confidence >= 0.90
        assert result.routed_to_review is False
        assert result.approved is True

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_missing_vendor_routes_to_review(self, MockWrapper):
        """An invoice with missing vendor_name scores below 0.85 and routes to review."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_low_confidence_response())

        structured = _make_structured(vendor_name="UNKNOWN")
        raw = _make_extraction(vendor_name=None)
        result = score_invoice(structured, raw)

        assert result.overall_confidence < 0.85
        assert result.routed_to_review is True
        assert result.approved is False

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_line_item_mismatch_generates_flag(self, MockWrapper):
        """An invoice where line items don't sum to total generates a QAFlag."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_line_item_mismatch_response())

        structured = _make_structured()
        raw = _make_extraction()
        result = score_invoice(structured, raw)

        assert len(result.flags) >= 1
        assert any(f.issue_type == IssueType.INCONSISTENT for f in result.flags)

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_routed_to_review_when_below_threshold(self, MockWrapper):
        """qa_result.routed_to_review is True when overall_confidence < threshold."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_low_confidence_response())

        result = score_invoice(_make_structured(), _make_extraction())
        assert result.routed_to_review is True

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_not_routed_when_above_threshold(self, MockWrapper):
        """qa_result.routed_to_review is False when overall_confidence >= threshold."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_high_confidence_response())

        result = score_invoice(_make_structured(), _make_extraction())
        assert result.routed_to_review is False

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_every_field_has_score(self, MockWrapper):
        """Every field in StructuredInvoice has a corresponding entry in field_scores."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_high_confidence_response())

        result = score_invoice(_make_structured(), _make_extraction())
        scored_fields = {fs.field_name for fs in result.field_scores}
        expected = {
            "invoice_number", "invoice_date", "due_date", "vendor_name",
            "vendor_address", "currency", "subtotal", "tax", "total", "line_items",
        }
        assert scored_fields == expected

    @patch("src.agents.qa_agent.config")
    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_custom_threshold_changes_routing(self, MockWrapper, mock_config):
        """Changing config.qa_confidence_threshold to 0.50 changes routing behavior."""
        mock_config.qa_confidence_threshold = 0.50
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance

        # Use the low-confidence response (min score 0.10 for vendor_address)
        # But with threshold 0.50, this should STILL route to review
        mock_instance.complete.return_value = json.dumps(_low_confidence_response())
        result = score_invoice(_make_structured(), _make_extraction())
        assert result.routed_to_review is True  # 0.10 < 0.50

        # Now use a response where all scores are >= 0.50 but < 0.85
        moderate_response = _high_confidence_response()
        for field in moderate_response["field_scores"]:
            moderate_response["field_scores"][field]["confidence"] = 0.60
        mock_instance.complete.return_value = json.dumps(moderate_response)

        result = score_invoice(_make_structured(), _make_extraction())
        assert result.routed_to_review is False  # 0.60 >= 0.50

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_overall_confidence_is_minimum(self, MockWrapper):
        """Overall confidence equals the minimum of all field scores."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance

        response = _high_confidence_response()
        # Set one field very low
        response["field_scores"]["vendor_address"]["confidence"] = 0.30
        mock_instance.complete.return_value = json.dumps(response)

        result = score_invoice(_make_structured(), _make_extraction())
        assert result.overall_confidence == 0.30

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_result_has_correct_invoice_id(self, MockWrapper):
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_high_confidence_response())

        structured = _make_structured()
        result = score_invoice(structured, _make_extraction())
        assert result.invoice_id == structured.invoice_id

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_handles_markdown_fenced_response(self, MockWrapper):
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        fenced = f"```json\n{json.dumps(_high_confidence_response())}\n```"
        mock_instance.complete.return_value = fenced

        result = score_invoice(_make_structured(), _make_extraction())
        assert isinstance(result, QAResult)
        assert result.overall_confidence >= 0.90

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_handles_bad_json_gracefully(self, MockWrapper):
        """Bad JSON response results in all-zero scores and routes to review."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = "not valid json {{{"

        result = score_invoice(_make_structured(), _make_extraction())
        assert result.overall_confidence == 0.0
        assert result.routed_to_review is True

    @patch("src.agents.qa_agent.AnthropicWrapper")
    def test_qa_result_serialization_roundtrip(self, MockWrapper):
        """QAResult survives JSON serialization and deserialization."""
        mock_instance = MagicMock()
        MockWrapper.return_value = mock_instance
        mock_instance.complete.return_value = json.dumps(_high_confidence_response())

        result = score_invoice(_make_structured(), _make_extraction())
        json_str = result.model_dump_json()
        restored = QAResult.model_validate_json(json_str)
        assert restored.overall_confidence == result.overall_confidence
        assert len(restored.field_scores) == len(result.field_scores)

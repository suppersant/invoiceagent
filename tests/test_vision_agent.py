"""Tests for the vision extraction agent."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image

from src.agents.prompts.vision_extraction import (
    VISION_EXTRACTION_SYSTEM_PROMPT,
    VISION_EXTRACTION_USER_MESSAGE,
)
from src.agents.vision_agent import (
    VisionExtractionError,
    _build_image_blocks,
    _parse_response,
    extract_from_pages,
)
from src.models.extraction import IngestResult, PageImage, VisionExtractionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_page_image(tmp_path: Path, page_number: int) -> PageImage:
    """Create a minimal PNG image file and return a PageImage model."""
    img = Image.new("RGB", (100, 100), color="white")
    img_path = tmp_path / f"page_{page_number}.png"
    img.save(img_path)
    return PageImage(
        page_number=page_number,
        image_path=str(img_path),
        width=100,
        height=100,
    )


def _make_ingest_result(tmp_path: Path, page_count: int = 1) -> IngestResult:
    """Create an IngestResult with the given number of page images."""
    pages = [_make_page_image(tmp_path, i + 1) for i in range(page_count)]
    return IngestResult(
        source_file="/fake/invoice.pdf",
        page_count=page_count,
        pages=pages,
    )


def _sample_api_response(num_items: int = 2, multi_page: bool = False) -> str:
    """Build a sample JSON response mimicking Claude Vision output."""
    items = []
    for i in range(num_items):
        items.append(
            {
                "description": f"Widget Type {chr(65 + i)}",
                "quantity": str(i + 1),
                "unit_price": f"{(i + 1) * 25}.00",
                "amount": f"{(i + 1) * 25}.00",
                "sku": f"WDG-{chr(65 + i)}",
                "page": 2 if multi_page and i >= num_items // 2 else 1,
            }
        )

    data: dict[str, Any] = {
        "vendor_name": {"value": "Acme Industrial Supply", "page": 1},
        "vendor_address": {"value": "123 Factory Ln, Detroit MI 48201", "page": 1},
        "invoice_number": {"value": "INV-2026-0042", "page": 1},
        "invoice_date": {"value": "2026-03-15", "page": 1},
        "due_date": {"value": "2026-04-14", "page": 1},
        "po_number": {"value": "PO-8810", "page": 1},
        "bill_to_name": {"value": "Baker Manufacturing", "page": 1},
        "bill_to_address": {"value": "456 Industrial Blvd, Toledo OH 43604", "page": 1},
        "subtotal": {"value": "75.00", "page": 2 if multi_page else 1},
        "tax": {"value": "6.00", "page": 2 if multi_page else 1},
        "total": {"value": "81.00", "page": 2 if multi_page else 1},
        "currency": {"value": "USD", "page": 1},
        "line_items": items,
        "raw_text": "Acme Industrial Supply\nINV-2026-0042\n...",
        "notes": [],
    }
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Tests — extract_from_pages
# ---------------------------------------------------------------------------


class TestExtractFromPages:
    """Tests for the main extract_from_pages function."""

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_returns_vision_extraction_result(self, mock_wrapper_cls, tmp_path):
        """extract_from_pages returns a valid VisionExtractionResult."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        ingest = _make_ingest_result(tmp_path)

        result = extract_from_pages(ingest)

        assert isinstance(result, VisionExtractionResult)
        assert result.ingest_id == ingest.ingest_id

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_extracts_vendor_name(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert result.vendor_name == "Acme Industrial Supply"

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_extracts_invoice_number(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert result.invoice_number == "INV-2026-0042"

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_extracts_dates(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert result.invoice_date == date(2026, 3, 15)
        assert result.due_date == date(2026, 4, 14)

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_extracts_totals_as_decimal(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert result.total == Decimal("81.00")
        assert result.subtotal == Decimal("75.00")
        assert result.tax == Decimal("6.00")

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_extracts_all_line_items(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response(
            num_items=3
        )
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert len(result.raw_line_items) == 3
        assert result.raw_line_items[0]["description"] == "Widget Type A"
        assert result.raw_line_items[0]["sku"] == "WDG-A"

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_source_regions_populated(self, mock_wrapper_cls, tmp_path):
        """Every extracted field has a corresponding SourceRegion."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert len(result.source_regions) > 0
        pages = {sr.page_number for sr in result.source_regions}
        assert 1 in pages

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_source_regions_include_page_numbers(self, mock_wrapper_cls, tmp_path):
        """Each SourceRegion includes a valid page_number."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        for sr in result.source_regions:
            assert sr.page_number >= 1
            assert sr.text_snippet  # not empty

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_multi_page_extracts_from_all_pages(self, mock_wrapper_cls, tmp_path):
        """Multi-page invoice extracts data from ALL pages."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response(
            num_items=4, multi_page=True
        )
        ingest = _make_ingest_result(tmp_path, page_count=2)
        result = extract_from_pages(ingest)

        # Should have source regions from both page 1 and page 2
        pages_found = {sr.page_number for sr in result.source_regions}
        assert 1 in pages_found
        assert 2 in pages_found

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_multi_page_sends_all_images_in_single_call(
        self, mock_wrapper_cls, tmp_path
    ):
        """All page images are sent in a single API call."""
        mock_instance = mock_wrapper_cls.return_value
        mock_instance.vision.return_value = _sample_api_response()
        ingest = _make_ingest_result(tmp_path, page_count=3)

        extract_from_pages(ingest)

        mock_instance.vision.assert_called_once()
        call_args = mock_instance.vision.call_args
        images = call_args.kwargs.get("images") or call_args[1]
        assert len(images) == 3

    def test_empty_pages_raises_error(self, tmp_path):
        """IngestResult with no pages raises VisionExtractionError."""
        ingest = IngestResult(
            source_file="/fake/invoice.pdf",
            page_count=1,
            pages=[],
        )
        with pytest.raises(VisionExtractionError, match="no page images"):
            extract_from_pages(ingest)

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_missing_image_file_raises_error(self, mock_wrapper_cls, tmp_path):
        """Missing page image file raises VisionExtractionError."""
        ingest = IngestResult(
            source_file="/fake/invoice.pdf",
            page_count=1,
            pages=[
                PageImage(
                    page_number=1,
                    image_path=str(tmp_path / "nonexistent.png"),
                )
            ],
        )
        with pytest.raises(VisionExtractionError, match="not found"):
            extract_from_pages(ingest)

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_api_error_raises_extraction_error(self, mock_wrapper_cls, tmp_path):
        """API failures are wrapped in VisionExtractionError."""
        mock_wrapper_cls.return_value.vision.side_effect = RuntimeError("API down")
        ingest = _make_ingest_result(tmp_path)
        with pytest.raises(VisionExtractionError, match="API call failed"):
            extract_from_pages(ingest)

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_does_not_validate_or_score(self, mock_wrapper_cls, tmp_path):
        """Agent does NOT attempt to validate or score data — only extracts."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        # VisionExtractionResult has no confidence_score field — that's the QA agent's job
        assert not hasattr(result, "confidence_score")

    @patch("src.agents.vision_agent.AnthropicWrapper")
    def test_raw_text_preserved(self, mock_wrapper_cls, tmp_path):
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        result = extract_from_pages(_make_ingest_result(tmp_path))
        assert result.raw_text is not None
        assert "Acme" in result.raw_text


# ---------------------------------------------------------------------------
# Tests — _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for JSON response parsing."""

    def test_parses_clean_json(self):
        raw = '{"vendor_name": {"value": "Test", "page": 1}}'
        result = _parse_response(raw, "test-id")
        assert result["vendor_name"]["value"] == "Test"

    def test_strips_markdown_fencing(self):
        raw = '```json\n{"total": {"value": "100", "page": 1}}\n```'
        result = _parse_response(raw, "test-id")
        assert result["total"]["value"] == "100"

    def test_invalid_json_raises_error(self):
        with pytest.raises(VisionExtractionError, match="parse"):
            _parse_response("not json at all", "test-id")

    def test_non_object_raises_error(self):
        with pytest.raises(VisionExtractionError, match="not a JSON object"):
            _parse_response("[1, 2, 3]", "test-id")


# ---------------------------------------------------------------------------
# Tests — _build_image_blocks
# ---------------------------------------------------------------------------


class TestBuildImageBlocks:
    """Tests for image encoding."""

    def test_encodes_png_as_base64(self, tmp_path):
        ingest = _make_ingest_result(tmp_path, page_count=1)
        blocks = _build_image_blocks(ingest)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["media_type"] == "image/png"
        assert blocks[0]["source"]["type"] == "base64"
        assert len(blocks[0]["source"]["data"]) > 0

    def test_encodes_multiple_pages(self, tmp_path):
        ingest = _make_ingest_result(tmp_path, page_count=3)
        blocks = _build_image_blocks(ingest)
        assert len(blocks) == 3

    def test_unsupported_format_raises_error(self, tmp_path):
        bmp_path = tmp_path / "page.bmp"
        Image.new("RGB", (10, 10)).save(bmp_path)
        ingest = IngestResult(
            source_file="/fake/invoice.pdf",
            page_count=1,
            pages=[PageImage(page_number=1, image_path=str(bmp_path))],
        )
        with pytest.raises(VisionExtractionError, match="Unsupported image format"):
            _build_image_blocks(ingest)


# ---------------------------------------------------------------------------
# Tests — prompt separation
# ---------------------------------------------------------------------------


class TestPromptSeparation:
    """Verify that the prompt is separate from agent logic."""

    def test_system_prompt_is_non_empty_string(self):
        assert isinstance(VISION_EXTRACTION_SYSTEM_PROMPT, str)
        assert len(VISION_EXTRACTION_SYSTEM_PROMPT) > 100

    def test_user_message_is_non_empty_string(self):
        assert isinstance(VISION_EXTRACTION_USER_MESSAGE, str)
        assert len(VISION_EXTRACTION_USER_MESSAGE) > 10

    def test_prompt_mentions_key_fields(self):
        """Prompt must instruct extraction of all required fields."""
        for field in (
            "vendor_name", "invoice_number", "invoice_date", "due_date",
            "subtotal", "tax", "total", "currency", "line_items",
        ):
            assert field in VISION_EXTRACTION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests — logging compliance (Constitution [D1])
# ---------------------------------------------------------------------------


class TestLoggingCompliance:
    """Verify that no financial data leaks into logs."""

    @patch("src.agents.vision_agent.AnthropicWrapper")
    @patch("src.agents.vision_agent._logger")
    def test_logs_duration_but_not_financial_data(
        self, mock_logger, mock_wrapper_cls, tmp_path
    ):
        """Log messages include duration_ms but not dollar amounts."""
        mock_wrapper_cls.return_value.vision.return_value = _sample_api_response()
        extract_from_pages(_make_ingest_result(tmp_path))

        # Find the "vision_extraction_complete" log call
        info_calls = mock_logger.info.call_args_list
        assert len(info_calls) > 0

        for call in info_calls:
            args = call[0]
            kwargs = call[1]
            # Should log duration
            if "extraction_duration_ms" in kwargs:
                assert isinstance(kwargs["extraction_duration_ms"], int)
            # Should NOT log any dollar amounts
            all_values = str(kwargs)
            assert "81.00" not in all_values
            assert "75.00" not in all_values

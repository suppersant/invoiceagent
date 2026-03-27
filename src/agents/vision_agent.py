"""Vision extraction agent — reads invoice page images via Claude Vision.

Extracts all visible text and structured data from invoice images.
Does NOT validate, score, or restructure data (that is downstream agents' job).
"""

from __future__ import annotations

import base64
import json
import time
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.agents.prompts.vision_extraction import (
    VISION_EXTRACTION_SYSTEM_PROMPT,
    VISION_EXTRACTION_USER_MESSAGE,
)
from src.models.extraction import IngestResult, SourceRegion, VisionExtractionResult
from src.utils.anthropic_client import AnthropicWrapper
from src.utils.logging import get_logger

_logger = get_logger("vision_agent")

# MIME types for images the Anthropic API accepts
_IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class VisionExtractionError(Exception):
    """Raised when the vision extraction agent fails."""


def extract_from_pages(ingest_result: IngestResult) -> VisionExtractionResult:
    """Extract invoice data from page images using Claude Vision.

    Sends all page images in a single API call (multi-page support) and
    parses the structured JSON response into a VisionExtractionResult.

    Args:
        ingest_result: The IngestResult from the ingestion pipeline containing
            page images to process.

    Returns:
        A VisionExtractionResult with all extracted fields and source regions.

    Raises:
        VisionExtractionError: If extraction fails (missing images, API error,
            or unparseable response).
    """
    if not ingest_result.pages:
        raise VisionExtractionError("IngestResult contains no page images")

    start_time = time.monotonic()

    # Build image content blocks for all pages
    image_blocks = _build_image_blocks(ingest_result)

    # Call Claude Vision
    client = AnthropicWrapper()
    try:
        raw_response = client.vision(
            system_prompt=VISION_EXTRACTION_SYSTEM_PROMPT,
            images=image_blocks,
            user_message=VISION_EXTRACTION_USER_MESSAGE,
            max_tokens=8192,
        )
    except Exception as exc:
        _logger.error(
            "vision_extraction_failed",
            invoice_id=str(ingest_result.ingest_id),
            page_count=ingest_result.page_count,
        )
        raise VisionExtractionError(f"Claude Vision API call failed: {exc}") from exc

    duration_ms = round((time.monotonic() - start_time) * 1000)

    # Parse the JSON response
    parsed = _parse_response(raw_response, str(ingest_result.ingest_id))

    # Build the result model
    result = _build_result(ingest_result, parsed)

    _logger.info(
        "vision_extraction_complete",
        invoice_id=str(ingest_result.ingest_id),
        page_count=ingest_result.page_count,
        extraction_duration_ms=duration_ms,
    )

    return result


def _build_image_blocks(ingest_result: IngestResult) -> list[dict[str, Any]]:
    """Encode each page image as a base64 content block for the API."""
    blocks: list[dict[str, Any]] = []

    for page in ingest_result.pages:
        image_path = Path(page.image_path)
        if not image_path.exists():
            raise VisionExtractionError(
                f"Page image not found: {image_path} (page {page.page_number})"
            )

        suffix = image_path.suffix.lower()
        media_type = _IMAGE_MIME_MAP.get(suffix)
        if media_type is None:
            raise VisionExtractionError(
                f"Unsupported image format '{suffix}' for page {page.page_number}"
            )

        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            }
        )

    return blocks


def _parse_response(raw_response: str, invoice_id: str) -> dict[str, Any]:
    """Parse the JSON response from Claude, stripping any markdown fencing."""
    text = raw_response.strip()

    # Strip markdown code fencing if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        _logger.error(
            "vision_response_parse_failed",
            invoice_id=invoice_id,
        )
        raise VisionExtractionError(
            f"Failed to parse vision extraction JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise VisionExtractionError("Vision extraction response is not a JSON object")

    return parsed


def _safe_decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal, returning None if not possible."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _safe_date(value: Any) -> date | None:
    """Parse a YYYY-MM-DD string to a date, returning None on failure."""
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _extract_value(field: Any) -> Any:
    """Extract the 'value' from a field dict, or return the field as-is."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def _extract_page(field: Any) -> int | None:
    """Extract the 'page' number from a field dict."""
    if isinstance(field, dict) and "page" in field:
        return field["page"]
    return None


def _build_result(
    ingest_result: IngestResult, parsed: dict[str, Any]
) -> VisionExtractionResult:
    """Map the parsed JSON dict into a VisionExtractionResult."""
    source_regions: list[SourceRegion] = []

    def _track(field_name: str, raw_field: Any) -> None:
        """Add a SourceRegion entry for a field if page info is available."""
        page = _extract_page(raw_field)
        value = _extract_value(raw_field)
        if page is not None and value is not None:
            source_regions.append(
                SourceRegion(
                    page_number=page,
                    text_snippet=f"{field_name}: {value}",
                )
            )

    # Track header fields
    for field_name in (
        "vendor_name", "vendor_address", "invoice_number", "invoice_date",
        "due_date", "po_number", "bill_to_name", "bill_to_address",
        "subtotal", "tax", "total", "currency",
    ):
        raw_field = parsed.get(field_name)
        if raw_field is not None:
            _track(field_name, raw_field)

    # Track line items
    raw_line_items: list[dict] = []
    for item in parsed.get("line_items", []):
        page = item.get("page")
        clean_item = {
            "description": item.get("description"),
            "quantity": item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "amount": item.get("amount"),
            "sku": item.get("sku"),
            "page": page,
        }
        raw_line_items.append(clean_item)
        if page is not None:
            source_regions.append(
                SourceRegion(
                    page_number=page,
                    text_snippet=f"line_item: {item.get('description', 'unknown')}",
                )
            )

    return VisionExtractionResult(
        ingest_id=ingest_result.ingest_id,
        vendor_name=_extract_value(parsed.get("vendor_name")),
        vendor_address=_extract_value(parsed.get("vendor_address")),
        invoice_number=_extract_value(parsed.get("invoice_number")),
        invoice_date=_safe_date(_extract_value(parsed.get("invoice_date"))),
        due_date=_safe_date(_extract_value(parsed.get("due_date"))),
        currency=_extract_value(parsed.get("currency")) or "USD",
        total=_safe_decimal(_extract_value(parsed.get("total"))),
        subtotal=_safe_decimal(_extract_value(parsed.get("subtotal"))),
        tax=_safe_decimal(_extract_value(parsed.get("tax"))),
        raw_line_items=raw_line_items,
        source_regions=source_regions,
        raw_text=parsed.get("raw_text"),
    )

"""Structuring agent — maps raw extraction output to StructuredInvoice."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from src.agents.prompts.structuring import STRUCTURING_SYSTEM_PROMPT
from src.models.extraction import VisionExtractionResult
from src.models.invoice import InvoiceMetadata, LineItem, StructuredInvoice
from src.utils.anthropic_client import AnthropicWrapper
from src.utils.logging import get_logger

_logger = get_logger("structuring_agent")


def _parse_date(value: Any) -> date | None:
    """Parse a date string in various formats into datetime.date."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    from dateutil import parser as dateutil_parser

    try:
        return dateutil_parser.parse(raw, dayfirst=False).date()
    except (ValueError, OverflowError):
        return None


def _parse_decimal(value: Any) -> Decimal | None:
    """Parse a monetary string into a two-decimal-place Decimal."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    # Strip currency symbols and whitespace
    for ch in ("$", "€", "£", "¥", ",", " "):
        raw = raw.replace(ch, "")
    # Remove trailing currency codes like "USD"
    for code in ("USD", "EUR", "GBP", "JPY", "CAD"):
        raw = raw.replace(code, "").strip()

    try:
        return Decimal(raw).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _build_raw_fields(extraction: VisionExtractionResult) -> dict[str, Any]:
    """Build a dict of raw fields from the VisionExtractionResult for the LLM."""
    fields: dict[str, Any] = {}
    if extraction.invoice_number:
        fields["invoice_number"] = extraction.invoice_number
    if extraction.invoice_date:
        fields["invoice_date"] = str(extraction.invoice_date)
    if extraction.due_date:
        fields["due_date"] = str(extraction.due_date)
    if extraction.vendor_name:
        fields["vendor_name"] = extraction.vendor_name
    if extraction.vendor_address:
        fields["vendor_address"] = extraction.vendor_address
    if extraction.currency:
        fields["currency"] = extraction.currency
    if extraction.total is not None:
        fields["total"] = str(extraction.total)
    if extraction.subtotal is not None:
        fields["subtotal"] = str(extraction.subtotal)
    if extraction.tax is not None:
        fields["tax"] = str(extraction.tax)
    if extraction.raw_line_items:
        fields["line_items"] = extraction.raw_line_items
    if extraction.raw_text:
        fields["raw_text"] = extraction.raw_text
    return fields


def _build_invoice_from_response(
    response_json: dict[str, Any],
    extraction: VisionExtractionResult,
) -> StructuredInvoice:
    """Convert Claude's JSON response into a validated StructuredInvoice."""
    line_items: list[LineItem] = []
    for item in response_json.get("line_items") or []:
        amount = _parse_decimal(item.get("amount"))
        quantity = _parse_decimal(item.get("quantity"))
        unit_price = _parse_decimal(item.get("unit_price"))
        if amount is None:
            continue
        line_items.append(
            LineItem(
                description=item.get("description", ""),
                quantity=quantity or Decimal("1.00"),
                unit_price=unit_price or amount,
                amount=amount,
                source_page=int(item.get("source_page", 1)),
            )
        )

    confidence_flags: list[str] = list(response_json.get("confidence_flags") or [])

    subtotal = _parse_decimal(response_json.get("subtotal"))
    if subtotal is not None and line_items:
        items_sum = sum((li.amount for li in line_items), Decimal("0.00"))
        if abs(items_sum - subtotal) > Decimal("0.05"):
            flag = (
                f"line_item_sum_mismatch: "
                f"Line items sum to {items_sum} but subtotal is {subtotal}"
            )
            if not any("line_item_sum_mismatch" in f for f in confidence_flags):
                confidence_flags.append(flag)

    invoice_number = response_json.get("invoice_number") or "UNKNOWN"
    invoice_date = _parse_date(response_json.get("invoice_date")) or date.today()
    vendor_name = response_json.get("vendor_name") or "UNKNOWN"
    total = _parse_decimal(response_json.get("total")) or Decimal("0.00")

    metadata = InvoiceMetadata(
        source_file=str(extraction.ingest_id),
        ingest_id=extraction.ingest_id,
    )

    return StructuredInvoice(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=_parse_date(response_json.get("due_date")),
        vendor_name=vendor_name,
        vendor_address=response_json.get("vendor_address"),
        currency=response_json.get("currency", "USD"),
        subtotal=subtotal,
        tax=_parse_decimal(response_json.get("tax")),
        total=total,
        line_items=line_items,
        metadata=metadata,
        confidence_flags=confidence_flags,
        raw_extraction=extraction.model_dump(mode="json"),
    )


def structure_invoice(extraction: VisionExtractionResult) -> StructuredInvoice:
    """Take raw vision extraction and return a normalized StructuredInvoice.

    Sends the extracted fields to Claude for schema mapping, then parses
    and validates the response locally.
    """
    raw_fields = _build_raw_fields(extraction)

    user_message = (
        "Here are the raw extracted fields from an invoice. "
        "Map them to the structured schema.\n\n"
        f"```json\n{json.dumps(raw_fields, indent=2, default=str)}\n```"
    )

    _logger.info("structuring_start", file=str(extraction.ingest_id))

    wrapper = AnthropicWrapper()
    response_text = wrapper.complete(STRUCTURING_SYSTEM_PROMPT, user_message)

    # Strip markdown fences if present
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)

    try:
        response_json = json.loads(stripped)
    except json.JSONDecodeError as exc:
        _logger.error("structuring_parse_error", error=str(exc))
        metadata = InvoiceMetadata(
            source_file=str(extraction.ingest_id),
            ingest_id=extraction.ingest_id,
        )
        return StructuredInvoice(
            invoice_number="PARSE_ERROR",
            invoice_date=date.today(),
            vendor_name="UNKNOWN",
            total=Decimal("0.00"),
            metadata=metadata,
            confidence_flags=[f"structuring_parse_error: {exc}"],
            raw_extraction=extraction.model_dump(mode="json"),
        )

    invoice = _build_invoice_from_response(response_json, extraction)
    _logger.info(
        "structuring_complete",
        invoice_number=invoice.invoice_number,
        flag_count=len(invoice.confidence_flags),
    )
    return invoice

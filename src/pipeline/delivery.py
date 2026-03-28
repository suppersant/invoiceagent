"""Delivery module — generates CSV and JSON output from approved invoices.

Takes QA-approved StructuredInvoice data and writes clean CSV (for AP clerks
opening in Excel) and JSON (for future QuickBooks integration) to the client's
processed output directory.

Constitution references:
  [B5] Raw invoices stored as-is; processed outputs to separate directories.
  [C3] No output generated for invoices that have not passed QA.
  [C4] All agent executions logged with required fields.
  [D1] No financial data in logs.
  [E1] Delivery means "file created", not "file emailed".
  [F8] Output path: data/clients/{client_id}/processed/
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

from src.config import config
from src.models.delivery import DeliveryResult
from src.models.invoice import StructuredInvoice
from src.models.qa import QAResult
from src.utils.logging import get_logger

_logger = get_logger("delivery")

# CSV column order — matches task spec for AP clerk usability
_CSV_COLUMNS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "po_number",
    "line_description",
    "line_quantity",
    "line_unit_price",
    "line_amount",
    "subtotal",
    "tax",
    "total",
    "currency",
    "confidence_score",
]

# UTF-8 BOM so Excel opens the file without garbled characters
_UTF8_BOM = "\ufeff"


def deliver_results(
    qa_result: QAResult,
    structured_invoice: StructuredInvoice,
    client_id: str,
) -> DeliveryResult:
    """Generate CSV and JSON output files for an approved invoice.

    Args:
        qa_result: QA result that must be approved (not routed to review).
        structured_invoice: The fully structured invoice data.
        client_id: Client identifier for tenant-isolated output directory.

    Returns:
        DeliveryResult with paths to both generated files.

    Raises:
        ValueError: If the invoice has not passed QA (Constitution [C3]).
    """
    start = time.monotonic()
    invoice_id = str(structured_invoice.invoice_id)

    # Constitution [C3]: No output for invoices that have not passed QA
    if qa_result.routed_to_review and not qa_result.approved:
        raise ValueError(
            f"Invoice {invoice_id} has not passed QA — cannot deliver. "
            "Route through review queue first."
        )

    # Build output directory: data/clients/{client_id}/processed/{invoice_id}/
    output_dir = config.data_dir / "clients" / client_id / "processed" / invoice_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{invoice_id}.csv"
    json_path = output_dir / f"{invoice_id}.json"

    # Generate CSV
    _write_csv(csv_path, structured_invoice, qa_result)

    # Generate JSON — full StructuredInvoice dump
    _write_json(json_path, structured_invoice)

    duration_ms = round((time.monotonic() - start) * 1000)

    # Constitution [C4]: Log delivery with required fields (no financial data [D1])
    _logger.info(
        "delivery_complete",
        invoice_id=invoice_id,
        client_id=client_id,
        csv_path=str(csv_path),
        json_path=str(json_path),
        delivery_timestamp=time.time(),
        duration_ms=duration_ms,
        record_count=len(structured_invoice.line_items),
    )

    return DeliveryResult(
        invoice_id=structured_invoice.invoice_id,
        csv_path=str(csv_path),
        json_path=str(json_path),
        success=True,
        record_count=len(structured_invoice.line_items),
    )


def _write_csv(path: Path, invoice: StructuredInvoice, qa_result: QAResult) -> None:
    """Write a UTF-8 BOM CSV with one row per line item."""
    with path.open("w", newline="", encoding="utf-8") as f:
        # Write BOM for Excel compatibility
        f.write(_UTF8_BOM)

        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for item in invoice.line_items:
            writer.writerow({
                "invoice_number": invoice.invoice_number,
                "invoice_date": str(invoice.invoice_date),
                "due_date": str(invoice.due_date) if invoice.due_date else "",
                "vendor_name": invoice.vendor_name,
                "po_number": "",  # Not in current model; reserved column
                "line_description": item.description,
                "line_quantity": str(item.quantity),
                "line_unit_price": str(item.unit_price),
                "line_amount": str(item.amount),
                "subtotal": str(invoice.subtotal) if invoice.subtotal is not None else "",
                "tax": str(invoice.tax) if invoice.tax is not None else "",
                "total": str(invoice.total),
                "currency": invoice.currency,
                "confidence_score": str(qa_result.overall_confidence),
            })


def _write_json(path: Path, invoice: StructuredInvoice) -> None:
    """Write the full StructuredInvoice as indented JSON."""
    path.write_text(invoice.model_dump_json(indent=2), encoding="utf-8")

"""Human review queue — stores low-confidence invoices for operator review.

Invoices that fail QA confidence thresholds are routed here. Operators can
view extractions alongside original PDFs, make corrections, and approve
for delivery. Items are never deleted (Constitution [B1]).
"""

from __future__ import annotations

import json
from typing import Any

from src.models.invoice import StructuredInvoice
from src.models.qa import QAResult
from src.utils.database import DatabaseManager
from src.utils.logging import get_logger

_logger = get_logger("review_queue")


class ReviewQueue:
    """Manages the human review queue for low-confidence invoices.

    No delete method exists — reviewed items are marked "approved" or
    "rejected", never removed (Constitution [B1]).
    """

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self._db = db or DatabaseManager()

    def add(
        self,
        qa_result: QAResult,
        structured_invoice: StructuredInvoice,
        original_pdf_path: str,
        client_id: str,
    ) -> int:
        """Add a low-confidence invoice to the review queue.

        Args:
            qa_result: The QA result that triggered review routing.
            structured_invoice: The structured invoice data for review.
            original_pdf_path: Path to the original PDF file.
            client_id: Client identifier for tenant isolation.

        Returns:
            The review queue item ID.
        """
        invoice_id = str(qa_result.invoice_id)

        review_id = self._db.add_review_item(
            invoice_id=invoice_id,
            client_id=client_id,
            structured_invoice_json=structured_invoice.model_dump_json(),
            qa_result_json=qa_result.model_dump_json(),
            original_pdf_path=original_pdf_path,
            reason=_build_reason(qa_result),
        )

        _logger.info(
            "review_item_added",
            invoice_id=invoice_id,
            client_id=client_id,
            review_id=review_id,
            overall_confidence=qa_result.overall_confidence,
            flag_count=len(qa_result.flags),
        )

        return review_id

    def list_pending(self, client_id: str | None = None) -> list[dict[str, Any]]:
        """List all pending review items, optionally filtered by client.

        Args:
            client_id: If provided, only return items for this client.

        Returns:
            List of review queue items with status "pending".
        """
        return self._db.get_review_items(status="pending", client_id=client_id)

    def list_all(self, client_id: str | None = None) -> list[dict[str, Any]]:
        """List all review items regardless of status.

        Args:
            client_id: If provided, only return items for this client.

        Returns:
            List of all review queue items.
        """
        return self._db.get_review_items(status=None, client_id=client_id)

    def get(self, invoice_id: str) -> dict[str, Any] | None:
        """Get a single review queue item by invoice ID.

        Args:
            invoice_id: The invoice UUID to look up.

        Returns:
            The review item dict, or None if not found.
        """
        items = self._db.get_review_items_by_invoice(invoice_id)
        return items[0] if items else None

    def approve(
        self,
        invoice_id: str,
        *,
        corrections: dict[str, Any] | None = None,
        operator_notes: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        """Approve a review item, optionally with corrections.

        Corrections are stored as a separate field — the original extraction
        is never overwritten (Constitution [B1]).

        Args:
            invoice_id: The invoice UUID to approve.
            corrections: Optional dict of field corrections.
            operator_notes: Optional notes from the reviewer.
            reviewer: Optional reviewer identifier.
        """
        corrections_json = json.dumps(corrections) if corrections else None

        self._db.update_review_item(
            invoice_id=invoice_id,
            status="approved",
            corrections_json=corrections_json,
            operator_notes=operator_notes,
            reviewer=reviewer,
        )

        _logger.info(
            "review_item_approved",
            invoice_id=invoice_id,
            has_corrections=corrections is not None,
            reviewer=reviewer,
        )

    def reject(
        self,
        invoice_id: str,
        *,
        operator_notes: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        """Reject a review item.

        Args:
            invoice_id: The invoice UUID to reject.
            operator_notes: Optional notes from the reviewer.
            reviewer: Optional reviewer identifier.
        """
        self._db.update_review_item(
            invoice_id=invoice_id,
            status="rejected",
            operator_notes=operator_notes,
            reviewer=reviewer,
        )

        _logger.info(
            "review_item_rejected",
            invoice_id=invoice_id,
            reviewer=reviewer,
        )


def _build_reason(qa_result: QAResult) -> str:
    """Build a human-readable reason for the review routing."""
    parts = [f"Overall confidence {qa_result.overall_confidence:.2f} below threshold"]
    for flag in qa_result.flags:
        parts.append(f"{flag.field_name}: {flag.message}")
    return "; ".join(parts)

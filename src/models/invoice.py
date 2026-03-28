"""Core invoice data models — the canonical schema for structured invoices."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class InvoiceMetadata(BaseModel):
    """Metadata about an invoice's processing journey."""

    source_file: str = Field(..., description="Original file path or URI")
    ingest_id: UUID = Field(..., description="UUID from the ingestion stage")
    processed_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="When the invoice was structured",
    )
    page_count: int = Field(default=1, ge=1, description="Number of pages in source document")
    pipeline_version: str = Field(default="1.0.0", description="Pipeline version that produced this")


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(..., description="Item description")
    quantity: Decimal = Field(..., description="Quantity ordered/delivered")
    unit_price: Decimal = Field(..., description="Price per unit")
    amount: Decimal = Field(..., description="Line total (quantity x unit_price)")
    sku: str | None = Field(default=None, description="SKU or product code")
    tax: Decimal | None = Field(default=None, description="Tax amount for this line")
    gl_code: str | None = Field(default=None, description="General ledger account code")
    source_page: int = Field(..., ge=1, description="Page number where this item was found")


class StructuredInvoice(BaseModel):
    """A fully structured, validated invoice — the canonical output of the structuring agent."""

    invoice_id: UUID = Field(default_factory=uuid4, description="Unique invoice identifier")
    invoice_number: str = Field(..., description="Invoice number from the document")
    invoice_date: datetime.date = Field(..., description="Date on the invoice")
    due_date: datetime.date | None = Field(default=None, description="Payment due date")
    vendor_name: str = Field(..., description="Vendor / supplier name")
    vendor_address: str | None = Field(default=None, description="Vendor address")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    subtotal: Decimal | None = Field(default=None, description="Subtotal before tax")
    tax: Decimal | None = Field(default=None, description="Total tax amount")
    total: Decimal = Field(..., description="Invoice total — must be Decimal, not float")
    line_items: list[LineItem] = Field(default_factory=list, description="Invoice line items")
    metadata: InvoiceMetadata = Field(..., description="Processing metadata")
    confidence_flags: list[str] = Field(
        default_factory=list,
        description="Flags for fields that could not be confidently mapped",
    )
    raw_extraction: dict[str, Any] = Field(
        default_factory=dict,
        description="Original extraction data preserved for traceability [B1]",
    )

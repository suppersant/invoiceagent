"""Data models for the ingestion and vision extraction stages."""

from __future__ import annotations

import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SourceRegion(BaseModel):
    """Traces an extracted value back to its location in the source document."""

    page_number: int = Field(..., ge=1, description="1-indexed page where the data was found")
    text_snippet: str = Field(..., description="Verbatim text from the source region")


class PageImage(BaseModel):
    """A single page image extracted from an ingested document."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    image_path: str = Field(..., description="Path to the extracted page image file")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")


class IngestResult(BaseModel):
    """Result of ingesting a raw document (PDF, image) into the pipeline."""

    ingest_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this ingestion")
    source_file: str = Field(..., description="Original file path or URI")
    ingested_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="Timestamp of ingestion",
    )
    page_count: int = Field(..., ge=1, description="Total pages in the document")
    pages: list[PageImage] = Field(default_factory=list, description="Extracted page images")
    mime_type: str = Field(default="application/pdf", description="MIME type of the source file")


class VisionExtractionResult(BaseModel):
    """Raw extraction output from the Claude vision agent."""

    ingest_id: UUID = Field(..., description="References the IngestResult this came from")
    extracted_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="Timestamp of extraction",
    )
    vendor_name: str | None = Field(default=None, description="Vendor / supplier name")
    vendor_address: str | None = Field(default=None, description="Vendor address")
    invoice_number: str | None = Field(default=None, description="Invoice number")
    invoice_date: datetime.date | None = Field(default=None, description="Invoice date")
    due_date: datetime.date | None = Field(default=None, description="Payment due date")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    total: Decimal | None = Field(default=None, description="Invoice total amount")
    subtotal: Decimal | None = Field(default=None, description="Subtotal before tax")
    tax: Decimal | None = Field(default=None, description="Tax amount")
    raw_line_items: list[dict] = Field(
        default_factory=list,
        description="Line items as raw dicts before structured validation",
    )
    source_regions: list[SourceRegion] = Field(
        default_factory=list,
        description="Source regions for traceability",
    )
    raw_text: str | None = Field(default=None, description="Full raw text extracted by vision")

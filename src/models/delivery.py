"""Data models for the delivery stage."""

from __future__ import annotations

import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class DeliveryFormat(str, Enum):
    """Supported output formats."""

    CSV = "csv"
    JSON = "json"


class DeliveryResult(BaseModel):
    """Result of delivering a processed invoice to its destination."""

    invoice_id: UUID = Field(..., description="UUID of the delivered StructuredInvoice")
    delivered_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="When the delivery completed",
    )
    format: DeliveryFormat = Field(..., description="Output format used")
    output_path: str = Field(..., description="Path or URI where the output was written")
    success: bool = Field(default=True, description="Whether delivery succeeded")
    error_message: str | None = Field(default=None, description="Error details if delivery failed")
    record_count: int = Field(default=0, ge=0, description="Number of line items delivered")

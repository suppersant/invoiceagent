"""Data models for the QA / confidence-scoring stage."""

from __future__ import annotations

import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Categories of QA issues."""

    MISSING = "missing"
    LOW_CONFIDENCE = "low_confidence"
    INCONSISTENT = "inconsistent"


class QAFlag(BaseModel):
    """A single quality issue flagged during QA."""

    field_name: str = Field(..., description="The field that has an issue")
    issue_type: IssueType = Field(..., description="Type of issue detected")
    message: str = Field(..., description="Human-readable description of the issue")


class FieldScore(BaseModel):
    """Confidence score for a single extracted field."""

    field_name: str = Field(..., description="Name of the scored field")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    source_page: int | None = Field(default=None, ge=1, description="Page the field was extracted from")


class QAResult(BaseModel):
    """Output of the QA agent — confidence scores and flags for a structured invoice."""

    invoice_id: UUID = Field(..., description="UUID of the StructuredInvoice being scored")
    reviewed_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="When the QA review was performed",
    )
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence 0.0-1.0")
    field_scores: list[FieldScore] = Field(default_factory=list, description="Per-field confidence scores")
    flags: list[QAFlag] = Field(default_factory=list, description="Quality issues found")
    approved: bool = Field(default=False, description="Whether the invoice passed QA")

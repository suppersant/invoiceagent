"""QA confidence scoring agent — reviews structured invoices and assigns scores.

Compares structured output against raw extraction, checks plausibility and
consistency, and determines whether invoices pass QA or route to human review.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.agents.prompts.qa_scoring import QA_SCORING_SYSTEM_PROMPT, QA_SCORING_USER_TEMPLATE
from src.config import config
from src.models.extraction import VisionExtractionResult
from src.models.invoice import StructuredInvoice
from src.models.qa import FieldScore, IssueType, QAFlag, QAResult
from src.utils.anthropic_client import AnthropicWrapper
from src.utils.logging import get_logger

_logger = get_logger("qa_agent")

# Fields that must be scored on every invoice
_SCORED_FIELDS = (
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "vendor_address",
    "currency",
    "subtotal",
    "tax",
    "total",
    "line_items",
)

# Valid issue type values from the IssueType enum
_VALID_ISSUE_TYPES = {e.value for e in IssueType}


def score_invoice(
    structured: StructuredInvoice,
    raw: VisionExtractionResult,
) -> QAResult:
    """Score a structured invoice for confidence and route accordingly.

    Args:
        structured: The structured invoice from the structuring agent.
        raw: The raw vision extraction result for cross-checking.

    Returns:
        A QAResult with per-field scores, flags, and routing decision.
    """
    start_time = time.monotonic()

    structured_json = structured.model_dump_json(indent=2)
    raw_json = raw.model_dump_json(indent=2)

    user_message = QA_SCORING_USER_TEMPLATE.format(
        structured_json=structured_json,
        raw_json=raw_json,
    )

    _logger.info("qa_start", invoice_id=str(structured.invoice_id))

    wrapper = AnthropicWrapper()
    response_text = wrapper.complete(
        QA_SCORING_SYSTEM_PROMPT,
        user_message,
        max_tokens=4096,
    )

    parsed = _parse_response(response_text, str(structured.invoice_id))
    field_scores = _build_field_scores(parsed, structured)
    flags = _build_flags(parsed)

    # Overall confidence = minimum of all field scores (conservative)
    overall_confidence = min(fs.confidence for fs in field_scores) if field_scores else 0.0

    # Route to review if below threshold (Constitution [C2])
    threshold = config.qa_confidence_threshold
    routed_to_review = overall_confidence < threshold
    approved = not routed_to_review

    duration_ms = round((time.monotonic() - start_time) * 1000)

    _logger.info(
        "qa_complete",
        invoice_id=str(structured.invoice_id),
        overall_confidence=overall_confidence,
        routed_to_review=routed_to_review,
        flag_count=len(flags),
        duration_ms=duration_ms,
    )

    return QAResult(
        invoice_id=structured.invoice_id,
        overall_confidence=overall_confidence,
        field_scores=field_scores,
        flags=flags,
        routed_to_review=routed_to_review,
        approved=approved,
    )


def _parse_response(response_text: str, invoice_id: str) -> dict[str, Any]:
    """Parse the JSON response from Claude, stripping markdown fences."""
    text = response_text.strip()

    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        _logger.error("qa_parse_error", invoice_id=invoice_id, error=str(exc))
        return {"field_scores": {}, "flags": []}

    if not isinstance(parsed, dict):
        _logger.error("qa_response_not_dict", invoice_id=invoice_id)
        return {"field_scores": {}, "flags": []}

    return parsed


def _build_field_scores(
    parsed: dict[str, Any],
    structured: StructuredInvoice,
) -> list[FieldScore]:
    """Build FieldScore objects from parsed response, ensuring all fields covered."""
    raw_scores = parsed.get("field_scores", {})
    field_scores: list[FieldScore] = []

    for field_name in _SCORED_FIELDS:
        score_data = raw_scores.get(field_name, {})
        if isinstance(score_data, dict):
            confidence = score_data.get("confidence", 0.0)
        else:
            confidence = 0.0

        # Clamp to valid range
        confidence = max(0.0, min(1.0, float(confidence)))

        # Determine source page for the field if available
        source_page = _get_source_page(field_name, structured)

        field_scores.append(
            FieldScore(
                field_name=field_name,
                confidence=confidence,
                source_page=source_page,
            )
        )

    return field_scores


def _get_source_page(field_name: str, structured: StructuredInvoice) -> int | None:
    """Extract the source page for a given field from the structured invoice."""
    if field_name == "line_items" and structured.line_items:
        return structured.line_items[0].source_page
    if structured.metadata and structured.metadata.page_count == 1:
        return 1
    return None


def _build_flags(parsed: dict[str, Any]) -> list[QAFlag]:
    """Build QAFlag objects from parsed response."""
    raw_flags = parsed.get("flags", [])
    flags: list[QAFlag] = []

    for flag_data in raw_flags:
        if not isinstance(flag_data, dict):
            continue

        field_name = flag_data.get("field_name", "unknown")
        issue_type_str = flag_data.get("issue_type", "low_confidence")
        message = flag_data.get("message", "")

        # Validate issue_type
        if issue_type_str not in _VALID_ISSUE_TYPES:
            issue_type_str = "low_confidence"

        flags.append(
            QAFlag(
                field_name=field_name,
                issue_type=IssueType(issue_type_str),
                message=message,
            )
        )

    return flags

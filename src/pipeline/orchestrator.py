"""Pipeline orchestrator — runs the complete invoice processing pipeline.

Calls each agent in strict sequence: ingest -> extract -> structure -> QA.
The orchestrator is the ONLY module that calls agents (Constitution [A3]).
Pipeline is strictly sequential (Constitution [A2]).
QA is never skipped (Constitution [E4]).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from src.agents.qa_agent import score_invoice
from src.agents.structuring_agent import structure_invoice
from src.agents.vision_agent import extract_from_pages
from src.config import config
from src.models.qa import QAResult
from src.pipeline.ingest import ingest_pdf
from src.utils.database import DatabaseManager
from src.utils.logging import get_logger

_logger = get_logger("orchestrator")


def process_invoice(pdf_path: Path, client_id: str) -> QAResult:
    """Run the full pipeline on a single PDF invoice.

    Pipeline stages: ingest -> extract -> structure -> QA.
    Each stage is logged with timing. On failure, the database records
    a "failed" status and the exception is re-raised (Constitution [E7]).

    Args:
        pdf_path: Path to the PDF file to process.
        client_id: Client identifier for tenant isolation.

    Returns:
        A QAResult with confidence scores and routing decision.

    Raises:
        Any exception from individual agents, after logging and recording
        the failure in the database.
    """
    pdf_path = Path(pdf_path)
    db = DatabaseManager()
    db.init_db()

    invoice_id: str | None = None

    # ── Stage 1: Ingest ─────────────────────────────────────────
    try:
        run_id = _start_run(db, "pending", client_id, "ingest")
        start = time.monotonic()

        ingest_result = ingest_pdf(pdf_path, client_id)
        invoice_id = str(ingest_result.ingest_id)

        # Register the invoice in the database
        db.add_invoice(
            invoice_id=invoice_id,
            client_id=client_id,
            source_file=str(pdf_path),
            status="ingested",
        )

        duration_ms = round((time.monotonic() - start) * 1000)
        _finish_run(db, run_id, "completed", duration_ms=duration_ms)

        _logger.info(
            "pipeline_stage_complete",
            stage="ingest",
            invoice_id=invoice_id,
            client_id=client_id,
            duration_ms=duration_ms,
        )
    except Exception:
        if run_id:
            _finish_run(db, run_id, "failed", error="Ingestion failed")
        _logger.error("pipeline_failed", stage="ingest", client_id=client_id)
        raise

    # ── Stage 2: Vision Extraction ──────────────────────────────
    try:
        run_id = _start_run(db, invoice_id, client_id, "vision_extraction")
        start = time.monotonic()

        extraction_result = extract_from_pages(ingest_result)

        db.update_invoice_status(invoice_id, "extracted")
        duration_ms = round((time.monotonic() - start) * 1000)
        _finish_run(db, run_id, "completed", duration_ms=duration_ms)

        _logger.info(
            "pipeline_stage_complete",
            stage="vision_extraction",
            invoice_id=invoice_id,
            client_id=client_id,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        db.update_invoice_status(invoice_id, "failed")
        _finish_run(db, run_id, "failed", error=str(exc))
        _logger.error(
            "pipeline_failed",
            stage="vision_extraction",
            invoice_id=invoice_id,
            client_id=client_id,
        )
        raise

    # ── Stage 3: Structuring ───────────────────────────────────
    try:
        run_id = _start_run(db, invoice_id, client_id, "structuring")
        start = time.monotonic()

        structured_invoice = structure_invoice(extraction_result)

        db.update_invoice_status(invoice_id, "structured")
        duration_ms = round((time.monotonic() - start) * 1000)
        _finish_run(db, run_id, "completed", duration_ms=duration_ms)

        _logger.info(
            "pipeline_stage_complete",
            stage="structuring",
            invoice_id=invoice_id,
            client_id=client_id,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        db.update_invoice_status(invoice_id, "failed")
        _finish_run(db, run_id, "failed", error=str(exc))
        _logger.error(
            "pipeline_failed",
            stage="structuring",
            invoice_id=invoice_id,
            client_id=client_id,
        )
        raise

    # ── Stage 4: QA Scoring ────────────────────────────────────
    try:
        run_id = _start_run(db, invoice_id, client_id, "qa")
        start = time.monotonic()

        qa_result = score_invoice(structured_invoice, extraction_result)

        # Determine final status based on QA routing
        final_status = "review_required" if qa_result.routed_to_review else "qa_complete"
        db.update_invoice_status(invoice_id, final_status)

        duration_ms = round((time.monotonic() - start) * 1000)
        _finish_run(
            db,
            run_id,
            "completed",
            duration_ms=duration_ms,
            confidence=qa_result.overall_confidence,
        )

        # Store QA results in database
        db.add_qa_result(
            invoice_id=invoice_id,
            client_id=client_id,
            overall_score=qa_result.overall_confidence,
            passed=qa_result.approved,
            flags_json=_serialize_flags(qa_result),
            field_scores_json=_serialize_field_scores(qa_result),
        )

        _logger.info(
            "pipeline_stage_complete",
            stage="qa",
            invoice_id=invoice_id,
            client_id=client_id,
            duration_ms=duration_ms,
            overall_confidence=qa_result.overall_confidence,
            routed_to_review=qa_result.routed_to_review,
        )
    except Exception as exc:
        db.update_invoice_status(invoice_id, "failed")
        _finish_run(db, run_id, "failed", error=str(exc))
        _logger.error(
            "pipeline_failed",
            stage="qa",
            invoice_id=invoice_id,
            client_id=client_id,
        )
        raise

    _logger.info(
        "pipeline_complete",
        invoice_id=invoice_id,
        client_id=client_id,
        final_status=final_status,
        overall_confidence=qa_result.overall_confidence,
    )

    return qa_result


def _start_run(db: DatabaseManager, invoice_id: str, client_id: str, agent_name: str) -> int:
    """Record the start of an agent processing run."""
    return db.add_processing_run(
        invoice_id=invoice_id,
        client_id=client_id,
        agent_name=agent_name,
        status="started",
    )


def _finish_run(
    db: DatabaseManager,
    run_id: int,
    status: str,
    *,
    duration_ms: int | None = None,
    confidence: float | None = None,
    error: str | None = None,
) -> None:
    """Update a processing run with its final status."""
    db.update_processing_status(
        run_id,
        status,
        duration_ms=duration_ms,
        confidence=confidence,
        error=error,
    )


def _serialize_flags(qa_result: QAResult) -> str:
    """Serialize QA flags to JSON string for database storage."""
    import json
    return json.dumps([f.model_dump() for f in qa_result.flags])


def _serialize_field_scores(qa_result: QAResult) -> str:
    """Serialize field scores to JSON string for database storage."""
    import json
    return json.dumps({fs.field_name: fs.confidence for fs in qa_result.field_scores})

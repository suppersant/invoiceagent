"""PDF ingestion module — entry point of the processing pipeline.

Converts a PDF file into page images suitable for Claude Vision,
copies the raw PDF to the client inbox, and returns an IngestResult.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from uuid import uuid4

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

from src.config import config
from src.models.extraction import IngestResult, PageImage
from src.utils.logging import get_logger

logger = get_logger("ingest_agent")

_POPPLER_PATH: str | None = None

# Auto-detect poppler on Windows
_WINGET_POPPLER = Path.home() / (
    "AppData/Local/Microsoft/WinGet/Packages/"
    "oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "poppler-25.07.0/Library/bin"
)
if _WINGET_POPPLER.exists():
    _POPPLER_PATH = str(_WINGET_POPPLER)

_DPI = 300


class IngestionError(Exception):
    """Raised when PDF ingestion fails."""


def ingest_pdf(pdf_path: Path, client_id: str) -> IngestResult:
    """Ingest a PDF file: validate, copy to client inbox, convert to page images.

    Args:
        pdf_path: Path to the source PDF file.
        client_id: Client identifier for tenant isolation.

    Returns:
        An IngestResult with page images and metadata.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        IngestionError: If the PDF is corrupt, password-protected, or has zero pages.
    """
    pdf_path = Path(pdf_path)
    start_time = time.monotonic()
    invoice_id = uuid4()

    # --- Validate input ---
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise IngestionError(f"Path is not a file: {pdf_path}")

    file_size = pdf_path.stat().st_size
    if file_size == 0:
        raise IngestionError(f"PDF file is empty (0 bytes): {pdf_path}")

    # --- Copy raw PDF to client inbox (Constitution [B5], [E2]) ---
    inbox_dir = config.data_dir / "clients" / client_id / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    inbox_path = inbox_dir / f"{invoice_id}.pdf"
    shutil.copy2(pdf_path, inbox_path)

    # --- Convert PDF to page images ---
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=_DPI,
            poppler_path=_POPPLER_PATH,
        )
    except PDFInfoNotInstalledError:
        raise IngestionError(
            "Poppler is not installed or not found. "
            "Install poppler and ensure it is on PATH."
        )
    except PDFPageCountError as exc:
        raise IngestionError(f"Failed to read PDF (corrupt or password-protected): {exc}")
    except Exception as exc:
        raise IngestionError(f"PDF conversion failed: {exc}")

    if not images:
        raise IngestionError(f"PDF produced zero pages: {pdf_path}")

    # --- Save page images and build PageImage list ---
    pages_dir = config.data_dir / "clients" / client_id / "pages" / str(invoice_id)
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_images: list[PageImage] = []
    for i, img in enumerate(images, start=1):
        page_path = pages_dir / f"page_{i:03d}.png"
        img.save(str(page_path), "PNG")
        page_images.append(
            PageImage(
                page_number=i,
                image_path=str(page_path),
                width=img.width,
                height=img.height,
            )
        )

    duration_ms = int((time.monotonic() - start_time) * 1000)

    # --- Log ingestion (Constitution [C4]) ---
    logger.info(
        "pdf_ingested",
        invoice_id=str(invoice_id),
        client_id=client_id,
        page_count=len(images),
        file_size_bytes=file_size,
        duration_ms=duration_ms,
    )

    return IngestResult(
        ingest_id=invoice_id,
        source_file=str(pdf_path),
        page_count=len(images),
        pages=page_images,
    )

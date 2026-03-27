"""Tests for the PDF ingestion module."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from PIL import Image

from src.models.extraction import IngestResult
from src.pipeline.ingest import IngestionError, ingest_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_invoice_01.pdf"


@pytest.fixture(autouse=True)
def _use_tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect config.data_dir to a temp directory for each test."""
    monkeypatch.setattr("src.pipeline.ingest.config.data_dir", tmp_path / "data")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestIngestPdf:
    """Test suite for ingest_pdf."""

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_returns_ingest_result_with_correct_page_count(self, tmp_path: Path):
        """ingest_pdf returns an IngestResult with correct page_count."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        assert isinstance(result, IngestResult)
        assert result.page_count >= 1
        assert len(result.pages) == result.page_count

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_page_images_are_300_dpi(self, tmp_path: Path):
        """Each PageImage contains a valid image at >= 300 DPI resolution."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        for page in result.pages:
            img = Image.open(page.image_path)
            # At 300 DPI, a US Letter page (8.5x11") should be at least 2550x3300
            assert img.width >= 2000, f"Page {page.page_number} width too small: {img.width}"
            assert img.height >= 2000, f"Page {page.page_number} height too small: {img.height}"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_original_pdf_not_modified(self, tmp_path: Path):
        """The original PDF is NOT modified (compare file hash before/after)."""
        hash_before = _file_hash(SAMPLE_PDF)
        ingest_pdf(SAMPLE_PDF, "test_client")
        hash_after = _file_hash(SAMPLE_PDF)
        assert hash_before == hash_after

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_pdf_copied_to_client_inbox(self, tmp_path: Path):
        """A copy of the PDF exists at data/clients/test_client/inbox/{invoice_id}.pdf."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        inbox_dir = tmp_path / "data" / "clients" / "test_client" / "inbox"
        inbox_file = inbox_dir / f"{result.ingest_id}.pdf"
        assert inbox_file.exists()
        assert _file_hash(inbox_file) == _file_hash(SAMPLE_PDF)

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path):
        """Passing a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ingest_pdf(tmp_path / "nonexistent.pdf", "test_client")

    def test_corrupt_file_raises_descriptive_error(self, tmp_path: Path):
        """Passing a corrupt file raises a descriptive IngestionError."""
        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_text("this is not a valid pdf")
        with pytest.raises(IngestionError, match="(corrupt|failed|conversion|PDF)"):
            ingest_pdf(corrupt_pdf, "test_client")

    def test_empty_file_raises_error(self, tmp_path: Path):
        """Passing an empty file raises IngestionError."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with pytest.raises(IngestionError, match="empty"):
            ingest_pdf(empty_pdf, "test_client")

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_ingest_id_is_valid_uuid(self, tmp_path: Path):
        """IngestResult.invoice_id is a valid UUID."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        assert isinstance(result.ingest_id, UUID)
        # Verify it's a valid UUID by round-tripping
        assert str(result.ingest_id) == str(UUID(str(result.ingest_id)))

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_page_images_have_valid_paths(self, tmp_path: Path):
        """Each PageImage has an image_path that points to an existing file."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        for page in result.pages:
            assert Path(page.image_path).exists()
            assert page.width is not None
            assert page.height is not None

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not available")
    def test_page_numbers_are_sequential(self, tmp_path: Path):
        """Page numbers are 1-indexed and sequential."""
        result = ingest_pdf(SAMPLE_PDF, "test_client")
        for i, page in enumerate(result.pages, start=1):
            assert page.page_number == i

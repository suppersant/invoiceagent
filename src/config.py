"""Central configuration module — single source of truth for all settings.

Loads all values from environment variables with sensible defaults.
Singleton: import `config` from this module everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class _Config:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self._loaded = False
        self._load()

    def _load(self) -> None:
        # --- Core settings ---
        self._anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
        self.database_path: Path = Path(os.getenv("DATABASE_PATH", "data/invoiceagent.db"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

        # --- Email settings ---
        self.email_host: str = os.getenv("EMAIL_HOST", "")
        self.email_user: str = os.getenv("EMAIL_USER", "")
        self.email_password: str = os.getenv("EMAIL_PASSWORD", "")

        # --- Quality thresholds ---
        self.qa_confidence_threshold: float = float(
            os.getenv("QA_CONFIDENCE_THRESHOLD", "0.85")
        )

        # --- Feature flags (all default to False) ---
        self.enable_email_ingestion: bool = _parse_bool(
            os.getenv("ENABLE_EMAIL_INGESTION", "false")
        )
        self.enable_auto_delivery: bool = _parse_bool(
            os.getenv("ENABLE_AUTO_DELIVERY", "false")
        )
        self.enable_ocr_fallback: bool = _parse_bool(
            os.getenv("ENABLE_OCR_FALLBACK", "false")
        )
        self.enable_duplicate_detection: bool = _parse_bool(
            os.getenv("ENABLE_DUPLICATE_DETECTION", "false")
        )

        self._loaded = True

    @property
    def anthropic_api_key(self) -> str:
        """Return the Anthropic API key or raise a clear error if not set."""
        if not self._anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it in your .env file or environment before running InvoiceAgent."
            )
        return self._anthropic_api_key


def _parse_bool(value: str) -> bool:
    """Parse a string to a boolean."""
    return value.strip().lower() in ("true", "1", "yes")


# Singleton instance — import this everywhere
config = _Config()

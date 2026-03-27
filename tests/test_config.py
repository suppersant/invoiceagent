"""Tests for the configuration module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfig:
    """Test suite for src.config."""

    def _make_config(self, env: dict[str, str] | None = None):
        """Create a fresh _Config with the given env vars."""
        base = {"ANTHROPIC_API_KEY": "sk-test-key"}
        if env is not None:
            base.update(env)
        with patch.dict(os.environ, base, clear=False):
            # Import fresh each time to avoid singleton caching
            from src.config import _Config
            return _Config()

    def test_import_singleton(self):
        """from src.config import config works."""
        from src.config import config
        assert config is not None

    def test_anthropic_api_key_raises_when_missing(self):
        """config.anthropic_api_key raises error if env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            from src.config import _Config
            cfg = _Config()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = cfg.anthropic_api_key

    def test_anthropic_api_key_returns_value(self):
        cfg = self._make_config({"ANTHROPIC_API_KEY": "sk-ant-test123"})
        assert cfg.anthropic_api_key == "sk-ant-test123"

    def test_qa_confidence_threshold_default(self):
        """config.qa_confidence_threshold returns 0.85 when env var not set."""
        cfg = self._make_config()
        assert cfg.qa_confidence_threshold == 0.85

    def test_qa_confidence_threshold_override(self):
        cfg = self._make_config({"QA_CONFIDENCE_THRESHOLD": "0.90"})
        assert cfg.qa_confidence_threshold == 0.90

    def test_enable_email_ingestion_default_false(self):
        """config.enable_email_ingestion returns False by default."""
        cfg = self._make_config()
        assert cfg.enable_email_ingestion is False

    def test_enable_email_ingestion_true(self):
        cfg = self._make_config({"ENABLE_EMAIL_INGESTION": "true"})
        assert cfg.enable_email_ingestion is True

    def test_data_dir_returns_path(self):
        """config.data_dir returns a pathlib.Path object."""
        cfg = self._make_config()
        assert isinstance(cfg.data_dir, Path)

    def test_database_path_returns_path(self):
        cfg = self._make_config()
        assert isinstance(cfg.database_path, Path)

    def test_all_feature_flags_default_false(self):
        """All feature flags accessible and default to False."""
        cfg = self._make_config()
        assert cfg.enable_email_ingestion is False
        assert cfg.enable_auto_delivery is False
        assert cfg.enable_ocr_fallback is False
        assert cfg.enable_duplicate_detection is False

    def test_log_level_default(self):
        cfg = self._make_config()
        assert cfg.log_level == "INFO"

    def test_log_level_override(self):
        cfg = self._make_config({"LOG_LEVEL": "debug"})
        assert cfg.log_level == "DEBUG"

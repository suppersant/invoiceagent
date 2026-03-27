"""Tests for the structured logging module."""

from __future__ import annotations

import json
import logging
import os
from io import StringIO
from unittest.mock import patch

import structlog

from src.utils.logging import _sanitize_processor, configure_logging, get_logger


class TestGetLogger:
    """Test suite for src.utils.logging."""

    def test_get_logger_returns_bound_logger(self):
        """get_logger returns a structlog BoundLogger."""
        logger = get_logger("vision_agent")
        assert logger is not None

    def test_logger_binds_agent_name(self, capsys):
        """logger = get_logger("vision_agent") produces logs with "agent": "vision_agent"."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            configure_logging()
            logger = get_logger("vision_agent")

            # Capture output via a stream handler
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            handler.setLevel(logging.DEBUG)

            stdlib_logger = logging.getLogger()
            stdlib_logger.addHandler(handler)
            stdlib_logger.setLevel(logging.DEBUG)

            try:
                logger.info("test event")
                output = stream.getvalue()
                parsed = json.loads(output.strip().split("\n")[-1])
                assert parsed["agent"] == "vision_agent"
            finally:
                stdlib_logger.removeHandler(handler)

    def test_log_entries_are_valid_json(self):
        """Log entries are valid JSON."""
        configure_logging()
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)

        stdlib_logger = logging.getLogger()
        stdlib_logger.addHandler(handler)
        stdlib_logger.setLevel(logging.DEBUG)

        try:
            logger = get_logger("test_agent")
            logger.info("hello world")
            output = stream.getvalue().strip()
            line = output.split("\n")[-1]
            parsed = json.loads(line)
            assert "event" in parsed
            assert parsed["event"] == "hello world"
        finally:
            stdlib_logger.removeHandler(handler)


class TestSanitization:
    """Test the financial data sanitization processor."""

    def test_total_is_redacted(self):
        """A log entry containing total=1500.00 has the value redacted."""
        event_dict = {"event": "processed", "total": 1500.00, "invoice_id": "abc"}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["total"] == "[REDACTED]"
        assert result["invoice_id"] == "abc"

    def test_vendor_name_is_redacted(self):
        """A log entry containing vendor_name="Acme Corp" has the value redacted."""
        event_dict = {"event": "processed", "vendor_name": "Acme Corp"}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["vendor_name"] == "[REDACTED]"

    def test_amount_is_redacted(self):
        event_dict = {"event": "test", "line_amount": 250.00}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["line_amount"] == "[REDACTED]"

    def test_price_is_redacted(self):
        event_dict = {"event": "test", "unit_price": 10.50}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["unit_price"] == "[REDACTED]"

    def test_dollar_is_redacted(self):
        event_dict = {"event": "test", "dollar_value": 99.99}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["dollar_value"] == "[REDACTED]"

    def test_non_sensitive_fields_unchanged(self):
        event_dict = {"event": "test", "invoice_id": "123", "agent": "qa"}
        result = _sanitize_processor(None, "info", event_dict)
        assert result["invoice_id"] == "123"
        assert result["agent"] == "qa"


class TestLogLevel:
    """Test that log level configuration works."""

    def test_error_level_suppresses_info(self):
        """config.log_level = "ERROR" suppresses INFO-level messages."""
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}, clear=False):
            from src.config import _Config
            cfg = _Config()
            assert cfg.log_level == "ERROR"

            configure_logging()
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            handler.setLevel(logging.ERROR)

            stdlib_logger = logging.getLogger()
            stdlib_logger.addHandler(handler)
            stdlib_logger.setLevel(logging.ERROR)

            try:
                logger = get_logger("test_agent")
                logger.info("should not appear")
                output = stream.getvalue()
                assert output.strip() == ""
            finally:
                stdlib_logger.removeHandler(handler)

"""Structured logging setup using structlog.

Provides JSON-formatted logs with automatic sanitization of financial data
per Constitution [D1]. Use get_logger() to obtain a bound logger.
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.config import config

# Fields whose values must be redacted from log output (Constitution [D1])
_SENSITIVE_KEYWORDS = {"amount", "total", "price", "dollar", "vendor_name"}


def _sanitize_processor(
    _logger: object, _method_name: str, event_dict: dict
) -> dict:
    """Strip/redact values for keys containing sensitive financial keywords."""
    for key in list(event_dict.keys()):
        if any(kw in key.lower() for kw in _SENSITIVE_KEYWORDS):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging() -> None:
    """Configure structlog and stdlib logging for the application."""
    log_level = getattr(logging, config.log_level, logging.INFO)

    # Configure stdlib logging (structlog wraps it)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _sanitize_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(agent_name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger pre-bound with the given agent name.

    Args:
        agent_name: Name of the agent or module requesting the logger.

    Returns:
        A structlog BoundLogger with ``agent`` automatically included.
    """
    configure_logging()
    return structlog.get_logger(agent=agent_name)

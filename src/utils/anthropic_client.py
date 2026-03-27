"""Thin wrapper around the Anthropic Python SDK.

Provides a singleton client with retry logic, convenience methods for text
and vision requests, and per-call logging (Constitution [C4]) without
exposing prompt content (Constitution [D1]).
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

from src.config import config
from src.utils.logging import get_logger

_logger = get_logger("anthropic_client")

DEFAULT_MODEL = "claude-sonnet-4-20250514"
OPUS_MODEL = "claude-opus-4-20250514"

# Retry settings for rate-limit (429) errors
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_MULTIPLIER = 2.0

# Default timeout in seconds
DEFAULT_TIMEOUT_S = 120.0


class AnthropicWrapper:
    """Singleton wrapper around the Anthropic SDK client.

    Usage::

        wrapper = AnthropicWrapper()
        result = wrapper.complete("You are helpful.", "Hello!")
    """

    _instance: AnthropicWrapper | None = None

    def __new__(cls) -> AnthropicWrapper:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        api_key = config.anthropic_api_key  # raises ValueError if missing
        self._client = anthropic.Anthropic(api_key=api_key)
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model_override: str | None = None,
        max_tokens: int = 4096,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> str:
        """Send a text completion request and return the assistant's reply.

        Args:
            system_prompt: System-level instruction.
            user_message: The user message content.
            model_override: Model to use instead of the default.
            max_tokens: Maximum tokens in the response.
            timeout: Request timeout in seconds.

        Returns:
            The text content of the assistant's response.
        """
        model = model_override or DEFAULT_MODEL
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]
        return self._send(
            model=model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def vision(
        self,
        system_prompt: str,
        images: list[dict[str, Any]],
        user_message: str,
        *,
        model_override: str | None = None,
        max_tokens: int = 4096,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> str:
        """Send a vision request with one or more images.

        Args:
            system_prompt: System-level instruction.
            images: List of image content blocks, each a dict with
                ``type``, ``source`` keys as expected by the Anthropic API.
            user_message: Accompanying text prompt.
            model_override: Model to use instead of the default.
            max_tokens: Maximum tokens in the response.
            timeout: Request timeout in seconds.

        Returns:
            The text content of the assistant's response.
        """
        model = model_override or DEFAULT_MODEL
        content: list[dict[str, Any]] = [*images, {"type": "text", "text": user_message}]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content},
        ]
        return self._send(
            model=model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Execute the API call with retry-on-429 and structured logging."""
        backoff = INITIAL_BACKOFF_S

        for attempt in range(1, MAX_RETRIES + 1):
            start = time.monotonic()
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    timeout=timeout,
                )
                duration_ms = round((time.monotonic() - start) * 1000)
                self._log_call(model, response.usage, duration_ms)
                return response.content[0].text

            except anthropic.RateLimitError:
                duration_ms = round((time.monotonic() - start) * 1000)
                _logger.warning(
                    "rate_limit_hit",
                    model=model,
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

            except anthropic.APIError:
                duration_ms = round((time.monotonic() - start) * 1000)
                _logger.error(
                    "api_error",
                    model=model,
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
                raise

        # Should never reach here, but satisfy type checker
        raise anthropic.APIError("max retries exhausted")  # pragma: no cover

    @staticmethod
    def _log_call(model: str, usage: Any, duration_ms: int) -> None:
        """Log API call metadata without any prompt/response content."""
        _logger.info(
            "api_call",
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            duration_ms=duration_ms,
        )

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton — for testing only."""
        cls._instance = None

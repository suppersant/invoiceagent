"""Tests for the Anthropic client wrapper (all mocked — no real API calls)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from src.utils.anthropic_client import (
    DEFAULT_MODEL,
    AnthropicWrapper,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test gets a fresh singleton."""
    AnthropicWrapper._reset()
    yield
    AnthropicWrapper._reset()


def _mock_response(text: str = "ok", input_tokens: int = 10, output_tokens: int = 5):
    """Build a minimal mock matching the Anthropic SDK response shape."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ------------------------------------------------------------------
# Singleton behaviour
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_singleton_returns_same_instance(mock_sdk, mock_cfg):
    mock_cfg.anthropic_api_key = "sk-test"
    a = AnthropicWrapper()
    b = AnthropicWrapper()
    assert a is b


# ------------------------------------------------------------------
# API key validation
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client.config")
def test_missing_api_key_raises(mock_cfg):
    type(mock_cfg).anthropic_api_key = property(
        lambda self: (_ for _ in ()).throw(
            ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        )
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicWrapper()


# ------------------------------------------------------------------
# complete()
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_complete_returns_string(mock_sdk, mock_cfg):
    mock_cfg.anthropic_api_key = "sk-test"
    mock_sdk.return_value.messages.create.return_value = _mock_response("Hello!")
    wrapper = AnthropicWrapper()
    result = wrapper.complete("system", "hi")
    assert result == "Hello!"
    assert isinstance(result, str)


@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_complete_uses_default_model(mock_sdk, mock_cfg):
    mock_cfg.anthropic_api_key = "sk-test"
    mock_sdk.return_value.messages.create.return_value = _mock_response()
    wrapper = AnthropicWrapper()
    wrapper.complete("sys", "msg")
    call_kwargs = mock_sdk.return_value.messages.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL


@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_complete_model_override(mock_sdk, mock_cfg):
    mock_cfg.anthropic_api_key = "sk-test"
    mock_sdk.return_value.messages.create.return_value = _mock_response()
    wrapper = AnthropicWrapper()
    wrapper.complete("sys", "msg", model_override="claude-opus-4-20250514")
    call_kwargs = mock_sdk.return_value.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-20250514"


# ------------------------------------------------------------------
# vision()
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_vision_returns_string(mock_sdk, mock_cfg):
    mock_cfg.anthropic_api_key = "sk-test"
    mock_sdk.return_value.messages.create.return_value = _mock_response("extracted")
    wrapper = AnthropicWrapper()
    image_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
    }
    result = wrapper.vision("system", [image_block], "extract this")
    assert result == "extracted"
    assert isinstance(result, str)


# ------------------------------------------------------------------
# Rate-limit retry
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client.time.sleep")  # don't actually sleep
@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_rate_limit_retries_then_succeeds(mock_sdk, mock_cfg, mock_sleep):
    mock_cfg.anthropic_api_key = "sk-test"
    create = mock_sdk.return_value.messages.create
    create.side_effect = [
        anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        ),
        _mock_response("retry ok"),
    ]
    wrapper = AnthropicWrapper()
    result = wrapper.complete("sys", "msg")
    assert result == "retry ok"
    assert create.call_count == 2
    mock_sleep.assert_called_once()


@patch("src.utils.anthropic_client.time.sleep")
@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_rate_limit_exhausts_retries(mock_sdk, mock_cfg, mock_sleep):
    mock_cfg.anthropic_api_key = "sk-test"
    create = mock_sdk.return_value.messages.create
    rate_err = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    create.side_effect = [rate_err, rate_err, rate_err]
    wrapper = AnthropicWrapper()
    with pytest.raises(anthropic.RateLimitError):
        wrapper.complete("sys", "msg")


# ------------------------------------------------------------------
# Logging — records metadata, NOT content
# ------------------------------------------------------------------

@patch("src.utils.anthropic_client._logger")
@patch("src.utils.anthropic_client.config")
@patch("src.utils.anthropic_client.anthropic.Anthropic")
def test_api_call_logging(mock_sdk, mock_cfg, mock_logger):
    mock_cfg.anthropic_api_key = "sk-test"
    mock_sdk.return_value.messages.create.return_value = _mock_response(
        text="secret data", input_tokens=42, output_tokens=17
    )
    wrapper = AnthropicWrapper()
    wrapper.complete("secret system prompt", "secret user message")

    mock_logger.info.assert_called_once()
    call_kwargs = mock_logger.info.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL
    assert call_kwargs["input_tokens"] == 42
    assert call_kwargs["output_tokens"] == 17
    assert "duration_ms" in call_kwargs

    # Ensure no prompt/response content leaked into logs
    logged_str = str(mock_logger.info.call_args)
    assert "secret" not in logged_str

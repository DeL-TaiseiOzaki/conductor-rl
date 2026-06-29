"""Tests for conductor_workflow.workers.

All network calls are mocked -- no real HTTP traffic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductor_workflow.workers import (
    WorkerConfig,
    _is_transient_error,
    _retry_with_backoff,
    build_client,
    call_judge,
    call_worker,
    resolve_model_slug,
)

# ---------------------------------------------------------------------------
# WorkerConfig / slug resolution
# ---------------------------------------------------------------------------


class TestResolveModelSlug:
    """Variant suffix application."""

    def test_nitro_variant_appended(self) -> None:
        cfg = WorkerConfig(
            worker_id=0,
            slug="deepseek/deepseek-v4-flash",
            latency_weight=1.0,
            cost_out_per_1m=0.18,
            openrouter_variant=":nitro",
        )
        assert resolve_model_slug(cfg) == "deepseek/deepseek-v4-flash:nitro"

    def test_online_variant_appended(self) -> None:
        cfg = WorkerConfig(
            worker_id=1,
            slug="minimax/minimax-m3",
            latency_weight=1.5,
            cost_out_per_1m=1.20,
            openrouter_variant=":online",
        )
        assert resolve_model_slug(cfg) == "minimax/minimax-m3:online"

    def test_empty_variant_no_suffix(self) -> None:
        cfg = WorkerConfig(
            worker_id=2,
            slug="deepseek/deepseek-v4-pro",
            latency_weight=3.0,
            cost_out_per_1m=0.87,
            openrouter_variant="",
        )
        assert resolve_model_slug(cfg) == "deepseek/deepseek-v4-pro"

    def test_whitespace_variant_no_suffix(self) -> None:
        cfg = WorkerConfig(
            worker_id=3,
            slug="z-ai/glm-5.2",
            latency_weight=3.0,
            cost_out_per_1m=4.40,
            openrouter_variant="  ",
        )
        assert resolve_model_slug(cfg) == "z-ai/glm-5.2"


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


class TestIsTransientError:
    """Classify errors as transient vs non-transient."""

    def test_429_is_transient(self) -> None:
        exc = Exception("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is True

    def test_500_is_transient(self) -> None:
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is True

    def test_502_is_transient(self) -> None:
        exc = Exception("bad gateway")
        exc.status_code = 502  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is True

    def test_timeout_in_name_is_transient(self) -> None:
        class TimeoutError(Exception):
            pass

        assert _is_transient_error(TimeoutError("timed out")) is True

    def test_connection_error_is_transient(self) -> None:
        class ConnectionError(Exception):
            pass

        assert _is_transient_error(ConnectionError("reset")) is True

    def test_400_is_not_transient(self) -> None:
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is False

    def test_generic_error_is_not_transient(self) -> None:
        assert _is_transient_error(ValueError("bad input")) is False


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Retry logic for transient failures."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        async def factory() -> str:
            return "hello"

        result = await _retry_with_backoff(factory, max_retries=3, initial_backoff=0.01)
        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_retry_on_transient_then_succeed(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("rate limit")
                exc.status_code = 429  # type: ignore[attr-defined]
                raise exc
            return "recovered"

        result = await _retry_with_backoff(factory, max_retries=3, initial_backoff=0.01)
        assert result.success is True
        assert result.output == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries_returns_transient_failure(self) -> None:
        async def factory() -> str:
            exc = Exception("always fails")
            exc.status_code = 503  # type: ignore[attr-defined]
            raise exc

        result = await _retry_with_backoff(factory, max_retries=2, initial_backoff=0.01)
        assert result.success is False
        assert result.transient_failure is True
        assert "transient failure" in result.error_message

    @pytest.mark.asyncio
    async def test_non_transient_error_no_retry(self) -> None:
        call_count = 0

        async def factory() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        result = await _retry_with_backoff(factory, max_retries=3, initial_backoff=0.01)
        assert result.success is False
        assert result.transient_failure is False
        assert call_count == 1  # No retries


# ---------------------------------------------------------------------------
# call_worker (mocked)
# ---------------------------------------------------------------------------


class TestCallWorker:
    """Worker calls with mocked AsyncOpenAI client."""

    @pytest.mark.asyncio
    async def test_successful_call(self) -> None:
        # Arrange
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            openrouter_variant=":nitro",
        )
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "answer text"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Act
        result = await call_worker(cfg, "test prompt", client=mock_client)

        # Assert
        assert result.success is True
        assert result.output == "answer text"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test/model:nitro"

    @pytest.mark.asyncio
    async def test_call_with_semaphore(self) -> None:
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
        )
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        sem = asyncio.Semaphore(1)
        result = await call_worker(cfg, "prompt", client=mock_client, semaphore=sem)
        assert result.success is True


# ---------------------------------------------------------------------------
# call_judge (mocked)
# ---------------------------------------------------------------------------


class TestCallJudge:
    """Judge calls with mocked client."""

    @pytest.mark.asyncio
    async def test_judge_call_success(self) -> None:
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "YES"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await call_judge(
            "Is 1/2 == 0.5?",
            judge_slug="test/judge",
            client=mock_client,
        )
        assert result.success is True
        assert result.output == "YES"

    @pytest.mark.asyncio
    async def test_judge_uses_env_override(self) -> None:
        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "NO"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"JUDGE_MODEL": "custom/judge"}):
            result = await call_judge(
                "test",
                client=mock_client,
            )
        assert result.success is True
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom/judge"


# ---------------------------------------------------------------------------
# No real HTTP
# ---------------------------------------------------------------------------


class TestNoRealHttp:
    """Verify no actual network calls occur."""

    @pytest.mark.asyncio
    async def test_build_client_requires_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                build_client()

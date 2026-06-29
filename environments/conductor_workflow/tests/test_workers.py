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
    clear_cache,
    resolve_model_slug,
    set_cache_enabled,
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
            cost_in_per_1m=0.09,
            openrouter_variant=":nitro",
        )
        assert resolve_model_slug(cfg) == "deepseek/deepseek-v4-flash:nitro"

    def test_online_variant_appended(self) -> None:
        cfg = WorkerConfig(
            worker_id=1,
            slug="minimax/minimax-m3",
            latency_weight=1.5,
            cost_out_per_1m=1.20,
            cost_in_per_1m=0.30,
            openrouter_variant=":online",
        )
        assert resolve_model_slug(cfg) == "minimax/minimax-m3:online"

    def test_empty_variant_no_suffix(self) -> None:
        cfg = WorkerConfig(
            worker_id=2,
            slug="deepseek/deepseek-v4-pro",
            latency_weight=3.0,
            cost_out_per_1m=0.87,
            cost_in_per_1m=0.435,
            openrouter_variant="",
        )
        assert resolve_model_slug(cfg) == "deepseek/deepseek-v4-pro"

    def test_whitespace_variant_no_suffix(self) -> None:
        cfg = WorkerConfig(
            worker_id=3,
            slug="z-ai/glm-5.2",
            latency_weight=3.0,
            cost_out_per_1m=4.40,
            cost_in_per_1m=0.94,
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
        async def factory() -> tuple[str, int, int]:
            return ("hello", 10, 5)

        result = await _retry_with_backoff(factory, max_retries=3, initial_backoff=0.01)
        assert result.success is True
        assert result.output == "hello"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_retry_on_transient_then_succeed(self) -> None:
        call_count = 0

        async def factory() -> tuple[str, int, int]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("rate limit")
                exc.status_code = 429  # type: ignore[attr-defined]
                raise exc
            return ("recovered", 20, 10)

        result = await _retry_with_backoff(factory, max_retries=3, initial_backoff=0.01)
        assert result.success is True
        assert result.output == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries_returns_transient_failure(self) -> None:
        async def factory() -> tuple[str, int, int]:
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

        async def factory() -> tuple[str, int, int]:
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


def _make_mock_client(
    content: str = "answer text",
    prompt_tokens: int = 50,
    completion_tokens: int = 30,
) -> AsyncMock:
    """Create a mock AsyncOpenAI client with usage data."""
    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


class TestCallWorker:
    """Worker calls with mocked AsyncOpenAI client."""

    def setup_method(self) -> None:
        """Clear cache and ensure it is enabled before each test."""
        clear_cache()
        set_cache_enabled(True)

    def teardown_method(self) -> None:
        """Reset cache state."""
        clear_cache()
        set_cache_enabled(True)

    @pytest.mark.asyncio
    async def test_successful_call_captures_usage(self) -> None:
        # Arrange
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
            openrouter_variant=":nitro",
        )
        mock_client = _make_mock_client("answer text", 50, 30)

        # Act
        result = await call_worker(cfg, "test prompt", client=mock_client)

        # Assert
        assert result.success is True
        assert result.output == "answer text"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 30
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test/model:nitro"

    @pytest.mark.asyncio
    async def test_cache_hit_returns_stored_result_no_api_call(self) -> None:
        """Cache hit returns stored WorkerResult with token counts, no API call."""
        # Arrange
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
        )
        mock_client = _make_mock_client("cached answer", 100, 50)

        # Act: first call populates cache
        result1 = await call_worker(cfg, "same prompt", client=mock_client)
        assert result1.success is True
        assert result1.prompt_tokens == 100

        # Act: second call with same params should hit cache
        result2 = await call_worker(cfg, "same prompt", client=mock_client)

        # Assert: same result, client called only ONCE total
        assert result2.success is True
        assert result2.output == "cached answer"
        assert result2.prompt_tokens == 100
        assert result2.completion_tokens == 50
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_opt_out_makes_fresh_call(self) -> None:
        """When cache is disabled, each call hits the API."""
        # Arrange
        set_cache_enabled(False)
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
        )
        mock_client = _make_mock_client("fresh answer", 80, 40)

        # Act: two calls with cache disabled
        await call_worker(cfg, "same prompt", client=mock_client)
        await call_worker(cfg, "same prompt", client=mock_client)

        # Assert: client called twice
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_prompt(self) -> None:
        """Different prompts are separate cache entries."""
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
        )
        mock_client = _make_mock_client("answer", 50, 30)

        await call_worker(cfg, "prompt A", client=mock_client)
        await call_worker(cfg, "prompt B", client=mock_client)

        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache_invalidates_entries(self) -> None:
        """clear_cache() removes all cached entries."""
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
        )
        mock_client = _make_mock_client("answer", 50, 30)

        await call_worker(cfg, "prompt", client=mock_client)
        clear_cache()
        await call_worker(cfg, "prompt", client=mock_client)

        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_call_with_semaphore(self) -> None:
        cfg = WorkerConfig(
            worker_id=0,
            slug="test/model",
            latency_weight=1.0,
            cost_out_per_1m=0.5,
            cost_in_per_1m=0.1,
        )
        mock_client = _make_mock_client("ok", 10, 5)

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
        mock_client = _make_mock_client("YES", 20, 5)

        result = await call_judge(
            "Is 1/2 == 0.5?",
            judge_slug="test/judge",
            client=mock_client,
        )
        assert result.success is True
        assert result.output == "YES"
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_judge_uses_env_override(self) -> None:
        mock_client = _make_mock_client("NO", 15, 3)

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

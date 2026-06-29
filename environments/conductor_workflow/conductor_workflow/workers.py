"""Async OpenRouter worker and judge client.

Provides ``call_worker`` and ``call_judge`` for routing LLM calls
through OpenRouter.  All calls are async, concurrency-limited via
semaphores, and retry transient failures (429, 5xx, timeouts) with
bounded exponential backoff.

Worker-call cache (keyed on resolved slug + prompt + max_tokens +
temperature) avoids redundant API spend when the same call is repeated
across rollouts in a GRPO group.  **The cache saves real wallet spend
but the modeled cost (token counts) is still carried in the cached
WorkerResult** so the reward function reflects deployment cost.
Deterministic with temperature=0.

Security:
    OPENROUTER_API_KEY is read lazily at call time from ``os.environ``.
    It is never hardcoded or logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Retry policy for transient errors
MAX_RETRIES: int = 3
INITIAL_BACKOFF_S: float = 1.0
BACKOFF_MULTIPLIER: float = 2.0

# Hard per-call wall-clock cap for a worker request (seconds). Bounds a
# hanging/slow provider so it degrades to a transient failure, not a stall.
DEFAULT_WORKER_TIMEOUT_S: float = 120.0

# HTTP status codes considered transient
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker-call cache
# ---------------------------------------------------------------------------

# Module-level cache: (resolved_slug, prompt, max_tokens, temperature) -> WorkerResult
_worker_cache: dict[tuple[str, str, int, float], WorkerResult] = {}

# Global flag to disable the cache (e.g. for tests that want fresh calls)
_cache_enabled: bool = True


def clear_cache() -> None:
    """Clear the worker-call cache."""
    _worker_cache.clear()


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable the worker-call cache globally."""
    global _cache_enabled
    _cache_enabled = enabled


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerResult:
    """Result of a worker call.

    Attributes:
        output: The model response text (empty on failure).
        success: True if the call succeeded (possibly after retries).
        transient_failure: True if ALL retries were exhausted due to
            transient errors.  NOT a Conductor-caused failure.
        error_message: Human-readable error (empty on success).
        prompt_tokens: Input tokens consumed (from API usage).
        completion_tokens: Output tokens consumed (from API usage).
    """

    output: str
    success: bool
    transient_failure: bool = False
    error_message: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Client factory (lazy key resolution)
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Read OPENROUTER_API_KEY lazily from the environment.

    Raises a clear error at call time, NOT import time.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Set it before calling any worker or judge function."
        )
    return key


def build_client(
    base_url: str = OPENROUTER_BASE_URL,
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at OpenRouter."""
    resolved_key = api_key if api_key else _get_api_key()
    return AsyncOpenAI(base_url=base_url, api_key=resolved_key)


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the exception represents a transient HTTP failure."""
    # openai library wraps HTTP errors in APIStatusError
    status = getattr(exc, "status_code", None)
    if status is not None and status in TRANSIENT_STATUS_CODES:
        return True
    # Timeout errors from httpx / openai
    exc_type_name = type(exc).__name__
    if "Timeout" in exc_type_name or "timeout" in str(exc).lower():
        return True
    # Connection errors
    if "Connection" in exc_type_name:
        return True
    return False


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


async def _retry_with_backoff(
    coro_factory: Any,
    *,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_S,
    multiplier: float = BACKOFF_MULTIPLIER,
) -> WorkerResult:
    """Execute *coro_factory()* with bounded exponential backoff on transient errors.

    ``coro_factory`` is a zero-arg callable returning a fresh coroutine
    each time (so we can retry).  The factory must return
    ``(text, prompt_tokens, completion_tokens)``.
    """
    backoff = initial_backoff
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            text, p_tok, c_tok = await coro_factory()
            return WorkerResult(
                output=text,
                success=True,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
            )
        except Exception as exc:
            last_error = exc
            if _is_transient_error(exc) and attempt < max_retries:
                logger.warning(
                    "Transient error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                await asyncio.sleep(backoff)
                backoff *= multiplier
                continue
            if _is_transient_error(exc):
                # Exhausted retries on transient error
                return WorkerResult(
                    output="",
                    success=False,
                    transient_failure=True,
                    error_message=f"transient failure after {max_retries + 1} attempts: {exc}",
                )
            # Non-transient error (Conductor-caused or unknown)
            return WorkerResult(
                output="",
                success=False,
                transient_failure=False,
                error_message=str(exc),
            )

    # Should not reach here, but satisfy type checker
    return WorkerResult(
        output="",
        success=False,
        transient_failure=True,
        error_message=f"exhausted retries: {last_error}",
    )


# ---------------------------------------------------------------------------
# Worker config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerConfig:
    """Per-worker configuration from default.yaml."""

    worker_id: int
    slug: str
    latency_weight: float
    cost_out_per_1m: float
    cost_in_per_1m: float = 0.0
    openrouter_variant: str = ""


def resolve_model_slug(config: WorkerConfig) -> str:
    """Apply the openrouter_variant suffix to the slug."""
    variant = config.openrouter_variant.strip()
    if variant:
        return f"{config.slug}{variant}"
    return config.slug


# ---------------------------------------------------------------------------
# Public API: call_worker
# ---------------------------------------------------------------------------


async def call_worker(
    config: WorkerConfig,
    prompt: str,
    *,
    client: AsyncOpenAI | None = None,
    semaphore: asyncio.Semaphore | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
) -> WorkerResult:
    """Call a worker model via OpenRouter.

    Uses a module-level cache keyed on ``(resolved_slug, prompt,
    max_tokens, temperature)``.  On hit the stored ``WorkerResult``
    (including its token counts) is returned **without** an API call.
    The cache saves real wallet spend; the modeled cost (token usage
    carried in the result) still reflects what deployment would cost.

    Args:
        config: Worker configuration.
        prompt: The full prompt (task + subtask instruction + deps context).
        client: Optional pre-built client (for testing / sharing).
        semaphore: Optional concurrency limiter.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        ``WorkerResult`` with the model's text output and token usage.
    """
    model = resolve_model_slug(config)
    cache_key = (model, prompt, max_tokens, temperature)

    if _cache_enabled and cache_key in _worker_cache:
        logger.debug("Cache hit for %s (prompt len %d)", model, len(prompt))
        return _worker_cache[cache_key]

    resolved_client = client or build_client()

    async def _do_call() -> tuple[str, int, int]:
        # Hard per-call timeout so a hanging/slow provider (e.g. the :online
        # web-search variant, or a node with limited egress) becomes a bounded
        # transient failure (retried, then NOT charged to f_exec) instead of
        # stalling the whole rollout indefinitely.
        if semaphore is not None:
            async with semaphore:
                return await asyncio.wait_for(
                    _raw_chat_call(
                        resolved_client, model, prompt, max_tokens, temperature
                    ),
                    timeout=timeout_s,
                )
        return await asyncio.wait_for(
            _raw_chat_call(
                resolved_client, model, prompt, max_tokens, temperature
            ),
            timeout=timeout_s,
        )

    result = await _retry_with_backoff(_do_call)

    if _cache_enabled and result.success:
        _worker_cache[cache_key] = result

    return result


async def _raw_chat_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    """Make a raw chat completion call and return (text, prompt_tokens, completion_tokens)."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = getattr(response, "usage", None)
    p_tok = getattr(usage, "prompt_tokens", 0) or 0
    c_tok = getattr(usage, "completion_tokens", 0) or 0
    return text, p_tok, c_tok


# ---------------------------------------------------------------------------
# Public API: call_judge
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_SLUG: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_JUDGE_MAX_CONCURRENCY: int = 4
DEFAULT_JUDGE_TIMEOUT_S: int = 60


async def call_judge(
    prompt: str,
    *,
    judge_slug: str | None = None,
    client: AsyncOpenAI | None = None,
    semaphore: asyncio.Semaphore | None = None,
    timeout_s: int = DEFAULT_JUDGE_TIMEOUT_S,
    max_tokens: int = 512,
) -> WorkerResult:
    """Call the judge model (Nemotron / env override).

    The judge slug can be overridden via the JUDGE_MODEL env var.
    Judge cost is 0 (free tier) so token counts are captured but not
    priced in the reward.
    """
    slug = judge_slug or os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_SLUG)
    resolved_client = client or build_client()

    async def _do_call() -> tuple[str, int, int]:
        if semaphore is not None:
            async with semaphore:
                return await asyncio.wait_for(
                    _raw_chat_call(resolved_client, slug, prompt, max_tokens, 0.0),
                    timeout=timeout_s,
                )
        return await asyncio.wait_for(
            _raw_chat_call(resolved_client, slug, prompt, max_tokens, 0.0),
            timeout=timeout_s,
        )

    return await _retry_with_backoff(_do_call)

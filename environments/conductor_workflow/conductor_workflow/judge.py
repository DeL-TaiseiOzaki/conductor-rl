"""Concrete TinyV fallback backed by workers.call_judge.

Implements the ``TinyVFallback`` protocol from ``graders.math_verify``
by calling the configured judge model (Nemotron by default) via
OpenRouter.  Fully async; injectable for testing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from conductor_workflow.workers import call_judge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE: str = """You are a mathematical equivalence judge.

Determine whether the following two mathematical expressions/answers are equivalent.
Consider: different notations, simplification levels, equivalent forms (e.g. 1/2 = 0.5).

Candidate answer: {candidate}
Gold answer: {gold}

Respond with ONLY "YES" if they are mathematically equivalent, or "NO" if they are not.
"""


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class NemotronJudge:
    """TinyV fallback using the Nemotron judge model via OpenRouter.

    Satisfies the ``TinyVFallback`` protocol.
    """

    def __init__(
        self,
        *,
        judge_slug: str | None = None,
        client: Any = None,
        semaphore: asyncio.Semaphore | None = None,
        timeout_s: int = 60,
    ) -> None:
        self._judge_slug = judge_slug
        self._client = client
        self._semaphore = semaphore
        self._timeout_s = timeout_s

    async def check_equivalence(
        self,
        candidate_answer: str,
        gold_answer: str,
        *,
        context: str | None = None,
    ) -> bool:
        """Call the judge model to check mathematical equivalence."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            candidate=candidate_answer,
            gold=gold_answer,
        )

        result = await call_judge(
            prompt,
            judge_slug=self._judge_slug,
            client=self._client,
            semaphore=self._semaphore,
            timeout_s=self._timeout_s,
        )

        if not result.success:
            logger.warning("Judge call failed: %s", result.error_message)
            return False

        # Parse YES/NO from response
        response_text = result.output.strip().upper()
        return response_text.startswith("YES")

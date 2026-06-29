"""Verifiers SingleTurnEnv wiring for Conductor-RL.

Builds the dataset, parser, rubric (with four async reward functions),
and returns a fully configured ``vf.SingleTurnEnv``.

Separated from ``__init__.py`` so that importing the package for
pure parser/reward usage does not pull in verifiers/datasets.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from conductor_workflow.config import (
    ConductorConfig,
    load_config,
)
from conductor_workflow.executor import ExecutionResult, execute_dag
from conductor_workflow.graders.code_exec import extract_code, grade_code
from conductor_workflow.graders.math_verify import grade_math_async
from conductor_workflow.graders.mcq_exact import grade_mcq
from conductor_workflow.judge import NemotronJudge
from conductor_workflow.parser import ParseResult, parse_workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Packaged asset helpers
# ---------------------------------------------------------------------------

_ASSETS_REF = importlib.resources.files("conductor_workflow") / "assets"


def _packaged_pilot_text() -> str:
    """Read the bundled pilot.jsonl via importlib.resources."""
    return (_ASSETS_REF / "pilot" / "pilot.jsonl").read_text(encoding="utf-8")


def _packaged_system_prompt_text() -> str:
    """Read the bundled system prompt via importlib.resources."""
    return (_ASSETS_REF / "conductor_system_prompt.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

CORRECT_THRESHOLD_BY_VERIFIER: dict[str, float] = {
    "code_exec": 0.5,
    "mcq_exact": 1.0,
    "math_verify": 1.0,
}


def _load_pilot_dataset(
    dataset_path: str | Path | None = None,
    system_prompt: str = "",
    clusters: list[str] | None = None,
) -> Dataset:
    """Load pilot JSONL into an HF Dataset with verifiers-compatible columns.

    Required output columns: prompt (list[dict]), answer (str), info (dict).

    Args:
        dataset_path: Explicit path to a JSONL file.  If *None*, reads
            the bundled ``assets/pilot/pilot.jsonl`` from the installed
            package via ``importlib.resources``.
        system_prompt: System prompt text prepended to every row.
        clusters: Optional cluster filter (e.g. ``["code"]``).
    """
    if dataset_path is not None:
        path = Path(dataset_path)
        with open(path, encoding="utf-8") as fh:
            raw_text = fh.read()
    else:
        raw_text = _packaged_pilot_text()

    rows: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)

        # Filter by cluster if requested
        if clusters and item.get("cluster") not in clusters:
            continue

        # Build chat-format prompt with system prompt
        prompt_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]

        # Info carries all metadata needed by reward functions.
        # _task_prompt = the original task text (also the user message),
        # used by the executor to build each worker's context. Stored here
        # so reward funcs get it from info without re-parsing prompt messages.
        info: dict[str, Any] = {
            "id": item.get("id", ""),
            "cluster": item.get("cluster", ""),
            "verifier": item.get("verifier", ""),
            "verifier_spec": item.get("verifier_spec", {}),
            "gold": item.get("gold"),
            "difficulty": item.get("difficulty", ""),
            "_task_prompt": item["prompt"],
        }

        rows.append(
            {
                "prompt": prompt_messages,
                "answer": item.get("gold") or "",
                "info": info,
            }
        )

    if not rows:
        source = str(dataset_path) if dataset_path is not None else "<bundled>"
        raise ValueError(f"No items loaded from {source} (clusters={clusters})")

    return Dataset.from_list(rows)


def _load_system_prompt(prompt_path: str | Path | None = None) -> str:
    """Load the system prompt from a markdown file.

    Args:
        prompt_path: Explicit file path.  If *None*, reads the bundled
            ``assets/conductor_system_prompt.md`` via ``importlib.resources``.
    """
    if prompt_path is not None:
        path = Path(prompt_path)
        return path.read_text(encoding="utf-8").strip()
    return _packaged_system_prompt_text().strip()


# ---------------------------------------------------------------------------
# WorkflowParser (verifiers.Parser subclass)
# ---------------------------------------------------------------------------


class WorkflowParser(vf.Parser):
    """Parse ```workflow JSON blocks from Conductor model output.

    Subclasses ``vf.Parser`` to integrate with the Rubric's
    dependency injection (``parser`` parameter in reward functions).
    """

    def parse(self, text: str) -> ParseResult:
        """Extract and validate the workflow JSON from raw text."""
        return parse_workflow(text)

    def parse_answer(self, completion: Any) -> str | None:
        """Extract final answer text from completion messages.

        Returns the raw text of the last assistant message (not the
        parsed workflow -- reward functions call ``parse`` separately
        to get the full ``ParseResult`` with ``f_fmt``).
        """
        if isinstance(completion, str):
            return completion
        # completion is list[dict] (Messages)
        assistant_msgs = [
            m
            for m in completion
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "assistant"
        ]
        if not assistant_msgs:
            return None
        msg = assistant_msgs[-1]
        content = (
            msg.get("content", "")
            if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )
        return self._content_to_text(content) if content else None

    def get_format_reward_func(self) -> Any:
        """Return a reward function for format adherence (f_fmt).

        Uses the parser's graded partial-credit scoring.
        """

        async def format_reward(
            completion: list[dict[str, str]],
            **kwargs: Any,
        ) -> float:
            text = ""
            if completion:
                msg = completion[-1]
                text = msg.get("content", "") if isinstance(msg, dict) else ""
            result = parse_workflow(text)
            return result.f_fmt

        return format_reward


# ---------------------------------------------------------------------------
# Async reward functions
# ---------------------------------------------------------------------------

DAG_RESULT_KEY = "_dag_result"
PARSE_RESULT_KEY = "_parse_result"


async def _ensure_dag_executed(
    state: dict[str, Any],
    completion: list[dict[str, str]],
    info: dict[str, Any],
    config: ConductorConfig,
    client: Any = None,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[ParseResult, ExecutionResult]:
    """Parse + execute the DAG, caching results in state.

    Called by multiple reward functions; only executes once per rollout.
    """
    if DAG_RESULT_KEY in state and PARSE_RESULT_KEY in state:
        return state[PARSE_RESULT_KEY], state[DAG_RESULT_KEY]

    # Extract text from last assistant message
    text = ""
    if completion:
        msg = completion[-1]
        text = msg.get("content", "") if isinstance(msg, dict) else ""

    parse_result = parse_workflow(text)
    state[PARSE_RESULT_KEY] = parse_result

    if not parse_result.valid:
        exec_result = ExecutionResult(f_exec=0.0, b_eff=1.0)
        state[DAG_RESULT_KEY] = exec_result
        return parse_result, exec_result

    # Get task prompt from info (original user task text)
    task_prompt = info.get("verifier_spec", {}).get("task_prompt", "")
    if not task_prompt:
        # Fall back: reconstruct from the user message if available
        task_prompt = info.get("_task_prompt", "")

    exec_result = await execute_dag(
        parse_result,
        task_prompt,
        config.worker_configs,
        client=client,
        semaphore=semaphore,
        lambda_latency=config.lambda_latency,
        mu_cost=config.mu_cost,
    )
    state[DAG_RESULT_KEY] = exec_result
    return parse_result, exec_result


async def _grade_correctness(
    final_output: str,
    info: dict[str, Any],
    config: ConductorConfig,
    judge: NemotronJudge | None = None,
) -> float:
    """Dispatch to the correct grader based on verifier type."""
    verifier = info.get("verifier", "")
    verifier_spec = info.get("verifier_spec", {})
    gold = info.get("gold")

    if verifier == "code_exec":
        tests = verifier_spec.get("tests", [])
        time_limit = verifier_spec.get("time_limit_s", 5)
        candidate_code = extract_code(final_output)
        result = grade_code(candidate_code, tests, time_limit_s=time_limit)
        if config.code_s_correct == "fraction":
            return result.s_correct
        return 1.0 if result.all_pass else 0.0

    if verifier == "mcq_exact":
        if gold is None:
            return 0.0
        return grade_mcq(final_output, str(gold))

    if verifier == "math_verify":
        if gold is None:
            return 0.0
        return await grade_math_async(
            final_output,
            str(gold),
            tiny_v_fallback=judge if config.judge.enabled else None,
            only_on_uncertain=config.judge.only_on_uncertain,
        )

    logger.warning("Unknown verifier type: %s", verifier)
    return 0.0


# ---------------------------------------------------------------------------
# Rubric reward function factories
# ---------------------------------------------------------------------------


def _make_reward_functions(
    config: ConductorConfig,
    client: Any = None,
    judge: NemotronJudge | None = None,
) -> list[tuple[Any, float]]:
    """Create the four async reward functions with their weights.

    Returns list of (func, weight) tuples in execution order:
    format -> correctness -> execution -> efficiency.

    Execution order matters: correctness caches the DAG result
    in state for execution and efficiency to reuse.
    """
    worker_semaphore = asyncio.Semaphore(32)

    # 1. Format reward
    async def format_reward(
        completion: list[dict[str, str]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        text = ""
        if completion:
            msg = completion[-1]
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        pr = parse_workflow(text)
        state[PARSE_RESULT_KEY] = pr
        return pr.f_fmt

    # 2. Correctness reward (triggers DAG execution)
    async def correctness_reward(
        completion: list[dict[str, str]],
        info: dict[str, Any],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        _pr, exec_result = await _ensure_dag_executed(
            state,
            completion,
            info,
            config,
            client=client,
            semaphore=worker_semaphore,
        )
        return await _grade_correctness(
            exec_result.final_output,
            info,
            config,
            judge=judge,
        )

    # 3. Execution feasibility reward
    async def execution_reward(
        completion: list[dict[str, str]],
        info: dict[str, Any],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        _pr, exec_result = await _ensure_dag_executed(
            state,
            completion,
            info,
            config,
            client=client,
            semaphore=worker_semaphore,
        )
        return exec_result.f_exec

    # 4. Efficiency bonus (gated by correctness in compute_reward,
    #    but here we just return the raw b_eff)
    async def efficiency_reward(
        completion: list[dict[str, str]],
        info: dict[str, Any],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        _pr, exec_result = await _ensure_dag_executed(
            state,
            completion,
            info,
            config,
            client=client,
            semaphore=worker_semaphore,
        )
        return exec_result.b_eff

    return [
        (format_reward, config.reward_weights.w_fmt),
        (correctness_reward, config.reward_weights.w_corr),
        (execution_reward, config.reward_weights.w_exec),
        (efficiency_reward, config.reward_weights.w_eff),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_environment(**kwargs: Any) -> vf.SingleTurnEnv:
    """Build and return the Conductor-RL SingleTurnEnv.

    Args (via kwargs):
        dataset_path: Override path to pilot JSONL.
        config_path: Override path to YAML config.
        clusters: Filter dataset to specific clusters.
        worker_client: Override AsyncOpenAI client (for testing).
        judge_client: Override AsyncOpenAI client for judge (for testing).
        skip_key_check: Skip OPENROUTER_API_KEY validation.

    Returns:
        Configured ``vf.SingleTurnEnv``.
    """
    config_path = kwargs.get("config_path")
    config = load_config(config_path)

    # Optionally validate API key (skip in tests)
    skip_key = kwargs.get("skip_key_check", False)
    if not skip_key:
        vf.ensure_keys(["OPENROUTER_API_KEY"])

    # Load system prompt — use packaged asset unless an explicit override
    # path is provided via kwargs or the config specifies an absolute path.
    prompt_path_override = kwargs.get("prompt_path")
    system_prompt = _load_system_prompt(prompt_path_override)

    # Load dataset — use packaged asset unless caller passes dataset_path.
    dataset_path = kwargs.get("dataset_path")
    clusters = kwargs.get("clusters", config.clusters)
    dataset = _load_pilot_dataset(dataset_path, system_prompt, clusters)

    # Build clients
    worker_client = kwargs.get("worker_client")
    judge_client = kwargs.get("judge_client")

    # Build judge
    judge: NemotronJudge | None = None
    if config.judge.enabled:
        judge_sem = asyncio.Semaphore(config.judge.max_concurrency)
        judge = NemotronJudge(
            judge_slug=config.judge.slug,
            client=judge_client,
            semaphore=judge_sem,
            timeout_s=config.judge.timeout_s,
        )

    # Build parser
    parser = WorkflowParser()

    # Build rubric with reward functions
    rubric = vf.Rubric(parser=parser)
    reward_funcs = _make_reward_functions(
        config,
        client=worker_client,
        judge=judge,
    )
    for func, weight in reward_funcs:
        rubric.add_reward_func(func, weight=weight)

    # Build environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
    )

    return env

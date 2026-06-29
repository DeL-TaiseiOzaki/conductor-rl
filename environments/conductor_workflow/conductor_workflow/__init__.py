"""Conductor-workflow: verifiers SingleTurnEnv for multi-agent orchestration.

Hub name: DeL-TaiseiOzaki/conductor-workflow

Modules:
    parser      -- parse + validate the ```workflow JSON block
    reward      -- tiered shaped reward (s_correct, f_fmt, f_exec, b_eff)
    graders/    -- per-cluster verifiers (code_exec, mcq_exact, math_verify)
    workers     -- async OpenRouter client (worker + judge calls)
    executor    -- async DAG executor with efficiency accounting
    judge       -- concrete TinyV fallback (NemotronJudge)
    config      -- YAML config loader

Entry point:
    ``load_environment(**kwargs)`` builds a ``verifiers.SingleTurnEnv``
    with the pilot dataset wired to a Rubric carrying four async reward
    functions: format, correctness, execution, efficiency.
"""

from conductor_workflow.parser import ParseResult, parse_workflow
from conductor_workflow.reward import compute_reward

__all__ = [
    "ParseResult",
    "compute_reward",
    "load_environment",
    "parse_workflow",
]


def load_environment(**kwargs: object) -> object:
    """Load the Conductor-RL SingleTurnEnv.

    Lazy import to avoid pulling in verifiers/datasets at import time
    when only using the pure parser/reward modules.

    Accepted kwargs:
        dataset_path: str -- path to pilot JSONL (default: config).
        config_path: str -- path to YAML config (default: configs/default.yaml).
        clusters: list[str] -- filter to specific clusters.
        worker_client: AsyncOpenAI -- override for testing.
        judge_client: AsyncOpenAI -- override for testing.
        skip_key_check: bool -- skip OPENROUTER_API_KEY validation.

    Returns:
        A ``verifiers.SingleTurnEnv`` instance.
    """
    # Deferred import to keep top-level lightweight
    from conductor_workflow._env_wiring import build_environment

    return build_environment(**kwargs)

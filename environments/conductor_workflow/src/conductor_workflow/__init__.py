"""Conductor-workflow: pure, framework-independent core logic.

Hub name: DeL-TaiseiOzaki/conductor-workflow

Modules:
    parser      -- parse + validate the ```workflow JSON block
    reward      -- tiered shaped reward (s_correct, f_fmt, f_exec, b_eff)
    graders/    -- per-cluster verifiers (code_exec, mcq_exact, math_verify)

Planned wiring (next phase -- ``load_environment``):
    - Subclass ``verifiers.Parser`` as ``WorkflowParser``
      (``parse`` extracts the ```workflow JSON block;
       ``get_format_reward_func`` returns a graded partial-credit scorer).
    - Build a ``verifiers.Rubric`` with four async reward functions
      registered in order: format -> execution -> correctness -> efficiency.
    - Wrap in ``verifiers.SingleTurnEnv`` (one turn = one workflow generation).
    - ``load_environment`` accepts ``dataset_path``, ``openrouter_base_url``,
      ``worker_models`` kwargs and validates ``OPENROUTER_API_KEY``.
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

    Not yet wired -- see module docstring for planned implementation.
    """
    raise NotImplementedError(
        "load_environment is not yet wired. "
        "Phase 2 will integrate verifiers.SingleTurnEnv + Rubric + "
        "async worker executor. See module docstring for the plan."
    )

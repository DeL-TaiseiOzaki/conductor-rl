"""Tiered shaped reward for Conductor-RL GRPO training.

Formula (from ``docs/reward-spec.md``):

    R = w_corr * s_correct
      + w_fmt  * f_fmt
      + w_exec * f_exec
      + w_eff  * b_eff * 1[correct]

where ``1[correct]`` gates the efficiency bonus: it is 1 when
``s_correct >= correct_threshold``, else 0.

**Invariant**: ``max(wrong) = w_fmt + w_exec = 0.2 < min(correct)``.
When correct (s_correct >= threshold, minimum s_correct = threshold = 0.5
for code fraction scoring), the minimum R = 0.5*1.0 + 0 + 0 + 0 = 0.5,
which exceeds 0.2.  For binary verifiers (threshold = 1.0),
min(correct) = 1.0 + 0 + 0 + 0 = 1.0 >> 0.2.

``correct_threshold`` design decision:
  - For binary verifiers (mcq_exact, math_verify): threshold = 1.0
    (the only possible correct score).
  - For fractional code verifier: threshold = 0.5 is recommended so
    partial solutions that pass >= 50 % of tests still get the efficiency
    bonus.  The invariant holds because min(correct) = 0.5 > 0.2.
  - Callers should set the threshold per item's verifier type.  The
    default here is 1.0 (safe for all verifier types).

Pure, synchronous, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Default weights (locked v1 from configs/default.yaml)
# ---------------------------------------------------------------------------

DEFAULT_W_CORR: float = 1.0
DEFAULT_W_FMT: float = 0.1
DEFAULT_W_EXEC: float = 0.1
DEFAULT_W_EFF: float = 0.2

DEFAULT_CORRECT_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Weights container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    """Named weight bundle, avoiding magic numbers."""

    w_corr: float = DEFAULT_W_CORR
    w_fmt: float = DEFAULT_W_FMT
    w_exec: float = DEFAULT_W_EXEC
    w_eff: float = DEFAULT_W_EFF


DEFAULT_WEIGHTS: RewardWeights = RewardWeights()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reward(
    s_correct: float,
    f_fmt: float,
    f_exec: float,
    b_eff: float,
    *,
    weights: RewardWeights | None = None,
    correct_threshold: float = DEFAULT_CORRECT_THRESHOLD,
) -> float:
    """Compute the tiered shaped reward.

    Args:
        s_correct: Correctness score in [0, 1].
        f_fmt: Format validity score in [0, 1].
        f_exec: Execution feasibility score in [0, 1].
        b_eff: Efficiency bonus in [0, 1], gated by correctness.
        weights: Reward term weights (defaults to locked v1 values).
        correct_threshold: Minimum ``s_correct`` to activate the
            efficiency gate.  Default 1.0 (binary verifiers).

    Returns:
        Scalar reward R >= 0.

    Raises:
        ValueError: If any input is outside its valid range.
    """
    _validate_unit("s_correct", s_correct)
    _validate_unit("f_fmt", f_fmt)
    _validate_unit("f_exec", f_exec)
    _validate_unit("b_eff", b_eff)

    if not (0.0 <= correct_threshold <= 1.0):
        raise ValueError(
            f"correct_threshold must be in [0, 1], got {correct_threshold}"
        )

    w = weights if weights is not None else DEFAULT_WEIGHTS

    is_correct = 1.0 if s_correct >= correct_threshold else 0.0

    reward = (
        w.w_corr * s_correct
        + w.w_fmt * f_fmt
        + w.w_exec * f_exec
        + w.w_eff * b_eff * is_correct
    )
    return reward


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_unit(name: str, value: float) -> None:
    """Raise ``ValueError`` if *value* is not in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")

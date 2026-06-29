"""Tests for conductor_workflow.reward."""

from __future__ import annotations

import pytest

from conductor_workflow.reward import (
    DEFAULT_CORRECT_THRESHOLD,
    DEFAULT_WEIGHTS,
    RewardWeights,
    compute_reward,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestComputeRewardHappyPath:
    """Basic reward computation with default weights."""

    def test_fully_correct_perfect_scores(self) -> None:
        # Arrange / Act
        r = compute_reward(s_correct=1.0, f_fmt=1.0, f_exec=1.0, b_eff=1.0)

        # Assert: 1.0*1.0 + 0.1*1.0 + 0.1*1.0 + 0.2*1.0*1 = 1.4
        assert r == pytest.approx(1.4)

    def test_fully_wrong_max_format(self) -> None:
        # max(wrong) = w_fmt + w_exec = 0.2
        r = compute_reward(s_correct=0.0, f_fmt=1.0, f_exec=1.0, b_eff=1.0)
        assert r == pytest.approx(0.2)

    def test_correct_but_no_bonuses(self) -> None:
        r = compute_reward(s_correct=1.0, f_fmt=0.0, f_exec=0.0, b_eff=0.0)
        assert r == pytest.approx(1.0)

    def test_all_zeros(self) -> None:
        r = compute_reward(s_correct=0.0, f_fmt=0.0, f_exec=0.0, b_eff=0.0)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Invariant: max(wrong) < min(correct)
# ---------------------------------------------------------------------------


class TestRewardInvariant:
    """The tiered reward invariant must hold."""

    def test_max_wrong_less_than_min_correct_binary(self) -> None:
        """With binary verifier (threshold=1.0)."""
        max_wrong = compute_reward(s_correct=0.0, f_fmt=1.0, f_exec=1.0, b_eff=1.0)
        min_correct = compute_reward(s_correct=1.0, f_fmt=0.0, f_exec=0.0, b_eff=0.0)
        assert max_wrong < min_correct

    def test_max_wrong_less_than_min_correct_fractional(self) -> None:
        """With fractional code verifier (threshold=0.5).

        The spec invariant ``max(wrong) = w_fmt + w_exec = 0.2`` assumes
        s_correct=0 for "maximally wrong".  With fractional scoring,
        a near-threshold wrong answer (e.g. s_correct=0.49) can exceed
        min(correct) in raw score, but the spec invariant is about the
        zero-correct worst case, which always holds.
        """
        min_correct = compute_reward(
            s_correct=0.5,
            f_fmt=0.0,
            f_exec=0.0,
            b_eff=0.0,
            correct_threshold=0.5,
        )
        # Spec invariant: worst-case wrong (s_correct=0) with perfect
        # format + exec still only gets 0.2
        max_wrong_worst = compute_reward(
            s_correct=0.0,
            f_fmt=1.0,
            f_exec=1.0,
            b_eff=1.0,
            correct_threshold=0.5,
        )
        assert max_wrong_worst == pytest.approx(0.2)
        assert max_wrong_worst < min_correct


# ---------------------------------------------------------------------------
# Efficiency gating
# ---------------------------------------------------------------------------


class TestEfficiencyGating:
    """b_eff is only counted when correct."""

    def test_efficiency_gated_when_wrong(self) -> None:
        r_with_eff = compute_reward(s_correct=0.0, f_fmt=0.0, f_exec=0.0, b_eff=1.0)
        r_without_eff = compute_reward(s_correct=0.0, f_fmt=0.0, f_exec=0.0, b_eff=0.0)
        assert r_with_eff == r_without_eff

    def test_efficiency_active_when_correct(self) -> None:
        r_with_eff = compute_reward(s_correct=1.0, f_fmt=0.0, f_exec=0.0, b_eff=1.0)
        r_without_eff = compute_reward(s_correct=1.0, f_fmt=0.0, f_exec=0.0, b_eff=0.0)
        assert r_with_eff > r_without_eff

    def test_custom_threshold(self) -> None:
        # s_correct=0.6 with threshold=0.5 -> correct
        r = compute_reward(
            s_correct=0.6,
            f_fmt=0.0,
            f_exec=0.0,
            b_eff=1.0,
            correct_threshold=0.5,
        )
        # 1.0*0.6 + 0 + 0 + 0.2*1.0*1 = 0.8
        assert r == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------


class TestCustomWeights:
    """Custom weight bundles."""

    def test_custom_weights_applied(self) -> None:
        w = RewardWeights(w_corr=2.0, w_fmt=0.5, w_exec=0.5, w_eff=0.0)
        r = compute_reward(s_correct=1.0, f_fmt=1.0, f_exec=1.0, b_eff=1.0, weights=w)
        assert r == pytest.approx(3.0)

    def test_default_weights_match_spec(self) -> None:
        assert DEFAULT_WEIGHTS.w_corr == 1.0
        assert DEFAULT_WEIGHTS.w_fmt == 0.1
        assert DEFAULT_WEIGHTS.w_exec == 0.1
        assert DEFAULT_WEIGHTS.w_eff == 0.2

    def test_default_threshold_is_one(self) -> None:
        assert DEFAULT_CORRECT_THRESHOLD == 1.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestComputeRewardErrors:
    """Invalid inputs should raise ValueError."""

    def test_s_correct_negative(self) -> None:
        with pytest.raises(ValueError, match="s_correct"):
            compute_reward(s_correct=-0.1, f_fmt=0.0, f_exec=0.0, b_eff=0.0)

    def test_s_correct_above_one(self) -> None:
        with pytest.raises(ValueError, match="s_correct"):
            compute_reward(s_correct=1.1, f_fmt=0.0, f_exec=0.0, b_eff=0.0)

    def test_f_fmt_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="f_fmt"):
            compute_reward(s_correct=0.0, f_fmt=2.0, f_exec=0.0, b_eff=0.0)

    def test_f_exec_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="f_exec"):
            compute_reward(s_correct=0.0, f_fmt=0.0, f_exec=-0.5, b_eff=0.0)

    def test_b_eff_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="b_eff"):
            compute_reward(s_correct=0.0, f_fmt=0.0, f_exec=0.0, b_eff=1.5)

    def test_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="correct_threshold"):
            compute_reward(
                s_correct=0.0,
                f_fmt=0.0,
                f_exec=0.0,
                b_eff=0.0,
                correct_threshold=1.5,
            )

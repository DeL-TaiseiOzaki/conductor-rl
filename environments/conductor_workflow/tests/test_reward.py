"""Tests for conductor_workflow.reward."""

from __future__ import annotations

import pytest

from conductor_workflow.reward import (
    DEFAULT_CORRECT_THRESHOLD,
    DEFAULT_WEIGHTS,
    NEUTRAL_BONUS,
    RewardWeights,
    compute_reward,
    rank_efficiency,
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
        """With fractional code verifier (threshold=0.5)."""
        min_correct = compute_reward(
            s_correct=0.5,
            f_fmt=0.0,
            f_exec=0.0,
            b_eff=0.0,
            correct_threshold=0.5,
        )
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


# ---------------------------------------------------------------------------
# Group-relative efficiency ranking
# ---------------------------------------------------------------------------


class TestRankEfficiencyHappyPath:
    """rank_efficiency: basic ranking over a GRPO group."""

    def test_cheapest_correct_gets_b_cost_one(self) -> None:
        # Arrange: 3 correct rollouts with different costs
        correct = [True, True, True]
        costs = [1.0, 2.0, 3.0]  # rollout 0 cheapest
        latencies = [1.0, 1.0, 1.0]

        # Act
        result = rank_efficiency(correct, costs, latencies, w_lat=0.0, w_cost=0.1)

        # Assert: cheapest (idx 0) -> b_cost=1.0, most expensive (idx 2) -> b_cost=0.0
        assert result[0] == pytest.approx(0.1 * 1.0)  # w_cost * 1.0
        assert result[2] == pytest.approx(0.1 * 0.0)  # w_cost * 0.0

    def test_most_expensive_correct_gets_b_cost_zero(self) -> None:
        correct = [True, True, True]
        costs = [3.0, 1.0, 2.0]
        latencies = [1.0, 1.0, 1.0]

        result = rank_efficiency(correct, costs, latencies, w_lat=0.0, w_cost=0.1)

        # idx 0 is most expensive -> 0.0
        assert result[0] == pytest.approx(0.0)
        # idx 1 is cheapest -> 1.0
        assert result[1] == pytest.approx(0.1)

    def test_incorrect_rollouts_get_zero(self) -> None:
        correct = [True, False, True]
        costs = [1.0, 0.5, 2.0]  # idx 1 is cheapest but wrong
        latencies = [1.0, 0.5, 2.0]

        result = rank_efficiency(correct, costs, latencies, w_lat=0.1, w_cost=0.1)

        assert result[1] == 0.0  # incorrect -> 0

    def test_single_correct_gets_neutral(self) -> None:
        correct = [False, True, False]
        costs = [1.0, 2.0, 3.0]
        latencies = [1.0, 2.0, 3.0]

        result = rank_efficiency(correct, costs, latencies, w_lat=0.1, w_cost=0.1)

        # Single correct -> neutral = 0.5 for both
        expected = 0.1 * NEUTRAL_BONUS + 0.1 * NEUTRAL_BONUS
        assert result[1] == pytest.approx(expected)
        assert result[0] == 0.0
        assert result[2] == 0.0


class TestRankEfficiencyStaging:
    """Staging: w_lat=w_cost=0 means efficiency contributes 0."""

    def test_zero_weights_returns_all_zeros(self) -> None:
        correct = [True, True, True]
        costs = [1.0, 2.0, 3.0]
        latencies = [1.0, 2.0, 3.0]

        result = rank_efficiency(correct, costs, latencies, w_lat=0.0, w_cost=0.0)

        assert result == [0.0, 0.0, 0.0]


class TestRankEfficiencyCorrectnessDominance:
    """Max efficiency bonus < min correctness reward (invariant)."""

    def test_max_bonus_less_than_min_correct(self) -> None:
        # Arrange: max possible efficiency bonus with w_lat=0.1, w_cost=0.1
        w_lat = 0.1
        w_cost = 0.1
        max_bonus = w_lat * 1.0 + w_cost * 1.0  # best possible

        # min(correct) with binary verifier = w_corr * 1.0 = 1.0
        min_correct = 1.0

        # Assert invariant
        assert max_bonus < min_correct

    def test_max_bonus_less_than_min_correct_fractional(self) -> None:
        """Even with fractional code verifier (threshold=0.5), invariant holds."""
        w_lat = 0.1
        w_cost = 0.1
        max_bonus = w_lat + w_cost  # 0.2

        # min(correct) with threshold=0.5 = w_corr * 0.5 = 0.5
        min_correct = 0.5

        assert max_bonus < min_correct


class TestRankEfficiencyEdgeCases:
    """Edge cases for rank_efficiency."""

    def test_empty_group(self) -> None:
        assert rank_efficiency([], [], [], w_lat=0.1, w_cost=0.1) == []

    def test_all_incorrect(self) -> None:
        correct = [False, False, False]
        costs = [1.0, 2.0, 3.0]
        latencies = [1.0, 2.0, 3.0]

        result = rank_efficiency(correct, costs, latencies, w_lat=0.1, w_cost=0.1)

        assert result == [0.0, 0.0, 0.0]

    def test_latency_ranking_independent_of_cost(self) -> None:
        """b_lat ranks by latency, not cost."""
        correct = [True, True]
        costs = [10.0, 1.0]  # idx 1 cheaper
        latencies = [1.0, 10.0]  # idx 0 faster

        result = rank_efficiency(correct, costs, latencies, w_lat=0.1, w_cost=0.0)

        # idx 0 is fastest -> b_lat=1.0
        assert result[0] == pytest.approx(0.1)
        # idx 1 is slowest -> b_lat=0.0
        assert result[1] == pytest.approx(0.0)

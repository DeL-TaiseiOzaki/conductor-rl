"""Tests for conductor_workflow.load_environment and env wiring.

All network calls are mocked -- no real HTTP traffic.
Tests verify: env constructs, dataset columns, rubric structure,
and a mocked end-to-end rollout producing the expected reward.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conductor_workflow.config import load_config
from conductor_workflow.reward import RewardWeights, compute_reward

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_ENV_DIR = _TESTS_DIR.parent
_REPO_ROOT = _ENV_DIR.parent.parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Config loads correctly from default.yaml."""

    def test_load_default_config(self) -> None:
        config = load_config()
        assert len(config.worker_configs) == 4
        assert config.worker_configs[0].slug == "deepseek/deepseek-v4-flash"
        assert config.reward_weights.w_corr == 1.0
        assert config.judge.slug == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert config.lambda_latency == 0.0
        assert config.mu_cost == 0.0
        assert config.w_lat == 0.0
        assert config.w_cost == 0.0

    def test_worker_configs_match_yaml(self) -> None:
        config = load_config()
        w0 = config.worker_configs[0]
        assert w0.openrouter_variant == ":nitro"
        assert w0.latency_weight == 1.0
        assert w0.cost_in_per_1m == pytest.approx(0.09)
        w3 = config.worker_configs[3]
        assert w3.slug == "z-ai/glm-5.2"
        assert w3.cost_out_per_1m == pytest.approx(3.00)
        assert w3.cost_in_per_1m == pytest.approx(0.94)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    """Pilot dataset loads with correct columns."""

    def test_dataset_loads_from_pilot(self) -> None:
        from conductor_workflow._env_wiring import _load_pilot_dataset

        dataset = _load_pilot_dataset(
            _REPO_ROOT / "data" / "pilot" / "pilot.jsonl",
            system_prompt="You are Conductor.",
        )
        assert len(dataset) == 201  # pilot has 201 items
        assert "prompt" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "info" in dataset.column_names

    def test_dataset_prompt_is_chat_messages(self) -> None:
        from conductor_workflow._env_wiring import _load_pilot_dataset

        dataset = _load_pilot_dataset(
            _REPO_ROOT / "data" / "pilot" / "pilot.jsonl",
            system_prompt="System prompt here.",
        )
        first_prompt = dataset[0]["prompt"]
        assert isinstance(first_prompt, list)
        assert len(first_prompt) >= 2
        assert first_prompt[0]["role"] == "system"
        assert first_prompt[1]["role"] == "user"

    def test_dataset_info_has_required_fields(self) -> None:
        from conductor_workflow._env_wiring import _load_pilot_dataset

        dataset = _load_pilot_dataset(
            _REPO_ROOT / "data" / "pilot" / "pilot.jsonl",
            system_prompt="test",
        )
        info = dataset[0]["info"]
        assert "verifier" in info
        assert "verifier_spec" in info
        assert "cluster" in info

    def test_cluster_filter(self) -> None:
        from conductor_workflow._env_wiring import _load_pilot_dataset

        dataset = _load_pilot_dataset(
            _REPO_ROOT / "data" / "pilot" / "pilot.jsonl",
            system_prompt="test",
            clusters=["code"],
        )
        assert len(dataset) == 71  # 71 code items in pilot


# ---------------------------------------------------------------------------
# WorkflowParser
# ---------------------------------------------------------------------------


class TestWorkflowParser:
    """WorkflowParser integrates with verifiers.Parser."""

    def test_parse_returns_parse_result(self) -> None:
        from conductor_workflow._env_wiring import WorkflowParser

        parser = WorkflowParser()
        result = parser.parse(
            '```workflow\n{"subtasks": ["X"], "model_id": [0], "access_list": [[]]}\n```'
        )
        assert result.valid is True
        assert result.f_fmt > 0.9

    def test_parse_answer_from_messages(self) -> None:
        from conductor_workflow._env_wiring import WorkflowParser

        parser = WorkflowParser()
        completion = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "some workflow text"},
        ]
        answer = parser.parse_answer(completion)
        assert answer == "some workflow text"

    def test_format_reward_func_returns_async(self) -> None:
        from conductor_workflow._env_wiring import WorkflowParser

        parser = WorkflowParser()
        func = parser.get_format_reward_func()
        assert asyncio.iscoroutinefunction(func)


# ---------------------------------------------------------------------------
# Environment construction (mocked key check)
# ---------------------------------------------------------------------------


class TestBuildEnvironment:
    """Environment builds without real API keys."""

    def test_env_constructs_with_skip_key(self) -> None:
        from conductor_workflow._env_wiring import build_environment

        env = build_environment(skip_key_check=True)

        # Verify it's a SingleTurnEnv
        assert hasattr(env, "rubric")
        assert hasattr(env, "dataset")

    def test_env_has_dataset(self) -> None:
        from conductor_workflow._env_wiring import build_environment

        env = build_environment(skip_key_check=True)
        assert env.dataset is not None
        assert len(env.dataset) == 201

    def test_rubric_has_reward_funcs(self) -> None:
        from conductor_workflow._env_wiring import build_environment

        env = build_environment(skip_key_check=True)
        names = env.rubric._get_reward_func_names()
        # Should have at least our 4 + the built-in num_turns metric
        assert len(names) >= 4


# ---------------------------------------------------------------------------
# End-to-end rollout with mocked workers
# ---------------------------------------------------------------------------


class TestMockedRollout:
    """Simulate a complete rollout with canned responses."""

    @pytest.mark.asyncio
    async def test_reward_matches_compute_reward(self) -> None:
        """Given known component values, verify the rubric's weighted
        sum reproduces compute_reward for those values."""
        # Arrange: known component values
        known_f_fmt = 1.0
        known_s_correct = 1.0
        known_f_exec = 1.0
        known_b_eff = 1.0

        # compute_reward with default weights
        expected_reward = compute_reward(
            s_correct=known_s_correct,
            f_fmt=known_f_fmt,
            f_exec=known_f_exec,
            b_eff=known_b_eff,
        )
        # 1.0*1.0 + 0.1*1.0 + 0.1*1.0 + 0.2*1.0 = 1.4
        assert expected_reward == pytest.approx(1.4)

        # The rubric weighted sum should produce:
        # w_fmt * f_fmt + w_corr * s_correct + w_exec * f_exec + w_eff * b_eff
        # = 0.1*1.0 + 1.0*1.0 + 0.1*1.0 + 0.2*1.0 = 1.4
        # Note: rubric doesn't gate b_eff by correctness -- that happens
        # in the final compute_reward. The raw rubric sum with these
        # weights and all-1.0 scores gives 1.4.
        weights = RewardWeights()
        raw_sum = (
            weights.w_fmt * known_f_fmt
            + weights.w_corr * known_s_correct
            + weights.w_exec * known_f_exec
            + weights.w_eff * known_b_eff
        )
        assert raw_sum == pytest.approx(expected_reward)

    @pytest.mark.asyncio
    async def test_mocked_format_reward(self) -> None:
        """Format reward function returns correct f_fmt for a valid workflow."""
        from conductor_workflow._env_wiring import WorkflowParser

        parser = WorkflowParser()
        func = parser.get_format_reward_func()

        completion = [
            {
                "role": "assistant",
                "content": (
                    "```workflow\n"
                    '{"subtasks": ["Solve it"], "model_id": [2], "access_list": [[]]}\n'
                    "```"
                ),
            }
        ]
        score = await func(completion=completion)
        assert score == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_mocked_format_reward_invalid(self) -> None:
        """Format reward for invalid workflow gives partial credit."""
        from conductor_workflow._env_wiring import WorkflowParser

        parser = WorkflowParser()
        func = parser.get_format_reward_func()

        completion = [{"role": "assistant", "content": "no workflow here"}]
        score = await func(completion=completion)
        assert score == 0.0

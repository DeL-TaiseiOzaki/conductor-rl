"""Tests for conductor_workflow.executor.

All worker calls are mocked -- no real HTTP traffic.
Tests cover: single-route, sequential chain, parallel-aggregate DAGs,
correct context assembly, f_exec accounting, latency = critical path,
cost = sum, and b_eff normalization.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from conductor_workflow.executor import (
    NodeResult,
    _build_execution_levels,
    _build_worker_prompt,
    compute_b_eff,
    compute_baseline_proxies,
    compute_critical_path_latency,
    compute_f_exec,
    compute_total_cost,
    execute_dag,
)
from conductor_workflow.parser import ParseResult, SubtaskNode
from conductor_workflow.workers import WorkerConfig, WorkerResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORKER_CONFIGS: dict[int, WorkerConfig] = {
    0: WorkerConfig(worker_id=0, slug="fast", latency_weight=1.0, cost_out_per_1m=0.18),
    1: WorkerConfig(worker_id=1, slug="mm", latency_weight=1.5, cost_out_per_1m=1.20),
    2: WorkerConfig(worker_id=2, slug="pro", latency_weight=3.0, cost_out_per_1m=0.87),
    3: WorkerConfig(worker_id=3, slug="glm", latency_weight=3.0, cost_out_per_1m=4.40),
}


def _make_parse_result(
    nodes: list[SubtaskNode],
    valid: bool = True,
    f_fmt: float = 1.0,
) -> ParseResult:
    return ParseResult(valid=valid, f_fmt=f_fmt, nodes=nodes)


# ---------------------------------------------------------------------------
# Critical path latency
# ---------------------------------------------------------------------------


class TestCriticalPathLatency:
    """Latency = critical path; parallel branches take max."""

    def test_single_node(self) -> None:
        nodes = [SubtaskNode(index=0, instruction="x", model_id=2, deps=[])]
        assert compute_critical_path_latency(nodes, WORKER_CONFIGS) == 3.0

    def test_sequential_chain_sums(self) -> None:
        # 0 (model 0, lat=1) -> 1 (model 2, lat=3)
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=2, deps=[0]),
        ]
        assert compute_critical_path_latency(nodes, WORKER_CONFIGS) == 4.0

    def test_parallel_branches_take_max(self) -> None:
        # 0 (model 0, lat=1) and 1 (model 2, lat=3) in parallel
        # 2 (model 0, lat=1) depends on both -> finish = max(1,3) + 1 = 4
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=2, deps=[]),
            SubtaskNode(index=2, instruction="c", model_id=0, deps=[0, 1]),
        ]
        assert compute_critical_path_latency(nodes, WORKER_CONFIGS) == 4.0

    def test_empty_nodes(self) -> None:
        assert compute_critical_path_latency([], WORKER_CONFIGS) == 0.0


# ---------------------------------------------------------------------------
# Total cost
# ---------------------------------------------------------------------------


class TestTotalCost:
    """Cost = sum of all node cost weights."""

    def test_single_node_cost(self) -> None:
        nodes = [SubtaskNode(index=0, instruction="x", model_id=3, deps=[])]
        assert compute_total_cost(nodes, WORKER_CONFIGS) == pytest.approx(4.40)

    def test_parallel_nodes_sum(self) -> None:
        # Parallel nodes: cost is sum, not max
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=2, deps=[]),
        ]
        assert compute_total_cost(nodes, WORKER_CONFIGS) == pytest.approx(0.18 + 0.87)

    def test_chain_cost_sums(self) -> None:
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=3, deps=[0]),
        ]
        assert compute_total_cost(nodes, WORKER_CONFIGS) == pytest.approx(0.18 + 4.40)


# ---------------------------------------------------------------------------
# f_exec computation
# ---------------------------------------------------------------------------


class TestComputeFExec:
    """f_exec = fraction of executable (well-specified) calls."""

    def test_all_executable(self) -> None:
        results = [
            NodeResult(index=0, executable=True, success=True),
            NodeResult(index=1, executable=True, success=True),
        ]
        assert compute_f_exec(results) == 1.0

    def test_one_conductor_failure(self) -> None:
        results = [
            NodeResult(index=0, executable=True, success=True),
            NodeResult(index=1, executable=False, success=False),
        ]
        assert compute_f_exec(results) == 0.5

    def test_transient_failure_still_executable(self) -> None:
        # Transient failures do NOT lower f_exec
        results = [
            NodeResult(index=0, executable=True, success=True),
            NodeResult(index=1, executable=True, success=False, transient_failure=True),
        ]
        assert compute_f_exec(results) == 1.0

    def test_empty(self) -> None:
        assert compute_f_exec([]) == 0.0


# ---------------------------------------------------------------------------
# b_eff computation
# ---------------------------------------------------------------------------


class TestComputeBEff:
    """b_eff normalization against baseline."""

    def test_lambda_mu_zero_gives_one(self) -> None:
        # With lambda=mu=0, b_eff=1 for any DAG
        b = compute_b_eff(10.0, 100.0, 3.0, 4.40, lambda_latency=0.0, mu_cost=0.0)
        assert b == pytest.approx(1.0)

    def test_baseline_exact_gives_one(self) -> None:
        b = compute_b_eff(3.0, 4.40, 3.0, 4.40, lambda_latency=1.0, mu_cost=1.0)
        assert b == pytest.approx(1.0)

    def test_faster_than_baseline_gives_one(self) -> None:
        # Faster/cheaper: p_lat and p_cost clamped at 0
        b = compute_b_eff(1.0, 0.18, 3.0, 4.40, lambda_latency=1.0, mu_cost=1.0)
        assert b == pytest.approx(1.0)

    def test_slower_than_baseline_less_than_one(self) -> None:
        b = compute_b_eff(6.0, 4.40, 3.0, 4.40, lambda_latency=1.0, mu_cost=0.0)
        expected = math.exp(-(1.0 * max(0, 6.0 / 3.0 - 1)))
        assert b == pytest.approx(expected)
        assert b < 1.0

    def test_more_expensive_than_baseline(self) -> None:
        b = compute_b_eff(3.0, 8.80, 3.0, 4.40, lambda_latency=0.0, mu_cost=1.0)
        expected = math.exp(-(1.0 * max(0, 8.80 / 4.40 - 1)))
        assert b == pytest.approx(expected)
        assert b < 1.0


# ---------------------------------------------------------------------------
# Baseline proxies
# ---------------------------------------------------------------------------


class TestBaselineProxies:
    """Baseline = strongest worker (max latency, then max cost on tie)."""

    def test_strongest_worker_is_glm(self) -> None:
        lat, cost = compute_baseline_proxies(WORKER_CONFIGS)
        # Workers 2 and 3 both have latency=3.0; GLM (3) has higher cost
        assert lat == 3.0
        assert cost == 4.40


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


class TestBuildWorkerPrompt:
    """Prompt assembly for workers."""

    def test_no_deps_only_task_and_instruction(self) -> None:
        node = SubtaskNode(index=0, instruction="Solve it", model_id=0, deps=[])
        prompt = _build_worker_prompt("Task text", node, {})
        assert "Task text" in prompt
        assert "Solve it" in prompt
        assert "Previous Subtasks" not in prompt

    def test_with_deps_includes_outputs(self) -> None:
        node = SubtaskNode(index=2, instruction="Aggregate", model_id=0, deps=[0, 1])
        dep_outputs = {0: "output_0", 1: "output_1"}
        prompt = _build_worker_prompt("Task text", node, dep_outputs)
        assert "output_0" in prompt
        assert "output_1" in prompt
        assert "subtask 0" in prompt
        assert "subtask 1" in prompt


# ---------------------------------------------------------------------------
# Execution levels
# ---------------------------------------------------------------------------


class TestBuildExecutionLevels:
    """Topological level grouping."""

    def test_single_level(self) -> None:
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=1, deps=[]),
        ]
        levels = _build_execution_levels(nodes)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_two_levels(self) -> None:
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=1, deps=[0]),
        ]
        levels = _build_execution_levels(nodes)
        assert len(levels) == 2

    def test_fan_out_fan_in(self) -> None:
        nodes = [
            SubtaskNode(index=0, instruction="a", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="b", model_id=1, deps=[]),
            SubtaskNode(index=2, instruction="c", model_id=0, deps=[0, 1]),
        ]
        levels = _build_execution_levels(nodes)
        assert len(levels) == 2
        assert len(levels[0]) == 2  # parallel
        assert len(levels[1]) == 1  # aggregator


# ---------------------------------------------------------------------------
# Full DAG execution (mocked workers)
# ---------------------------------------------------------------------------


def _mock_call_worker_factory(responses: dict[int, str]):
    """Create a mock call_worker that returns predetermined responses."""

    async def mock_call_worker(config, prompt, *, client=None, semaphore=None):
        worker_id = config.worker_id
        if worker_id in responses:
            return WorkerResult(output=responses[worker_id], success=True)
        return WorkerResult(output="", success=False, error_message="no mock")

    return mock_call_worker


class TestExecuteDag:
    """Full DAG execution with mocked worker calls."""

    @pytest.mark.asyncio
    async def test_single_route(self) -> None:
        # Arrange: single subtask using worker 2
        nodes = [SubtaskNode(index=0, instruction="Solve", model_id=2, deps=[])]
        pr = _make_parse_result(nodes)

        with patch(
            "conductor_workflow.executor.call_worker",
            new=_mock_call_worker_factory({2: "The answer is 42"}),
        ):
            # Act
            result = await execute_dag(pr, "Task", WORKER_CONFIGS)

        # Assert
        assert result.final_output == "The answer is 42"
        assert result.f_exec == 1.0
        assert result.latency_proxy == 3.0
        assert result.cost_proxy == pytest.approx(0.87)

    @pytest.mark.asyncio
    async def test_sequential_chain(self) -> None:
        # Arrange: 0 (model 2) -> 1 (model 0)
        nodes = [
            SubtaskNode(index=0, instruction="Draft", model_id=2, deps=[]),
            SubtaskNode(index=1, instruction="Refine", model_id=0, deps=[0]),
        ]
        pr = _make_parse_result(nodes)

        with patch(
            "conductor_workflow.executor.call_worker",
            new=_mock_call_worker_factory({2: "draft output", 0: "final output"}),
        ):
            result = await execute_dag(pr, "Task", WORKER_CONFIGS)

        assert result.final_output == "final output"
        assert result.f_exec == 1.0
        assert result.latency_proxy == 4.0  # 3.0 + 1.0
        assert result.cost_proxy == pytest.approx(0.87 + 0.18)

    @pytest.mark.asyncio
    async def test_parallel_aggregate(self) -> None:
        # Arrange: 0 (model 2) and 1 (model 3) in parallel, then 2 (model 0) aggregates
        nodes = [
            SubtaskNode(index=0, instruction="Solve A", model_id=2, deps=[]),
            SubtaskNode(index=1, instruction="Solve B", model_id=3, deps=[]),
            SubtaskNode(index=2, instruction="Aggregate", model_id=0, deps=[0, 1]),
        ]
        pr = _make_parse_result(nodes)

        with patch(
            "conductor_workflow.executor.call_worker",
            new=_mock_call_worker_factory({2: "A=5", 3: "B=5", 0: "Answer: 5"}),
        ):
            result = await execute_dag(pr, "Task", WORKER_CONFIGS)

        assert result.final_output == "Answer: 5"
        assert result.f_exec == 1.0
        # Latency: max(3.0, 3.0) + 1.0 = 4.0
        assert result.latency_proxy == 4.0
        # Cost: 0.87 + 4.40 + 0.18 = 5.45
        assert result.cost_proxy == pytest.approx(0.87 + 4.40 + 0.18)

    @pytest.mark.asyncio
    async def test_invalid_parse_result(self) -> None:
        pr = ParseResult(valid=False, f_fmt=0.3)
        result = await execute_dag(pr, "Task", WORKER_CONFIGS)
        assert result.final_output == ""
        assert result.f_exec == 0.0

    @pytest.mark.asyncio
    async def test_transient_failure_does_not_lower_f_exec(self) -> None:
        # Arrange: node 0 succeeds, node 1 has transient failure
        nodes = [
            SubtaskNode(index=0, instruction="A", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="B", model_id=2, deps=[]),
        ]
        pr = _make_parse_result(nodes)

        async def mock_call(config, prompt, *, client=None, semaphore=None):
            if config.worker_id == 0:
                return WorkerResult(output="ok", success=True)
            return WorkerResult(
                output="",
                success=False,
                transient_failure=True,
                error_message="503 timeout",
            )

        with patch("conductor_workflow.executor.call_worker", new=mock_call):
            result = await execute_dag(pr, "Task", WORKER_CONFIGS)

        # Both are executable (transient doesn't count against f_exec)
        assert result.f_exec == 1.0
        # Node 1 failed, so its output is not available
        assert len(result.node_results) == 2

    @pytest.mark.asyncio
    async def test_b_eff_with_lambda_mu_zero(self) -> None:
        # With default lambda=mu=0, b_eff should always be 1.0
        nodes = [
            SubtaskNode(index=0, instruction="A", model_id=0, deps=[]),
            SubtaskNode(index=1, instruction="B", model_id=2, deps=[]),
            SubtaskNode(index=2, instruction="C", model_id=3, deps=[0, 1]),
        ]
        pr = _make_parse_result(nodes)

        with patch(
            "conductor_workflow.executor.call_worker",
            new=_mock_call_worker_factory({0: "a", 2: "b", 3: "c"}),
        ):
            result = await execute_dag(
                pr,
                "Task",
                WORKER_CONFIGS,
                lambda_latency=0.0,
                mu_cost=0.0,
            )

        assert result.b_eff == pytest.approx(1.0)

r"""Async DAG executor for Conductor workflows.

Executes a parsed workflow DAG by calling worker models via OpenRouter,
assembling context from dependency outputs, and computing efficiency
proxies for the reward function.

Efficiency Accounting (from Codex design review)
-------------------------------------------------

**Latency (critical-path)**::

    finish[i] = max(finish[j] for j in deps[i], default=0) + latency_weight[model_id[i]]
    L_dag     = max(finish[i] for all i)

Parallel branches take ``max`` (they run concurrently); sequential
chains sum.  This matches wall-clock time behaviour.

**Cost (total)**::

    C_dag = sum(cost_weight[model_id[i]] for all i)

Every call is billed regardless of parallelism.

**f_exec (execution feasibility)**::

    f_exec = executable_nodes / total_nodes

A node is *executable* if the Conductor specified a valid call AND the
required dependency outputs are available.  Transient API failures
(429, 5xx, timeouts) are retried and, even if ultimately unrecoverable,
do NOT lower ``f_exec`` -- they are infrastructure noise, not
Conductor-caused mis-specification.

**b_eff (efficiency bonus, baseline-relative)**::

    p_lat  = max(0, L_dag / L_base - 1)
    p_cost = max(0, C_dag / C_base - 1)
    b_eff  = exp(-(lambda * p_lat + mu * p_cost))

With ``lambda = mu = 0`` (staging OFF), ``b_eff = 1`` for every DAG,
producing a constant contribution ``w_eff * 1`` for correct answers.
In GRPO the group-relative advantage cancels constants, so there is
**no efficiency gradient signal** until lambda/mu are staged in.

Baseline = single strongest worker alone:
``L_base = max(latency_weight)``, ``C_base = cost of that worker``.
Configured via ``reward.baseline`` in ``default.yaml``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from conductor_workflow.parser import ParseResult, SubtaskNode
from conductor_workflow.workers import (
    WorkerConfig,
    WorkerResult,
    call_worker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Execution result for a single DAG node."""

    index: int
    output: str = ""
    success: bool = False
    executable: bool = True
    transient_failure: bool = False


@dataclass
class ExecutionResult:
    """Full DAG execution result.

    Attributes:
        final_output: Text output of the last node (the answer).
        node_results: Per-node execution details.
        f_exec: Fraction of well-specified, executable calls in [0,1].
        latency_proxy: Critical-path latency (sum along longest path).
        cost_proxy: Total cost (sum of all node cost weights).
        b_eff: Efficiency bonus in [0,1], baseline-relative.
    """

    final_output: str = ""
    node_results: list[NodeResult] = field(default_factory=list)
    f_exec: float = 0.0
    latency_proxy: float = 0.0
    cost_proxy: float = 0.0
    b_eff: float = 1.0


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _build_worker_prompt(
    task_prompt: str,
    node: SubtaskNode,
    dep_outputs: dict[int, str],
) -> str:
    """Build the full prompt for a worker node.

    Each worker sees: original task + subtask instruction + outputs
    of nodes in its access_list.
    """
    parts: list[str] = [
        f"## Original Task\n{task_prompt}",
        f"\n## Subtask Instruction\n{node.instruction}",
    ]
    if node.deps:
        dep_section = "\n## Context from Previous Subtasks\n"
        for dep_idx in node.deps:
            dep_text = dep_outputs.get(dep_idx, "[unavailable]")
            dep_section += f"### Output of subtask {dep_idx}\n{dep_text}\n"
        parts.append(dep_section)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Latency / cost computation
# ---------------------------------------------------------------------------


def compute_critical_path_latency(
    nodes: list[SubtaskNode],
    worker_configs: dict[int, WorkerConfig],
) -> float:
    """Compute critical-path latency over the DAG.

    ``finish[i] = max(finish[j] for j in deps[i]) + latency_weight[i]``
    Parallel branches take max; sequential chains sum.
    """
    if not nodes:
        return 0.0

    finish: dict[int, float] = {}
    for node in nodes:
        cfg = worker_configs.get(node.model_id)
        node_latency = cfg.latency_weight if cfg else 0.0
        parent_max = max(
            (finish.get(dep, 0.0) for dep in node.deps),
            default=0.0,
        )
        finish[node.index] = parent_max + node_latency

    return max(finish.values()) if finish else 0.0


def compute_total_cost(
    nodes: list[SubtaskNode],
    worker_configs: dict[int, WorkerConfig],
) -> float:
    """Compute total cost proxy: sum of all node cost weights."""
    total = 0.0
    for node in nodes:
        cfg = worker_configs.get(node.model_id)
        if cfg:
            total += cfg.cost_out_per_1m
    return total


# ---------------------------------------------------------------------------
# f_exec computation
# ---------------------------------------------------------------------------


def compute_f_exec(node_results: list[NodeResult]) -> float:
    """Compute fraction of executable (well-specified) worker calls.

    Nodes that failed due to transient errors still count as executable.
    Only Conductor-caused failures lower f_exec.
    """
    if not node_results:
        return 0.0
    executable_count = sum(1 for nr in node_results if nr.executable)
    return executable_count / len(node_results)


# ---------------------------------------------------------------------------
# b_eff computation
# ---------------------------------------------------------------------------


def compute_baseline_proxies(
    worker_configs: dict[int, WorkerConfig],
) -> tuple[float, float]:
    """Compute baseline (strongest-worker-alone) latency and cost.

    Baseline = single call to the worker with the highest latency_weight.
    On ties, pick the most expensive (conservative baseline).
    """
    if not worker_configs:
        return 1.0, 1.0

    strongest = max(
        worker_configs.values(),
        key=lambda w: (w.latency_weight, w.cost_out_per_1m),
    )
    return strongest.latency_weight, strongest.cost_out_per_1m


def compute_b_eff(
    latency_proxy: float,
    cost_proxy: float,
    baseline_latency: float,
    baseline_cost: float,
    *,
    lambda_latency: float = 0.0,
    mu_cost: float = 0.0,
) -> float:
    """Compute efficiency bonus b_eff in [0, 1].

    b_eff = exp(-(lambda * max(0, L/L_base - 1) + mu * max(0, C/C_base - 1)))

    With lambda=mu=0, b_eff=1 (constant, no signal).
    """
    if baseline_latency <= 0 or baseline_cost <= 0:
        return 1.0

    p_lat = max(0.0, latency_proxy / baseline_latency - 1.0)
    p_cost = max(0.0, cost_proxy / baseline_cost - 1.0)

    exponent = -(lambda_latency * p_lat + mu_cost * p_cost)
    return math.exp(exponent)


# ---------------------------------------------------------------------------
# DAG executor
# ---------------------------------------------------------------------------


async def execute_dag(
    parse_result: ParseResult,
    task_prompt: str,
    worker_configs: dict[int, WorkerConfig],
    *,
    client: Any = None,
    semaphore: asyncio.Semaphore | None = None,
    lambda_latency: float = 0.0,
    mu_cost: float = 0.0,
) -> ExecutionResult:
    """Execute a parsed workflow DAG.

    Nodes are executed in topological order.  Independent nodes (no
    dependency chain) are run concurrently via ``asyncio.gather``.
    The last node's output is the final answer.

    Args:
        parse_result: Validated DAG from the parser.
        task_prompt: The original task text.
        worker_configs: Worker id -> config mapping.
        client: Optional AsyncOpenAI client (for mocking).
        semaphore: Optional concurrency limiter.
        lambda_latency: Latency penalty coefficient (0 = off).
        mu_cost: Cost penalty coefficient (0 = off).

    Returns:
        ``ExecutionResult`` with final output and all metrics.
    """
    if not parse_result.valid or not parse_result.nodes:
        return ExecutionResult(f_exec=0.0, b_eff=1.0)

    nodes = parse_result.nodes
    node_outputs: dict[int, str] = {}
    node_results: list[NodeResult] = []

    # Group nodes into levels (topological layers) for parallel execution
    levels = _build_execution_levels(nodes)

    for level in levels:
        tasks = []
        for node in level:
            # Check if all deps have available outputs
            deps_available = all(dep in node_outputs for dep in node.deps)
            if not deps_available:
                # Dep missing due to upstream failure -> Conductor-caused
                nr = NodeResult(
                    index=node.index,
                    executable=False,
                    success=False,
                )
                node_results.append(nr)
                continue

            cfg = worker_configs.get(node.model_id)
            if cfg is None:
                # Invalid model_id -> Conductor-caused
                nr = NodeResult(
                    index=node.index,
                    executable=False,
                    success=False,
                )
                node_results.append(nr)
                continue

            prompt = _build_worker_prompt(task_prompt, node, node_outputs)
            tasks.append((node, cfg, prompt))

        if not tasks:
            continue

        # Execute all tasks in this level concurrently
        coros = [
            call_worker(cfg, prompt, client=client, semaphore=semaphore)
            for (_, cfg, prompt) in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (node, _cfg, _prompt), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                # Unexpected exception -> treat as transient
                nr = NodeResult(
                    index=node.index,
                    executable=True,
                    success=False,
                    transient_failure=True,
                )
                node_results.append(nr)
            elif isinstance(result, WorkerResult):
                if result.success:
                    node_outputs[node.index] = result.output
                    nr = NodeResult(
                        index=node.index,
                        output=result.output,
                        executable=True,
                        success=True,
                    )
                elif result.transient_failure:
                    # Transient: executable but failed
                    nr = NodeResult(
                        index=node.index,
                        executable=True,
                        success=False,
                        transient_failure=True,
                    )
                else:
                    # Non-transient failure (bad request etc.)
                    nr = NodeResult(
                        index=node.index,
                        executable=False,
                        success=False,
                    )
                node_results.append(nr)

    # Compute metrics
    f_exec = compute_f_exec(node_results)
    latency_proxy = compute_critical_path_latency(nodes, worker_configs)
    cost_proxy = compute_total_cost(nodes, worker_configs)
    baseline_lat, baseline_cost = compute_baseline_proxies(worker_configs)
    b_eff = compute_b_eff(
        latency_proxy,
        cost_proxy,
        baseline_lat,
        baseline_cost,
        lambda_latency=lambda_latency,
        mu_cost=mu_cost,
    )

    # Final output is the last node's output
    last_index = nodes[-1].index
    final_output = node_outputs.get(last_index, "")

    return ExecutionResult(
        final_output=final_output,
        node_results=node_results,
        f_exec=f_exec,
        latency_proxy=latency_proxy,
        cost_proxy=cost_proxy,
        b_eff=b_eff,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_execution_levels(
    nodes: list[SubtaskNode],
) -> list[list[SubtaskNode]]:
    """Group nodes into topological levels for parallel execution.

    Nodes with all deps in earlier levels can run concurrently.
    """
    if not nodes:
        return []

    completed: set[int] = set()
    remaining = list(nodes)
    levels: list[list[SubtaskNode]] = []

    while remaining:
        current_level: list[SubtaskNode] = []
        still_remaining: list[SubtaskNode] = []

        for node in remaining:
            if all(dep in completed for dep in node.deps):
                current_level.append(node)
            else:
                still_remaining.append(node)

        if not current_level:
            # Prevent infinite loop on malformed DAG
            break

        levels.append(current_level)
        for node in current_level:
            completed.add(node.index)
        remaining = still_remaining

    return levels

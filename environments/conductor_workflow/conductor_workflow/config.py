"""Configuration loading for Conductor-RL.

Reads ``configs/default.yaml`` and provides typed accessors for
worker configs, reward weights, judge settings, etc.

Pure, synchronous, no side effects.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from conductor_workflow.reward import RewardWeights
from conductor_workflow.workers import WorkerConfig

# ---------------------------------------------------------------------------
# Packaged asset resolution
# ---------------------------------------------------------------------------


def _packaged_config_text() -> str:
    """Read the bundled default.yaml via importlib.resources."""
    assets = importlib.resources.files("conductor_workflow") / "assets"
    return (assets / "default.yaml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeConfig:
    """Judge (TinyV) configuration."""

    slug: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    enabled: bool = True
    only_on_uncertain: bool = True
    max_concurrency: int = 4
    timeout_s: int = 60


# ---------------------------------------------------------------------------
# Full config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConductorConfig:
    """Top-level configuration bundle."""

    worker_configs: dict[int, WorkerConfig]
    reward_weights: RewardWeights
    lambda_latency: float
    mu_cost: float
    judge: JudgeConfig
    system_prompt_path: str
    pilot_data_path: str
    clusters: list[str]
    code_s_correct: str  # "binary" | "fraction"
    baseline: str
    # Group-relative efficiency weights (staged, default 0.0 = off)
    w_lat: float = 0.0
    w_cost: float = 0.0


def load_config(config_path: Path | str | None = None) -> ConductorConfig:
    """Load and parse the YAML config.

    Args:
        config_path: Path to the YAML config file.
            If *None*, reads the bundled ``assets/default.yaml`` via
            ``importlib.resources`` (works from an installed wheel).
            If given, reads the file at that path.

    Returns:
        Parsed ``ConductorConfig``.
    """
    if config_path is not None:
        path = Path(config_path)
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
    else:
        raw = yaml.safe_load(_packaged_config_text())

    # Workers
    workers_raw = raw.get("workers", [])
    worker_configs: dict[int, WorkerConfig] = {}
    for w in workers_raw:
        wid = w["id"]
        worker_configs[wid] = WorkerConfig(
            worker_id=wid,
            slug=w["slug"],
            latency_weight=float(w["latency_weight"]),
            cost_out_per_1m=float(w["cost_out_per_1m"]),
            cost_in_per_1m=float(w.get("cost_in_per_1m", 0.0)),
            openrouter_variant=w.get("openrouter_variant", ""),
        )

    # Reward
    reward_raw = raw.get("reward", {})
    reward_weights = RewardWeights(
        w_corr=float(reward_raw.get("w_corr", 1.0)),
        w_fmt=float(reward_raw.get("w_fmt", 0.1)),
        w_exec=float(reward_raw.get("w_exec", 0.1)),
        w_eff=float(reward_raw.get("w_eff", 0.2)),
    )

    # Judge
    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        slug=judge_raw.get("slug", JudgeConfig.slug),
        enabled=judge_raw.get("enabled", True),
        only_on_uncertain=judge_raw.get("only_on_uncertain", True),
        max_concurrency=judge_raw.get("max_concurrency", 4),
        timeout_s=judge_raw.get("timeout_s", 60),
    )

    # Data
    data_raw = raw.get("data", {})
    conductor_raw = raw.get("conductor", {})

    return ConductorConfig(
        worker_configs=worker_configs,
        reward_weights=reward_weights,
        lambda_latency=float(reward_raw.get("lambda_latency", 0.0)),
        mu_cost=float(reward_raw.get("mu_cost", 0.0)),
        judge=judge,
        system_prompt_path=conductor_raw.get(
            "system_prompt", "prompts/conductor_system_prompt.md"
        ),
        pilot_data_path=data_raw.get("pilot", "data/pilot/pilot.jsonl"),
        clusters=data_raw.get("clusters", ["code", "science_mcq", "hard_math"]),
        code_s_correct=reward_raw.get("code_s_correct", "fraction"),
        baseline=reward_raw.get("baseline", "strongest_worker_alone"),
        w_lat=float(reward_raw.get("w_lat", 0.0)),
        w_cost=float(reward_raw.get("w_cost", 0.0)),
    )

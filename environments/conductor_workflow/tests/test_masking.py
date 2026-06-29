"""Tests for worker anonymization (masking).

Asserts the loaded system prompt contains NONE of the concrete model/lab
names and still contains the workflow format rules.
"""

from __future__ import annotations

import re

import pytest

from conductor_workflow._env_wiring import _load_system_prompt

# ---------------------------------------------------------------------------
# Forbidden names (case-insensitive)
# ---------------------------------------------------------------------------

# Patterns that must NOT appear in the system prompt.
# Use word-boundary regex to avoid false positives like "produce" matching "Pro".
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Flash", re.compile(r"\bFlash\b", re.IGNORECASE)),
    ("M3", re.compile(r"\bM3\b", re.IGNORECASE)),
    ("Pro", re.compile(r"\bPro\b", re.IGNORECASE)),
    ("GLM", re.compile(r"\bGLM\b", re.IGNORECASE)),
    ("DeepSeek", re.compile(r"\bDeepSeek\b", re.IGNORECASE)),
    ("MiniMax", re.compile(r"\bMiniMax\b", re.IGNORECASE)),
    ("z-ai", re.compile(r"\bz-ai\b", re.IGNORECASE)),
]

# Simple substring checks for slugs (these won't false-positive)
FORBIDDEN_SUBSTRINGS: list[str] = [
    "deepseek/",
    "minimax/",
    "z-ai/",
    "glm-",
    "minimax-m3",
    "deepseek-v4",
]


class TestSystemPromptMasking:
    """System prompt must not leak concrete model/lab names."""

    @pytest.fixture(scope="class")
    def system_prompt(self) -> str:
        """Load the bundled system prompt."""
        return _load_system_prompt()

    def test_no_forbidden_patterns_in_prompt(self, system_prompt: str) -> None:
        """Assert NONE of the forbidden model/lab names appear (word boundary)."""
        for name, pattern in FORBIDDEN_PATTERNS:
            assert not pattern.search(system_prompt), (
                f"Forbidden name '{name}' found in system prompt"
            )

    def test_no_forbidden_slugs_in_prompt(self, system_prompt: str) -> None:
        """Assert no model slugs appear in the system prompt."""
        lower_prompt = system_prompt.lower()
        for slug in FORBIDDEN_SUBSTRINGS:
            assert slug.lower() not in lower_prompt, (
                f"Forbidden slug fragment '{slug}' found in system prompt"
            )

    def test_prompt_still_has_workflow_format_rules(self, system_prompt: str) -> None:
        """The workflow format rules must still be present."""
        assert "```workflow" in system_prompt
        assert '"subtasks"' in system_prompt
        assert '"model_id"' in system_prompt
        assert '"access_list"' in system_prompt

    def test_prompt_has_worker_table(self, system_prompt: str) -> None:
        """Workers table with capability profiles is present."""
        assert "| id |" in system_prompt
        assert "fastest" in system_prompt
        assert "multimodal" in system_prompt.lower()

    def test_prompt_has_orchestration_patterns(self, system_prompt: str) -> None:
        """Key orchestration patterns are preserved."""
        assert "Single route" in system_prompt
        assert "Sequential chain" in system_prompt
        assert "Parallel + aggregate" in system_prompt

    def test_workers_identified_by_id_only(self, system_prompt: str) -> None:
        """Workers should be referenced by numeric id, not by name."""
        # Should have "worker 0", "worker 1", etc.
        assert (
            "worker 0" in system_prompt.lower() or "worker 1" in system_prompt.lower()
        )


class TestConfigDoesNotLeakSlugsToPrompt:
    """Config keeps internal slug mapping; prompt never sees them."""

    def test_config_has_slugs_internally(self) -> None:
        """WorkerConfig still stores real slugs for API routing."""
        from conductor_workflow.config import load_config

        config = load_config()
        # Slugs exist in the config
        assert "deepseek" in config.worker_configs[0].slug
        # But the system prompt loaded by the env does not contain them
        prompt = _load_system_prompt()
        assert "deepseek" not in prompt.lower()

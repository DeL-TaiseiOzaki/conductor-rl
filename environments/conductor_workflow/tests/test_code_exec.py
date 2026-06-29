"""Tests for conductor_workflow.graders.code_exec."""

from __future__ import annotations

from typing import Any

import pytest

from conductor_workflow.graders.code_exec import (
    grade_code,
    normalize_output,
)

# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


class TestNormalizeOutput:
    """Normalization: per-line rstrip + overall rstrip."""

    def test_trailing_spaces_stripped(self) -> None:
        assert normalize_output("hello   ") == "hello"

    def test_trailing_newline_stripped(self) -> None:
        assert normalize_output("hello\n") == "hello"

    def test_per_line_rstrip(self) -> None:
        assert normalize_output("a  \nb  \n") == "a\nb"

    def test_empty_string(self) -> None:
        assert normalize_output("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_output("   \n  \n") == ""

    def test_preserves_leading_spaces(self) -> None:
        assert normalize_output("  hello\n  world\n") == "  hello\n  world"


# ---------------------------------------------------------------------------
# Happy path with simple programs
# ---------------------------------------------------------------------------


class TestGradeCodeHappyPath:
    """Correct programs should score 1.0."""

    def test_simple_echo(self) -> None:
        # Arrange
        code = "print(input())"
        tests = [
            {"input": "hello\n", "output": "hello"},
            {"input": "world\n", "output": "world"},
        ]

        # Act
        result = grade_code(code, tests)

        # Assert
        assert result.s_correct == pytest.approx(1.0)
        assert result.all_pass is True
        assert result.passed == 2
        assert result.total == 2

    def test_addition_program(self) -> None:
        code = "a, b = map(int, input().split()); print(a + b)"
        tests = [
            {"input": "1 2\n", "output": "3"},
            {"input": "10 20\n", "output": "30"},
        ]
        result = grade_code(code, tests)
        assert result.s_correct == pytest.approx(1.0)
        assert result.all_pass is True


# ---------------------------------------------------------------------------
# Partial scoring
# ---------------------------------------------------------------------------


class TestGradeCodePartialScoring:
    """Partially correct programs should score between 0 and 1."""

    def test_half_correct(self) -> None:
        # This code always prints "hello"
        code = 'print("hello")'
        tests = [
            {"input": "\n", "output": "hello"},
            {"input": "\n", "output": "world"},
        ]
        result = grade_code(code, tests)
        assert result.s_correct == pytest.approx(0.5)
        assert result.all_pass is False
        assert result.passed == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestGradeCodeErrors:
    """Programs with errors should score 0."""

    def test_syntax_error(self) -> None:
        code = "def oops(:"
        tests = [{"input": "\n", "output": "anything"}]
        result = grade_code(code, tests)
        assert result.s_correct == pytest.approx(0.0)

    def test_runtime_error(self) -> None:
        code = "raise RuntimeError('boom')"
        tests = [{"input": "\n", "output": "anything"}]
        result = grade_code(code, tests)
        assert result.s_correct == pytest.approx(0.0)

    def test_empty_tests_list(self) -> None:
        result = grade_code("print('hi')", [])
        assert result.s_correct == pytest.approx(0.0)
        assert result.total == 0

    def test_timeout_scores_zero(self) -> None:
        code = "import time; time.sleep(30)"
        tests = [{"input": "\n", "output": "anything"}]
        result = grade_code(code, tests, time_limit_s=1)
        assert result.s_correct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Security edge cases
# ---------------------------------------------------------------------------


class TestGradeCodeSecurity:
    """Security-related edge cases."""

    def test_memory_bomb_handled(self) -> None:
        # Try to allocate a huge list -- should be killed by RLIMIT_AS
        code = "x = [0] * (10**9)"
        tests = [{"input": "\n", "output": "anything"}]
        result = grade_code(code, tests, memory_limit_bytes=64 * 1024 * 1024)
        assert result.s_correct == pytest.approx(0.0)

    def test_infinite_loop_killed(self) -> None:
        code = "while True: pass"
        tests = [{"input": "\n", "output": "anything"}]
        result = grade_code(code, tests, time_limit_s=1)
        assert result.s_correct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Real pilot data
# ---------------------------------------------------------------------------


class TestGradeCodePilotData:
    """Test against real pilot items."""

    def test_code_0001_correct_solution(self, first_code_item: dict[str, Any]) -> None:
        """A correct reference solution for code-0001 (alternating sum)
        should score 1.0."""
        # Arrange
        correct_code = (
            "n = int(input())\n"
            "nums = list(map(int, input().split()))\n"
            "s = 0\n"
            "for i, x in enumerate(nums):\n"
            "    s += x if i % 2 == 0 else -x\n"
            "print(s)\n"
        )
        tests = first_code_item["verifier_spec"]["tests"]

        # Act
        result = grade_code(correct_code, tests)

        # Assert
        assert result.s_correct == pytest.approx(1.0)
        assert result.all_pass is True

    def test_code_0001_wrong_solution(self, first_code_item: dict[str, Any]) -> None:
        """An obviously wrong solution should score < 1.0."""
        wrong_code = "print(0)"
        tests = first_code_item["verifier_spec"]["tests"]
        result = grade_code(wrong_code, tests)
        assert result.s_correct < 1.0

    def test_code_0002_correct_solution(self, code_items: list[dict[str, Any]]) -> None:
        """A correct solution for code-0002 (vowel count) should score 1.0."""
        correct_code = (
            "line = input()\nprint(sum(1 for c in line if c in 'aeiouAEIOU'))\n"
        )
        tests = code_items[1]["verifier_spec"]["tests"]
        result = grade_code(correct_code, tests)
        assert result.s_correct == pytest.approx(1.0)

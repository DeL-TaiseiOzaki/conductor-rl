"""Tests for conductor_workflow.graders.code_exec."""

from __future__ import annotations

from typing import Any

import pytest

from conductor_workflow.graders.code_exec import (
    extract_code,
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


# ---------------------------------------------------------------------------
# extract_code helper
# ---------------------------------------------------------------------------


class TestExtractCode:
    """Tests for extract_code() markdown fence extraction."""

    def test_python_tagged_block_extracted(self) -> None:
        # Arrange
        text = "Here is my solution:\n```python\nprint(input())\n```\nThis should work."

        # Act
        result = extract_code(text)

        # Assert
        assert result == "print(input())"

    def test_py_tag_variant(self) -> None:
        text = "Solution:\n```py\nx = 1\nprint(x)\n```\n"
        result = extract_code(text)
        assert result == "x = 1\nprint(x)"

    def test_untagged_block_extracted(self) -> None:
        text = "Try this:\n```\nprint('hello')\n```\n"
        result = extract_code(text)
        assert result == "print('hello')"

    def test_multiple_blocks_picks_last_python(self) -> None:
        text = (
            "First attempt:\n"
            "```python\nprint('wrong')\n```\n"
            "Corrected:\n"
            "```python\nprint('right')\n```\n"
        )
        result = extract_code(text)
        assert result == "print('right')"

    def test_multiple_mixed_blocks_picks_python(self) -> None:
        text = "Config:\n```json\n{}\n```\nCode:\n```python\nprint(42)\n```\n"
        result = extract_code(text)
        assert result == "print(42)"

    def test_no_fence_returns_input(self) -> None:
        bare = "print(input())"
        result = extract_code(bare)
        assert result == bare

    def test_empty_string_returns_empty(self) -> None:
        assert extract_code("") == ""

    def test_untagged_largest_block_selected(self) -> None:
        text = "```\na\n```\n```\nfoo\nbar\nbaz\n```\n"
        result = extract_code(text)
        assert result == "foo\nbar\nbaz"


# ---------------------------------------------------------------------------
# Fenced-block grading integration
# ---------------------------------------------------------------------------


ECHO_CODE = "print(input())"
ECHO_TESTS: list[dict[str, str]] = [
    {"input": "hello\n", "output": "hello"},
    {"input": "world\n", "output": "world"},
]


class TestGradeCodeFencedInput:
    """grade_code on pre-extracted fenced blocks should match bare code."""

    def test_fenced_python_grades_same_as_bare(self) -> None:
        # Arrange
        fenced = f"Here is the solution:\n```python\n{ECHO_CODE}\n```\n"
        extracted = extract_code(fenced)

        # Act
        result_bare = grade_code(ECHO_CODE, ECHO_TESTS)
        result_extracted = grade_code(extracted, ECHO_TESTS)

        # Assert
        assert result_bare.s_correct == pytest.approx(1.0)
        assert result_extracted.s_correct == pytest.approx(1.0)

    def test_fenced_py_tag_grades_correctly(self) -> None:
        fenced = f"```py\n{ECHO_CODE}\n```\n"
        extracted = extract_code(fenced)
        result = grade_code(extracted, ECHO_TESTS)
        assert result.s_correct == pytest.approx(1.0)

    def test_raw_fenced_without_extraction_fails(self) -> None:
        """Without extract_code, fenced text causes SyntaxError -> 0."""
        fenced = f"Here:\n```python\n{ECHO_CODE}\n```\n"
        result = grade_code(fenced, ECHO_TESTS)
        assert result.s_correct == pytest.approx(0.0)

    def test_prose_only_no_code_scores_zero(self) -> None:
        prose = "I think the answer involves printing the input back."
        extracted = extract_code(prose)
        result = grade_code(extracted, ECHO_TESTS)
        assert result.s_correct == pytest.approx(0.0)

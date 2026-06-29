"""Tests for conductor_workflow.graders.math_verify."""

from __future__ import annotations

from typing import Any

from conductor_workflow.graders.math_verify import (
    extract_math_answer,
    grade_math,
)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


class TestExtractMathAnswer:
    """Test answer extraction from various formats."""

    def test_boxed_answer(self) -> None:
        text = r"Therefore $\boxed{50/51}$"
        assert extract_math_answer(text) == "50/51"

    def test_last_boxed_wins(self) -> None:
        text = r"First \boxed{wrong}, then \boxed{42}"
        assert extract_math_answer(text) == "42"

    def test_inline_dollar(self) -> None:
        text = "The result is $\\frac{50}{51}$"
        assert extract_math_answer(text) is not None

    def test_equals_at_end(self) -> None:
        text = "After simplification:\nresult = 42"
        assert extract_math_answer(text) == "42"

    def test_last_line_fallback(self) -> None:
        text = "Step 1: compute\nStep 2: simplify\n72"
        assert extract_math_answer(text) == "72"

    def test_empty_input(self) -> None:
        assert extract_math_answer("") is None

    def test_whitespace_only(self) -> None:
        assert extract_math_answer("   ") is None


# ---------------------------------------------------------------------------
# SymPy equivalence - happy path
# ---------------------------------------------------------------------------


class TestGradeMathHappyPath:
    """Equivalent answers should score 1.0."""

    def test_exact_integer_match(self) -> None:
        assert grade_math(r"\boxed{42}", "42") == 1.0

    def test_fraction_vs_decimal(self) -> None:
        # 1/2 == 0.5
        assert grade_math(r"\boxed{1/2}", "0.5") == 1.0

    def test_fraction_string_match(self) -> None:
        assert grade_math(r"\boxed{50/51}", "50/51") == 1.0

    def test_unsimplified_fraction(self) -> None:
        # 100/102 == 50/51
        assert grade_math(r"\boxed{100/102}", "50/51") == 1.0

    def test_integer_as_fraction(self) -> None:
        # 72/1 == 72
        assert grade_math(r"\boxed{72}", "72") == 1.0

    def test_negative_number(self) -> None:
        assert grade_math(r"\boxed{-5}", "-5") == 1.0

    def test_zero(self) -> None:
        assert grade_math(r"\boxed{0}", "0") == 1.0

    def test_tolerance_near_match(self) -> None:
        # 0.333333 is within 1e-6 of 1/3 only if enough digits
        assert grade_math(r"\boxed{0.333333333}", "1/3") == 1.0


# ---------------------------------------------------------------------------
# SymPy equivalence - wrong answers
# ---------------------------------------------------------------------------


class TestGradeMathWrongAnswers:
    """Non-equivalent answers should score 0.0."""

    def test_different_integers(self) -> None:
        assert grade_math(r"\boxed{41}", "42") == 0.0

    def test_different_fractions(self) -> None:
        assert grade_math(r"\boxed{49/51}", "50/51") == 0.0

    def test_extraction_failure(self) -> None:
        assert grade_math("I give up", "42") == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGradeMathEdgeCases:
    """Edge cases and boundary values."""

    def test_empty_candidate(self) -> None:
        assert grade_math("", "42") == 0.0

    def test_unparseable_gold(self) -> None:
        # If gold can't be parsed, score 0
        assert grade_math(r"\boxed{42}", "not_a_number_xyz") == 0.0

    def test_unparseable_candidate(self) -> None:
        assert grade_math(r"\boxed{hello_world}", "42") == 0.0

    def test_large_integer(self) -> None:
        assert grade_math(r"\boxed{1170}", "1170") == 1.0

    def test_tiny_v_fallback_not_called(self) -> None:
        """The fallback interface exists but is never invoked."""

        # Pass a mock that would fail if called
        class FakeFallback:
            def check_equivalence(
                self,
                candidate_answer: str,
                gold_answer: str,
                *,
                context: str | None = None,
            ) -> bool:
                raise AssertionError("Should not be called")

        # Should still work (fallback not invoked)
        result = grade_math(r"\boxed{42}", "42", tiny_v_fallback=FakeFallback())
        assert result == 1.0


# ---------------------------------------------------------------------------
# Real pilot data
# ---------------------------------------------------------------------------


class TestGradeMathPilotData:
    """Test against real pilot items."""

    def test_math_0001_correct_answer(self, first_math_item: dict[str, Any]) -> None:
        """math-0001 gold is '50/51' (telescoping sum)."""
        gold = first_math_item["gold"]
        tolerance = first_math_item["verifier_spec"].get("tolerance", 1e-6)

        # Correct answer in boxed form
        candidate = r"By telescoping, $\sum = 1 - 1/51 = \boxed{50/51}$"
        assert grade_math(candidate, gold, tolerance=tolerance) == 1.0

    def test_math_0001_wrong_answer(self, first_math_item: dict[str, Any]) -> None:
        gold = first_math_item["gold"]
        candidate = r"\boxed{49/51}"
        assert grade_math(candidate, gold) == 0.0

    def test_math_0001_decimal_equivalent(
        self, first_math_item: dict[str, Any]
    ) -> None:
        """Decimal form of 50/51 should also be accepted."""
        gold = first_math_item["gold"]
        tolerance = first_math_item["verifier_spec"].get("tolerance", 1e-6)
        # 50/51 = 0.9803921568627451...
        candidate = r"\boxed{0.9803921568627451}"
        assert grade_math(candidate, gold, tolerance=tolerance) == 1.0

    def test_multiple_math_items_parseable_gold(
        self, hard_math_items: list[dict[str, Any]]
    ) -> None:
        """All math item golds should be parseable by SymPy."""
        from conductor_workflow.graders.math_verify import _safe_sympify

        for item in hard_math_items:
            gold = item["gold"]
            expr = _safe_sympify(gold)
            assert expr is not None, (
                f"{item['id']} gold {gold!r} is not SymPy-parseable"
            )

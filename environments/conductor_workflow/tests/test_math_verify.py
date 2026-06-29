"""Tests for conductor_workflow.graders.math_verify."""

from __future__ import annotations

from typing import Any

import pytest

from conductor_workflow.graders.math_verify import (
    _extract_boxed_contents,
    _normalize_latex,
    extract_math_answer,
    grade_math,
    grade_math_async,
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


# ---------------------------------------------------------------------------
# Brace-balanced boxed extraction
# ---------------------------------------------------------------------------


class TestExtractBoxedContents:
    """Test brace-balanced extraction from \\boxed{...}."""

    def test_simple_boxed(self) -> None:
        assert _extract_boxed_contents(r"\boxed{42}") == ["42"]

    def test_nested_frac(self) -> None:
        result = _extract_boxed_contents(r"\boxed{\frac{625}{861}}")
        assert result == [r"\frac{625}{861}"]

    def test_deeply_nested(self) -> None:
        result = _extract_boxed_contents(r"\boxed{\frac{\sqrt{2}}{3}}")
        assert result == [r"\frac{\sqrt{2}}{3}"]

    def test_multiple_boxed_returns_all(self) -> None:
        text = r"First \boxed{1} then \boxed{\frac{2}{3}}"
        result = _extract_boxed_contents(text)
        assert len(result) == 2
        assert result[0] == "1"
        assert result[1] == r"\frac{2}{3}"

    def test_no_boxed_returns_empty(self) -> None:
        assert _extract_boxed_contents("no boxed here") == []


# ---------------------------------------------------------------------------
# LaTeX normalisation
# ---------------------------------------------------------------------------


class TestNormalizeLatex:
    """Test LaTeX-to-SymPy normalisation."""

    def test_frac(self) -> None:
        assert _normalize_latex(r"\frac{50}{51}") == "(50)/(51)"

    def test_dfrac(self) -> None:
        assert _normalize_latex(r"\dfrac{1}{2}") == "(1)/(2)"

    def test_tfrac(self) -> None:
        assert _normalize_latex(r"\tfrac{3}{4}") == "(3)/(4)"

    def test_sqrt(self) -> None:
        assert _normalize_latex(r"\sqrt{2}") == "sqrt(2)"

    def test_strip_dollar(self) -> None:
        assert _normalize_latex(r"$50/51$") == "50/51"

    def test_strip_paren_latex(self) -> None:
        assert _normalize_latex(r"\(50/51\)") == "50/51"

    def test_strip_bracket_latex(self) -> None:
        assert _normalize_latex(r"\[50/51\]") == "50/51"

    def test_cdot_and_times(self) -> None:
        result = _normalize_latex(r"3 \cdot 5 \times 2")
        assert result == "3 * 5 * 2"

    def test_strip_left_right(self) -> None:
        result = _normalize_latex(r"\left(\frac{1}{2}\right)")
        assert result == "((1)/(2))"

    def test_plain_string_unchanged(self) -> None:
        assert _normalize_latex("50/51") == "50/51"


# ---------------------------------------------------------------------------
# LaTeX extraction regression (extract_math_answer)
# ---------------------------------------------------------------------------


class TestExtractMathAnswerLatex:
    """Extraction of LaTeX forms that previously failed."""

    def test_boxed_frac_nested_braces(self) -> None:
        """Boxed \\frac must capture the full expression including inner braces."""
        text = r"\boxed{\frac{625}{861}}"
        result = extract_math_answer(text)
        assert result == r"\frac{625}{861}"

    def test_boxed_dfrac(self) -> None:
        text = r"\boxed{\dfrac{1}{2}}"
        result = extract_math_answer(text)
        assert result == r"\dfrac{1}{2}"

    def test_boxed_sqrt(self) -> None:
        text = r"\boxed{\sqrt{2}}"
        result = extract_math_answer(text)
        assert result == r"\sqrt{2}"

    def test_paren_latex_fallback(self) -> None:
        r"""Text with \(...\) but no boxed should extract the math."""
        text = r"The answer is \(\frac{50}{51}\)"
        result = extract_math_answer(text)
        assert result is not None
        assert "50" in result and "51" in result

    def test_trailing_fraction_no_boxed(self) -> None:
        """Plain 'answer: 50/51' should extract 50/51."""
        text = "answer: 50/51"
        result = extract_math_answer(text)
        assert result is not None
        assert "50/51" in result


# ---------------------------------------------------------------------------
# LaTeX grading regression (the reported bug)
# ---------------------------------------------------------------------------


class TestGradeMathLatexRegression:
    r"""Regression tests for LaTeX \frac, \dfrac, \sqrt, etc.

    These are the exact cases from the bug report that previously returned
    0.0 (false negative) instead of 1.0.
    """

    def test_boxed_frac_625_861(self) -> None:
        """Bug case 1: \\boxed{\\frac{625}{861}} vs '625/861'."""
        candidate = r"...\boxed{\frac{625}{861}}"
        assert grade_math(candidate, "625/861") == 1.0

    def test_boxed_plain_625_861(self) -> None:
        """Bug case 2: \\boxed{625/861} vs '625/861' (was already passing)."""
        candidate = r"\boxed{625/861}"
        assert grade_math(candidate, "625/861") == 1.0

    def test_boxed_frac_50_51(self) -> None:
        """Bug case 3: \\boxed{\\frac{50}{51}} vs '50/51'."""
        candidate = r"\boxed{\frac{50}{51}}"
        assert grade_math(candidate, "50/51") == 1.0

    def test_no_boxed_plain_fraction(self) -> None:
        """Bug case 4: 'answer: 50/51' vs '50/51' (no \\boxed)."""
        candidate = "answer: 50/51"
        assert grade_math(candidate, "50/51") == 1.0

    def test_boxed_dfrac_half_vs_decimal(self) -> None:
        r"""\\boxed{\\dfrac{1}{2}} vs '0.5'."""
        candidate = r"\boxed{\dfrac{1}{2}}"
        assert grade_math(candidate, "0.5") == 1.0

    def test_boxed_sqrt2_vs_power(self) -> None:
        r"""\\boxed{\\sqrt{2}} vs '2**0.5' within tolerance."""
        candidate = r"\boxed{\sqrt{2}}"
        assert grade_math(candidate, "2**0.5") == 1.0

    def test_wrong_frac_still_zero(self) -> None:
        """A wrong LaTeX fraction must still score 0.0."""
        candidate = r"\boxed{\frac{49}{51}}"
        assert grade_math(candidate, "50/51") == 0.0

    def test_wrong_dfrac_still_zero(self) -> None:
        """A wrong dfrac must still score 0.0."""
        candidate = r"\boxed{\dfrac{1}{3}}"
        assert grade_math(candidate, "0.5") == 0.0

    def test_deeply_nested_latex(self) -> None:
        r"""\\boxed{\\frac{\\sqrt{2}}{3}} should be graded correctly."""
        candidate = r"\boxed{\frac{\sqrt{2}}{3}}"
        # sqrt(2)/3 ~ 0.4714
        assert grade_math(candidate, "0.4714045207910317") == 1.0

    def test_latex_with_left_right(self) -> None:
        r"""\\left and \\right delimiters should not break grading."""
        candidate = r"\boxed{\left(\frac{1}{2}\right)}"
        assert grade_math(candidate, "0.5") == 1.0

    def test_dollar_wrapped_frac(self) -> None:
        """Inline $\\frac{a}{b}$ without \\boxed."""
        candidate = r"The answer is $\frac{50}{51}$"
        assert grade_math(candidate, "50/51") == 1.0

    def test_long_response_with_boxed_frac(self) -> None:
        """Realistic long response ending with boxed LaTeX fraction."""
        candidate = (
            "We start by decomposing the sum:\n"
            r"$\sum_{k=2}^{51} \frac{1}{k(k+1)} = \sum_{k=2}^{51}"
            r"\left(\frac{1}{k} - \frac{1}{k+1}\right)$"
            "\n\nTelescoping gives:\n"
            r"$= \frac{1}{2} - \frac{1}{52} = \frac{25}{52}$"
            "\n\nTherefore the answer is "
            r"$\boxed{\frac{625}{861}}$"
        )
        assert grade_math(candidate, "625/861") == 1.0


# ---------------------------------------------------------------------------
# Async path regression
# ---------------------------------------------------------------------------


class TestGradeMathAsyncLatex:
    """Verify async path also handles LaTeX correctly."""

    @pytest.mark.asyncio
    async def test_boxed_frac_async(self) -> None:
        """Async path: \\boxed{\\frac{50}{51}} vs '50/51'."""
        result = await grade_math_async(r"\boxed{\frac{50}{51}}", "50/51")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_no_boxed_async(self) -> None:
        """Async path: 'answer: 50/51' vs '50/51'."""
        result = await grade_math_async("answer: 50/51", "50/51")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_wrong_answer_async(self) -> None:
        """Async path: wrong answer still 0.0."""
        result = await grade_math_async(r"\boxed{\frac{49}{51}}", "50/51")
        assert result == 0.0

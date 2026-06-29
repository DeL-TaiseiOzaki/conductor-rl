"""Tests for conductor_workflow.graders.mcq_exact."""

from __future__ import annotations

from typing import Any

import pytest

from conductor_workflow.graders.mcq_exact import extract_letter, grade_mcq

# ---------------------------------------------------------------------------
# Letter extraction - happy path
# ---------------------------------------------------------------------------


class TestExtractLetterHappyPath:
    """Standard extraction patterns."""

    def test_boxed_letter(self) -> None:
        assert extract_letter(r"The answer is \boxed{A}") == "A"

    def test_answer_is_paren(self) -> None:
        assert extract_letter("The answer is (B)") == "B"

    def test_answer_is_no_paren(self) -> None:
        assert extract_letter("The answer is C") == "C"

    def test_bold_markdown(self) -> None:
        assert extract_letter("Therefore, **D**") == "D"

    def test_trailing_standalone_letter(self) -> None:
        assert extract_letter("After analysis:\nA") == "A"

    def test_lowercase_normalized(self) -> None:
        assert extract_letter(r"\boxed{b}") == "B"

    def test_last_match_wins(self) -> None:
        text = r"First I thought \boxed{A} but actually \boxed{C}"
        assert extract_letter(text) == "C"


# ---------------------------------------------------------------------------
# Letter extraction - edge cases
# ---------------------------------------------------------------------------


class TestExtractLetterEdgeCases:
    """Edge cases for extraction."""

    def test_empty_string(self) -> None:
        assert extract_letter("") is None

    def test_no_letter_found(self) -> None:
        assert extract_letter("I have no idea what the answer could be.") is None

    def test_letter_in_word_not_extracted(self) -> None:
        # "A" appears in "Analysis" but should not be extracted as answer
        # unless it's in a recognized pattern
        result = extract_letter("My detailed analysis of the problem.")
        assert result is None

    def test_standalone_letter_with_period(self) -> None:
        assert extract_letter("The correct option is:\nB.") == "B"

    def test_standalone_letter_with_paren(self) -> None:
        assert extract_letter("Choose:\n(C)") == "C"

    def test_answer_is_with_colon(self) -> None:
        assert extract_letter("The answer is: D") == "D"


# ---------------------------------------------------------------------------
# grade_mcq - happy path
# ---------------------------------------------------------------------------


class TestGradeMcqHappyPath:
    """Correct and incorrect grading."""

    def test_correct_answer(self) -> None:
        assert grade_mcq("The answer is (A)", "A") == 1.0

    def test_wrong_answer(self) -> None:
        assert grade_mcq("The answer is (B)", "A") == 0.0

    def test_extraction_failure_scores_zero(self) -> None:
        assert grade_mcq("I don't know", "A") == 0.0


# ---------------------------------------------------------------------------
# grade_mcq - error cases
# ---------------------------------------------------------------------------


class TestGradeMcqErrors:
    """Invalid gold letters should raise ValueError."""

    def test_invalid_gold_letter(self) -> None:
        with pytest.raises(ValueError, match="A-D"):
            grade_mcq("anything", "E")

    def test_empty_gold_letter(self) -> None:
        with pytest.raises(ValueError, match="A-D"):
            grade_mcq("anything", "")


# ---------------------------------------------------------------------------
# Real pilot data
# ---------------------------------------------------------------------------


class TestGradeMcqPilotData:
    """Test against real pilot items."""

    def test_sci_0001_correct_answer(self, first_mcq_item: dict[str, Any]) -> None:
        """sci-0001 gold is 'A' (Carnot efficiency 40%)."""
        gold = first_mcq_item["gold"]
        # Simulate a correct response
        candidate = (
            f"The Carnot efficiency is 1 - 300/500 = 0.40 = 40%. The answer is ({gold})"
        )
        assert grade_mcq(candidate, gold) == 1.0

    def test_sci_0001_wrong_answer(self, first_mcq_item: dict[str, Any]) -> None:
        gold = first_mcq_item["gold"]
        candidate = "I think the answer is (D)"
        # gold is A, candidate says D
        if gold != "D":
            assert grade_mcq(candidate, gold) == 0.0

    def test_multiple_mcq_items_gold_valid(
        self, science_mcq_items: list[dict[str, Any]]
    ) -> None:
        """All MCQ items should have a valid gold letter."""
        for item in science_mcq_items:
            assert item["gold"] in {"A", "B", "C", "D"}, (
                f"{item['id']} has invalid gold: {item['gold']!r}"
            )

"""Tests for conductor_workflow.parser."""

from __future__ import annotations

import pytest

from conductor_workflow.parser import parse_workflow

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestParseWorkflowHappyPath:
    """Valid workflow blocks should parse correctly."""

    def test_single_subtask_returns_valid(self) -> None:
        # Arrange
        text = (
            "Here is the plan:\n"
            "```workflow\n"
            '{"subtasks": ["Solve it"], "model_id": [2], "access_list": [[]]}\n'
            "```"
        )

        # Act
        result = parse_workflow(text)

        # Assert
        assert result.valid is True
        assert result.f_fmt == pytest.approx(1.0, abs=0.01)
        assert len(result.nodes) == 1
        assert result.nodes[0].instruction == "Solve it"
        assert result.nodes[0].model_id == 2
        assert result.nodes[0].deps == []
        assert result.errors == []

    def test_multi_subtask_chain(self) -> None:
        # Arrange
        text = (
            "```workflow\n"
            '{"subtasks": ["Draft", "Review", "Final"],'
            ' "model_id": [0, 2, 3],'
            ' "access_list": [[], [0], [0, 1]]}\n'
            "```"
        )

        # Act
        result = parse_workflow(text)

        # Assert
        assert result.valid is True
        assert len(result.nodes) == 3
        assert result.nodes[2].deps == [0, 1]

    def test_surrounding_text_ignored(self) -> None:
        # Arrange
        text = (
            "I will use model 2.\n\n"
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [1], "access_list": [[]]}\n'
            "```\n\n"
            "That should work."
        )

        # Act
        result = parse_workflow(text)

        # Assert
        assert result.valid is True
        assert result.nodes[0].model_id == 1

    def test_last_node_is_final_answer(self) -> None:
        # Arrange
        text = (
            "```workflow\n"
            '{"subtasks": ["Step1", "Step2"], "model_id": [0, 2],'
            ' "access_list": [[], [0]]}\n'
            "```"
        )

        # Act
        result = parse_workflow(text)

        # Assert
        assert result.nodes[-1].index == 1
        assert result.nodes[-1].instruction == "Step2"


# ---------------------------------------------------------------------------
# Partial credit (f_fmt grading)
# ---------------------------------------------------------------------------


class TestParseWorkflowPartialCredit:
    """Invalid blocks should receive partial credit proportional to
    how many checks pass."""

    def test_no_block_scores_zero(self) -> None:
        result = parse_workflow("No workflow here")
        assert result.valid is False
        assert result.f_fmt == 0.0

    def test_empty_input_scores_zero(self) -> None:
        result = parse_workflow("")
        assert result.valid is False
        assert result.f_fmt == 0.0

    def test_block_found_but_invalid_json(self) -> None:
        text = "```workflow\n{not valid json}\n```"
        result = parse_workflow(text)
        assert result.valid is False
        assert result.f_fmt > 0.0  # block found credit
        assert result.f_fmt < 0.3  # but not much more

    def test_valid_json_but_missing_keys(self) -> None:
        text = '```workflow\n{"subtasks": ["X"]}\n```'
        result = parse_workflow(text)
        assert result.valid is False
        # block found + valid JSON + partial keys
        assert result.f_fmt > 0.2

    def test_unequal_lengths(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["A", "B"], "model_id": [0],'
            ' "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False
        assert result.f_fmt > 0.3  # block + json + keys

    def test_invalid_model_id_partial(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["A", "B"], "model_id": [0, 99],'
            ' "access_list": [[], [0]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False
        # Should have substantial partial credit (most checks pass)
        assert result.f_fmt > 0.5

    def test_invalid_access_list_backward_ref(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["A", "B"], "model_id": [0, 1],'
            ' "access_list": [[], [1]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False
        # access_list[1] references index 1 (self), violating DAG constraint
        assert result.f_fmt > 0.6

    def test_valid_workflow_scores_one(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [0], "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is True
        assert result.f_fmt == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestParseWorkflowEdgeCases:
    """Edge cases and boundary conditions."""

    def test_model_id_boundary_zero(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [0], "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is True

    def test_model_id_boundary_three(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [3], "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is True

    def test_model_id_four_is_invalid(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [4], "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False

    def test_model_id_negative_is_invalid(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [-1], "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False

    def test_access_list_first_node_must_be_empty(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": ["X"], "model_id": [0], "access_list": [[0]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False

    def test_non_list_values_rejected(self) -> None:
        text = (
            "```workflow\n"
            '{"subtasks": "not a list", "model_id": [0],'
            ' "access_list": [[]]}\n'
            "```"
        )
        result = parse_workflow(text)
        assert result.valid is False

    def test_json_array_root_rejected(self) -> None:
        text = "```workflow\n[1, 2, 3]\n```"
        result = parse_workflow(text)
        assert result.valid is False

    def test_result_is_frozen_dataclass(self) -> None:
        result = parse_workflow("```workflow\n{}\n```")
        with pytest.raises(AttributeError):
            result.valid = True  # type: ignore[misc]

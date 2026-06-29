r"""Parse and validate Conductor workflow JSON blocks.

Extracts a fenced ``\`\`\`workflow ... \`\`\``` block from raw model
completion text, validates structural constraints (equal-length arrays,
legal model_id range, acyclic access_list), and computes a graded
format score ``f_fmt in [0, 1]`` with partial credit.

Pure, synchronous, no side effects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKFLOW_BLOCK_RE: re.Pattern[str] = re.compile(
    r"```workflow\s*\n(.*?)\n\s*```", re.DOTALL
)

REQUIRED_KEYS: frozenset[str] = frozenset({"subtasks", "model_id", "access_list"})

VALID_MODEL_IDS: frozenset[int] = frozenset({0, 1, 2, 3})

# Partial-credit weights (must sum to 1.0).
_W_BLOCK_FOUND = 0.10
_W_VALID_JSON = 0.15
_W_KEYS_PRESENT = 0.15
_W_EQUAL_LENGTH = 0.15
_W_MODEL_RANGE = 0.15
_W_ACCESS_VALID = 0.15
_W_AT_LEAST_ONE = 0.15


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtaskNode:
    """A single node in the workflow DAG."""

    index: int
    instruction: str
    model_id: int
    deps: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a Conductor workflow completion.

    Attributes:
        valid: True only when ALL structural checks pass.
        f_fmt: Graded format score in [0, 1] (partial credit).
        nodes: Ordered list of DAG nodes (empty when invalid).
        raw_json: The parsed JSON dict (None if extraction failed).
        errors: Human-readable list of validation failures.
    """

    valid: bool
    f_fmt: float
    nodes: list[SubtaskNode] = field(default_factory=list)
    raw_json: dict | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_workflow(text: str) -> ParseResult:
    """Parse and validate a workflow block from raw completion text.

    Returns a ``ParseResult`` carrying the DAG, validity flag, and
    graded ``f_fmt`` score.
    """
    score = 0.0
    errors: list[str] = []

    # --- Check 1: fenced block found ---
    if not text:
        return ParseResult(valid=False, f_fmt=0.0, errors=["empty input"])

    match = WORKFLOW_BLOCK_RE.search(text)
    if match is None:
        return ParseResult(
            valid=False, f_fmt=0.0, errors=["no ```workflow block found"]
        )
    score += _W_BLOCK_FOUND

    # --- Check 2: valid JSON ---
    raw_content = match.group(1)
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON: {exc}")
        return ParseResult(valid=False, f_fmt=score, errors=errors)

    if not isinstance(data, dict):
        errors.append("JSON root is not an object")
        return ParseResult(valid=False, f_fmt=score, errors=errors)
    score += _W_VALID_JSON

    # --- Check 3: required keys ---
    missing_keys = REQUIRED_KEYS - data.keys()
    if missing_keys:
        errors.append(f"missing keys: {sorted(missing_keys)}")
        # partial: credit per present key
        present_ratio = (len(REQUIRED_KEYS) - len(missing_keys)) / len(REQUIRED_KEYS)
        score += _W_KEYS_PRESENT * present_ratio
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)
    score += _W_KEYS_PRESENT

    subtasks = data["subtasks"]
    model_ids = data["model_id"]
    access_lists = data["access_list"]

    # All three must be lists
    if not (
        isinstance(subtasks, list)
        and isinstance(model_ids, list)
        and isinstance(access_lists, list)
    ):
        errors.append("subtasks/model_id/access_list must all be lists")
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)

    # --- Check 4: equal length ---
    lengths = {len(subtasks), len(model_ids), len(access_lists)}
    if len(lengths) != 1:
        errors.append(f"array lengths differ: {lengths}")
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)
    score += _W_EQUAL_LENGTH

    n = len(subtasks)

    # --- Check 5: at least 1 subtask ---
    if n < 1:
        errors.append("must have at least 1 subtask")
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)
    score += _W_AT_LEAST_ONE

    # --- Check 6: model_id range ---
    bad_models = [
        (i, mid)
        for i, mid in enumerate(model_ids)
        if not isinstance(mid, int) or mid not in VALID_MODEL_IDS
    ]
    if bad_models:
        errors.append(f"invalid model_id at indices: {bad_models}")
        # partial: fraction of valid entries
        valid_ratio = (n - len(bad_models)) / n
        score += _W_MODEL_RANGE * valid_ratio
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)
    score += _W_MODEL_RANGE

    # --- Check 7: access_list validity (acyclic DAG) ---
    bad_access: list[tuple[int, object]] = []
    for i, deps in enumerate(access_lists):
        if not isinstance(deps, list):
            bad_access.append((i, deps))
            continue
        for dep in deps:
            if not isinstance(dep, int) or dep < 0 or dep >= i:
                bad_access.append((i, dep))

    if bad_access:
        errors.append(f"invalid access_list entries: {bad_access}")
        valid_ratio = (n - len(bad_access)) / n
        score += _W_ACCESS_VALID * valid_ratio
        return ParseResult(valid=False, f_fmt=score, raw_json=data, errors=errors)
    score += _W_ACCESS_VALID

    # --- All checks pass: build DAG ---
    nodes: list[SubtaskNode] = []
    for i in range(n):
        instruction = subtasks[i] if isinstance(subtasks[i], str) else str(subtasks[i])
        nodes.append(
            SubtaskNode(
                index=i,
                instruction=instruction,
                model_id=model_ids[i],
                deps=list(access_lists[i]),
            )
        )

    return ParseResult(
        valid=True,
        f_fmt=min(score, 1.0),
        nodes=nodes,
        raw_json=data,
        errors=[],
    )

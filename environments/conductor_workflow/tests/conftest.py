"""Shared fixtures for conductor-workflow tests.

Loads real pilot items from ``data/pilot/*.jsonl`` (relative path from
the package root up to the repo data directory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# environments/conductor_workflow/tests/conftest.py  -> ../../.. -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PILOT_DIR = _REPO_ROOT / "data" / "pilot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pilot_dir() -> Path:
    """Path to the pilot data directory."""
    assert _PILOT_DIR.exists(), f"Pilot data not found at {_PILOT_DIR}"
    return _PILOT_DIR


@pytest.fixture(scope="session")
def code_items(pilot_dir: Path) -> list[dict[str, Any]]:
    """All code cluster items from pilot data."""
    return _load_jsonl(pilot_dir / "code.jsonl")


@pytest.fixture(scope="session")
def science_mcq_items(pilot_dir: Path) -> list[dict[str, Any]]:
    """All science_mcq cluster items from pilot data."""
    return _load_jsonl(pilot_dir / "science_mcq.jsonl")


@pytest.fixture(scope="session")
def hard_math_items(pilot_dir: Path) -> list[dict[str, Any]]:
    """All hard_math cluster items from pilot data."""
    return _load_jsonl(pilot_dir / "hard_math.jsonl")


@pytest.fixture(scope="session")
def first_code_item(code_items: list[dict[str, Any]]) -> dict[str, Any]:
    """First code item (code-0001: alternating sum)."""
    return code_items[0]


@pytest.fixture(scope="session")
def first_mcq_item(science_mcq_items: list[dict[str, Any]]) -> dict[str, Any]:
    """First science_mcq item (sci-0001: Carnot efficiency)."""
    return science_mcq_items[0]


@pytest.fixture(scope="session")
def first_math_item(hard_math_items: list[dict[str, Any]]) -> dict[str, Any]:
    """First hard_math item (math-0001: telescoping sum)."""
    return hard_math_items[0]

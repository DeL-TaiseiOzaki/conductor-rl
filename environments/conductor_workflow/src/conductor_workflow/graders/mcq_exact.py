"""Multiple-choice letter extraction and exact match grader.

Extracts the final answer letter (A-D) from candidate text using
multiple robust patterns, then compares to the gold letter.
Returns ``s_correct in {0, 1}``.

Extraction strategy (checked in priority order):
1. ``\\boxed{X}``  -- LaTeX boxed answer
2. ``answer is (X)`` / ``answer is X`` -- explicit declaration
3. ``The answer is X`` -- common phrasing
4. ``**X**`` -- bold markdown letter at end
5. Option-label on last line: ``A) ...``, ``A. ...``, ``A: ...``
6. Trailing standalone capital letter (last non-whitespace char
   on the last non-empty line, if it is A-D)

False negatives are mostly extraction failures (per data-spec);
the multi-pattern approach minimizes them.

Pure, synchronous, no side effects.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LETTERS: frozenset[str] = frozenset({"A", "B", "C", "D"})

# Patterns ordered by specificity (most specific first).
# Each pattern should capture the letter in group 1.
_BOXED_RE = re.compile(r"\\boxed\{([A-Da-d])\}")
_ANSWER_IS_PAREN_RE = re.compile(r"[Aa]nswer\s+is\s*\(?([A-Da-d])\)?", re.IGNORECASE)
_THE_ANSWER_IS_RE = re.compile(
    r"[Tt]he\s+answer\s+is\s*:?\s*\(?([A-Da-d])\)?", re.IGNORECASE
)
_BOLD_LETTER_RE = re.compile(r"\*\*([A-Da-d])\*\*")
# Option-label form: "A) ...", "A. ...", "A: ..." at the start of a line
# Captures a letter immediately followed by ), ., or : (optionally with space).
_OPTION_LABEL_RE = re.compile(r"(?:^|\n)\s*([A-Da-d])\s*[).:]", re.MULTILINE)
_OPTION_LETTER_RE = re.compile(r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_letter(text: str) -> str | None:
    """Extract the final MCQ answer letter from *text*.

    Returns an uppercase letter A-D, or None if extraction fails.
    """
    if not text:
        return None

    # Strategy: apply patterns from most specific to least specific.
    # For patterns that can match multiple times, take the LAST match
    # (the model's final answer is typically at the end).

    # 1. \\boxed{X}
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].upper()

    # 2. "answer is (X)" / "answer is X"
    matches = _ANSWER_IS_PAREN_RE.findall(text)
    if matches:
        return matches[-1].upper()

    # 3. "The answer is X"
    matches = _THE_ANSWER_IS_RE.findall(text)
    if matches:
        return matches[-1].upper()

    # 4. **X** (bold markdown)
    matches = _BOLD_LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()

    # 5. Option-label on the last non-empty line: "A) ...", "A. ...", "A: ..."
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        m = _OPTION_LABEL_RE.search("\n" + last_line)
        if m:
            return m.group(1).upper()

    # 6. Trailing standalone letter on last non-empty line
    if lines:
        last_line = lines[-1]
        # Check if last line is just a single letter (possibly with parens/period)
        clean = re.sub(r"[().\s]", "", last_line)
        if len(clean) == 1 and clean.upper() in VALID_LETTERS:
            return clean.upper()

    return None


def grade_mcq(candidate_text: str, gold_letter: str) -> float:
    """Grade a multiple-choice answer.

    Args:
        candidate_text: The model's full response text.
        gold_letter: The correct letter (A-D).

    Returns:
        ``s_correct``: 1.0 if extracted letter matches gold, else 0.0.

    Raises:
        ValueError: If ``gold_letter`` is not A-D.
    """
    gold_upper = gold_letter.strip().upper()
    if gold_upper not in VALID_LETTERS:
        raise ValueError(f"gold_letter must be A-D, got {gold_letter!r}")

    extracted = extract_letter(candidate_text)
    if extracted is None:
        return 0.0

    return 1.0 if extracted == gold_upper else 0.0

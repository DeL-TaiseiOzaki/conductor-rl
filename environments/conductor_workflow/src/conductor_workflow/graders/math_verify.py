"""SymPy-based mathematical equivalence grader.

Extracts the final answer from candidate text (preferring ``\\boxed{}``),
compares to the gold answer via SymPy symbolic equivalence with a
numerical tolerance fallback.  Handles equivalent forms such as
``1/2`` vs ``0.5``, unsimplified expressions, etc.

A ``TinyVFallback`` protocol is provided for a lightweight LLM verifier
that catches false negatives from symbolic comparison.  The synchronous
``grade_math`` never calls it (preserving Phase 1 behaviour).  The async
``grade_math_async`` calls the fallback ONLY when SymPy is uncertain
(can't parse or disagrees) and ``only_on_uncertain=True`` (default).

Pure (SymPy only) for the sync path; async path may invoke network.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from sympy import Abs, N, Rational, oo, simplify, sympify
from sympy.core.expr import Expr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE: float = 1e-6

# Extraction patterns (most specific first)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_DOLLAR_RE = re.compile(r"\$([^$]+)\$")
_EQUALS_RE = re.compile(r"=\s*([^\s,;.]+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# TinyV fallback interface (stub -- not called in tests)
# ---------------------------------------------------------------------------


@runtime_checkable
class TinyVFallback(Protocol):
    """Interface for a lightweight LLM verifier fallback.

    Implementations make async network calls to a judge model that
    decides whether two mathematical expressions are equivalent.
    Sync ``grade_math`` never calls it; async ``grade_math_async`` does.
    """

    async def check_equivalence(
        self,
        candidate_answer: str,
        gold_answer: str,
        *,
        context: str | None = None,
    ) -> bool:
        """Return True if the two answers are mathematically equivalent."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_math_answer(text: str) -> str | None:
    """Extract the final mathematical answer from *text*.

    Priority:
    1. Last ``\\boxed{...}`` expression.
    2. Last inline ``$...$`` expression.
    3. Last ``= <value>`` on its own line.
    4. Last non-empty line (stripped).

    Returns the raw string (not yet parsed by SymPy).
    """
    if not text:
        return None

    # 1. \\boxed{...} -- take the last one (final answer)
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 2. Inline $...$
    matches = _DOLLAR_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 3. = <value> at end of a line
    matches = _EQUALS_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 4. Last non-empty line
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return None


# ---------------------------------------------------------------------------
# SymPy comparison
# ---------------------------------------------------------------------------


def _safe_sympify(expr_str: str) -> Expr | None:
    """Attempt to parse *expr_str* as a SymPy expression.

    Returns None on failure (unparseable input).
    """
    if not expr_str:
        return None

    # Clean common LaTeX artifacts
    cleaned = expr_str.replace("\\", "").replace("{", "").replace("}", "")
    cleaned = cleaned.replace("dfrac", "").replace("frac", "")
    cleaned = cleaned.replace("left", "").replace("right", "")
    cleaned = cleaned.replace("cdot", "*").replace("times", "*")
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # Try direct sympify first
    try:
        return sympify(cleaned, rational=True)  # ty: ignore[no-matching-overload]  # SymPy stubs lack rational kwarg
    except Exception:
        pass

    # Try as a Python literal (handles "50/51" etc.)
    try:
        # Rational("50/51") handles fraction strings
        return Rational(cleaned)
    except Exception:
        pass

    return None


def _symbolic_equal(
    candidate_expr: Expr,
    gold_expr: Expr,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Check symbolic equivalence, with numerical fallback."""
    # 1. Exact symbolic equality after simplification
    try:
        diff = simplify(candidate_expr - gold_expr)
        if diff == 0:
            return True
    except Exception:
        pass

    # 2. Numerical comparison with tolerance
    try:
        candidate_val = complex(N(candidate_expr))
        gold_val = complex(N(gold_expr))

        # Handle infinite values
        if candidate_val == gold_val:
            return True

        # Both must be finite for tolerance comparison
        if not (abs(candidate_val.real) < float(oo) and abs(gold_val.real) < float(oo)):
            return False

        # Absolute difference
        num_diff = abs(candidate_val - gold_val)
        if num_diff <= tolerance:
            return True

        # Relative difference (avoid division by zero)
        if abs(gold_val) > tolerance:
            rel_diff = num_diff / abs(gold_val)
            if rel_diff <= tolerance:
                return True
    except Exception:
        pass

    # 3. Check via Abs(diff) (symbolic absolute difference)
    try:
        abs_diff = Abs(candidate_expr - gold_expr)
        if simplify(abs_diff) == 0:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_math(
    candidate_text: str,
    gold_answer: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    tiny_v_fallback: TinyVFallback | None = None,
) -> float:
    """Grade a mathematical answer via SymPy equivalence.

    Args:
        candidate_text: The model's full response text.
        gold_answer: The gold answer string (SymPy-parseable).
        tolerance: Numerical tolerance for equivalence check.
        tiny_v_fallback: Optional LLM fallback verifier.
            **Not called in current implementation** -- interface only.

    Returns:
        ``s_correct``: 1.0 if equivalent, else 0.0.
    """
    extracted = extract_math_answer(candidate_text)
    if extracted is None:
        return 0.0

    candidate_expr = _safe_sympify(extracted)
    gold_expr = _safe_sympify(gold_answer)

    if candidate_expr is None or gold_expr is None:
        # Cannot parse one or both -- future TinyV fallback would go here.
        # tiny_v_fallback is intentionally NOT called (no network in tests).
        return 0.0

    if _symbolic_equal(candidate_expr, gold_expr, tolerance=tolerance):
        return 1.0

    # Sync path: never calls the fallback (no network).
    return 0.0


# ---------------------------------------------------------------------------
# Async API (with TinyV fallback)
# ---------------------------------------------------------------------------


async def grade_math_async(
    candidate_text: str,
    gold_answer: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    tiny_v_fallback: TinyVFallback | None = None,
    only_on_uncertain: bool = True,
) -> float:
    """Grade a mathematical answer, with optional async LLM fallback.

    The SymPy path runs first (synchronously).  If SymPy confirms
    equivalence, returns 1.0 immediately.  The fallback is invoked
    ONLY when SymPy is uncertain (unparseable or disagrees) and
    ``only_on_uncertain`` is True and a fallback is provided.

    Args:
        candidate_text: The model's full response text.
        gold_answer: The gold answer string.
        tolerance: Numerical tolerance for SymPy comparison.
        tiny_v_fallback: Optional async LLM fallback verifier.
        only_on_uncertain: Only call fallback when SymPy can't decide.

    Returns:
        ``s_correct``: 1.0 if equivalent, else 0.0.
    """
    extracted = extract_math_answer(candidate_text)
    if extracted is None:
        return 0.0

    candidate_expr = _safe_sympify(extracted)
    gold_expr = _safe_sympify(gold_answer)

    sympy_uncertain = candidate_expr is None or gold_expr is None

    if not sympy_uncertain:
        assert candidate_expr is not None  # for type checker
        assert gold_expr is not None
        if _symbolic_equal(candidate_expr, gold_expr, tolerance=tolerance):
            return 1.0
        # SymPy says not equal -- this is also "uncertain" for fallback
        sympy_uncertain = True

    # Fallback: call the judge if available and appropriate
    if tiny_v_fallback is not None and (not only_on_uncertain or sympy_uncertain):
        try:
            is_equiv = await tiny_v_fallback.check_equivalence(
                candidate_answer=extracted,
                gold_answer=gold_answer,
            )
            return 1.0 if is_equiv else 0.0
        except Exception:
            # Judge failure -> conservative: not equivalent
            return 0.0

    return 0.0

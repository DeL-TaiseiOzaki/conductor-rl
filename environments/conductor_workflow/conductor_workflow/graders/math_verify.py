"""Mathematical equivalence grader with LaTeX-aware parsing.

Uses the ``math-verify`` library as the primary comparison engine for
robust LaTeX extraction (``\\boxed{\\frac{a}{b}}``, ``\\dfrac``,
``\\sqrt``, etc.) and symbolic equivalence checking.  Falls back to
hand-rolled SymPy comparison when ``math-verify`` cannot parse.

A ``TinyVFallback`` protocol is provided for a lightweight LLM verifier
that catches false negatives from symbolic comparison.  The synchronous
``grade_math`` never calls it (preserving Phase 1 behaviour).  The async
``grade_math_async`` calls the fallback ONLY when the deterministic check
is uncertain and ``only_on_uncertain=True`` (default).

Pure (deterministic) for the sync path; async path may invoke network.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from math_verify import parse as mv_parse
from math_verify import verify as mv_verify
from sympy import Abs, N, Rational, oo, simplify, sympify
from sympy.core.expr import Expr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE: float = 1e-6

# Float rounding precision for math-verify (number of decimal places)
_MV_FLOAT_ROUNDING: int = 6

# Disable internal timeouts in math-verify so it is safe to call from
# worker threads (GRPO training uses ThreadPoolExecutor).  The caller is
# responsible for overall timeout management.
_MV_PARSE_TIMEOUT: int = 0
_MV_VERIFY_TIMEOUT: int | None = 0

# Extraction patterns (most specific first)
_BOXED_NESTED_RE = re.compile(r"\\boxed\{")
# Simple (flat) boxed -- used only when brace-balanced extraction fails
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_DOLLAR_RE = re.compile(r"\$([^$]+)\$")
_PAREN_LATEX_RE = re.compile(r"\\\((.+?)\\\)")
_BRACKET_LATEX_RE = re.compile(r"\\\[(.+?)\\\]")
_EQUALS_RE = re.compile(r"=\s*([^\s,;.]+)\s*$", re.MULTILINE)
# Trailing number or simple fraction (e.g. "answer: 50/51", "the answer is 42")
_TRAILING_EXPR_RE = re.compile(
    r"(?:^|[\s:])(\-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*\.?\s*$",
    re.MULTILINE,
)

# LaTeX commands to normalise for the SymPy fallback
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\{([^}]*)\}\{([^}]*)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^}]*)\}")
_LATEX_STRIP_COMMANDS = re.compile(
    r"\\(?:left|right|displaystyle|text|mathrm|operatorname)"
)


# ---------------------------------------------------------------------------
# TinyV fallback interface (stub -- not called in sync path)
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
# Brace-balanced boxed extraction
# ---------------------------------------------------------------------------


def _extract_boxed_contents(text: str) -> list[str]:
    r"""Extract contents of all ``\boxed{...}`` groups, handling nested braces.

    Unlike a simple ``[^}]+`` regex, this correctly captures
    ``\boxed{\frac{a}{b}}`` as ``\frac{a}{b}`` (not ``\frac{a``).
    """
    results: list[str] = []
    for match in _BOXED_NESTED_RE.finditer(text):
        start = match.end()  # position right after the opening '{'
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[start : pos - 1])
    return results


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_math_answer(text: str) -> str | None:
    r"""Extract the final mathematical answer from *text*.

    Priority:
    1. Last ``\boxed{...}`` expression (brace-balanced for nested LaTeX).
    2. Last inline ``$...$`` expression.
    3. Last ``\(...\)`` or ``\[...\]`` LaTeX math expression.
    4. Last ``= <value>`` on its own line.
    5. Last trailing number or fraction (e.g. ``answer: 50/51``).
    6. Last non-empty line (stripped).

    Returns the raw string (not yet parsed by SymPy).
    """
    if not text:
        return None

    # 1. \boxed{...} -- brace-balanced, take the last one (final answer)
    boxed = _extract_boxed_contents(text)
    if boxed:
        return boxed[-1].strip()

    # 2. Inline $...$
    matches = _DOLLAR_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 3. \(...\) or \[...\]
    matches = _PAREN_LATEX_RE.findall(text)
    if matches:
        return matches[-1].strip()
    matches = _BRACKET_LATEX_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 4. = <value> at end of a line
    matches = _EQUALS_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 5. Trailing number/fraction (handles "answer: 50/51")
    matches = _TRAILING_EXPR_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 6. Last non-empty line
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return None


# ---------------------------------------------------------------------------
# math-verify primary path
# ---------------------------------------------------------------------------


def _try_math_verify(candidate_text: str, gold_answer: str) -> bool | None:
    """Compare candidate and gold using the ``math-verify`` library.

    Returns ``True`` (equivalent), ``False`` (definitely not equivalent),
    or ``None`` (could not parse one or both -- uncertain).
    """
    try:
        parsed_candidate = mv_parse(
            candidate_text,
            parsing_timeout=_MV_PARSE_TIMEOUT,
        )
        parsed_gold = mv_parse(
            gold_answer,
            parsing_timeout=_MV_PARSE_TIMEOUT,
        )
    except Exception:
        logger.debug("math-verify parse raised an exception", exc_info=True)
        return None

    if not parsed_candidate or not parsed_gold:
        return None

    try:
        return mv_verify(
            gold=parsed_gold,
            target=parsed_candidate,
            float_rounding=_MV_FLOAT_ROUNDING,
            timeout_seconds=_MV_VERIFY_TIMEOUT,
        )
    except Exception:
        logger.debug("math-verify verify raised an exception", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# LaTeX normalisation (for SymPy fallback)
# ---------------------------------------------------------------------------


def _normalize_latex(expr_str: str) -> str:
    r"""Normalise LaTeX markup into a SymPy-parseable string.

    Transforms:
    - ``\frac{a}{b}`` / ``\dfrac`` / ``\tfrac``  ->  ``(a)/(b)``
    - ``\sqrt{x}``  ->  ``sqrt(x)``
    - Strips ``\left``, ``\right``, ``\displaystyle``, etc.
    - Strips surrounding ``$``, ``\(``, ``\)``, ``\[``, ``\]``
    - Replaces ``\cdot`` / ``\times`` with ``*``
    - Removes remaining ``\`` and ``{`` / ``}``
    """
    s = expr_str.strip()

    # Strip surrounding math delimiters
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()

    # \frac{a}{b} -> (a)/(b)
    while _LATEX_FRAC_RE.search(s):
        s = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", s)

    # \sqrt{x} -> sqrt(x)
    s = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)

    # Strip decorative commands
    s = _LATEX_STRIP_COMMANDS.sub("", s)

    # \cdot, \times -> *
    s = s.replace("\\cdot", "*").replace("\\times", "*")

    # Remove remaining backslashes and braces
    s = s.replace("\\", "").replace("{", "").replace("}", "")

    return s.strip()


# ---------------------------------------------------------------------------
# SymPy comparison (fallback)
# ---------------------------------------------------------------------------


def _safe_sympify(expr_str: str) -> Expr | None:
    """Attempt to parse *expr_str* as a SymPy expression.

    Applies LaTeX normalisation before parsing.
    Returns None on failure (unparseable input).
    """
    if not expr_str:
        return None

    cleaned = _normalize_latex(expr_str)

    if not cleaned:
        return None

    # Try direct sympify first
    try:
        return sympify(cleaned, rational=True)  # ty: ignore[no-matching-overload]  # SymPy stubs lack rational kwarg
    except Exception:
        pass

    # Try as a Python literal (handles "50/51" etc.)
    try:
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
# Core grading logic (shared by sync and async paths)
# ---------------------------------------------------------------------------


def _deterministic_grade(
    candidate_text: str,
    gold_answer: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float | None:
    """Run the deterministic (no-network) grading pipeline.

    Returns:
        ``1.0`` if equivalent, ``0.0`` if definitely not equivalent,
        or ``None`` if uncertain (both paths failed to parse).
    """
    # -- Layer 1: math-verify (LaTeX-aware, handles \frac, \sqrt, etc.) --
    mv_result = _try_math_verify(candidate_text, gold_answer)
    if mv_result is True:
        return 1.0
    if mv_result is False:
        # math-verify parsed both but says not equal.  Still try SymPy
        # fallback in case of edge-case disagreement, but treat this as
        # a strong signal.
        pass

    # -- Layer 2: hand-rolled extraction + SymPy --
    extracted = extract_math_answer(candidate_text)
    if extracted is None:
        # Nothing extractable and math-verify also failed
        if mv_result is False:
            return 0.0
        return None

    candidate_expr = _safe_sympify(extracted)
    gold_expr = _safe_sympify(gold_answer)

    if candidate_expr is None or gold_expr is None:
        # Could not parse -- uncertain unless math-verify already decided
        if mv_result is False:
            return 0.0
        return None

    if _symbolic_equal(candidate_expr, gold_expr, tolerance=tolerance):
        return 1.0

    # Both layers say not equal (or SymPy says no)
    return 0.0


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
    """Grade a mathematical answer via deterministic equivalence check.

    Uses ``math-verify`` (LaTeX-aware) as the primary engine, with a
    SymPy fallback.  The sync path never calls the LLM fallback.

    Args:
        candidate_text: The model's full response text.
        gold_answer: The gold answer string.
        tolerance: Numerical tolerance for equivalence check.
        tiny_v_fallback: Optional LLM fallback verifier.
            **Not called in sync path** -- interface preserved for compat.

    Returns:
        ``s_correct``: 1.0 if equivalent, else 0.0.
    """
    result = _deterministic_grade(candidate_text, gold_answer, tolerance)
    if result is not None:
        return result

    # Uncertain: sync path never calls the fallback (no network).
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

    The deterministic path (math-verify + SymPy) runs first.  If it
    confirms equivalence, returns 1.0 immediately.  The fallback is
    invoked ONLY when the deterministic check is uncertain (cannot parse)
    and ``only_on_uncertain`` is True and a fallback is provided.

    Args:
        candidate_text: The model's full response text.
        gold_answer: The gold answer string.
        tolerance: Numerical tolerance for comparison.
        tiny_v_fallback: Optional async LLM fallback verifier.
        only_on_uncertain: Only call fallback when deterministic check
            cannot decide.

    Returns:
        ``s_correct``: 1.0 if equivalent, else 0.0.
    """
    result = _deterministic_grade(candidate_text, gold_answer, tolerance)

    deterministic_uncertain = result is None

    if result is not None and result == 1.0:
        return 1.0

    # If deterministic says 0.0 or is uncertain, consider fallback
    if tiny_v_fallback is not None and (
        not only_on_uncertain or deterministic_uncertain
    ):
        extracted = extract_math_answer(candidate_text)
        candidate_for_judge = extracted if extracted is not None else candidate_text
        try:
            is_equiv = await tiny_v_fallback.check_equivalence(
                candidate_answer=candidate_for_judge,
                gold_answer=gold_answer,
            )
            return 1.0 if is_equiv else 0.0
        except Exception:
            # Judge failure -> conservative: not equivalent
            return 0.0

    if result is not None:
        return result
    return 0.0

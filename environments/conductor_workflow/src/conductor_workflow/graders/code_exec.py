"""Sandboxed stdin/stdout code execution grader.

Executes a candidate Python program against a list of test cases
(each with ``input`` / ``output``), compares stdout to expected output
after normalization, and returns ``s_correct`` as the fraction of tests
passed (continuous score; also exposes an all-pass binary flag).

Security / Isolation
--------------------
This module executes **untrusted code**.  The sandbox uses:

1. **Separate subprocess** (``subprocess.Popen``) -- crash/exit does not
   affect the grader process.
2. **Hard timeout** via ``subprocess.communicate(timeout=...)`` +
   ``SIGKILL`` on expiry.
3. **Resource limits** via ``resource.setrlimit`` in ``preexec_fn``:
   - ``RLIMIT_CPU``   : CPU seconds capped at ``time_limit_s``.
   - ``RLIMIT_AS``    : Virtual memory capped at ``memory_limit_bytes``
     (default 256 MiB).
   - ``RLIMIT_CORE``  : Core dumps disabled (0).
   - ``RLIMIT_NPROC`` : Fork-bomb mitigation (max 0 new processes, i.e.
     the child itself only; NOTE: this is per-uid, so may be too
     restrictive on shared hosts -- set to a small positive if needed).
4. **Best-effort network denial**: The child inherits no sockets from the
   parent; ``RLIMIT_NPROC=0`` prevents spawning helpers.  True network
   namespace isolation requires root (``unshare -n``) or container
   tooling (firejail / nsjail / bubblewrap), none of which are available
   on this host.

**Known limitations** (document for reviewers):
- Without a network namespace, the child *can* open sockets.
- ``RLIMIT_AS`` affects the entire virtual address space, which may
  cause spurious ``MemoryError`` on some Python builds that mmap
  aggressively (rare at 256 MiB).
- ``RLIMIT_NPROC`` is per-uid, not per-process; on a shared login node
  this could starve other user processes if set to 0.  We therefore
  default to a small positive value (``MAX_CHILD_PROCESSES``).
- No filesystem read/write restrictions beyond the inherent lack of
  write permissions in the working directory (we ``cwd`` to ``/tmp``).

Pure (apart from subprocess invocation), synchronous.
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIME_LIMIT_S: int = 5
DEFAULT_MEMORY_LIMIT_BYTES: int = 256 * 1024 * 1024  # 256 MiB
MAX_CHILD_PROCESSES: int = 4  # small but non-zero for shared hosts
SANDBOX_CWD: str = "/tmp"

# Regex for fenced code blocks: ```lang\n...\n```
_FENCED_BLOCK_RE = re.compile(
    r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```",
    re.DOTALL,
)

# Language tags recognised as Python
_PYTHON_TAGS: frozenset[str] = frozenset({"python", "py", "python3", "py3"})


# ---------------------------------------------------------------------------
# Code extraction helper
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """Extract Python source from worker output that may contain markdown fences.

    Strategy (in priority order):
    1. If one or more **python-tagged** fenced blocks exist, return the LAST one.
    2. Else if any un-tagged / other-language fenced blocks exist, return the
       LARGEST (by character count).
    3. Else return the input unchanged (assumed to be bare source already).
    """
    if not text:
        return text

    blocks = _FENCED_BLOCK_RE.findall(text)
    if not blocks:
        return text

    # blocks is list of (lang_tag, content)
    python_blocks = [content for tag, content in blocks if tag.lower() in _PYTHON_TAGS]
    if python_blocks:
        return python_blocks[-1].strip()

    # No python-tagged block — pick the largest block by character count
    all_contents = [content for _, content in blocks]
    largest = max(all_contents, key=len)
    return largest.strip()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeExecResult:
    """Result of grading a candidate program.

    Attributes:
        s_correct: Fraction of tests passed, in [0, 1].
        all_pass: True iff every test passed.
        passed: Number of tests that passed.
        total: Total number of tests.
        details: Per-test pass/fail + captured stdout / stderr.
    """

    s_correct: float
    all_pass: bool
    passed: int
    total: int
    details: list[dict[str, object]]


# ---------------------------------------------------------------------------
# Output normalization (must match worker normalization; see data-spec)
# ---------------------------------------------------------------------------


def normalize_output(raw: str) -> str:
    """Normalize output: per-line rstrip + overall rstrip."""
    lines = raw.split("\n")
    stripped_lines = [line.rstrip() for line in lines]
    return "\n".join(stripped_lines).rstrip()


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _make_preexec_fn(
    cpu_limit_s: int,
    mem_limit_bytes: int,
) -> Callable[[], None]:
    """Return a preexec_fn that sets resource limits in the child."""

    def _set_limits() -> None:
        # CPU time (hard kill by kernel)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
        # Virtual memory
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        # No core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Limit child processes (fork-bomb mitigation)
        resource.setrlimit(
            resource.RLIMIT_NPROC,
            (MAX_CHILD_PROCESSES, MAX_CHILD_PROCESSES),
        )

    return _set_limits


def _run_program(
    code: str,
    stdin_data: str,
    time_limit_s: int = DEFAULT_TIME_LIMIT_S,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> tuple[str, str, int | None]:
    """Execute *code* as a Python script with *stdin_data*.

    Returns (stdout, stderr, returncode).  On timeout, returncode is None.
    """
    # Wall-clock timeout slightly exceeds CPU limit to allow kernel cleanup
    wall_timeout = time_limit_s + 2

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=SANDBOX_CWD,
            env=_minimal_env(),
            preexec_fn=_make_preexec_fn(time_limit_s, memory_limit_bytes),
        )
    except OSError as exc:
        return "", str(exc), None

    try:
        stdout_bytes, stderr_bytes = proc.communicate(
            input=stdin_data.encode(), timeout=wall_timeout
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return "", "TimeoutExpired", None

    assert isinstance(stdout_bytes, bytes)  # Popen without text=True
    assert isinstance(stderr_bytes, bytes)
    stdout_text = stdout_bytes.decode(errors="replace")
    stderr_text = stderr_bytes.decode(errors="replace")
    return stdout_text, stderr_text, proc.returncode


def _minimal_env() -> dict[str, str]:
    """Build a minimal environment for the child process."""
    env: dict[str, str] = {}
    # Inherit only essential variables
    for key in ("PATH", "HOME", "LANG", "LC_ALL"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_code(
    candidate_code: str,
    tests: list[dict[str, str]],
    *,
    time_limit_s: int = DEFAULT_TIME_LIMIT_S,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> CodeExecResult:
    """Grade a candidate Python program against test cases.

    Args:
        candidate_code: Python source code to execute.
        tests: List of ``{"input": ..., "output": ...}`` test cases.
        time_limit_s: Per-test CPU time limit in seconds.
        memory_limit_bytes: Per-test virtual memory limit.

    Returns:
        ``CodeExecResult`` with fraction-of-tests-passed score.
    """
    if not tests:
        return CodeExecResult(
            s_correct=0.0, all_pass=False, passed=0, total=0, details=[]
        )

    details: list[dict[str, object]] = []
    passed = 0

    for test in tests:
        stdin_data = test.get("input", "")
        expected_raw = test.get("output", "")
        expected = normalize_output(expected_raw)

        stdout, stderr, returncode = _run_program(
            candidate_code,
            stdin_data,
            time_limit_s=time_limit_s,
            memory_limit_bytes=memory_limit_bytes,
        )

        actual = normalize_output(stdout)
        is_pass = actual == expected

        if is_pass:
            passed += 1

        details.append(
            {
                "pass": is_pass,
                "expected": expected,
                "actual": actual,
                "stderr": stderr[:500],  # truncate for safety
                "returncode": returncode,
            }
        )

    total = len(tests)
    s_correct = passed / total if total > 0 else 0.0

    return CodeExecResult(
        s_correct=s_correct,
        all_pass=(passed == total),
        passed=passed,
        total=total,
        details=details,
    )

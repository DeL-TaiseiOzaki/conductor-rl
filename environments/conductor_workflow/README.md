# conductor-workflow

Conductor-RL: Multi-agent workflow orchestration environment for GRPO training.

**Hub name**: `DeL-TaiseiOzaki/conductor-workflow`

## What this package is

Pure, framework-independent core logic for the Conductor-RL verifiers environment:

- **parser** -- extract and validate `\`\`\`workflow` JSON blocks from model completions, with graded partial-credit `f_fmt` scoring.
- **reward** -- tiered shaped reward: `R = w_corr*s_correct + w_fmt*f_fmt + w_exec*f_exec + w_eff*b_eff*1[correct]`.
- **graders/** -- per-cluster verifiers:
  - `code_exec` -- sandboxed stdin/stdout execution (subprocess + rlimit).
  - `mcq_exact` -- letter extraction + exact match.
  - `math_verify` -- SymPy equivalence with TinyV LLM fallback interface (stub).

## Build order

1. **This package** (Phase 1): pure logic, no network, no GPU, fully unit-tested.
2. **Phase 2**: `workers.py` + `executor.py` + `load_environment()` wiring (SingleTurnEnv + Rubric + async OpenRouter calls).
3. **Phase 3**: `prime env push` to Hub, prime-rl training config.

## Development

```bash
cd environments/conductor_workflow
uv venv && uv pip install -e ".[dev]"
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

# src/conductor — implementation plan (TODO)

Implement here (Python, `uv`). **No code committed yet** — this is the module spec for the implementing agent. Contracts live in `docs/data-spec.md` (verifiers) and `docs/reward-spec.md` (reward).

- **`workers.py`** — OpenRouter client. Async, parallel calls; per-worker `openrouter_variant` (`:nitro` / `:online`); retry + fallback on transient errors. Transient failures must NOT propagate into the reward's `f_exec` (retry instead).
- **`parser.py`** — parse the Conductor output: extract the ` ```workflow ` JSON block; validate (3 equal-length arrays; `model_id ∈ {0..3}`; `access_list[i] ⊆ {0..i-1}`); return a DAG, or a format error carrying partial-credit info for `f_fmt`.
- **`executor.py`** — run the DAG: topological order; run independent subtasks in parallel (`asyncio.gather`) so the latency proxy (critical path) is real; each worker's context = original task + subtask instruction + access_list outputs; the **last subtask's output is the final answer**.
- **`verifiers/`** — `code_exec.py` (sandboxed stdin/stdout; line + overall `rstrip`), `mcq_exact.py` (letter extraction), `math_verify.py` (SymPy + TinyV fallback). See `docs/data-spec.md`.
- **`reward.py`** — tiered shaped reward (`s_correct`, `f_fmt`, `f_exec`, `b_eff`). See `docs/reward-spec.md`.
- **`train_grpo.py`** — GRPO loop; vLLM serving of the Conductor + weight sync (cf. vLLM native RL APIs); difficulty-matched groups; curriculum (easy→hard, single-route→chain→parallel→search-aggregate).

Suggested order: `parser.py` + `reward.py` + verifiers (small, unit-testable) → `workers.py` + `executor.py` → `train_grpo.py`.

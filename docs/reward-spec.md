# Reward spec (Conductor GRPO)

Outcome-style reward computed per rollout (one Conductor-generated workflow, executed by the workers). **Tiered / shaped** so correctness dominates, process milestones give dense signal for cold-start, and efficiency is a **group-relative** bonus **only when correct**.

## Formula

```
R_i = w_corr * s_correct_i
    + w_fmt  * f_fmt_i
    + w_exec * f_exec_i
    + (w_lat * b_lat_i + w_cost * b_cost_i) * 1[correct_i]
```

Locked weights (v2): `w_corr=1.0`, `w_fmt=0.1`, `w_exec=0.1`.
Staging weights: `w_lat=0.0`, `w_cost=0.0` (efficiency OFF at start; each ramps to ~0.1).

**Invariant:** `max(wrong) = w_fmt + w_exec = 0.2 < min(correct)`. Max efficiency bonus = w_lat + w_cost <= 0.2 < min(correct). Correctness always dominates.

## Per-rollout terms

- **`s_correct`** in {0,1} (or [0,1]) -- from the per-cluster verifier (`docs/data-spec.md`):
  - `code` -- all tests pass (binary) **or fraction of tests passed (continuous)**.
  - `science_mcq` -- exact letter match (binary).
  - `hard_math` -- SymPy equivalence vs gold, + TinyV LLM fallback (binary).
- **`f_fmt`** in [0,1] -- workflow validity, graded partial credit: parseable JSON; 3 arrays equal length; `model_id in {0..3}`; `access_list[i] subset {0..i-1}` (acyclic).
- **`f_exec`** in [0,1] -- fraction of worker calls that were well-specified & executable (Conductor-controllable). Transient API failures are retried by the executor, NOT penalized here.

## Group-relative efficiency bonus (new in v2)

The efficiency bonus is **GROUP-RELATIVE** (no absolute C_ref baseline). Within each GRPO group (the G rollouts for one prompt), among the CORRECT rollouts only:

### Cost ranking

- `C_i` = REAL dollar cost of rollout i = sum over its worker calls of `(prompt_tokens * cost_in_per_1m + completion_tokens * cost_out_per_1m) / 1e6`. Judge (Nemotron) calls cost 0.
- Among correct rollouts, rank by cost (ascending, cheapest = rank 0):
  - `b_cost_i = (n_correct - 1 - rank_i) / (n_correct - 1)` so cheapest = 1.0, most expensive = 0.0.
  - If `n_correct == 1`: `b_cost = 0.5` (neutral).
  - Incorrect rollouts: `b_cost = 0`.

### Latency ranking

- `L_i` = deterministic latency = critical-path sum of per-worker `latency_weight` (parallel branches -> max). Already computed in executor.
- `b_lat_i` analogous to cost ranking on latency (fastest = 1.0).

### Combined

```
efficiency_i = (w_lat * b_lat_i + w_cost * b_cost_i) * 1[correct_i]
```

### Worker-call cache

A module-level cache keyed on `(resolved_slug, prompt, max_tokens, temperature)` avoids redundant API spend when the same call is repeated across rollouts. **The cache saves real wallet spend but the modeled cost (token counts) is still carried** so the reward function reflects deployment cost. Deterministic with temperature=0.

## Staging (avoid fragile multi-objective tuning)

- Keep `w_lat`, `w_cost` at **0** to start (accuracy-first). Once accuracy is stable, introduce **cost**, then **latency**.
- Recommended ramp: 0 -> 0.05 -> 0.1 per weight.
- The old `lambda_latency`/`mu_cost` baseline-relative mechanism is preserved for backward compatibility but the group-relative design is preferred.

## GRPO notes

- Group-relative advantage `A_i = (R_i - mean)/std`. Constant terms cancel within a group, so `f_fmt`/`f_exec` **auto-neutralize once mastered** (cold-start help without asymptotic distortion).
- The efficiency bonus is implemented as a **group-level reward function** in verifiers (plural params like `states` returning `list[float]`) so it has access to the full group for ranking.
- Use **difficulty-matched groups** so each group has solvable items (`std>0`); otherwise hard-cluster groups collapse (all wrong -> all 0.2 -> no gradient). Recommended `G=8-16`.

## Anti reward-hacking

- **Do NOT** add a "use >=2 workers" term -- it invites gaming. Single-route (and, later, self-answer) are valid cheapest paths; the correct-only gating makes them safe.
- `code` verifier: use private + generated / HARDTESTS-style tests; drop problems whose only public test equals the in-prompt example.

> Parent design: vault note `LLM-Orchestration-RL/conductor-rl-memo.md`. Verifier semantics: `docs/data-spec.md`.

# Reward spec (Conductor GRPO)

Outcome-style reward computed per rollout (one Conductor-generated workflow, executed by the workers). **Tiered / shaped** so correctness dominates, process milestones give dense signal for cold-start, and efficiency is a bonus **only when correct**.

## Formula

```
R = w_corr·s_correct + w_fmt·f_fmt + w_exec·f_exec + w_eff·b_eff·1[correct]
```

Locked weights (v1): `w_corr=1.0`, `w_fmt=0.1`, `w_exec=0.1`, `w_eff=0.2`.

**Invariant:** `max(wrong) = w_fmt + w_exec = 0.2  <  min(correct) ≈ 1.2`. Correctness always dominates.

## Terms

- **`s_correct`** ∈ {0,1} (or [0,1]) — from the per-cluster verifier (`docs/data-spec.md`):
  - `code` — all tests pass (binary) **or fraction of tests passed (continuous — recommended**; eases hard-cluster signal collapse).
  - `science_mcq` — exact letter match (binary).
  - `hard_math` — SymPy equivalence vs gold, `+ TinyV` LLM fallback for equivalent-form false negatives (binary).
- **`f_fmt`** ∈ [0,1] — workflow validity, graded partial credit (not just 0/1): parseable JSON; 3 arrays equal length; `model_id ∈ {0..3}`; `access_list[i] ⊆ {0..i-1}` (→ acyclic).
- **`f_exec`** ∈ [0,1] — fraction of worker calls that were **well-specified & executable** (Conductor-controllable). Transient API failures are **retried by the executor, NOT penalized here** (else infra noise leaks into the reward).
- **`b_eff`** ∈ [0,1] — efficiency, **baseline-relative**, gated by `1[correct]`:
  - latency proxy = **critical path** of per-model latency weights (parallel steps → `max`).
  - cost proxy = **sum** of per-call dollar weights (parallel steps → `sum`; every call is billed).
  - normalize against the *single-strongest-worker-alone* baseline; `b_eff=1` ⇒ much faster/cheaper than baseline.
  - asymmetry (latency `max`, cost `sum`) makes the Conductor see parallelization as "faster but pricier" correctly.

## Staging (avoid fragile multi-objective tuning)

- Keep `λ, μ` (inside `b_eff`) at **0** to start (accuracy-first). Once accuracy is stable, introduce **latency**, then **cost**.
- Optional **threshold form**: penalize only beyond a latency/cost budget (don't penalize already-cheap correct workflows).

## GRPO notes

- Group-relative advantage `A_i = (R_i − mean)/std`. Constant terms cancel within a group, so `f_fmt`/`f_exec` **auto-neutralize once mastered** (cold-start help without asymptotic distortion).
- Use **difficulty-matched groups** so each group has solvable items (`std>0`); otherwise hard-cluster groups collapse (all wrong → all 0.2 → no gradient). Recommended `G=8–16`.

## Anti reward-hacking

- **Do NOT** add a "use ≥2 workers" term — it invites gaming. Single-route (and, later, self-answer) are valid cheapest paths; the correct-only gating makes them safe.
- `code` verifier: use private + generated / HARDTESTS-style tests; drop problems whose only public test equals the in-prompt example.

> Parent design: vault note `LLM-Orchestration-RL/conductor-rl-自前構築メモ.md` (§報酬設計). Verifier semantics: `docs/data-spec.md`.

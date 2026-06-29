# Design Document — 要件定義書 (Requirements & Macro Design)

> **Role:** Macro-level requirements and design — *what* this project builds and *why*.
> **Document map:** Orchestrator contract → [CLAUDE.md](../../CLAUDE.md) ·
> Micro work progress → [PROGRESS.md](../../PROGRESS.md) · Design narrative SoT → `build.md` (repo root).
> Grounding research → [.claude/docs/research/](research/).

## 背景・目的 (Background & Purpose)

Train a small **Conductor** LLM (`Qwen/Qwen3.5-4B`) via **GRPO** to orchestrate a pool of cheap
OpenRouter worker LLMs — composing workflows (route / chain / parallel-aggregate / search→summarize /
solve→verify→fix) so tasks are solved **correctly, fast, and cheap**. Follows the Sakana
Fugu / Trinity / Conductor line ("a tiny orchestrator conducts large workers to elicit collective
intelligence"). **North star:** beat **Fugu-Ultra** on its own eval suite using only cheap OSS workers —
win on the cost-efficiency frontier and on absolute scores where OSS is near-frontier (coding).

## スコープ (Scope)

### In Scope

- A `verifiers`-library **Environment** (`vf.SingleTurnEnv`) implementing the Conductor task: one model
  generation = a workflow DAG; async reward parses → executes the DAG over OpenRouter workers → grades → tiered reward.
- Publishing that environment to the **Prime Intellect Environments Hub** (`DeL-TaiseiOzaki/conductor-workflow`).
- **prime-rl** GRPO training (trainer / inference / orchestrator) on a **multi-node Slurm** H100 cluster.
- Phase 1 clusters (text-only, static verification): `code`, `science_mcq`, `hard_math`. Pilot = 201 items.

### Out of Scope (for now)

- Charts/multimodal (pilot-2) and agentic SWE/Terminal clusters (Phase 2, Docker + multi-turn).
- A custom GRPO trainer — **prime-rl replaces the old `src/conductor/train_grpo.py` plan** (obsolete).
- Frontier (Claude/Gemini/GPT) workers — cost; OSS-only by decision (one frontier worker may be allowed for hardest subtasks later).

## 機能要件 (Functional Requirements)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Conductor outputs a ```workflow JSON block = 3 equal-length arrays (`subtasks`/`model_id`/`access_list`); last subtask = final answer | P0 | `prompts/conductor_system_prompt.md` |
| FR-2 | Parser validates the workflow (equal length, `model_id∈{0..3}`, `access_list[i]⊆{0..i-1}` acyclic) and yields **graded f_fmt** | P0 | partial credit, not 0/1 |
| FR-3 | Executor runs the DAG; independent subtasks in parallel (`asyncio.gather`) so latency proxy = real critical path | P0 | async, OpenRouter |
| FR-4 | Workers client: async parallel OpenRouter calls, per-worker variant (`:nitro`/`:online`), retry + fallback; transient failures retried (NOT penalized in f_exec) | P0 | `httpx.AsyncClient` / `AsyncOpenAI` only |
| FR-5 | Graders: code (sandbox pass-rate) / mcq (letter exact) / math (SymPy + TinyV fallback) → `s_correct` | P0 | `docs/data-spec.md` |
| FR-6 | Tiered reward `R = w_corr·s_correct + w_fmt·f_fmt + w_exec·f_exec + w_eff·b_eff·1[correct]` | P0 | `docs/reward-spec.md` |
| FR-7 | `load_environment(**kwargs)` exposes the env to the Hub / prime-rl | P0 | hatchling package |
| FR-8 | prime-rl configs (rl.toml + slurm.toml) + sbatch launch for multi-node H100 | P1 | next phase |

## 非機能要件 (Non-Functional Requirements)

| Category | Requirement | Metric / Target |
|----------|-------------|-----------------|
| Performance | Reward hot path fully async (no sync HTTP/sleep) | tolerate 100s+ concurrent rollouts; no serialization |
| Reproducibility | Pinned deps via `uv.lock`; deterministic graders | bad-JSON 0; grader/worker output normalization identical |
| Security | Code grader executes UNTRUSTED model code | subprocess + rlimit + timeout + best-effort net-deny; document isolation limits |
| Cost | Reward gates speed/cost bonus on correctness only | no "use ≥2 workers" term (anti reward-hacking) |
| Maintainability | One file = one responsibility, 200–400 LOC, typed | `.claude/rules/` |

## アーキテクチャ (Architecture)

```
[Conductor: Qwen3.5-4B] ──vLLM serve──┐  (1 generation = workflow JSON)
                                       ▼
prime-rl 3 processes: trainer(FSDP2) | inference(vLLM) | orchestrator(asyncio CPU)
                                       │  orchestrator pulls rollouts from the env
                                       ▼
[verifiers Environment  vf.SingleTurnEnv]   ← pushed to Environments Hub
   async reward = parser → DAG executor (OpenRouter workers, parallel) → grader → tiered reward
```

- **Env referenced by Hub id**: prime-rl TOML `[[orchestrator.train.env]] id = "DeL-TaiseiOzaki/conductor-workflow"`, `args={...}` → `load_environment` kwargs. Env pre-installed via `prime env install`.
- **GPU topology (4B)**: single node enough — 6 GPU vLLM inference + 2 GPU FSDP trainer (matches official Qwen3-4B example). **Multi-node = rollout throughput** (more inference nodes hide external-API reward latency — our reward is API-latency-bound).
- **Weight sync**: filesystem (NFS, default & safe) or NCCL broadcast. `/home` is NFS-shared across nodes ✓.
- **Off-policy**: `orchestrator.max_off_policy_steps` (default 8); IS correction; validated to async level 4 (INTELLECT-2).

### Component layout

| Component | Location | Phase |
|-----------|----------|-------|
| parser / reward / graders (pure, tested) | `environments/conductor_workflow/src/.../{parser,reward,graders}` | **1 (now)** |
| workers / executor / `load_environment` wiring | same package | 2 |
| prime-rl configs + sbatch | `configs/`, `scripts/` | 3 |

## 技術選定 (Tech Stack & Rationale)

| Area | Technology | Rationale | Alternatives Considered |
|------|------------|-----------|-------------------------|
| RL framework | **prime-rl** | async rollout fits API-latency-bound reward; 3-proc split; Slurm/multi-node; Hub env integration | TRL (sync GRPO — worse for async ext calls), OpenRLHF (Ray) |
| Env definition | **`verifiers` (Prime Intellect)** + Environments Hub | native `async def` reward; `SingleTurnEnv`; graded format reward; Hub publish/version | hand-rolled env |
| Conductor serving | vLLM | prime-rl inference engine; weight sync support | SGLang |
| Pkg / tooling | uv, hatchling, ruff, ty, pytest | repo `.claude/rules/dev-environment.md` | — |
| Graders | SymPy / math-verify; subprocess sandbox | `docs/data-spec.md` | nsjail/firejail if available |

## 制約 (Constraints)

- OSS-only workers (no Claude/Gemini/GPT) — cost. High-speed slot unified on DeepSeek (diversity traded for launch speed).
- prime-rl pins: Python ~=3.12, vLLM ≥0.23, torch ≥2.9, CUDA 12.8 (uv-managed; host python is 3.10).
- Shared NFS mandatory for multi-node (have `/home` 2.5T NFS, `/gcs` gcsfuse 1PB).
- Slurm nodes are cloud-autoscaled (`idle~`) — need GPU/NFS readiness pre-checks before training.
- `OPENROUTER_API_KEY` required for executor/worker phase (not yet set).

## Key Decisions

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| prime-rl + `verifiers` + Environments Hub | async reward fit; least custom infra; multi-node Slurm | TRL, OpenRLHF, custom trainer | 2026-06-29 |
| Env = `vf.SingleTurnEnv` w/ async reward funcs | one generation = whole workflow; heavy async scoring in Rubric | ToolEnv, MultiTurnEnv | 2026-06-29 |
| Drop custom `train_grpo.py` | prime-rl owns training/serving/weight-sync | self-built GRPO loop | 2026-06-29 |
| Single-node compute for 4B; multi-node for throughput | 4B fits 1 node; reward is API-bound so scale inference nodes | multi-node trainer sharding | 2026-06-29 |
| filesystem weight sync (NFS) default | `/home` NFS shared; safe; NCCL optional later | NCCL broadcast | 2026-06-29 |
| internal graders named `graders/` not `verifiers/` | avoid clash with the `verifiers` library | — | 2026-06-29 |

## TODO / Open Questions

- [ ] Phase 1 (now): bootstrap env package + parser/reward/graders + unit tests (in progress).
- [ ] Phase 2: workers.py + executor.py + `load_environment` wiring (Rubric); local `vf-eval`.
- [ ] Phase 3: prime-rl rl.toml/slurm.toml + sbatch; `prime env push`; smoke train on debug partition.
- [ ] Set `OPENROUTER_API_KEY`; confirm 4 worker slugs' `supported_parameters` via OpenRouter Models API.
- [ ] Resolve: per-rollout timeout for long DAG (30–60s); `SingleTurnEnv` `parser=` kwarg; v0.1.14 `Taskset/Harness` deprecation watch.
- [ ] Build-order risk: code grader gold is reference-defined (pilot OK; curate for real training).

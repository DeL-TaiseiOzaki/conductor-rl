# conductor-rl

Train a small **Conductor** (`Qwen/Qwen3.5-4B`) to orchestrate a pool of OpenRouter worker LLMs via **GRPO**, so it solves tasks **correctly, fast, and cheaply** by composing workflows (route / chain / parallel-aggregate / search→summarize / solve→verify→fix).

**North star:** beat **Sakana Fugu-Ultra** on its own eval suite (SWE-Bench Pro, Terminal-Bench 2.1, LiveCodeBench, GPQA-Diamond, HLE, CharXiv) using only **cheap OSS workers** — win on the cost-efficiency frontier, and on absolute scores where OSS is already near-frontier (coding).

> **Design SoT** (single source of truth) is the Obsidian vault note `MY_MEMORY/Others/Learning/LLM-Orchestration-RL/`. This repo carries the implementation + handed-off specs/data.

## Approach (decided)

- **Conductor** = `Qwen/Qwen3.5-4B` (local vLLM/SGLang). It outputs a workflow as 3 equal-length arrays `subtasks / model_id / access_list` (a DAG); the last subtask's output is the answer. See `prompts/conductor_system_prompt.md`.
- **Workers** (OpenRouter): `0 deepseek-v4-flash` (fast/cheap, aggregator) · `1 minimax-m3` (multimodal/tool) · `2 deepseek-v4-pro` (reasoning/code) · `3 z-ai/glm-5.2` (reasoning/code, diverse). No Claude (cost).
- **GRPO full-workflow, no SFT.** Cold-start handled by a crafted system prompt + dense format reward + easy→hard curriculum.
- **Reward** = tiered shaped: `w_corr·s_correct + w_fmt·f_fmt + w_exec·f_exec + w_eff·b_eff·1[correct]` (weights `1.0 / 0.1 / 0.1 / 0.2`). Speed/cost bonus only when correct. See `docs/reward-spec.md`.
- **Training data** = OOD-by-construction generated pilot (`data/pilot/pilot.jsonl`, 201 items, all gold-verified). Scale-up curates from CodeContests / SuperGPQA / OlympiadBench / ReachQA.

## Layout

```
prompts/         conductor system prompt (loaded verbatim)
data/pilot/      201 verified RL items (code / science_mcq / hard_math) + manifests
docs/            data-spec.md (schema + verifier rules), reward-spec.md (reward design)
configs/         default.yaml (workers, reward weights, GRPO hyperparams)
src/conductor/   implementation (TODO — see src/conductor/README.md)
```

## Status

- [x] Design · reward design · system prompt · pilot data (201, all `gold_confidence=1.0`)
- [ ] Verifiers (code sandbox / mcq exact / math-verify + TinyV fallback)
- [ ] Workflow parser + async DAG executor + OpenRouter wrapper
- [ ] Tiered reward function
- [ ] GRPO loop (vLLM serving + weight sync)

## Quickstart (TODO)

Implementation pending. Target stack: Python (uv), vLLM/SGLang to serve the Conductor, OpenRouter for workers, a GRPO trainer (verl / TRL / OpenRLHF — TBD). Copy `.env.example` → `.env` and set `OPENROUTER_API_KEY`.

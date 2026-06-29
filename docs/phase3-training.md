# Phase 3: prime-rl GRPO Training Runbook

> Last updated: 2026-06-29
> prime-rl version: v0.6.0 (pin to a specific commit for reproducibility)

## Overview

Train Qwen3.5-4B as a Conductor orchestrator via GRPO on a single 8xH100 node,
using the `o-taisei/conductor-workflow` environment published on the Prime
Intellect Environments Hub. Budget: $50 worker-API spend (OpenRouter).

## Prerequisites

### Software

| Component | Version | Notes |
|-----------|---------|-------|
| Python | ~=3.12.0 | Pinned in prime-rl pyproject.toml |
| PyTorch | >=2.9.0 | CUDA 12.8 wheels via uv |
| vLLM | >=0.23.0 | Bundled with prime-rl |
| transformers | 5.6.2 | Pinned |
| uv | >=0.11.1 | Package manager |
| prime CLI | latest | `pip install prime-cli` or via prime-rl |

### Hardware

| Resource | Spec |
|----------|------|
| GPU | 8x NVIDIA H100 80GB (single node) |
| Partition | `a3megatpa` (8 nodes) or `debug` (2 nodes) |
| Shared FS | `/home` NFS 2.5T (mounted on all nodes) |
| Network | InfiniBand or high-bandwidth ethernet (for NCCL, if multi-node) |

### Cluster Pre-flight

Nodes are cloud-autoscaled (`idle~`). Before submitting jobs:

1. **Check node availability**: `sinfo -p a3megatpa` -- nodes should be `idle` or `alloc`, not `down~`
2. **Verify GPU visibility**: `srun -p a3megatpa -N1 --gres=gpu:1 nvidia-smi`
3. **Verify NFS**: `srun -p a3megatpa -N1 ls /home/` -- should show shared contents
4. **Wake nodes** (if needed): submit a short dummy job to trigger provisioning, then wait

## Installation

### 1. Clone prime-rl onto shared NFS

```bash
cd /home
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
```

### 2. Install dependencies

```bash
uv sync --all-extras
```

### 3. Flash Attention 3 (H100 Hopper)

```bash
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation
```

### 4. Install the Conductor-Workflow environment

```bash
prime env install o-taisei/conductor-workflow
```

Verify:

```bash
uv run python -c "import conductor_workflow; print('OK')"
```

### 5. Set up secrets

```bash
# Copy the template
cp /home/conductor-rl/configs/secrets.env.example .env

# Edit .env and fill in:
#   OPENROUTER_API_KEY=sk-or-...
#   WANDB_API_KEY=...  (optional but recommended)
```

The prime-rl sbatch template sources `.env` from the project root before
launching. The conductor-workflow environment reads `OPENROUTER_API_KEY` via
`os.environ` at rollout time. Do NOT commit `.env`.

## Launch Sequence

### Step 1: Dry-run (validate config)

```bash
cd /home/prime-rl
uv run rl \
    @ /home/conductor-rl/configs/rl.toml \
    @ /home/conductor-rl/configs/slurm.toml \
    --output-dir /home/outputs/conductor-rl \
    --dry-run
```

This validates the TOML config, generates the sbatch script, and prints it
without submitting.

### Step 2: Smoke test (2 steps, ~$0.05)

```bash
uv run rl \
    @ /home/conductor-rl/configs/rl.toml \
    @ /home/conductor-rl/configs/smoke.toml \
    --output-dir /home/outputs/conductor-rl/smoke
```

Validates end-to-end: vLLM serves model, orchestrator drives rollouts through
the environment, workers are called via OpenRouter, rewards are computed,
trainer updates weights. Expected cost: ~$0.05 (160 worker calls).

### Step 3: Full training (100 steps, ~$30-50)

```bash
uv run rl \
    @ /home/conductor-rl/configs/rl.toml \
    @ /home/conductor-rl/configs/slurm.toml \
    --output-dir /home/outputs/conductor-rl
```

Or without Slurm (interactive on a GPU node):

```bash
srun -p a3megatpa -N1 --gres=gpu:8 --exclusive --pty bash
cd /home/prime-rl
source .env
uv run rl \
    @ /home/conductor-rl/configs/rl.toml \
    --output-dir /home/outputs/conductor-rl
```

### Monitoring

```bash
# Slurm job status
squeue -u $USER

# Logs (Slurm output)
tail -f /home/outputs/conductor-rl/job_*.log

# tmux helper (if available in prime-rl)
bash scripts/tmux.sh conductor-rl /home/outputs/conductor-rl

# Weights & Biases dashboard
# https://wandb.ai/<your-entity>/conductor-rl
```

Key metrics to watch:

| Metric | What it means | Action if abnormal |
|--------|---------------|-------------------|
| `reward/mean` | Rising = learning | Flat after 20 steps: check reward signal |
| `entropy` | Diversity of policy | Too low = mode collapse; too high = not converging |
| `optim/grad_norm` | Gradient magnitude | Spikes precede divergence; reduce lr |
| `time/wait_for_batch` | Orchestrator waiting | High = inference bottleneck; add inference replicas |
| `time/wait_for_ckpt` | Trainer bottleneck | High = FSDP slow; check GPU util |
| `mismatch_kl/all/mean` | Off-policy staleness | Rising = reduce async level |
| format compliance | f_fmt > 0 fraction | Low = model not learning workflow JSON format |

## GPU / Node Topology

### Single-node (default, recommended for pilot)

```
Node 0 (8xH100):
  GPU 0-5: vLLM inference (DP=6, 6 replicas)
  GPU 6-7: FSDP trainer (2 GPUs, sharded 4B model)
  CPU:     Orchestrator (asyncio event loop)
```

Config:
```toml
[deployment]
num_train_gpus = 2
num_infer_gpus = 6
```

Memory estimate (4B model, H100 80GB):
- vLLM inference (BF16): ~8-10 GB per replica -- plenty of headroom
- FSDP trainer (BF16 + AdamW states): ~40-50 GB sharded across 2 GPUs

### Multi-node (throughput scaling)

For more inference throughput (hide API latency), add inference nodes:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 2    # or up to 7
gpus_per_node = 8

[slurm]
job_name = "conductor-rl"
```

This gives 16 vLLM replicas (2 nodes x 8 GPUs) while keeping trainer on 1 node.
More replicas = more concurrent rollouts = more tolerance for OpenRouter API latency.

Multi-node does NOT increase $ worker spend (same total rollouts, just faster).

### Weight sync

| Method | Config | When to use |
|--------|--------|-------------|
| Filesystem (NFS) | Default (no config needed) | Single-node or multi-node with shared NFS |
| NCCL broadcast | `[weight_broadcast] type = "nccl"` | Multi-node with fast interconnect |

Our `/home` is NFS-shared -- filesystem sync is the safe default. NCCL is
lower-latency but requires NCCL connectivity between trainer and inference nodes.

## $50 Budget Math

### Parameters

| Parameter | Pilot (budget) | Full-scale (post-pilot) |
|-----------|----------------|------------------------|
| max_steps | 100 | 200 |
| batch_size | 128 | 256 |
| group_size (G) | 8 | 12 |
| Total rollouts | 100 x 128 x 8 = 102,400 | 200 x 256 x 12 = 614,400 |

### Cost estimate (pilot)

```
Total rollouts:          102,400
Avg subtasks/workflow:   2.5  (pilot data: 1-5 subtasks typical)
Worker calls/rollout:    2.5
Total worker calls:      256,000

Cache hit rate (est):    60%  (temp=0, 201 items, many repeated prompts)
Effective calls:         102,400

Avg tokens/call:         ~150 in + ~500 out
Avg cost/call:           ~$0.00015 (weighted: mostly DeepSeek V4 Flash @ $0.18/1M out)
                         + occasional V4 Pro ($0.87/1M) or GLM-5.2 ($3.00/1M)
Blended avg:             ~$0.0003/call

Estimated total:         102,400 * $0.0003 = ~$30.72
```

With pessimistic assumptions (lower cache rate, more expensive worker mix):
**~$30-45**, within $50 budget.

### If over budget, cut list (in priority order)

1. Reduce `max_steps` from 100 to 75 or 50
2. Reduce `batch_size` from 128 to 64
3. Filter to cheapest cluster only: `args = { clusters = ["science_mcq"] }`
4. Reduce `group_size` from 8 to 6 (reduces GRPO signal quality)

### Full-scale target (post-pilot, when budget allows)

```toml
max_steps = 200
[orchestrator]
batch_size = 256
group_size = 12
```

Estimated cost: ~$100-180 (needs expanded budget).

## Staging Plan

### Phase 3a: Accuracy-only (current config)

Reward weights: `w_corr=1.0, w_fmt=0.1, w_exec=0.1, w_lat=0.0, w_cost=0.0`

Train until accuracy (s_correct) stabilizes. Monitor format compliance (f_fmt)
-- it should rise to >0.8 within 20-30 steps.

### Phase 3b: Add efficiency (after accuracy stabilizes)

Update `configs/default.yaml` in the conductor-workflow environment:

```yaml
reward:
  w_lat: 0.05    # start small
  w_cost: 0.05
```

Re-push the env: `prime env push` and restart training (or resume from checkpoint).

This stages the efficiency signal so the model first learns to solve tasks
correctly, then learns to do so efficiently (avoid reward hacking where the
model optimizes for cheap but wrong).

## Troubleshooting

### "OPENROUTER_API_KEY not set" warning

The environment warns at construction but does not crash. Rollouts will fail
when workers are actually called. Ensure `.env` has the key and it's sourced.

### vLLM OOM

Reduce `inference.parallel.dp` from 6 to 4 and increase `num_train_gpus` to 4.
Or enable FP8 quantization for inference (not yet configured).

### Stale rollouts (rising mismatch_kl)

The 4B model updates fast -- staleness should be minimal. If `mismatch_kl`
rises, reduce `orchestrator.max_off_policy_steps` (default 8) to 4.

### Slurm job fails immediately

1. Check `sinfo` -- nodes may be `down~` (need provisioning)
2. Check `.env` is at the prime-rl project root
3. Check `uv sync --all-extras` was run in the prime-rl venv
4. Run `--dry-run` and inspect the generated sbatch script

### Zero-advantage filter drops all groups

If all G rollouts get the same reward (all fail format or all get same score),
the zero_advantage filter drops them. This is expected early in training when
the model hasn't learned the workflow format. If it persists past step 20,
check that the environment is working correctly.

## Files Reference

| File | Purpose |
|------|---------|
| `configs/rl.toml` | Core GRPO config (model, deployment, orchestrator, trainer) |
| `configs/slurm.toml` | Slurm overlay (job_name, output_dir) |
| `configs/smoke.toml` | Smoke test override (2 steps, minimal cost) |
| `configs/secrets.env.example` | Template for secrets (.env) |
| `scripts/train.sh` | Launch script with preflight/smoke/train commands |
| `docs/phase3-training.md` | This runbook |

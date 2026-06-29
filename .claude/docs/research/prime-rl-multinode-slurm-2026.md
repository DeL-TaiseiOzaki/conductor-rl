# prime-rl Multi-Node Slurm Training Research

> Research date: 2026-06-29
> prime-rl version: v0.6.0 (released 2024-06-23; repo moves fast -- pin to a commit)
> Python: ~=3.12.0 | torch >=2.9.0 | vllm >=0.23.0 | transformers 5.6.2

---

## Summary (5 bullets)

1. **Three-process architecture**: trainer (FSDP2, weight updates), inference (vLLM, rollout generation), orchestrator (lightweight CPU, data plane + weight relay). Fully disaggregated -- each is a separate executable launched by the `uv run rl` entrypoint.
2. **Environment referenced by Hub id**: `[[orchestrator.train.env]] id = "your-env-id"` with `args = { key = "value" }`. Environment must be installed via `prime env install <id>` into the venv before launch. The orchestrator calls `load_environment(**args)` at runtime.
3. **Weight sync**: Two backends -- filesystem (default: trainer saves safetensors checkpoint to shared dir, orchestrator detects it, calls `/update_weights` on vLLM) or NCCL broadcast (`[weight_broadcast] type = "nccl"` for direct GPU-to-GPU transfer). LoRA mode transfers only adapter params (sub-millisecond). Off-policy staleness bounded by `orchestrator.max_off_policy_steps` (default 8).
4. **GPU topology for 4B model on 8xH100**: Recommended single-node split is 6 inference GPUs + 2 trainer GPUs (matches wiki_search example for Qwen3-4B). Multi-node mainly adds throughput (more parallel rollouts). For our 8-node cluster: 1 node for training (2-8 GPUs FSDP) + 1-7 nodes for inference is realistic.
5. **Slurm launch**: prime-rl auto-generates sbatch scripts from TOML config. Add `[slurm] job_name = "..."` and `[deployment] type = "multi_node"` blocks. Launch with `uv run rl @ rl.toml @ slurm.toml`. The `--dry-run` flag outputs the generated sbatch without submitting.

---

## Architecture (trainer / inference / orchestrator)

### Three Cooperating Processes

```
                    +-----------------+
                    |  Orchestrator   |  (CPU-only, lightweight)
                    |  - Drives rollouts via renderers
                    |  - Computes advantages
                    |  - Packs rollouts into training batches
                    |  - Relays weight updates trainer -> inference
                    +---------+-------+
                              |
              +---------------+----------------+
              |                                |
     +--------v--------+            +----------v---------+
     | Inference (vLLM) |            |  Trainer (FSDP2)   |
     | - /v1/generate   |            |  - Consumes packed  |
     | - /update_weights|            |    rollout batches  |
     | - /reload_weights|            |  - Steps optimizer  |
     | - FP8 quantized  |            |  - Saves checkpoints|
     +---------+--------+            +----------+---------+
               |                                |
          GPU pool 1                       GPU pool 2
    (rollout generation)              (gradient computation)
```

### Roles

| Component | What it does | Process type | GPU usage |
|-----------|-------------|--------------|-----------|
| **Inference** | vLLM server with OpenAI-compatible API. Serves `/v1/generate` for rollout token generation. Custom endpoints: `/update_weights` (load new policy), `/reload_weights` (reset to base), `/init_broadcaster` (init NCCL group). | Persistent HTTP server | Dedicated GPU(s), supports TP + DP |
| **Orchestrator** | CPU process. Owns the data plane. Drives multi-turn rollouts without re-tokenization. Computes per-group advantages. Packs rollouts into training batches. Relays weight checkpoints from trainer to inference. Manages train + eval environments. | asyncio event loop | CPU only |
| **Trainer** | FSDP2 process group (one process per GPU via torchrun). Consumes packed rollout batches, computes GRPO loss, steps optimizer. Saves DCP checkpoints. Supports EP, CP, FP8, LoRA, compile. | torchrun distributed | Dedicated GPU(s) |

### Communication

| Link | Default transport | Alternative |
|------|------------------|-------------|
| Trainer <-> Orchestrator (rollouts) | Local filesystem | ZMQ (for multi-host without shared FS) |
| Trainer -> Inference (weights) | Filesystem safetensors + HTTP `/update_weights` | NCCL broadcast |
| Orchestrator -> Inference (rollout requests) | HTTP (OpenAI-compatible API) | -- |

### Key insight: fully async pipeline
- Inference generates rollouts using policy version N-1
- Trainer trains on those rollouts to produce policy N
- Weight transfer overlaps with next round of inference
- Result: near-zero GPU idle time

---

## Referencing the Hub Environment

### Environment lifecycle

1. **Build** your env with `verifiers` library:
   ```
   environments/my_env/
   |- my_env.py        # contains load_environment() -> vf.Environment
   |- pyproject.toml   # deps including verifiers>=0.1.8
   |- README.md
   ```

2. **Test locally**:
   ```bash
   prime env install my-env                    # from local directory
   uv run vf-eval my-env -n 20 --max-tokens 512
   ```

3. **Push to Hub**:
   ```bash
   prime login
   prime env push
   ```

4. **Install in prime-rl venv before training**:
   ```bash
   prime env install primeintellect/my-env     # from Hub
   uv run python -c "import my_env"            # verify importable
   ```

5. **Reference in TOML config**:
   ```toml
   [[orchestrator.train.env]]
   id = "primeintellect/my-env"          # Hub identifier or local module name
   name = "my-env"                        # display name (must be unique if same id used twice)
   ratio = 1.0                            # sampling weight (for multi-env)
   args = { worker_url = "https://...", dataset_path = "path/to/data", concurrency = 256 }
   ```

### The `load_environment()` function

Every verifiers environment module exports `load_environment(**kwargs) -> vf.Environment`.
The `args` dict from the TOML config is unpacked as kwargs to this function.
For our case with async HTTP reward scoring, the environment should use `httpx.AsyncClient` (never sync `requests`).

### Eval environments

```toml
[[orchestrator.eval.env]]
id = "primeintellect/my-env"
name = "my-env-eval"
args = { dataset_name = "eval_split", concurrency = 64 }
```

### Multi-environment training (EnvGroup)

Multiple `[[orchestrator.train.env]]` entries with `ratio` fields create weighted sampling across environments. An injected `info['env_id']` routes rollout and scoring logic.

---

## Config Files (annotated minimal example)

### Single unified `rl.toml` for our Qwen3.5-4B GRPO setup

```toml
# === Global ===
max_steps = 200
seq_len = 4096                          # max sequence length (prompt + completion)

# === Model ===
[model]
name = "Qwen/Qwen3.5-4B"               # HF model id

# === Deployment (single-node) ===
[deployment]
num_train_gpus = 2                      # GPUs for FSDP trainer
num_infer_gpus = 6                      # GPUs for vLLM inference

# === Deployment (multi-node -- use instead of above) ===
# [deployment]
# type = "multi_node"
# num_train_nodes = 1                   # nodes for FSDP trainer
# num_infer_nodes = 2                   # nodes for vLLM inference
# gpus_per_node = 8
# nodes_per_fsdp_group = 1             # how many nodes per FSDP group

# === Weight broadcast (multi-node) ===
# [weight_broadcast]
# type = "nccl"                         # "nccl" for direct GPU broadcast; omit for filesystem default

# === Slurm (multi-node) ===
# [slurm]
# job_name = "conductor-rl"
# partition = "megatpa"                  # optional, depends on cluster
# template_path = "path/to/custom.sh.j2"  # optional Jinja2 template

# === Logging ===
[wandb]
project = "conductor-rl"
name = "qwen3.5-4b-grpo"

# === Trainer ===
[trainer.optim]
lr = 1e-6                               # learning rate
weight_decay = 0.01

# [trainer.optim]                        # Alternative: Muon optimizer
# type = "muon"
# lr = 1e-6

# === Orchestrator ===
[orchestrator]
batch_size = 256                         # tasks (prompts) per trainer step (>= 64 recommended)
group_size = 12                          # rollouts per task (>= 8 recommended)
oversampling_factor = 1.0                # generate extra rollouts to compensate filtering
# max_off_policy_steps = 8              # staleness bound (default 8)

[orchestrator.algo]
# type = "grpo"                          # default, no need to set
# kl_tau = 0.0                           # set to 0 to disable KL (our spec: no KL)
# adv_tau = 1.0                          # advantage temperature
# length_penalty = "None"                # or "tokens" / "turns"

[orchestrator.train.sampling]
max_completion_tokens = 2048             # max tokens per completion

[[orchestrator.train.env]]
id = "your-org/your-env-id"             # Hub environment id
name = "conductor-env"
args = { openrouter_api_key_env = "OPENROUTER_API_KEY", max_concurrent_requests = 64 }

[[orchestrator.pre_batch_filters]]
type = "zero_advantage"                 # drop groups where all rewards are identical
enforce = true

# === Eval ===
[orchestrator.eval]
interval = 25                            # eval every N steps

[[orchestrator.eval.env]]
id = "your-org/your-env-id"
name = "conductor-env-eval"
args = { split = "test", num_examples = 50 }

# === Inference ===
[inference]
# model settings auto-inherited from [model]

[inference.parallel]
# tp = 1                                 # tensor parallelism (must fit in one node)
# dp = 6                                 # data parallelism (number of vLLM replicas)

# === Checkpointing ===
[ckpt]
interval = 50                            # save every N steps
# keep_last = 3                          # retain last N checkpoints
# resume_step = -1                       # resume from latest

# === Orchestrator Renderer ===
[orchestrator.renderer]
name = "default"
```

### Key config fields reference

| Field | Section | Description | Default |
|-------|---------|-------------|---------|
| `max_steps` | top-level | Total training steps | -- |
| `seq_len` | top-level | Max sequence length (prompt + completion) | -- |
| `model.name` | `[model]` | HF model identifier | -- |
| `deployment.num_train_gpus` | `[deployment]` | Trainer GPU count (single-node) | 1 |
| `deployment.num_infer_gpus` | `[deployment]` | Inference GPU count (single-node) | 1 |
| `deployment.type` | `[deployment]` | `"multi_node"` for Slurm | single-node |
| `deployment.num_train_nodes` | `[deployment]` | Trainer node count (multi-node) | -- |
| `deployment.num_infer_nodes` | `[deployment]` | Inference node count (multi-node) | -- |
| `deployment.gpus_per_node` | `[deployment]` | GPUs per node | 8 |
| `weight_broadcast.type` | `[weight_broadcast]` | `"nccl"` or filesystem (default) | filesystem |
| `orchestrator.batch_size` | `[orchestrator]` | Prompts per trainer step | -- |
| `orchestrator.group_size` | `[orchestrator]` | Rollouts per prompt | -- |
| `orchestrator.max_off_policy_steps` | `[orchestrator]` | Max policy version drift | 8 |
| `orchestrator.algo.kl_tau` | `[orchestrator.algo]` | KL regularizer weight | 1e-3 |
| `orchestrator.algo.adv_tau` | `[orchestrator.algo]` | Advantage temperature | 1.0 |
| `orchestrator.train.env[].id` | `[[orchestrator.train.env]]` | Environment Hub id | -- |
| `orchestrator.train.env[].args` | `[[orchestrator.train.env]]` | kwargs to `load_environment()` | {} |
| `orchestrator.train.env[].ratio` | `[[orchestrator.train.env]]` | Sampling weight | 1.0 |
| `trainer.optim.lr` | `[trainer.optim]` | Learning rate | -- |
| `inference.parallel.tp` | `[inference.parallel]` | Tensor parallelism | 1 |
| `slurm.job_name` | `[slurm]` | Slurm job name | -- |
| `ckpt.interval` | `[ckpt]` | Checkpoint save interval (steps) | -- |

---

## Weight Sync & Off-Policy

### Weight transfer mechanism

prime-rl supports two weight broadcast backends, configured via `[weight_broadcast]`:

#### 1. Filesystem (default)
1. Trainer completes a training step and saves model weights as safetensors to `<output_dir>/weights/step_N/`
2. Orchestrator detects the new checkpoint
3. Orchestrator sends HTTP POST to vLLM's `/update_weights` endpoint
4. vLLM loads the new weights in-place (no server restart)
5. **Requirement**: Shared filesystem (NFS) accessible by all nodes

#### 2. NCCL broadcast (`[weight_broadcast] type = "nccl"`)
1. Trainer completes a step
2. Orchestrator calls `/init_broadcaster` on vLLM to set up NCCL process group (first time)
3. Weights are broadcast directly GPU-to-GPU via NCCL collective
4. Lower latency than filesystem, but requires NCCL connectivity between trainer and inference GPUs
5. **Note**: Not compatible with LoRA mode (LoRA uses filesystem + adapter-only sync)

#### 3. Shardcast (decentralized / cross-provider)
Used in INTELLECT-2 for globally distributed training. Shards weights via CDN-style relay servers. Not needed for single-cluster Slurm deployments.

#### FP8 weight transfer (v0.6.0+)
Trainer can send weights in FP8 format to inference, reducing transfer bandwidth.

### Off-policy handling

The async pipeline means inference always generates rollouts with a policy that is >= 1 step behind the trainer.

**Three staleness controls (hybrid approach)**:

1. **Version rejection**: `orchestrator.max_off_policy_steps` (default 8) -- rollouts generated by a policy more than N steps old are discarded
2. **Importance-sampling correction**: Trust-region IS weighting (`kl_tau`) compensates for policy drift
3. **Partial rollout cancellation**: When weights update mid-rollout, stale rollout groups are cancelled; inference continues serving active requests

**Monitoring**: Watch `mismatch_kl/all/mean` metric. Rising KL signals off-policy instability.

**INTELLECT-2 finding**: "Even with asynchrony levels of up to four, reward trajectory matches synchronous baseline."

### Practical implication for our setup
With ~200 steps and group_size 12, most rollouts will be at most 1-2 steps stale. The default `max_off_policy_steps = 8` is very conservative. For a 4B model where weight updates are fast, staleness is minimal.

---

## GPU Topology Recommendation (single vs multi node for 4B)

### Single node (8xH100): SUFFICIENT and RECOMMENDED for 4B

A 4B parameter model fits comfortably in a single H100's 80GB memory. The official `wiki_search` example (Qwen3-4B) uses exactly:
- **6 GPUs for inference** (vLLM, DP=6 for throughput)
- **2 GPUs for trainer** (FSDP across 2 GPUs)

This is the recommended split for our Qwen3.5-4B setup.

**Memory estimates (4B model)**:
- vLLM inference (FP8): ~4-5GB per instance
- FSDP trainer (BF16 + optimizer): ~40-50GB sharded across 2 GPUs
- Plenty of headroom on H100 80GB

### Multi-node: for THROUGHPUT, not necessity

For a 4B model, multi-node is about:
- **More parallel rollouts** (more vLLM replicas = higher throughput)
- **Faster convergence** (larger effective batch processed per wall-clock time)
- **NOT** about fitting the model (4B fits easily on 1-2 GPUs)

### Recommended topology for our 8-node cluster

**Option A: Conservative (1 node)**
```
Node 0: 6 GPUs inference + 2 GPUs trainer + orchestrator (CPU)
Nodes 1-7: idle
```
- Simplest setup, good for initial development and debugging
- ~200 steps at batch_size=256, group_size=12 is manageable

**Option B: Throughput-optimized (2-3 nodes)**
```
Node 0: 8 GPUs trainer (FSDP) + orchestrator (CPU)
Node 1: 8 GPUs inference (vLLM DP=8)
Node 2: 8 GPUs inference (vLLM DP=8) [optional, for more throughput]
```
Config:
```toml
[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 1   # or 2
gpus_per_node = 8
```
- Use this if rollout generation is the bottleneck (likely with external API reward scoring)
- More inference replicas = more concurrent rollouts = more tolerance for API latency

**Option C: Maximum throughput (4+ nodes)**
Only needed if the OpenRouter reward API has high latency and you need massive rollout parallelism to hide it.

### Our specific situation: external API reward scoring

Since reward scoring calls external OpenRouter APIs (network-bound), the bottleneck is NOT GPU compute for rewards but rather:
1. Rollout generation speed (vLLM inference) -- GPU bound
2. Reward API latency (OpenRouter HTTP calls) -- network bound

**Recommendation**: Start with Option A (single node, 6+2 split). If `time/wait_for_batch` metric is high (orchestrator waiting), add inference nodes.

---

## Slurm Launch Recipe

### How prime-rl's Slurm integration works

1. You add `[slurm]` and `[deployment] type = "multi_node"` blocks to your TOML config
2. `uv run rl @ rl.toml` detects the Slurm config, generates a `.sbatch` script via Jinja2 template
3. The generated script is submitted with `sbatch`
4. The script positions inference replicas, orchestrator, and trainer across allocated nodes
5. Rendezvous endpoints, IPs, ports, and filesystem paths are auto-configured

### Skeleton rl.toml for multi-node

```toml
# rl.toml
max_steps = 200
seq_len = 4096
output_dir = "/shared/outputs/conductor-rl"    # REQUIRED for Slurm (shared FS path)

[model]
name = "Qwen/Qwen3.5-4B"

[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 1
gpus_per_node = 8

[weight_broadcast]
type = "nccl"

[slurm]
job_name = "conductor-rl"
# partition = "megatpa"                 # uncomment if needed
# template_path = "custom_slurm.sh.j2" # uncomment for custom template

[wandb]
project = "conductor-rl"
name = "qwen3.5-4b-grpo"

[trainer.optim]
lr = 1e-6
weight_decay = 0.01

[orchestrator]
batch_size = 256
group_size = 12

[orchestrator.algo]
kl_tau = 0.0

[orchestrator.train.sampling]
max_completion_tokens = 2048

[[orchestrator.train.env]]
id = "your-org/conductor-env"
name = "conductor"
args = { openrouter_base_url = "https://openrouter.ai/api/v1" }

[[orchestrator.pre_batch_filters]]
type = "zero_advantage"
enforce = true

[ckpt]
interval = 50

[inference]

[orchestrator.renderer]
name = "default"
```

### Launch commands

```bash
# Dry-run: generate sbatch script without submitting
uv run rl @ rl.toml --dry-run --output-dir /shared/outputs/conductor-rl

# Submit to Slurm
uv run rl @ rl.toml

# The launcher will output:
#   - Path to generated .sbatch file
#   - Slurm job ID
#   - Log file paths for each component

# Monitor logs (if using tmux helper)
bash scripts/tmux.sh conductor-rl /shared/outputs/conductor-rl
```

### Manual multi-node launch (without Slurm auto-generation)

If the auto-generated sbatch doesn't fit your cluster, you can launch components separately:

```bash
# === On inference node(s) ===
uv run inference @ rl.toml \
  --model.name Qwen/Qwen3.5-4B \
  --inference.parallel.dp 8

# === On trainer node ===
# torchrun handles multi-GPU distribution
uv run torchrun \
  --nproc-per-node 8 \
  --nnodes 1 \
  --node-rank 0 \
  --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/rl/train.py @ rl.toml

# === On orchestrator node (can co-locate with trainer) ===
uv run orchestrator @ rl.toml
```

### Skeleton sbatch script (if writing manually)

```bash
#!/bin/bash
#SBATCH --job-name=conductor-rl
#SBATCH --partition=megatpa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/shared/outputs/conductor-rl/slurm-%j.log
#SBATCH --exclusive

# --- Environment ---
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0           # adjust to your network interface
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY}"

# --- Node allocation ---
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=29500
INFER_NODE=${NODELIST[1]}
TRAIN_NODE=${NODELIST[0]}

PROJECT_DIR="/shared/prime-rl"
OUTPUT_DIR="/shared/outputs/conductor-rl"
CONFIG="${PROJECT_DIR}/rl.toml"

cd ${PROJECT_DIR}

# --- Launch inference on node 1 ---
srun --nodes=1 --ntasks=1 --nodelist=${INFER_NODE} \
  bash -c "
    cd ${PROJECT_DIR} && \
    uv run inference @ ${CONFIG}
  " &

# --- Wait for inference server to be ready ---
sleep 30

# --- Launch trainer + orchestrator on node 0 ---
srun --nodes=1 --ntasks=1 --nodelist=${TRAIN_NODE} \
  bash -c "
    cd ${PROJECT_DIR} && \
    uv run torchrun \
      --nproc-per-node 8 \
      --nnodes 1 \
      --node-rank 0 \
      --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      --local-ranks-filter 0 \
      src/prime_rl/trainer/rl/train.py @ ${CONFIG} &
    sleep 10 && \
    uv run orchestrator @ ${CONFIG}
  " &

wait
```

**NOTE**: The above manual sbatch is a SKELETON. The preferred approach is to let prime-rl generate the sbatch via `uv run rl @ rl.toml` with `[slurm]` config. The auto-generated script handles rendezvous, IP discovery, port allocation, and component coordination automatically.

---

## Gotchas

### Python / uv setup
- **Python 3.12 only** (`~=3.12.0` pinned in pyproject.toml)
- Clone with submodules: `git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config`
- Install: `uv sync --all-extras`
- Flash Attention 3 (Hopper/H100): `uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation`
- **Do not use pip directly** -- all through uv

### CUDA / vLLM version pins
- vLLM >= 0.23.0 (very recent -- check compatibility)
- torch >= 2.9.0
- CUDA 12.8 wheels (configured in uv settings)
- `uv.lock` file pins all transitive deps -- `uv sync` is reproducible

### OpenRouter HTTP calls in reward (our specific case)

**Critical: async only**. The verifiers docs explicitly warn:
> "Synchronous calls block all concurrent rollouts."

Your reward function MUST use:
- `httpx.AsyncClient` (NOT `requests`)
- `await asyncio.sleep()` (NOT `time.sleep()`)
- `aiofiles` for any file I/O

**Concurrency control**:
- `set_concurrency(n)` in your environment to control parallel rollout workers
- Pass via `args = { concurrency = 256 }` in TOML
- Register custom `ThreadPoolExecutor` via `register_executor()` for CPU-bound work

**Rate limits**: OpenRouter has per-minute rate limits. Your environment should implement:
- Exponential backoff with `tenacity` or manual retry
- Connection pooling via a shared `httpx.AsyncClient`
- Timeout handling (set reasonable timeouts, e.g. 60s per reward call)

**Latency interaction with throughput**:
- The orchestrator won't block on slow rewards -- it runs many rollouts concurrently
- However, if ALL rewards are slow, the orchestrator's `time/wait_for_batch` will increase
- Monitor this metric; if high, increase inference replicas to generate more rollouts in parallel
- The `oversampling_factor` config can help: generate extra rollouts to compensate for slow/failed ones

### Shared filesystem requirement
- Multi-node RL requires shared NFS/GPFS mounted on all nodes
- Checkpoints, rollout data, and weight files are exchanged via this shared FS
- Without shared FS, use ZMQ transport (more complex setup)

### LoRA constraints
- LoRA mode requires filesystem weight broadcast (NCCL not supported for LoRA)
- Only adapter params are synced (sub-millisecond transfers)
- Full fine-tuning uses either filesystem or NCCL

### Monitoring
- `mismatch_kl/all/mean` trending up = off-policy instability (reduce async level or increase weight sync frequency)
- `entropy` too low = mode collapse; too high = not converging
- `optim/grad_norm` spikes precede divergence
- `time/wait_for_batch` high = orchestrator bottleneck (need more inference)
- `time/wait_for_ckpt` high = trainer bottleneck

### Re-tokenization warning
> "The trainer must see the exact tokens the server sampled -- re-tokenization across turns drifts under BPE round-trip."

Use `openai_chat_completions_token` client type (recommended for production RL).

### Batch size constraints
- `batch_size >= 64` to minimize overhead dominance
- `group_size >= 8` to ensure meaningful advantage signals
- Always `pin output_dir per run` to prevent checkpoint contamination

---

## Open Questions & Version Caveats

1. **Version drift**: prime-rl v0.6.0 is the latest tagged release as of research date. The repo moves fast (1392+ PRs). Pin to a specific commit hash for reproducibility.

2. **vLLM >= 0.23.0**: This is a very recent vLLM version. Verify compatibility with your CUDA driver and H100 setup. The repo uses `uv.lock` for reproducible installs.

3. **Custom Slurm template**: If the auto-generated sbatch doesn't work for your cluster (cloud-autoscaled nodes, specific partition names), use `slurm.template_path` to provide a custom Jinja2 template. Inspect the default template first via `--dry-run`.

4. **NCCL weight broadcast + multi-node**: When trainer and inference are on different nodes, NCCL broadcast requires inter-node NCCL connectivity (InfiniBand or high-bandwidth ethernet). Verify with `NCCL_DEBUG=INFO`. Filesystem broadcast is simpler but slightly higher latency.

5. **Teacher inference**: Not supported in multi-node deployment (per release notes). Not relevant for GRPO without KL reference model.

6. **Environment hot-reload**: Unclear whether environment code changes require restart or if they're picked up dynamically. Assume restart is needed.

7. **Cloud-autoscaled Slurm nodes**: Your nodes are `idle~` (cloud-autoscaled). Verify that the nodes are fully provisioned (GPUs visible, NFS mounted, NCCL working) before training starts. Consider adding a pre-check step in your sbatch script.

8. **Orchestrator co-location**: In multi-node mode, the orchestrator typically runs on the same node as the trainer (or on a separate CPU-only allocation). The docs say "trainer and orchestrator co-located for checkpoint consistency."

9. **Max concurrent environments**: With external API calls, you may hit OpenRouter's rate limits before GPU throughput limits. Size `concurrency` in env args carefully.

---

## Sources

- [prime-rl GitHub README](https://github.com/PrimeIntellect-ai/prime-rl)
- [prime-rl docs: Overview](https://docs.primeintellect.ai/prime-rl/overview)
- [prime-rl docs: Configuration](https://docs.primeintellect.ai/prime-rl/configuration)
- [prime-rl docs: Training](https://docs.primeintellect.ai/prime-rl/training)
- [prime-rl docs: Scaling](https://docs.primeintellect.ai/prime-rl/scaling)
- [prime-rl docs: Algorithms](https://docs.primeintellect.ai/prime-rl/algorithms)
- [prime-rl v0.6.0 Release Notes](https://github.com/PrimeIntellect-ai/prime-rl/releases)
- [prime-rl wiki_search example](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/wiki_search/README.md)
- [prime-rl wiki_search rl.toml](https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/examples/wiki_search/rl.toml)
- [prime-rl reverse_text rl.toml](https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/examples/reverse_text/rl.toml)
- [prime-rl INTELLECT-3.1 rl.toml](https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/examples/Intellect-3.1/rl.toml)
- [prime-rl INTELLECT-3.1 README](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/Intellect-3.1/README.md)
- [verifiers environments docs](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/environments.md)
- [verifiers training docs](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/training.md)
- [INTELLECT-2 paper (arxiv)](https://arxiv.org/html/2505.07291v1)
- [Anatomy of RL Frameworks (HuggingFace blog)](https://huggingface.co/blog/async-rl-training-landscape)
- [RL at 1T Scale blog](https://www.primeintellect.ai/blog/rl-at-1t-scale)
- [Environments Hub blog](https://www.primeintellect.ai/blog/environments)
- [prime-rl pyproject.toml](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/pyproject.toml)
- [Slurm Orchestration docs](https://docs.primeintellect.ai/tutorials-multi-node-cluster/slurm-orchestration)
- [prime CLI GitHub](https://github.com/PrimeIntellect-ai/prime)

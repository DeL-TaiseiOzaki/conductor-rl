# Prime Intellect Environment Authoring Contract (June 2026)

> Grounding research for implementing the Conductor-RL environment.
> All information verified against official sources as of 2026-06-29.
> **verifiers v0.1.14** (released 2026-05-07) is current stable.

---

## Summary

1. **Base class**: `SingleTurnEnv` is the correct fit for Conductor-RL — one model generation (the workflow JSON), then heavy async reward computation. `SingleTurnEnv` is internally `MultiTurnEnv(max_turns=1)`.
2. **Reward functions are `async def`** natively. The rubric calls them with `asyncio.gather`. External HTTP calls (OpenRouter workers) MUST use `httpx.AsyncClient` or `AsyncOpenAI` — sync calls block the entire event loop at scale.
3. **Parser**: Subclass `vf.Parser`, override `parse(text) -> Any` and `parse_answer(completion) -> str|None`. Override `get_format_reward_func()` to return a graded format reward (not the default 1.0 no-op).
4. **Package layout**: `prime env init <name>` scaffolds `<name>/` with `<name>.py` + `pyproject.toml` + `README.md`. The module exposes `def load_environment(**kwargs) -> vf.Environment`. Build system = hatchling.
5. **Publish**: `prime login` then `prime env push` (or `--auto-bump` for auto-versioning). Referenced as `owner/env-name` (e.g. `DeL-TaiseiOzaki/conductor-rl`). Install: `prime env install owner/env-name`. Training config: `[[env]] id = "owner/env-name"`.

---

## verifiers Environment API

### Class Hierarchy

```
vf.Environment (abstract)
  └── MultiTurnEnv
        ├── SingleTurnEnv          ← best fit for Conductor-RL
        ├── ToolEnv
        │     └── StatefulToolEnv
        ├── MCPEnv
        ├── SandboxEnv
        │     └── PythonEnv
        └── (experimental: GymEnv, CliAgentEnv, etc.)
```

### Why SingleTurnEnv

- One model generation = one rollout. The model outputs the entire workflow JSON in a single turn.
- The heavy async work (DAG execution, worker calls, verification) happens entirely inside reward functions — NOT in `env_response()`.
- `SingleTurnEnv` = `MultiTurnEnv` with `max_turns=1` and a placeholder `env_response`.

### Constructor Signature (SingleTurnEnv)

```python
vf.SingleTurnEnv(
    dataset: Dataset | DatasetBuilder,        # HF Dataset or callable
    rubric: Rubric,                           # scoring pipeline
    system_prompt: str | None = None,         # prepended if no system msg
    eval_dataset: Dataset | DatasetBuilder | None = None,
    # parser is NOT a direct constructor arg — pass via rubric
    **kwargs
)
```

Note: `parser` is passed to `Rubric(parser=...)` constructor, NOT directly to the env.

### Dataset Format

Standard HF `datasets.Dataset`. Required/optional columns:

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `prompt` | `list[dict]` | Yes (or `question`) | `[{"role": "user", "content": "..."}]` |
| `question` | `str` | Alternative to `prompt` | Auto-wrapped in user message |
| `answer` | `str` | Optional | Ground truth for reward functions |
| `info` | `dict` or JSON str | Optional | Arbitrary metadata (verifier_spec, gold, etc.) |

If both `prompt` and `question` exist, `prompt` takes precedence.

### Minimal Skeleton for Conductor-RL

```python
import verifiers as vf
from datasets import Dataset

def load_environment(
    dataset_path: str = "path/to/conductor_tasks.jsonl",
    **kwargs,
) -> vf.Environment:
    # Validate required API keys
    vf.ensure_keys(["OPENROUTER_API_KEY"])

    # Load dataset
    dataset = Dataset.from_json(dataset_path)
    # Expected columns: prompt (list[dict]), answer (str), info (dict w/ verifier_spec)

    # Parser for ```workflow blocks
    parser = WorkflowParser()

    # Reward functions (all async)
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(format_reward, weight=W_FMT)
    rubric.add_reward_func(execution_reward, weight=W_EXEC)
    rubric.add_reward_func(correctness_reward, weight=W_CORR)
    rubric.add_reward_func(efficiency_bonus, weight=W_EFF)

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=CONDUCTOR_SYSTEM_PROMPT,
        **kwargs,
    )
```

---

## Rubric & Async Reward

### Rubric Constructor

```python
vf.Rubric(
    funcs: list[Callable] | None = None,     # reward functions
    weights: list[float] | None = None,       # per-function weights (default 1.0 each)
    parser: vf.Parser | None = None,          # auto-injected as class_object
)
```

### Adding Functions

```python
rubric.add_reward_func(func, weight=1.0)   # contributes to final reward
rubric.add_metric(func)                     # weight=0, observation-only
rubric.add_class_object("name", obj)        # injectable into reward func params
```

### Reward Function Signature (Dependency Injection)

Functions request data by **parameter name** (name-based injection):

```python
async def my_reward(
    completion: list[dict],   # model output messages
    prompt: list[dict],       # input messages
    answer: str,              # from dataset "answer" column
    info: dict,               # from dataset "info" column
    state: vf.State,          # mutable per-rollout state dict
    parser: vf.Parser,        # injected from rubric.class_objects
) -> float:
    ...
```

Available injectable names:
- `completion`, `prompt`, `answer`, `info`, `state` — standard rollout data
- `parser` — auto-injected when `Rubric(parser=...)` is used
- Any key from `rubric.add_class_object(name, obj)` — e.g. `"http_client"`, `"judge"`
- For `JudgeRubric`: `judge`, `judge_client`, `judge_model`, `judge_prompt`, `judge_sampling_args`

### ASYNC IS NATIVE

**Reward functions MUST be `async def`**. The rubric calls them with `asyncio.gather` for parallelism. This is critical for our use case:

```python
async def correctness_reward(
    completion: list[dict],
    answer: str,
    info: dict,
    parser: vf.Parser,
    state: vf.State,
) -> float:
    """Parse workflow, execute DAG via async OpenRouter calls, verify."""
    workflow = parser.parse_answer(completion)
    if workflow is None:
        return 0.0

    # Execute DAG — this does async HTTP calls to OpenRouter workers
    dag_result = await execute_workflow_dag(workflow, info)
    state["dag_result"] = dag_result  # cache for downstream reward fns

    # Verify correctness
    verifier_spec = info.get("verifier_spec", {})
    return await verify_answer(dag_result.final_output, answer, verifier_spec)
```

### Sync Call Prohibition

At 2000+ concurrent rollouts, a 10ms sync call serializes to 20+ seconds of wall-clock blocking. **NEVER** use:
- `time.sleep()` → `await asyncio.sleep()`
- `requests.get()` / `httpx.Client` → `httpx.AsyncClient`
- `OpenAI()` → `AsyncOpenAI()`
- `copy.deepcopy()` → `await asyncio.to_thread(copy.deepcopy, ...)`

### Execution Order & State Sharing

Reward functions execute **in registration order**. The mutable `state` dict is shared, enabling earlier functions to cache values for later ones:

```python
# 1st: format_reward caches parse result in state
# 2nd: execution_reward uses cached parse, caches DAG result
# 3rd: correctness_reward uses cached DAG result
# 4th: efficiency_bonus uses cached DAG result + correctness
```

### Weighted Sum

Final reward = `sum(score_i * weight_i)` for all registered functions.

### Group-Level Reward Functions

For GRPO diversity bonuses, use plural parameter names:

```python
async def diversity_bonus(completions: list[list[dict]]) -> list[float]:
    """Return per-rollout scores for the entire group."""
    responses = [c[-1]["content"] for c in completions]
    unique = set(responses)
    return [0.2 if responses.count(r) == 1 else 0.0 for r in responses]
```

---

## Parser & Partial-Credit

### Base Parser Class

```python
class Parser:
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        self.extract_fn = extract_fn

    def parse(self, text: str) -> Any:
        """Core parse logic. Override this."""
        return self.extract_fn(text)

    def parse_answer(self, completion: Messages) -> str | None:
        """Extract answer from completion messages. Uses last assistant message."""
        ...

    def get_format_reward_func(self) -> Callable:
        """Return a reward function for format adherence. Base = always 1.0."""
        def format_reward_func(completion, **kwargs) -> float:
            return 1.0
        return format_reward_func
```

### Custom WorkflowParser Skeleton

```python
import json
import re
from typing import Any
from verifiers.parsers.parser import Parser
from verifiers.types import Messages

WORKFLOW_BLOCK_RE = re.compile(
    r"```workflow\s*\n(.*?)\n\s*```", re.DOTALL
)

class WorkflowParser(Parser):
    """Parse ```workflow JSON blocks from Conductor model output."""

    def parse(self, text: str) -> dict | None:
        """Extract and validate the workflow JSON from raw text."""
        if not text:
            return None
        match = WORKFLOW_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        # Structural validation
        if not isinstance(data, dict):
            return None
        required_keys = {"subtasks", "model_id", "access_list"}
        if not required_keys.issubset(data.keys()):
            return None
        return data

    def parse_answer(self, completion: Messages) -> dict | None:
        if isinstance(completion, str):
            return self.parse(completion)
        assistant_msgs = [m for m in completion if m.get("role") == "assistant"]
        if not assistant_msgs:
            return None
        content = assistant_msgs[-1].get("content", "")
        return self.parse(content)

    def get_format_reward_func(self):
        """Graded format reward with partial credit."""
        def format_reward(completion: list[dict], **kwargs) -> float:
            text = completion[-1].get("content", "") if completion else ""
            score = 0.0
            # Has fenced block at all?
            if "```workflow" in text:
                score += 0.2
            match = WORKFLOW_BLOCK_RE.search(text)
            if not match:
                return score
            score += 0.1  # valid fencing
            try:
                data = json.loads(match.group(1))
                score += 0.2  # valid JSON
            except json.JSONDecodeError:
                return score
            if not isinstance(data, dict):
                return score
            # Has required keys?
            required = {"subtasks", "model_id", "access_list"}
            present = required.intersection(data.keys())
            score += 0.2 * (len(present) / len(required))
            # Equal-length arrays?
            arrays = [data.get(k) for k in required if isinstance(data.get(k), list)]
            if len(arrays) == 3 and len(set(len(a) for a in arrays)) == 1:
                score += 0.3
            return min(score, 1.0)
        return format_reward
```

### Built-in Parsers

| Parser | Purpose | `get_format_reward_func()` |
|--------|---------|---------------------------|
| `Parser` (base) | Pass-through | Always 1.0 (no-op) |
| `XMLParser(tags=["think","answer"])` | Extract XML-tagged fields | Graded: field presence (40%), spacing (20%), start (20%), end (20%) |
| `ThinkParser` | Separate reasoning from answer | Format adherence scoring |

### Using Parser Format Reward in Rubric

```python
parser = WorkflowParser()
rubric = vf.Rubric(parser=parser)
# Option A: use the parser's built-in format reward
rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
# Option B: write a custom format reward that receives parser via DI
rubric.add_reward_func(custom_format_reward, weight=0.2)
```

---

## Hub Package Layout

### Generated by `prime env init conductor-workflow`

```
conductor-workflow/
├── conductor_workflow.py    # Module: must expose load_environment()
├── pyproject.toml           # Metadata, deps, build config
└── README.md                # Displayed on Environments Hub
```

For multi-file: `prime env init conductor-workflow --package` creates:
```
conductor-workflow/
├── conductor_workflow/
│   ├── __init__.py          # exports load_environment
│   └── conductor_workflow.py
├── pyproject.toml
└── README.md
```

### pyproject.toml Template

```toml
[project]
name = "conductor-workflow"
description = "Conductor-RL: Multi-agent workflow orchestration environment for GRPO training"
tags = ["single-turn", "multi-agent", "workflow", "orchestration", "grpo"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.14",
    "httpx>=0.27",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["conductor_workflow.py", "pyproject.toml"]
# For multi-file: include = ["conductor_workflow/", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 20
rollouts_per_example = 5
```

**Key rules:**
- `[tool.hatch.build].include` MUST list `pyproject.toml` explicitly.
- Do NOT use `[tool.uv.sources]` for URL deps — metadata doesn't transfer to wheels.
- Use PEP 508 for git deps: `"mylib @ git+https://github.com/org/repo.git"`.
- `[tool.verifiers.eval]` provides defaults for `prime eval run` / `vf-eval`.

### Dataset Handling

Datasets can be:
1. **Inline** in `load_environment()` via `Dataset.from_list([...])` or `Dataset.from_generator(fn)`
2. **Loaded from HF Hub**: `load_dataset("my-org/conductor-tasks", split="train")`
3. **Loaded from local JSONL**: `Dataset.from_json("path/to/data.jsonl")`
4. **Lazy via DatasetBuilder**: pass a callable `() -> Dataset` instead of a `Dataset`

For Conductor-RL: likely push dataset to HF Hub, reference in `load_environment()`.

### API Key Validation

```python
def load_environment(**kwargs) -> vf.Environment:
    vf.ensure_keys(["OPENROUTER_API_KEY"])
    # Raises MissingKeyError with instructions if not set
    ...
```

Keys can be set via: env vars, `.env` file, or the Environments Hub secrets panel.

---

## Publish & Reference Flow

### 1. Authentication

```bash
uv tool install prime          # or: pip install prime
prime login                    # opens browser auth flow
# Configure username at https://app.primeintellect.ai/dashboard/profile
```

### 2. Push to Hub

```bash
cd conductor-workflow/
prime env push                 # uploads under your username
prime env push --team myteam   # uploads under team account
prime env push --auto-bump     # auto-increment version
prime env push --visibility=PRIVATE  # private environment
```

### 3. Naming & Versioning

- Hub ID format: `owner/env-name` (e.g., `DeL-TaiseiOzaki/conductor-workflow`)
- Version from `pyproject.toml` `[project].version`
- `--auto-bump` increments patch version automatically
- All previous versions preserved

### 4. Install & Use

```bash
prime env install owner/conductor-workflow           # latest
prime env install owner/conductor-workflow@0.1.0     # specific version
prime env pull owner/conductor-workflow               # download source
prime env info owner/conductor-workflow               # view details
prime env inspect owner/conductor-workflow            # inspect source without download
```

### 5. Reference in Training Config (prime-rl TOML)

```toml
model = "Qwen/Qwen3.5-4B"
max_steps = 100
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 2048

[[orchestrator.train.env]]
id = "owner/conductor-workflow"
```

Multiple environments supported via repeated `[[orchestrator.train.env]]` sections.

### 6. Reference in Python

```python
from verifiers import load_environment
env = load_environment("owner/conductor-workflow")
results = env.evaluate(examples=100, rollouts_per_example=1)
```

---

## Local Eval Loop

### Development Workflow

```bash
# 1. Scaffold
prime env init conductor-workflow
cd conductor-workflow/

# 2. Implement load_environment() in conductor_workflow.py

# 3. Install locally (editable)
uv pip install -e .

# 4. Quick test with vf-eval (uses [tool.verifiers.eval] defaults)
uv run vf-eval conductor-workflow

# 5. Full eval with prime CLI
prime eval run conductor-workflow -m openai/gpt-4.1-mini
prime eval run conductor-workflow -m openai/gpt-4.1-mini -n 50 -r 3

# 6. Custom kwargs
prime eval run conductor-workflow -m vllm/Qwen/Qwen3.5-4B \
    --extra-env-kwargs '{"dataset_path": "data/pilot.jsonl"}'

# 7. Iterate, then push
prime env push --auto-bump
```

### vf-eval vs prime eval run

| Feature | `vf-eval` | `prime eval run` |
|---------|-----------|-----------------|
| Source | verifiers package | prime CLI |
| Local model | Yes | Yes |
| API model | Yes | Yes |
| Saves results | To `outputs/evals/` | To `outputs/evals/` |
| Hub integration | No | Yes (can push results) |

### Pointing at Custom Dataset + Workers

```python
def load_environment(
    dataset_path: str = "default/path.jsonl",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    worker_models: list[str] | None = None,
    **kwargs,
) -> vf.Environment:
    vf.ensure_keys(["OPENROUTER_API_KEY"])
    # All params can be overridden via --extra-env-kwargs
    ...
```

### Concurrency Tuning

```python
env = vf.SingleTurnEnv(...)
env.set_concurrency(256)  # resize internal executors
```

Or via CLI: `--extra-env-kwargs '{"concurrency": 256}'`

---

## Open Questions & Version Caveats

### Open Questions

1. **Reward function timeout**: No documented per-function timeout. Our DAG execution could take 30-60s per rollout. Need to verify if there's a rollout-level timeout (`timeout_seconds` param exists for some env types — check if `SingleTurnEnv` supports it).

2. **State serialization**: `state` dict is passed between reward functions, but unclear if it's serialized between processes or stays in-memory. For our case (caching DAG results), in-memory is fine, but worth verifying.

3. **Rubric constructor `parser` kwarg**: The MMLU example passes `parser=parser` to `Rubric()`, and also passes `parser=parser` to `SingleTurnEnv()`. Need to verify if `SingleTurnEnv` actually accepts a `parser` kwarg (it's not in the documented constructor signature — may be passed via `**kwargs` to `MultiTurnEnv`).

4. **`score_group` advantage calculation**: The rubric auto-calculates per-state advantage relative to batch mean. This interacts with GRPO — verify that `prime-rl` uses the rubric's advantage or computes its own.

5. **v0.1.14 Taskset/Harness API**: New composable `vf.Taskset`/`vf.Harness` pattern was introduced. The older `SingleTurnEnv` pattern still works but may eventually be superseded. Monitor for deprecation.

6. **Dataset size limits on Hub**: No documented maximum. Our pilot has 201 examples, which is fine. For larger datasets, pushing to HF Hub and referencing via `load_dataset()` is recommended.

### Version Caveats

- **verifiers v0.1.14** (2026-05-07): Current stable. Introduced composable Taskset/Harness API, model-family starter configs (includes Qwen 3.5).
- **verifiers v0.1.12** (2026-04-17): Major RLMEnv overhaul. If using older version, `env_response` semantics differ.
- **Python**: Requires >=3.10, <3.14.
- **prime CLI**: Install via `uv tool install prime` (v0.5.42+ recommended based on community envs).

---

## Sources

- [PrimeIntellect-ai/verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) — main library repo
- [verifiers docs/environments.md](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/environments.md) — comprehensive environment documentation
- [verifiers docs/training.md](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/training.md) — training config format
- [Prime Intellect Docs: Create & Upload Environment](https://docs.primeintellect.ai/tutorials-environments/create) — official tutorial
- [Prime Intellect Docs: Install & Use Environment](https://docs.primeintellect.ai/tutorials-environments/install) — install flow
- [Prime Intellect Docs: Environments Overview](https://docs.primeintellect.ai/verifiers/environments) — env types API reference
- [PrimeIntellect-ai/prime GitHub](https://github.com/PrimeIntellect-ai/prime) — CLI tool
- [PrimeIntellect-ai/prime-environments](https://github.com/PrimeIntellect-ai/prime-environments) — curated community envs
- [PrimeIntellect-ai/community-environments](https://github.com/PrimeIntellect-ai/community-environments) — community envs
- [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — training framework
- [verifiers on PyPI](https://pypi.org/project/verifiers/) — v0.1.14 (2026-05-07)
- [Environments Hub](https://app.primeintellect.ai/dashboard/environments) — browse/publish environments
- [Environments Hub blog post](https://www.primeintellect.ai/blog/environments) — overview
- [MMLU env example](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/mmlu/mmlu.py) — concrete SingleTurnEnv + custom Parser
- [Building Your First Environment (Medium)](https://medium.com/@alaminibrahim433/building-your-first-prime-intellect-environment-a-complete-guide-to-creating-rl-ready-evaluation-56933249ff43) — community tutorial with XMLParser + partial-credit example

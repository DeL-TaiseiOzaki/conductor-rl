#!/usr/bin/env bash
# =============================================================================
# Conductor-RL Phase 3 Training Launch Script
# =============================================================================
# Prerequisites:
#   1. prime-rl cloned and installed (see docs/phase3-training.md)
#   2. conductor-workflow env installed into the prime-rl venv
#   3. .env file with OPENROUTER_API_KEY at the prime-rl project root
#
# This script is run FROM the prime-rl project directory (not conductor-rl).
# It references conductor-rl configs via absolute paths.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDUCTOR_DIR="${CONDUCTOR_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PRIME_RL_DIR="${PRIME_RL_DIR:-/home/prime-rl}"

RL_CONFIG="${CONDUCTOR_DIR}/configs/rl.toml"
SLURM_CONFIG="${CONDUCTOR_DIR}/configs/slurm.toml"
SMOKE_CONFIG="${CONDUCTOR_DIR}/configs/smoke.toml"
OUTPUT_DIR="${OUTPUT_DIR:-/home/outputs/conductor-rl}"

echo "=== Conductor-RL Training Launch ==="
echo "  CONDUCTOR_DIR: ${CONDUCTOR_DIR}"
echo "  PRIME_RL_DIR:  ${PRIME_RL_DIR}"
echo "  OUTPUT_DIR:    ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight() {
    echo "--- Pre-flight checks ---"

    # Check prime-rl directory
    if [ ! -f "${PRIME_RL_DIR}/pyproject.toml" ]; then
        echo "ERROR: prime-rl not found at ${PRIME_RL_DIR}"
        echo "  Clone: git clone https://github.com/PrimeIntellect-ai/prime-rl.git ${PRIME_RL_DIR}"
        exit 1
    fi

    # Check .env file exists at prime-rl root
    if [ ! -f "${PRIME_RL_DIR}/.env" ]; then
        echo "ERROR: ${PRIME_RL_DIR}/.env not found."
        echo "  Copy configs/secrets.env.example to ${PRIME_RL_DIR}/.env and fill in keys."
        exit 1
    fi

    # Check OPENROUTER_API_KEY is set (either in env or .env)
    if ! grep -q "OPENROUTER_API_KEY=.\+" "${PRIME_RL_DIR}/.env" 2>/dev/null; then
        if [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo "WARNING: OPENROUTER_API_KEY not found in .env or environment."
            echo "  Rollouts will fail without it."
        fi
    fi

    # Check configs exist
    for cfg in "${RL_CONFIG}" "${SLURM_CONFIG}" "${SMOKE_CONFIG}"; do
        if [ ! -f "${cfg}" ]; then
            echo "ERROR: Config not found: ${cfg}"
            exit 1
        fi
    done

    # Check conductor-workflow env is importable
    cd "${PRIME_RL_DIR}"
    if ! uv run python -c "import conductor_workflow" 2>/dev/null; then
        echo "WARNING: conductor_workflow not importable in prime-rl venv."
        echo "  Run: cd ${PRIME_RL_DIR} && prime env install o-taisei/conductor-workflow"
    fi

    echo "Pre-flight OK."
    echo ""
}

# ---------------------------------------------------------------------------
# Step 1: Install environment (idempotent)
# ---------------------------------------------------------------------------
install_env() {
    echo "--- Step 1: Install conductor-workflow environment ---"
    cd "${PRIME_RL_DIR}"
    prime env install o-taisei/conductor-workflow
    echo "Verifying import..."
    uv run python -c "import conductor_workflow; print('conductor_workflow imported OK')"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Dry-run (validate config, generate sbatch, do NOT submit)
# ---------------------------------------------------------------------------
dry_run() {
    echo "--- Step 2: Dry-run (config validation) ---"
    cd "${PRIME_RL_DIR}"
    uv run rl \
        @ "${RL_CONFIG}" \
        @ "${SLURM_CONFIG}" \
        --output-dir "${OUTPUT_DIR}" \
        --dry-run
    echo ""
}

# ---------------------------------------------------------------------------
# Step 3: Smoke test (2 steps, minimal cost)
# ---------------------------------------------------------------------------
smoke() {
    echo "--- Step 3: Smoke test (2 steps, ~$0.05 worker spend) ---"
    cd "${PRIME_RL_DIR}"
    uv run rl \
        @ "${RL_CONFIG}" \
        @ "${SMOKE_CONFIG}" \
        --output-dir "${OUTPUT_DIR}/smoke"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 4: Full training run (100 steps, ~$30-50 worker spend)
# ---------------------------------------------------------------------------
train() {
    echo "--- Step 4: Full training run ---"
    cd "${PRIME_RL_DIR}"
    uv run rl \
        @ "${RL_CONFIG}" \
        @ "${SLURM_CONFIG}" \
        --output-dir "${OUTPUT_DIR}"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 4b: Full training (local, no Slurm)
# ---------------------------------------------------------------------------
train_local() {
    echo "--- Step 4b: Full training (local, no Slurm) ---"
    cd "${PRIME_RL_DIR}"
    uv run rl \
        @ "${RL_CONFIG}" \
        --output-dir "${OUTPUT_DIR}"
    echo ""
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 {preflight|install_env|dry_run|smoke|train|train_local|all}"
    echo ""
    echo "  preflight    - Check prerequisites"
    echo "  install_env  - Install conductor-workflow into prime-rl venv"
    echo "  dry_run      - Validate config + generate sbatch (no submit)"
    echo "  smoke        - Run 2-step smoke test (~\$0.05)"
    echo "  train        - Full 100-step training via Slurm (~\$30-50)"
    echo "  train_local  - Full 100-step training locally (no Slurm)"
    echo "  all          - preflight -> install_env -> dry_run -> smoke"
    echo ""
    echo "Environment variables:"
    echo "  CONDUCTOR_DIR  - Path to conductor-rl repo (default: script parent)"
    echo "  PRIME_RL_DIR   - Path to prime-rl clone (default: /home/prime-rl)"
    echo "  OUTPUT_DIR     - Output directory on NFS (default: /home/outputs/conductor-rl)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-}" in
    preflight)    preflight ;;
    install_env)  install_env ;;
    dry_run)      dry_run ;;
    smoke)        smoke ;;
    train)        train ;;
    train_local)  train_local ;;
    all)          preflight && install_env && dry_run && smoke ;;
    *)            usage ;;
esac

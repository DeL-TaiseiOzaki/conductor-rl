#!/usr/bin/env bash
# Sync canonical repo-root sources into the env package's bundled assets.
# Run before `prime env push` so the Hub-shipped package is up to date.
# (The package must be self-contained; repo-root files are the human-facing source.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${REPO_ROOT}/environments/conductor_workflow/conductor_workflow/assets"

mkdir -p "${ASSETS}/pilot"
cp "${REPO_ROOT}/data/pilot/pilot.jsonl"        "${ASSETS}/pilot/pilot.jsonl"
cp "${REPO_ROOT}/data/pilot/code.jsonl"         "${ASSETS}/pilot/code.jsonl"
cp "${REPO_ROOT}/data/pilot/science_mcq.jsonl"  "${ASSETS}/pilot/science_mcq.jsonl"
cp "${REPO_ROOT}/data/pilot/hard_math.jsonl"    "${ASSETS}/pilot/hard_math.jsonl"
cp "${REPO_ROOT}/prompts/conductor_system_prompt.md" "${ASSETS}/conductor_system_prompt.md"
cp "${REPO_ROOT}/configs/default.yaml"          "${ASSETS}/default.yaml"

echo "Synced assets -> ${ASSETS}"

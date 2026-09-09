#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

mkdir -p generated

"$PYTHON" experiments/vulex/build_cap_anchor_bank.py \
  --repo-worktree inputs/linux \
  --seed 0 \
  --records inputs/records/linux_records_target_function.jsonl \
  --visibility-split inputs/splits/visibility_split.json \
  --cwe-limit 30 \
  --per-cwe-count 12 \
  --prior-weight 0.30 \
  --repo-demand-count 512 \
  --candidate-pool-size 1024 \
  --out generated/cap_bank.json

"$PYTHON" experiments/vulex/build_sss_bank.py \
  --records inputs/records/linux_records_target_function.jsonl \
  --visibility-split inputs/splits/visibility_split.json \
  --cwe-source inputs/splits/cwe_order_source.json \
  --repo-demand-source-files inputs/splits/repo_demand_source_files.json \
  --codebert-model inputs/models/codebert-base \
  --checkpoint inputs/models/stagedvulbert-msp.bin \
  --staged-source third_party/StagedVulBERT \
  --embedding-cache generated/sss_embeddings.json \
  --out generated/sss_bank.json

OPENAI_BASE_URL="${VULEX_OPENAI_API_BASE:-https://api.openai.com/v1}" \
  "$PYTHON" experiments/vulex/build_snippets.py \
  --records inputs/records/linux_records_target_function.jsonl \
  --bank generated/sss_bank.json \
  --model gpt-5.5 \
  --cache-dir generated/snippet_cache \
  --out generated/sss_snippets.json

#!/usr/bin/env bash
# Orchestrates the standard cross-model benchmark matrix.
#
# For each model we run a sweep of measured context lengths using the same
# codec hyperparameters, then project to 16k-128k via kakeya_extrapolate.py.
#
# The three "measured" contexts (2k, 4k, 8k) are enough to pin the
# per-block skeleton cost and per-token encoded cost. The 16k-128k numbers
# are then exact (byte-accurate) projections, validated against real 16k/32k
# measurements on gemma-4-E2B-it in a previous run.
#
# Usage:
#   ./run_all_benchmarks.sh <model_path> <short_name>
#
# e.g. ./run_all_benchmarks.sh models/Qwen2.5-0.5B-Instruct qwen2_5_0_5b
set -euo pipefail

MODEL_PATH="$1"
SHORT="$2"
ATTN="${ATTN:-eager}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "reports/${SHORT}"

# Codec hyperparameters -- the "standard" used for the Gemma 4 reference run.
BLOCK=512
RESID=256
DRES=8
K=16
VAR=0.95

for CTX in 2048 4096 8192; do
  echo "==== [$SHORT] context=$CTX ===="
  python3 kakeya_benchmark.py \
    --model-path "$MODEL_PATH" \
    --model-name "$SHORT" \
    --context-tokens "$CTX" \
    --new-tokens "$([ "$CTX" -le 4096 ] && echo 8 || echo 4)" \
    --block-size "$BLOCK" \
    --residual-length "$RESID" \
    --d-res "$DRES" \
    --k-segments "$K" \
    --variance-ratio "$VAR" \
    --attn "$ATTN" \
    --dtype "$DTYPE" \
    --skip-baseline-prefill \
    --prefill-chunk 2048 \
    --report "reports/${SHORT}/bench_${CTX}.json"
done

# 16k / 32k / 64k / 128k / 256k projections from the 8k report.
python3 kakeya_extrapolate.py \
  --report "reports/${SHORT}/bench_8192.json" \
  --targets 16384,32768,65536,131072,262144 \
  --out "reports/${SHORT}/extrapolation.json"

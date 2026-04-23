#!/usr/bin/env bash
# Build the v1.3 Rust codec and run the vLLM-based PPL harness on
# Qwen2.5-0.5B with the smoke-sized config used by the HF harness so
# the two numbers can be compared directly.
#
# Requires: cargo in PATH; a Python environment with vllm, datasets,
# transformers, torch (CUDA). See benchmarks/README or the PR body.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
MODEL_NAME="${MODEL_NAME:-qwen2_5_0_5b}"
CTX_LEN="${CTX_LEN:-1024}"
N_EVAL="${N_EVAL:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BIT_WIDTH="${BIT_WIDTH:-2}"
N_PASSAGES="${N_PASSAGES:-2}"
VR="${VARIANCE_RATIO:-0.95}"
PCA_METHOD="${PCA_METHOD:-randomized}"
OUT_DIR="${OUT_DIR:-reports/v1_3_rsvd_rope/e2e_ppl_vllm_smoke}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

echo "[build] kakeyaturbo-bench (release)"
(cd kakeyaturbo && cargo build --release --bin kakeyaturbo-bench 1>&2)

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /venv/main/bin/python ] && [ "${PYTHON_BIN}" = "python3" ]; then
    PYTHON_BIN=/venv/main/bin/python
fi

echo "[run] e2e_ppl_validation_vllm.py (using $PYTHON_BIN)"
"$PYTHON_BIN" benchmarks/e2e_ppl_validation_vllm.py \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --ctx-len "$CTX_LEN" \
  --n-eval "$N_EVAL" \
  --block-size "$BLOCK_SIZE" \
  --bit-width "$BIT_WIDTH" \
  --variance-ratio "$VR" \
  --pca-method "$PCA_METHOD" \
  --n-passages "$N_PASSAGES" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --out-dir "$OUT_DIR"

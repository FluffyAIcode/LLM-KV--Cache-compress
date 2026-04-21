#!/usr/bin/env bash
# Re-fit Σ_q Cholesky + Lloyd-Max centroids on vLLM prefill snapshots
# (disjoint calibration split) and drop the products next to the
# ablation outputs so they can be swapped into the codec-pre_qp cell.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
N_PASSAGES="${N_PASSAGES:-8}"
CTX_LEN="${CTX_LEN:-2048}"
WIKI_SPLIT="${WIKI_SPLIT:-train}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
OUT_DIR="${OUT_DIR:-reports/v1_3_ppl/vllm_recalibrated}"
RIDGE="${RIDGE:-1e-3}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /venv/main/bin/python ] && [ "${PYTHON_BIN}" = "python3" ]; then
    PYTHON_BIN=/venv/main/bin/python
fi

echo "[run] vllm_calibration_refit.py (using $PYTHON_BIN)"
"$PYTHON_BIN" benchmarks/vllm_calibration_refit.py \
    --model-path "$MODEL_PATH" --out-dir "$OUT_DIR" \
    --n-passages "$N_PASSAGES" --ctx-len "$CTX_LEN" \
    --wikitext-split "$WIKI_SPLIT" --ridge "$RIDGE" \
    --k-bits 2 3 --v-bits 2 \
    --gpu-mem-util "$GPU_MEM_UTIL"

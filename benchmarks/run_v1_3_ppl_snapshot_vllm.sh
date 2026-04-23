#!/usr/bin/env bash
# Run the v1.3 PPL production cell on vLLM with HF's two-pass
# snapshot semantics (Scenario A: compress the KV cache after a
# clean prefill).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_NAME="${MODEL_NAME:-ds_distill_qwen_1_5b_snapshot}"
CTX_LEN="${CTX_LEN:-2048}"
N_EVAL="${N_EVAL:-64}"
N_PASSAGES="${N_PASSAGES:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BIT_WIDTH_K="${BIT_WIDTH_K:-3}"
BIT_WIDTH_V="${BIT_WIDTH_V:-2}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-2.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
OUT_DIR="${OUT_DIR:-reports/v1_3_ppl/snapshot_mode}"

Q_CALIB="${Q_CALIB:-reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors}"
K_CENTROIDS="${K_CENTROIDS:-reports/v1_4_q_pca/calibrated_codebook/ds_K_b${BIT_WIDTH_K}_centroids.f32}"
V_CENTROIDS="${V_CENTROIDS:-reports/v1_4_q_pca/calibrated_codebook/ds_V_b${BIT_WIDTH_V}_centroids.f32}"
BOUNDARY_LAYERS="${BOUNDARY_LAYERS:-0 1 7 14 26 27}"

echo "[build] kakeyaturbo-bench (release)"
(cd kakeyaturbo && cargo build --release --bin kakeyaturbo-bench 1>&2)

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /venv/main/bin/python ] && [ "${PYTHON_BIN}" = "python3" ]; then
    PYTHON_BIN=/venv/main/bin/python
fi

echo "[run] e2e_ppl_validation_vllm_snapshot.py"
"$PYTHON_BIN" benchmarks/e2e_ppl_validation_vllm_snapshot.py \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --ctx-len "$CTX_LEN" --n-eval "$N_EVAL" \
    --block-size "$BLOCK_SIZE" \
    --bit-width-k "$BIT_WIDTH_K" \
    --bit-width-v "$BIT_WIDTH_V" \
    --q-calib "$Q_CALIB" \
    --k-centroids "$K_CENTROIDS" \
    --v-centroids "$V_CENTROIDS" \
    --outlier-threshold "$OUTLIER_THRESHOLD" \
    --boundary-skip-layers $BOUNDARY_LAYERS \
    --n-passages "$N_PASSAGES" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --out-dir "$OUT_DIR"

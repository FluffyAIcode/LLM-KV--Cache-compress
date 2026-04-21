#!/usr/bin/env bash
# Ablation sweep for the HF vs vLLM gap.
# Runs these cells against a shared reference, pair-wise per passage:
#   identity-pre_qp  — whiten→identity codec→unwhiten  (tests noise-only)
#   codec-no_qp      — codec, no whitening
#   codec-pre_qp     — production recipe (pre-RoPE Q-precond)
#   codec-post_qp    — codec + post-RoPE Σ_q_post (self-calibrated online)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_NAME="${MODEL_NAME:-ds_distill_qwen_1_5b}"
CTX_LEN="${CTX_LEN:-2048}"
N_EVAL="${N_EVAL:-64}"
N_PASSAGES="${N_PASSAGES:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BIT_WIDTH_K="${BIT_WIDTH_K:-3}"
BIT_WIDTH_V="${BIT_WIDTH_V:-2}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-2.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
POST_ROPE_CALIB="${POST_ROPE_CALIB:-2}"
OUT_DIR="${OUT_DIR:-reports/v1_3_ppl/vllm_ablation}"

Q_CALIB_PRE="${Q_CALIB_PRE:-reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors}"
K_CENTROIDS="${K_CENTROIDS:-reports/v1_4_q_pca/calibrated_codebook/ds_K_b${BIT_WIDTH_K}_centroids.f32}"
V_CENTROIDS="${V_CENTROIDS:-reports/v1_4_q_pca/calibrated_codebook/ds_V_b${BIT_WIDTH_V}_centroids.f32}"
BOUNDARY_LAYERS="${BOUNDARY_LAYERS:-0 1 7 14 26 27}"

echo "[build] kakeyaturbo-bench (release)"
(cd kakeyaturbo && cargo build --release --bin kakeyaturbo-bench 1>&2)

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /venv/main/bin/python ] && [ "${PYTHON_BIN}" = "python3" ]; then
    PYTHON_BIN=/venv/main/bin/python
fi

echo "[run] e2e_ppl_validation_vllm_ablation.py (using $PYTHON_BIN)"
"$PYTHON_BIN" benchmarks/e2e_ppl_validation_vllm_ablation.py \
    --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
    --ctx-len "$CTX_LEN" --n-eval "$N_EVAL" \
    --block-size "$BLOCK_SIZE" \
    --bit-width-k "$BIT_WIDTH_K" --bit-width-v "$BIT_WIDTH_V" \
    --q-calib-pre-rope "$Q_CALIB_PRE" \
    --k-centroids "$K_CENTROIDS" --v-centroids "$V_CENTROIDS" \
    --outlier-threshold "$OUTLIER_THRESHOLD" \
    --boundary-skip-layers $BOUNDARY_LAYERS \
    --post-rope-qp-calib-passages "$POST_ROPE_CALIB" \
    --n-passages "$N_PASSAGES" --gpu-mem-util "$GPU_MEM_UTIL" \
    --out-dir "$OUT_DIR"

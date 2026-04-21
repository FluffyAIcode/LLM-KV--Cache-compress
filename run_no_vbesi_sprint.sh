#!/bin/bash
# Sprint: drop V Besi, keep V on kakeyaturbo RSVD (v1.3 original V path).
# Isolation: does V Besi contribute to Delta ppl, or is it purely cost?
# User's hypothesis: V Besi adds ratio cost without PPL benefit -- V RSVD
# is more byte-efficient and just as stable.
#
# Four cells:
#   NB3   = B3 recipe, V RSVD b=3 (drop V Besi)
#   NR3   = R3 recipe, V RSVD b=3 (drop V Besi)   [outlier T=1.5]
#   NB3v2 = NB3 with V bit width = 2 (V tolerates aggressive)
#   NR3v2 = NR3 with V bit width = 2
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/.hf
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODEL_PATH=/workspace/.hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
OUT=reports/v1_3_v_rsvd_noBesi
mkdir -p "$OUT"

COMMON=(
  --model-path "$MODEL_PATH"
  --ctx-len 2048 --n-eval 64 --n-passages 4 --block-size 512
  --bit-width 3 --pca-method randomized --variance-ratio 1.0 --rsvd-rank-factor 0.5
  --skeleton-dtype fp16 --compress kv
  --boundary-skip-layers 0 1 7 14 26 27 --boundary-mode bf16
  --q-precondition reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors
  --k-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32
)

echo "=== Cell NB3: B3 minus V Besi (K b=3 T=2.0, V RSVD b=3) ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" \
  --k-outlier-threshold 2.0 \
  --model-name NB3_noVBesi_T20 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/NB3_run.log"

echo "=== Cell NR3: R3 minus V Besi (K b=3 T=1.5, V RSVD b=3) ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" \
  --k-outlier-threshold 1.5 \
  --model-name NR3_noVBesi_T15 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/NR3_run.log"

echo "=== Cell NB3v2: NB3 with V bit-width = 2 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" \
  --k-outlier-threshold 2.0 \
  --bit-width-v 2 \
  --v-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \
  --model-name NB3v2_noVBesi_T20_Vb2 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/NB3v2_run.log"

echo "=== Cell NR3v2: NR3 with V bit-width = 2 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" \
  --k-outlier-threshold 1.5 \
  --bit-width-v 2 \
  --v-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \
  --model-name NR3v2_noVBesi_T15_Vb2 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/NR3v2_run.log"

echo "=== All 4 cells complete ==="
ls -la "$OUT"

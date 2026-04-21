#!/bin/bash
# Sprint part 2: V RSVD with --share-basis-v (layer-shared basis).
# This is the original v1.3 default for V; amortizing basis cost
# across all ctx tokens makes V RSVD far more byte-efficient than
# V Besi (which always pays per-vector cost).
#
# Four cells:
#   NB3-sv3   = NB3 + share-basis-v (V RSVD b=3 shared)
#   NR3-sv3   = NR3 + share-basis-v (V RSVD b=3 shared) 
#   NB3-sv2   = NB3 + share-basis-v + V bit=2 (aggressive V)
#   NR3-sv2   = NR3 + share-basis-v + V bit=2 (aggressive V)
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
  --share-basis-v
  --boundary-skip-layers 0 1 7 14 26 27 --boundary-mode bf16
  --q-precondition reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors
  --k-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_K_b3_centroids.f32
)

echo "=== Cell NB3sv3: V RSVD b=3 shared-basis + outlier T=2.0 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" --k-outlier-threshold 2.0 \
  --model-name NB3sv3_noVBesi_T20 --out-dir "$OUT" 2>&1 | tee "$OUT/NB3sv3_run.log"

echo "=== Cell NR3sv3: V RSVD b=3 shared-basis + outlier T=1.5 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" --k-outlier-threshold 1.5 \
  --model-name NR3sv3_noVBesi_T15 --out-dir "$OUT" 2>&1 | tee "$OUT/NR3sv3_run.log"

echo "=== Cell NB3sv2: V RSVD b=2 shared-basis + outlier T=2.0 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" --k-outlier-threshold 2.0 --bit-width-v 2 \
  --v-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \
  --model-name NB3sv2_noVBesi_T20_Vb2 --out-dir "$OUT" 2>&1 | tee "$OUT/NB3sv2_run.log"

echo "=== Cell NR3sv2: V RSVD b=2 shared-basis + outlier T=1.5 ==="
time python3 benchmarks/e2e_ppl_pre_rope.py \
  "${COMMON[@]}" --k-outlier-threshold 1.5 --bit-width-v 2 \
  --v-centroids-file reports/v1_4_q_pca/calibrated_codebook/ds_V_b2_centroids.f32 \
  --model-name NR3sv2_noVBesi_T15_Vb2 --out-dir "$OUT" 2>&1 | tee "$OUT/NR3sv2_run.log"

echo "=== All 4 share-basis-v cells complete ==="

#!/usr/bin/env bash
# v1.3 real benchmark matrix: bit_width=2 × {exact, randomized r=D/2} on
# every full-attn model in the 7-model corpus, at ctx=4096. The exact
# column re-uses ctx=4096 results already checked into
# reports/real_kakeyaturbo_bit_width_sweep/bw2/.
#
# No mock, no fallback -- the Rust release binary does the real PCA +
# K-means + WHT + Lloyd-Max work on real HF bf16 KV tensors.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="reports/v1_3_rsvd_rope/bench"
mkdir -p "$OUT"

# (model_path, short_name, head_dim, prefill_chunk)
declare -a MODELS=(
  "models/Qwen2.5-0.5B-Instruct qwen2_5_0_5b 64 0"
  "models/Qwen3-0.6B qwen3_0_6b 128 0"
  "models/gemma-4-E2B-it gemma4_e2b 512 0"
  "models/DeepSeek-R1-Distill-Qwen-1.5B deepseek_r1_distill_qwen_1_5b 128 0"
  "models/glm-edge-1.5b-chat glm_edge_1_5b 128 0"
  "models/SmolLM2-1.7B-Instruct smollm2_1_7b 64 1024"
  "models/glm-edge-4b-chat glm_edge_4b 128 1024"
)

CTX=4096
for entry in "${MODELS[@]}"; do
  read -r mp short hd chunk <<<"$entry"
  tgt=$(( hd / 2 ))
  out_dir="$OUT/${short}_rsvd_half/ctx_${CTX}"
  mkdir -p "$out_dir"
  echo "=== $short (D=${hd}, r=${tgt}) rsvd b=2 ==="
  python3 benchmarks/kakeyaturbo_v1_2_real_bench.py \
    --model-path "$mp" --model-name "$short" \
    --context-tokens "$CTX" --bit-width 2 \
    --pca-method randomized \
    --rsvd-target-rank "$tgt" \
    --rsvd-oversample 8 \
    --rsvd-power-iters 2 \
    --prefill-chunk "$chunk" \
    --out-dir "$out_dir" 2>&1 | tail -6
done

echo
echo "=== done, reports under $OUT ==="

#!/usr/bin/env bash
# Real v1.2 bit_width sweep at ctx=4096 over all 7 open-source models.
# Same monomorphic MSE codec as run_v1_2_full_matrix.sh; only bit_width varies.
#
# No mock, no fallback, no simplification -- the Rust release binary
# does real PCA + K-means + WHT + Lloyd-Max on real HF bf16 KV tensors.

set -euo pipefail

cd "$(dirname "$0")/.."

BW_LIST=("${@:-2 3 4}")
OUT="reports/real_kakeyaturbo_bit_width_sweep"
mkdir -p "$OUT"

# (model_path, short_name, prefill_chunk)
declare -a MODELS=(
  "models/Qwen2.5-0.5B-Instruct qwen2_5_0_5b 0"
  "models/Qwen3-0.6B qwen3_0_6b 0"
  "models/gemma-4-E2B-it gemma4_e2b 0"
  "models/DeepSeek-R1-Distill-Qwen-1.5B deepseek_r1_distill_qwen_1_5b 0"
  "models/glm-edge-1.5b-chat glm_edge_1_5b 0"
  "models/SmolLM2-1.7B-Instruct smollm2_1_7b 1024"
  "models/glm-edge-4b-chat glm_edge_4b 1024"
)

CTX=4096
for bw in "${BW_LIST[@]}"; do
  for entry in "${MODELS[@]}"; do
    read -r mp short chunk <<<"$entry"
    out_dir="$OUT/bw${bw}/$short/ctx_${CTX}"
    mkdir -p "$out_dir"
    echo "=== $short @ ctx=${CTX} bw=${bw} ==="
    python3 benchmarks/kakeyaturbo_v1_2_real_bench.py \
      --model-path "$mp" --model-name "$short" \
      --context-tokens "$CTX" \
      --bit-width "$bw" \
      --prefill-chunk "$chunk" \
      --out-dir "$out_dir" 2>&1 | tail -8
  done
done

echo
echo "=== done, reports under $OUT ==="

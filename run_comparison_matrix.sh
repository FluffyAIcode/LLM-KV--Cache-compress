#!/usr/bin/env bash
# Run compare_kakeya_vs_turboquant.py across every model + context combo
# we have numbers for in reports/.  Output lands in reports/compare/<model>/.
set -euo pipefail

BASE="${1:-/workspace}"
cd "$BASE"

mkdir -p reports/compare

# (model_path, short_name, contexts_to_run, prefill_chunk)
MODELS=(
  "models/Qwen2.5-0.5B-Instruct|qwen2_5_0_5b|2048,8192|0"
  "models/Qwen3-0.6B|qwen3_0_6b|2048,8192|0"
  "models/gemma-4-E2B-it|gemma4_e2b|2048,8192|0"
  "models/SmolLM2-1.7B-Instruct|smollm2_1_7b|2048,4096|1024"
  "models/DeepSeek-R1-Distill-Qwen-1.5B|deepseek_r1_distill_qwen_1_5b|2048,8192|0"
  "models/glm-edge-1.5b-chat|glm_edge_1_5b|2048,8192|0"
  "models/glm-edge-4b-chat|glm_edge_4b|2048,4096|1024"
)

for row in "${MODELS[@]}"; do
  IFS='|' read -r path short ctx_list chunk <<< "$row"
  for ctx in $(echo "$ctx_list" | tr ',' ' '); do
    mkdir -p "reports/compare/$short"
    out="reports/compare/$short/compare_${ctx}.json"
    echo "===== $short @ $ctx tokens (chunk=$chunk) ====="
    python3 compare_kakeya_vs_turboquant.py \
      --model-path "$path" --model-name "$short" \
      --context-tokens "$ctx" \
      --block-size 512 --residual-length 256 \
      --d-res 8 --k-segments 16 --variance-ratio 0.95 \
      --dtype bfloat16 --attn eager \
      --prefill-chunk "$chunk" \
      --out "$out"
  done
done

echo
echo "=== ALL COMPARISONS DONE ==="

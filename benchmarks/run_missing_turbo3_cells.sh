#!/usr/bin/env bash
# Fill in the missing turbo3 / turbo4 comparison data points so the
# v1.2 vs turbo3 matrix has full 7 models × 3 contexts coverage.
set -euo pipefail
cd /workspace

# (model_dir, short, missing_contexts, prefill_chunk_if_big_ctx)
declare -a JOBS=(
  "Qwen2.5-0.5B-Instruct|qwen2_5_0_5b|4096|0"
  "Qwen3-0.6B|qwen3_0_6b|4096|0"
  "gemma-4-E2B-it|gemma4_e2b|4096|0"
  "DeepSeek-R1-Distill-Qwen-1.5B|deepseek_r1_distill_qwen_1_5b|4096|0"
  "glm-edge-1.5b-chat|glm_edge_1_5b|4096|0"
  "SmolLM2-1.7B-Instruct|smollm2_1_7b|8192|1024"
  "glm-edge-4b-chat|glm_edge_4b|8192|1024"
)

for row in "${JOBS[@]}"; do
  IFS='|' read -r model short ctx chunk <<< "$row"
  out="reports/compare/$short/compare_${ctx}.json"
  echo ""
  echo "=== $short @ $ctx tokens (chunk=$chunk) $(date -u +%FT%TZ) ==="
  python3 compare_kakeya_vs_turboquant.py \
    --model-path "models/$model" \
    --model-name "$short" \
    --context-tokens "$ctx" \
    --block-size 512 \
    --residual-length 256 \
    --d-res 8 \
    --k-segments 16 \
    --variance-ratio 0.95 \
    --dtype bfloat16 \
    --attn eager \
    --prefill-chunk "$chunk" \
    --out "$out"
done
echo ""
echo "=== DONE $(date -u +%FT%TZ) ==="

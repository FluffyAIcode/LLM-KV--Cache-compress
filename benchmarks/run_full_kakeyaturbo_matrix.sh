#!/usr/bin/env bash
# Full 7-model x 3-context (2k/4k/8k) real kakeyaturbo benchmark.
# NO mock, NO fallback, NO simplification -- the Rust release binary
# does the real PCA+K-means+WHT+Lloyd-Max work on the real KV tensors
# captured from real HF forward passes in bf16.
set -euo pipefail

BASE="${BASE:-/workspace}"
OUT_ROOT="${OUT_ROOT:-$BASE/reports/real_kakeyaturbo/full}"
LOG="${LOG:-/tmp/real_kakeyaturbo_full.log}"

mkdir -p "$OUT_ROOT"
echo "=== kakeyaturbo FULL matrix start $(date -u +%FT%TZ) ===" | tee "$LOG"

# (model_dir | short | chunk@8k)
ROWS=(
  "Qwen2.5-0.5B-Instruct|qwen2_5_0_5b|0"
  "Qwen3-0.6B|qwen3_0_6b|0"
  "gemma-4-E2B-it|gemma4_e2b|0"
  "DeepSeek-R1-Distill-Qwen-1.5B|deepseek_r1_distill_qwen_1_5b|0"
  "glm-edge-1.5b-chat|glm_edge_1_5b|0"
  "SmolLM2-1.7B-Instruct|smollm2_1_7b|1024"
  "glm-edge-4b-chat|glm_edge_4b|1024"
)
CONTEXTS=(2048 4096 8192)

for row in "${ROWS[@]}"; do
  IFS='|' read -r model short chunk_8k <<< "$row"
  for ctx in "${CONTEXTS[@]}"; do
    outdir="$OUT_ROOT/$short/ctx_${ctx}"
    mkdir -p "$outdir"
    # Use chunked prefill only at 8k for the known-big models.
    chunk=0
    if [ "$ctx" -ge 8192 ] && [ "$chunk_8k" -gt 0 ]; then
      chunk="$chunk_8k"
    fi
    echo "" | tee -a "$LOG"
    echo "=== $short @ $ctx tokens (chunk=$chunk) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    python3 "$BASE/benchmarks/kakeyaturbo_real_bench.py" \
      --model-path "$BASE/models/$model" \
      --model-name "$short" \
      --context-tokens "$ctx" \
      --block-size 512 \
      --variance-ratio 0.95 \
      --k 16 \
      --bit-width 3 \
      --rotation-seed 3405691582 \
      --prefill-chunk "$chunk" \
      --out-dir "$outdir" \
      --verify 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "=== kakeyaturbo FULL matrix end $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo FULL_DONE > /tmp/kakeyaturbo_full_done.flag

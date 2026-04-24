#!/bin/bash
set -euo pipefail
cd /workspace/LLM-KV--Cache-compress
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export KAKEYA_SNAPSHOT_QWEN3=1

LOG=/tmp/live_audit_$(date -u +%Y%m%dT%H%M%SZ).log
echo "=== AUDIT RUN $(date -u +%FT%TZ) ==="             | tee -a $LOG
echo "=== host: $(hostname) ==="                        | tee -a $LOG
echo "=== nvidia-smi pre-run ==="                       | tee -a $LOG
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv | tee -a $LOG
echo                                                     | tee -a $LOG

# Start nvidia-smi sampling at 1 Hz in background.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > /tmp/gpu_trace_$$.csv &
SMI_PID=$!

# Run the benchmark (minimal: 2 passages on cached GLM-4-9B).
echo "=== LAUNCHING vLLM + snapshot harness $(date -u +%FT%TZ) ===" | tee -a $LOG
timeout 180 /venv/main/bin/python benchmarks/multimodel_v14_kv_128k_report.py \
  --model-path zai-org/GLM-4-9B-Chat \
  --model-name audit_glm \
  --q-values 38 --tq-b-values 6 \
  --ctx-len 1024 --n-eval 16 --n-passages 2 \
  --gpu-mem-util 0.60 \
  --trust-remote-code \
  --out-dir /tmp/audit_out 2>&1 | tee -a $LOG

kill $SMI_PID 2>/dev/null || true
sleep 1

echo                                                     | tee -a $LOG
echo "=== nvidia-smi post-run ==="                       | tee -a $LOG
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv | tee -a $LOG
echo                                                     | tee -a $LOG
echo "=== GPU trace excerpts (peak utilization) ==="     | tee -a $LOG
echo "First 3 samples:"                                  | tee -a $LOG
head -4 /tmp/gpu_trace_$$.csv                            | tee -a $LOG
echo "..."                                               | tee -a $LOG
echo "Lines where utilization > 50%:"                    | tee -a $LOG
awk -F"," "NR>1 && \$2+0>50" /tmp/gpu_trace_$$.csv | head -15 | tee -a $LOG
echo "..."                                               | tee -a $LOG
echo "Peak memory usage observed in trace:"              | tee -a $LOG
sort -t"," -k3 -n -r /tmp/gpu_trace_$$.csv | head -3     | tee -a $LOG

echo                                                     | tee -a $LOG
echo "=== Result JSON sha256 ==="                        | tee -a $LOG
sha256sum /tmp/audit_out/audit_glm_kv_128k.json 2>&1 | tee -a $LOG
echo                                                     | tee -a $LOG
echo "=== Key result fields from JSON ==="               | tee -a $LOG
/venv/main/bin/python -c "
import json
d = json.load(open(\"/tmp/audit_out/audit_glm_kv_128k.json\"))
print(f\"model:        {d[\"model\"]}\")
print(f\"num_layers:   {d[\"num_layers\"]}\")
print(f\"head_dim:     {d[\"head_dim\"]}\")
print(f\"num_kv_heads: {d[\"num_kv_heads\"]}\")
print(f\"n_passages:   {d[\"n_passages\"]}\")
print(\"per_passage raw entries:\")
for p in d[\"per_passage\"]:
    fc = p.get(\"fire_count\", \"?\")
    ch = p[\"channel\"]
    dp = p.get(\"delta_ppl\")
    pr = p.get(\"ppl_ref\")
    pa = p.get(\"ppl_alt\")
    ta = p.get(\"t_alt\")
    print(f\"  passage={p[\"passage\"]} ch={ch:<15} fires={fc:<3} ppl_ref={pr} ppl_alt={pa} delta_ppl={dp} t_alt={ta}\")
" | tee -a $LOG

echo
echo "=== log file: $LOG ==="
echo "=== gpu trace: /tmp/gpu_trace_$$.csv (keep) ==="
cp /tmp/gpu_trace_$$.csv /tmp/gpu_trace_kept.csv

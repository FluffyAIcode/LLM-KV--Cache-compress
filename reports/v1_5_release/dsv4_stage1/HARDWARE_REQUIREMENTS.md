# Stage 1 — DeepSeek-V4 live-vLLM evaluation: hardware requirements

**Status**: code scaffold complete (2026-04-25); execution blocked on
hardware. This document lists exactly what is needed to run the
Stage 1 evaluation, so that when appropriate hardware becomes
available (whether via vast.ai upgrade, a cloud Blackwell node, or a
dedicated DC rental) a user can go from "code on GitHub" to
"JSON + FINDINGS.md" in one session.

## 1. Hardware — minimum

| resource | minimum | recommended |
| --- | --- | --- |
| GPUs (V4-Flash, 284B total / 13B active) | **2× H200 SXM 141 GiB** (Hopper, SM 9.0) | **4× B200 or 4× B300** (Blackwell, SM 10.0) |
| GPUs (V4-Pro, 1.6T total / 49B active) | 8× H200 SXM 141 GiB | 8× B200 or 8× B300 |
| GPU interconnect | NVLink 4.0 or NVSwitch (needed for TP/EP) | NVSwitch |
| System RAM | 512 GiB | 1 TiB |
| Local disk (weights + HF cache + logs) | **500 GiB free** for V4-Flash, **1.5 TiB free** for V4-Pro | NVMe SSD |
| Network (initial download) | 1 Gbit/s sustained to HF hub | ≥ 10 Gbit/s |

Our current vast.ai dev instance (1× H200 SXM 141 GiB, 7 GiB disk
free) satisfies none of these for V4-Flash. V4-Pro is beyond the
scale we have ever provisioned in this project.

## 2. Software — minimum

| component | minimum version | notes |
| --- | --- | --- |
| vLLM | `0.19.x` **with DSV4 support merged**, or ship-ahead image `vllm/vllm-openai:deepseekv4-cu130` | PR #40760 is still `needs-rebase` as of 2026-04-25T01:00Z; use the docker image until merged |
| PyTorch | 2.8+ with native `float8_e4m3fn` + `float4_e2m1fn_x2` | the docker image pins this |
| CUDA | 13.0+ | provided by docker image |
| transformers | 5.5.2+ (for DSV4 tokenizer) | already on our dev box |
| Docker | 24+ | **NOT currently available on the vast.ai instance** |
| NCCL | 2.22+ | bundled with PyTorch |
| FlashMLA + FlashInfer | latest (docker image contents) | V4 attention requires both |

## 3. vLLM command — single-node V4-Flash

Minimum-viable command for B200x4 (from vLLM blog):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/kakeyalattice:/workspace/kakeyalattice \
  -v $(pwd)/vllm_backend:/workspace/vllm_backend \
  -e KAKEYA_SNAPSHOT_DSV4=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

For H200×2 (if Blackwell unavailable):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e KAKEYA_SNAPSHOT_DSV4=1 \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE", "custom_ops":["all"]}' \
  --trust-remote-code
```

(Hopper does not have native FP4 compute — the FP4 indexer cache
flag is dropped; FlashMLA falls back to FP8/BF16 codepath. Expect
2–3× slower decode than B200.)

## 4. Evaluation command — drop-in replacement for rigorous_eval.py

```bash
python benchmarks/dsv4_stage1/rigorous_eval_dsv4.py \
    --model-path deepseek-ai/DeepSeek-V4-Flash \
    --model-name dsv4_flash_nobdry \
    --mode inforward \
    --no-boundary \
    --q-values "" \
    --v15-q-values 4,10,38 \
    --tq-b-values "" \
    --kv-modes KV \
    --ctx-len 4096 \
    --n-eval 64 \
    --n-passages 32 \
    --kv-stream-filter all \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --out-dir reports/v1_5_release/dsv4_stage1
```

The wrapper (`benchmarks/dsv4_stage1/rigorous_eval_dsv4.py`) sets
`KAKEYA_SNAPSHOT_DSV4=1`, installs our DSV4 snapshot hook before
vLLM loads the model, then delegates to the standard
`benchmarks/rigorous_eval.py` with the V4-specific
`--kv-stream-filter` argument for per-KV-stream aggregation.

## 5. Expected deliverables after a successful run

```
reports/v1_5_release/dsv4_stage1/
├── FINDINGS.md                     (narrative + tables matching the
│                                    reports/v1_5_release/dsv4_stage0_5/
│                                    FINDINGS.md format)
├── dsv4_flash_nobdry_inforward.json
├── dsv4_flash_nobdry.log
└── per_layer_audit/
    └── dsv4_flash_layer_<L>.json   (per-layer K-MSE + non-Gaussian
                                     audit, sharded by layer_id for
                                     c4a / c128a / SWA streams)
```

## 6. Risks and open items

1. **FP8 / FP4 path on Hopper**: the vLLM V4 blog explicitly
   documents Blackwell optimisations; Hopper path is supported but
   untested for KakeyaLattice's in-forward splicing. The hook
   intercepts pre-FP8-quant, so in principle it works on both, but
   the FlashMLA kernel may assert on arch-specific block layouts.
2. **Shared-latent semantics**: V4 uses a single 512-dim latent for
   both K and V. Our hook feeds that latent to both the `K` and
   `V` slots of `_snapshot_capture_replace` (shared alias). The
   codec sees one tensor and returns one tensor; the downstream
   custom-op reads that single latent. This matches the V4
   architecture but differs from how the Qwen3/Gemma-4/GLM hooks
   treat K and V separately. Regression tests for this invariant
   are in `benchmarks/dsv4_stage1/test_dsv4_snapshot_hook.py`.
3. **MTP layers**: V4 has 1 MTP layer for V4-Flash and 1 for V4-Pro.
   Our layer-id extractor maps MTP layers to `10_000 + mtp_index`
   reserved ids. The boundary-skip policy should include MTP layers
   by default (use `--boundary-policy "first 2, last 2, mtp"` once
   added to rigorous_eval; currently MTP layers are not filtered).
4. **KV stream filter**: Stage 0.5 measured SWA / c4a-pool / c128a-pool
   separately; Stage 1 in-forward ONLY sees the shared pre-RoPE
   latent (single stream per token). The `--kv-stream-filter` flag
   subsets the per-layer rollup by compress_ratio attribute, so
   we can still report "SWA layer average MSE" vs "c4a layer
   average MSE" vs "c128a layer average MSE" even though the hook
   sees a single latent.
5. **Compilation + CUDA graphs**: V4 uses `cudagraph_mode="FULL_AND_PIECEWISE"`
   by default in the blog one-liner. Our hook wraps the forward in
   Python, which breaks CUDA-graph capture. We MUST pass
   `--compilation-config '{"cudagraph_mode":"NONE"}'` or
   `"PIECEWISE"` for the hook to fire. Piecewise should work; FULL
   will silently skip the hook.

## 7. Cost estimate (vast.ai, approximate, April 2026 pricing)

| config | hourly | 4-hour single-shot | 24-hour week-long campaign |
| --- | --- | --- | --- |
| 2× H200 SXM 141 GiB + 1 TiB SSD | $6–10/hr | $25–40 | $150–250 |
| 4× B200 + 2 TiB SSD (if available) | $12–20/hr | $50–80 | $300–500 |

(vast.ai pricing fluctuates hourly; use
`vastai search offers 'gpu_name=H200_SXM num_gpus>=2'` for live
quotes.)

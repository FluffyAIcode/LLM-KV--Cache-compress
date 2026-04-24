# v1.4 KakeyaLattice — Execution Audit Trail

**Date**: 2026-04-23T22:28:14Z
**Host**: vast.ai instance (NVIDIA H200, driver 580.95.05, CUDA 13.0)
**Stack**: vLLM `0.19.2rc1.dev100+gf946659ff.cu130` · PyTorch 2.11.0 · FlashAttention v3 · transformers 5.5.2

This directory contains the raw evidence that the numbers in the v1.4
reports (`reports/v1_4_release/kv_128k_report*/`,
`reports/v1_4_release/kv_128k_isoppl*/`) came from **real vLLM
prefill passes on a real GPU**, not from CPU simulators, mocks, or
offline replay.

## Files

| File | sha256 | What it is |
|:-----|:-------|:-----------|
| `live_audit_20260423T222814Z.log` | `09caf0be…cf1e1f` | Full stdout+stderr of an audit run (see below) |
| `audit_glm_kv_128k.json`          | `fe69502d…6f676a` | Measurement JSON produced by the same run |
| `gpu_trace.csv`                   | `20d0a58d…cb8da`  | 1 Hz `nvidia-smi` trace, sampled throughout |

## What the audit run did

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -l 1 > gpu_trace.csv &
python benchmarks/multimodel_v14_kv_128k_report.py \
    --model-path zai-org/GLM-4-9B-Chat \
    --model-name audit_glm \
    --q-values 38 --tq-b-values 6 \
    --ctx-len 1024 --n-eval 16 --n-passages 2 \
    --gpu-mem-util 0.60 \
    --trust-remote-code \
    --out-dir /tmp/audit_out
```

2 WikiText-103 passages × 3 codec channels (bf16 / v1.4 Q=38 K+V /
TQ b=6 K+V) on GLM-4-9B-Chat, with GPU utilisation + memory sampled
at 1 Hz throughout.

## What the log proves

### 1. Real GPU initialisation

From `live_audit_20260423T222814Z.log`:
```
=== nvidia-smi pre-run ===
name, driver_version, utilization.gpu [%], memory.used [MiB]
NVIDIA H200, 580.95.05, 0 %, 0 MiB
```

Memory starts at 0 MiB (no leftover process).

### 2. Real vLLM engine init

Excerpts from the same log:
```
INFO 04-23 22:28:20 [core.py:107] Initializing a V1 LLM engine (v0.19.2rc1.dev100+gf946659ff) ...
INFO 04-23 22:28:20 [cuda.py:368] Using FLASH_ATTN attention backend ...
INFO 04-23 22:28:20 [flash_attn.py:637] Using FlashAttention version 3
INFO 04-23 22:28:26 [weight_utils.py:615] Time spent downloading weights ...
INFO 04-23 22:28:32 [gpu_model_runner.py:4854] Model loading took 17.56 GiB memory and 5.71 s
INFO 04-23 22:28:34 [gpu_worker.py:440] Available KV cache memory: 63.82 GiB
INFO 04-23 22:28:34 [kv_cache_utils.py:1337] GPU KV cache size: 1,673,040 tokens
```

### 3. Snapshot hook install at plugin load

```
[snap-patch] Qwen3Attention.forward wrapped (capture / replace / off)
[snap-patch] Qwen2Attention.forward wrapped (for DeepSeek-R1-Distill-Qwen-1.5B)
[snap-patch] Gemma4Attention.forward wrapped (for Gemma 4 E2B/E4B/26B-A4B/31B)
[snap-patch] GLMAttention.forward wrapped (for GLM-4 / ChatGLM)
```

These four lines come from `kakeya_v1_4_snapshot.snapshot_hook` and
prove the vLLM plugin entry-point fired and monkey-patched the
actual Attention classes in the vLLM model-executor module.

### 4. Real prefill + real FA output

```
[model] zai-org/GLM-4-9B-Chat
[model] L=40 hd=128 kv_h=2
=== passage 1/2 ===
  [capture] layers=40 tokens=1040 ref_ppl=7.685 in 0.05s
  [bf16_pass   ] K2048b V2048b  Δppl= +0.000%  top1=100.00%  ... fires=40
  [v14_Q38_K+V ] K832b V832b  Δppl= -1.159%  top1= 93.75%  ... fires=40
  [tq_b6_K+V   ] K800b V800b  Δppl= +0.028%  top1=100.00%  ... fires=40
```

`fires=40` on every channel = the GLM Attention patch fired on all
40 transformer layers each time the alt-forward replaced K/V.  A
silent passthrough (e.g. CPU fallback) would have `fires=0`.

### 5. GPU actually worked hard

From `gpu_trace.csv` (selected samples):
```
2026/04/23 22:28:14.286, 0 %, 0 MiB           ← process not started
2026/04/23 22:28:26.295, 1 %, 819 MiB         ← CUDA ctx + kernels loaded
2026/04/23 22:28:27.296, 0 %, 18837 MiB       ← 17.56 GiB weights loaded
2026/04/23 22:28:33.297, 100 %, 21271 MiB     ← FlashInfer autotune spike
2026/04/23 22:28:35.297, 0 %, 87945 MiB       ← paged KV cache allocated
2026/04/23 22:28:36.298, 0 %, 87945 MiB       ← alt-forwards running
2026/04/23 22:28:37.298, 49 %, 84505 MiB      ← mid-alt-forward
2026/04/23 22:28:38.298, 36 %, 0 MiB          ← process exit (mem freed)
```

Total GPU memory consumption: ~88 GiB (of 143 GiB H200 VRAM).
Weight: 17.56 GiB. KV cache: 63.82 GiB.  This matches what a real
9B-parameter model + FlashAttention V3 + 128k-token KV cache would
need; it is not achievable on CPU.

### 6. Measurement numbers reproduce committed reports

From the same audit JSON (`audit_glm_kv_128k.json`):

| channel      | passage 0 ppl_ref   | passage 0 ppl_alt   | passage 0 delta_ppl | fires |
|:-------------|--------------------:|--------------------:|--------------------:|------:|
| bf16_pass    | 7.68523274918787    | 7.68523274918787    | 0.0                 | 40    |
| v14_Q38_K+V  | 7.68523274918787    | 7.59616873720817    | -0.011588980         | 40    |
| tq_b6_K+V    | 7.68523274918787    | 7.68737710991048    | +0.000279023         | 40    |

`ppl_ref` is the bf16 capture-pass PPL, `ppl_alt` is the replace-pass
PPL.  Each number is produced by `llm.generate(...)` → `prompt_logprobs`
returned from a real FlashAttention bf16 forward on 1040 tokens ×
40 layers × 2 KV heads × 128 head_dim.

The v1.4 Q=38 channel at CR=2.15× and the TQ b=6 channel at CR=2.21×
numbers in `reports/v1_4_release/kv_128k_report_tq_compare/`
(4-passage) / `kv_128k_isoppl_n8/` (8-passage) include these two
passages as a subset — running them again reproduces the
previously-reported numbers within bf16 FlashAttention
reproducibility noise.

## How to run the audit yourself

On any H200 (or comparable Hopper/Ampere GPU) with the repo checked out:

```bash
# 1. Environment (vast.ai image — or any CUDA 13 image with FA3).
pip install -e kakeyalattice
pip install -e vllm_backend   # installs the vllm.general_plugins entry point

# 2. Run the audit.
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv -l 1 > my_trace.csv &
SMI=$!

python benchmarks/multimodel_v14_kv_128k_report.py \
    --model-path zai-org/GLM-4-9B-Chat \
    --model-name audit \
    --q-values 38 --tq-b-values 6 \
    --ctx-len 1024 --n-eval 16 --n-passages 2 \
    --gpu-mem-util 0.60 --trust-remote-code \
    --out-dir audit_out 2>&1 | tee my_audit.log

kill $SMI
```

The log should contain:

- A vLLM `[core.py] Initializing a V1 LLM engine` banner
- The `[flash_attn.py] Using FlashAttention version 3` line
- A `[weight_utils.py] Time spent downloading weights` line (or a fast
  path if cached)
- The four `[snap-patch] *Attention.forward wrapped` lines
- Per-passage `[capture] layers=40 tokens=1040 ref_ppl=<float> in <sec>s`
- Per-channel `fires=40` — non-zero proves the monkey-patched forward
  actually ran on the GPU-resident K/V
- A final storage-ratio table matching this file's numbers

Your `my_trace.csv` should show:

- Memory jump to ~18 GiB (weight load) and then ~88 GiB (KV cache
  allocation + running prefill)
- GPU utilisation spikes to 50-100 % during prefill
- Memory drops back to 0 MiB on process exit

If the GPU never hits those memory / utilisation levels, the
benchmark did not actually run on the GPU and the numbers cannot be
trusted.  If the `fires=` count is below the expected (`num_layers -
boundary_size`), the snapshot hook did not intercept the forward and
the measurement channel has silently passed through the original
bf16 K/V instead of the compressed reconstructions.  The harness
itself already catches this case (`fatal: silent passthrough` in
`per_passage[].fatal`) — but visually confirming `fires = L_non_bdry`
is the simplest independent check.

## Supplementary: 19 log files in other directories

Each of the following directories contains a `*.log` file
capturing the full stdout of the corresponding benchmark run.
All sha256-stable from the commits that added them.

```
reports/v1_4_release/kv_128k_report/              × 4 logs (Qwen3, DeepSeek, Gemma, GLM; n=4 v1.4-only)
reports/v1_4_release/kv_128k_report_tq_compare/   × 4 logs (n=4 iso-bit vs TQ)
reports/v1_4_release/kv_128k_isoppl/              × 4 logs (n=4 dense sweep)
reports/v1_4_release/kv_128k_isoppl_n8/           × 4 logs (n=8 dense sweep, canonical)
reports/v1_4_release/streaming/logs/              × 3 logs (streaming proof / diag / latency)
```

All 19 include the vLLM engine init banner, the FlashAttention version
line, the snapshot-patch install lines, and the per-passage /
per-channel `fires=<L_non_bdry>` confirmation.  Any of them is an
independent verifiable record of a real vLLM GPU run.

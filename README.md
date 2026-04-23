# KakeyaLattice — v1.4 KV-Cache Compression

A GPU-native lattice-quantisation codec for transformer KV caches.
Measured across 4 open-source model families (Qwen3, DeepSeek, Gemma 4,
GLM-4) in real vLLM on an NVIDIA H200: **wins on K-MSE, V-MSE, and
|Δppl| against TurboQuant at matched bit budgets across 12 / 12
pairings**, with **+3 % to +38 % compression-ratio advantage** at
deployment-relevant quality thresholds (|Δppl| ≤ 2 %).

[Release notes, full comparison tables, and reproducibility commands
are on the v1.4 Release page.](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.4)

## What's in the box

```
kakeyaturbo-py/python/kakeyaturbo_py/
  v1_4_kakeya_zamir_lattice_gpu.py   — canonical V14KakeyaZamirLatticeGPU class
  bridge_b2_d4_tq_style.py           — core D4 + TurboQuant-style codec
  bridge_b_nested_lattice.py         — Conway-Sloane D4 closest-lattice-point
  spherical_codebooks.py             — codec interface
  __init__.py                        — re-exports V14KakeyaZamirLatticeGPU

vllm_backend/kakeya_v1_4_snapshot/
  snapshot_hook.py                   — Attention monkey-patches (Qwen3 / Qwen2
                                       / Gemma4 / GLM) for post-QK/V-norm,
                                       pre-RoPE K/V capture + replace
  plugin.py                          — vLLM entry point (gated by env var)
  pyproject.toml                     — installs the plugin into vLLM workers

benchmarks/
  multimodel_v14_kv_128k_report.py   — per-model 128k KV storage report
                                       (v1.4 + TurboQuant comparison)
  multimodel_v14_vs_tq.py            — iso-bit head-to-head (K-only variant)
  v14_streaming_proof.py             — streaming / online proof
  v14_streaming_diag.py              — batch-vs-streaming root-cause diagnostic
  v14_streaming_latency.py           — per-decode-step latency

reports/v1_4_release/
  kv_128k_report/                    — v1.4-only 128k KV storage tables
  kv_128k_report_tq_compare/         — iso-bit comparison vs TurboQuant
  kv_128k_isoppl_n8/                 — iso-PPL comparison vs TurboQuant
  streaming/                         — streaming / online capability report
  paper/                             — LaTeX paper
```

## Quick start

Install the pure-Python codec + the vLLM snapshot plugin:

```bash
pip install -e kakeyaturbo-py          # pure-Python, PyTorch-only
pip install -e vllm_backend             # installs the vllm.general_plugins entry point
```

Use the codec directly:

```python
import torch
from kakeyaturbo_py import V14KakeyaZamirLatticeGPU

cb = V14KakeyaZamirLatticeGPU(D=128, q_range=38, device="cuda")
K = torch.randn(2048, 8, 128, device="cuda", dtype=torch.float32) * 0.3
K_hat = cb.roundtrip(K)       # encode + decode round-trip, bits known in advance
print(cb.bits_per_token_per_head)   # 832 bits for D=128, Q=38
```

Run the multi-model head-to-head benchmark (real vLLM prefill, strict GPU):

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
python benchmarks/multimodel_v14_kv_128k_report.py \
    --model-path Qwen/Qwen3-4B --model-name qwen3_4b \
    --q-values 4,6,10,15,22,38,76,152 \
    --tq-b-values 3,4,5,6,7,8 \
    --ctx-len 2048 --n-eval 64 --n-passages 8 \
    --out-dir reports/v1_4_release/kv_128k_isoppl_n8
```

Add `--trust-remote-code` for GLM-4-9B-Chat.

## Key results (full tables in `reports/v1_4_release/`)

**iso-PPL compression advantage at |Δppl| ≤ 2 %** (dense Q/b sweep,
n=8 passages, 512 target tokens per channel):

| Model          | v1.4 CR | TQ CR | v1.4 advantage |
|:---------------|--------:|------:|---------------:|
| Qwen3-4B       | 2.77×   | 2.18× | **+26.9 %**    |
| GLM-4-9B-Chat  | 2.44×   | 1.77× | **+37.8 %**    |
| Gemma-4-E4B    | 3.04×   | 3.04× | tied (saturated) |
| DeepSeek-1.5B  | 2.43×   | 2.36× | **+3.3 %**     |

**iso-bit |Δppl| advantage at aggressive point** (v1.4 Q=10 vs TQ b=4,
n=4 passages, ~3.6-3.9× CR):

| Model          | v1.4 \|Δppl\| | TQ \|Δppl\| | v1.4 better by |
|:---------------|--------------:|------------:|---------------:|
| Qwen3-4B       | 1.45 %        | 6.58 %      | **4.5×**       |
| GLM-4-9B-Chat  | 6.52 %        | 10.74 %     | 1.6×           |
| Gemma-4-E4B    | 0.33 %        | 1.04 %      | **3.2×**       |
| DeepSeek-1.5B  | 2.22 %        | 3.47 %      | 1.6×           |

**Streaming latency** (per-decode-step, 1 new token × all layers × all
KV heads, batched): **~0.25 ms** across all 4 models × 3 operating
points.  At typical 15-30 ms bf16 decode step, codec overhead is
**< 2 %** of total decode latency.

## Streaming / online deployment

v1.4 has no cross-token state — the codec is a pure per-vector
function.  It supports streaming / online compression out of the box:
no calibration pass, no warmup, no buffering.  See
`reports/v1_4_release/streaming/V14_STREAMING_REPORT.md` for
measurements and integration notes.

## Compliance

All reported numbers are measured on real vLLM + real Hugging Face
weights + real WikiText-103 + real FlashAttention bf16 forward on
an NVIDIA H200.  No mocks, no simplifications, no fallbacks.

## License

Apache-2.0 (see `LICENSE`).

# v1.4 KakeyaLattice — Evidence Inventory

**Date pulled**: 2026-04-24
**Source**: vast.ai H200 instance, `/workspace/LLM-KV--Cache-compress/reports/v1_4_release/` + `/tmp/`
**Integrity**: see `MANIFEST.sha256` — 103 files, 5.15 MiB, all sha256-verified against vast at pull time

All files in this tree are raw outputs of real vLLM + real H200 GPU runs.  No mocks, no simulations.  Every log is the full stdout+stderr of an actual `llm.generate(...)` call chain, plus the vLLM engine startup banner + weight loading + FlashAttention version + snapshot-patch install lines + per-channel fires-count + per-passage PPL values.

## Directory structure

```
reports/v1_4_release/
├── MANIFEST.sha256                   ← integrity manifest (this file references)
├── INVENTORY.md                      ← this file
│
├── audit/                            ← end-to-end audit trail (see README.md)
│   ├── README.md                      audit reproduction guide
│   ├── audit_glm_kv_128k.json         measurement JSON from the audit run
│   ├── gpu_trace.csv                  1 Hz nvidia-smi trace during audit
│   ├── live_audit_20260423T222814Z.log audit run full stdout+stderr
│   ├── smoke_runs/                    11 harness smoke-test JSONs (early sanity)
│   └── tmp_artifacts/                 gpu_trace_*.csv + live_audit.sh
│
├── kv_128k_report/                   ← v1.4-only 128k KV storage (n=4, early run)
│   ├── *_kv_128k.json                 per-model measurement JSONs
│   ├── *.log                          per-model full log
│   └── V14_KV_128K_REPORT.md
│
├── kv_128k_report_tq_compare/        ← v1.4 vs TQ at matched bits (n=4, iso-bit)
│   ├── *_kv_128k.json
│   ├── *.log
│   └── V14_VS_TQ_KV_128K_REPORT.md
│
├── kv_128k_isoppl/                   ← iso-PPL dense Pareto sweep (n=4)
├── kv_128k_isoppl_n8/                ← iso-PPL dense Pareto sweep (n=8)
│   └── V14_VS_TQ_ISOPPL_REPORT.md
│
├── kv_128k_inforward/                ← early 4-passage in-forward (Apr 23 22:58)
│   └── deepseek_1p5b, glm4_9b        (superseded by rigorous_eval/inforward/)
│
├── rigorous_eval/                    ← the canonical n=32 data (RIGOROUS_EVAL_REPORT.md)
│   ├── RIGOROUS_EVAL_REPORT.md
│   ├── snapshot/                     snapshot mode n=32, 4 models
│   ├── inforward/                    in-forward mode n=32, 4 models
│   ├── ablation/                     6 ablation variants × Qwen3-4B n=32
│   ├── noboundary/                   no-boundary n=32 check (Gemma + DeepSeek)
│   ├── min_boundary/                 DeepSeek boundary-size sweep (k=0,2,4,6)
│   └── beta_aggressive/              PHASE_BETA_REPORT.md (Q=4 vs b=3 iso-bit)
│
├── niah/                             ← NIAH retrieval accuracy
│   └── *_inforward.log + *.json     (Qwen3, DeepSeek, GLM; Gemma: see harness fix)
│
├── streaming/                        ← streaming / online latency + bit-identity
│   ├── V14_STREAMING_REPORT.md
│   └── logs/
│       ├── v14_streaming_proof.log
│       ├── v14_streaming_diag.log
│       └── v14_streaming_latency.log
│
└── paper/                            ← LaTeX paper
```

## What each log contains

Every `*.log` under `rigorous_eval/`, `niah/`, `kv_128k_*/`, and `streaming/` is the **complete stdout+stderr** of a real vLLM run.  Each one includes:

1. **vLLM engine banner** — version, config, kv cache memory, FlashAttention version
   ```
   INFO 04-23 14:46:19 [core.py:107] Initializing a V1 LLM engine (v0.19.2rc1.dev100+gf946659ff) ...
   INFO 04-23 14:46:20 [flash_attn.py:637] Using FlashAttention version 3
   INFO 04-23 14:46:32 [gpu_worker.py:440] Available KV cache memory: 45.92 GiB
   ```

2. **Snapshot-patch install** (four lines per vLLM worker process):
   ```
   [snap-patch] Qwen3Attention.forward wrapped (capture / replace / off)
   [snap-patch] Qwen2Attention.forward wrapped (for DeepSeek-R1-Distill-Qwen-1.5B)
   [snap-patch] Gemma4Attention.forward wrapped (for Gemma 4 E2B/E4B/26B-A4B/31B)
   [snap-patch] GLMAttention.forward wrapped (for GLM-4 / ChatGLM)
   ```

3. **Weight download + load** (cache miss case):
   ```
   INFO 04-23 14:46:28 [weight_utils.py:615] Time spent downloading weights for Qwen/Qwen3-4B: 7.77 seconds
   INFO 04-23 14:46:31 [gpu_model_runner.py:4854] Model loading took 7.56 GiB memory and 10.19 seconds
   ```

4. **Per-passage ref_ppl** (the bf16 baseline PPL — `llm.generate()` return):
   ```
   === passage 1/32 ===
     [ref] ppl=27.698  (0.07s)
   ```

5. **Per-channel results**:
   ```
     [v14_Q38_KV                      ] Δppl= +1.216% top1=100.00% K-MSE(l0)=5.31e-04 V-MSE(l0)=6.04e-04 fires=36
   ```
   Note `fires=<N>` where N is the number of layers that actually went through the codec in that passage.  For a model with 36 layers and 4 boundary layers, `fires=32` confirms the codec actually ran on the GPU-resident K/V for every non-boundary layer.

6. **Aggregate table** at the end, including n=32 mean ± CI95.

## Replication quick checks

- `MANIFEST.sha256` lists every file's SHA256.  To verify: `cd reports/v1_4_release && sha256sum -c MANIFEST.sha256` (after reformatting if needed for your sha256sum version).
- Every measurement JSON has a `per_passage` list (one entry per passage × channel) with raw ppl_ref, ppl_alt, fire_count, and k/v rel-MSE — so a third party can recompute the aggregates from the raw entries.
- The corresponding `*.log` confirms (a) vLLM was used, (b) the model was loaded on GPU, (c) the codec fired on the right layers.

## Key sha256s

Pin these sha256s to catch accidental data corruption:

| file | sha256 (first 12) |
|:-|:-|
| `rigorous_eval/snapshot/qwen3_4b_snapshot.log` | `03e4aec89a7a` |
| `rigorous_eval/inforward/qwen3_4b_inforward.log` | `10bc94e635a3` |
| `rigorous_eval/beta_aggressive/qwen3_4b_beta_aggressive.log` | `6b62065c9294` |
| `niah/qwen3_4b_32k_inforward.log` (NIAH at 32k) | `2f917046629d` |
| `audit/live_audit_20260423T222814Z.log` | `09caf0bec11a` |

Full first-12-hex prefixes for every file are in `MANIFEST.sha256`.

## Reproducibility summary

Each log can be independently reproduced on any H200 (or comparable Hopper/Ampere GPU) with:

```bash
git clone FluffyAIcode/LLM-KV--Cache-compress
cd LLM-KV--Cache-compress
pip install -e kakeyalattice
pip install -e vllm_backend
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1
```

Then run the harness command documented in each report's "Reproducibility" section.  On an H200, a 32-passage dense sweep (18 channels) completes in 5-15 minutes per model (download time excluded).

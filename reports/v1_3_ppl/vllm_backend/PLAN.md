# v1.3 PPL as a production vLLM KV-cache backend

**Goal.** Deliver a deployable vLLM attention backend
`--kv-cache-dtype kakeya_v1_3_ppl` that actually saves GPU memory
and runs at near-baseline throughput (not a research harness), and
compare it head-to-head against the merged upstream TurboQuant
(PR #38479) on the SAME hardware, model, metric, and context.

**Why this plan exists.** PR #17's snapshot-mode harness measures
PPL but does not change vLLM's paged cache layout, uses a CPU
subprocess for the codec, and runs in V0 `enforce_eager=True`.
That is a research instrument, not a deployed backend. This branch
rebuilds the codec as a TurboQuant-style Triton attention backend
so the "performance on vLLM" comparison is meaningful.

## Reference points

- TurboQuant (PR #38479) merged 2026-04-15. Verified-stable preset
  is **`k8v4`**: FP8 keys + 4-bit uniform values + norm correction.
  Reported on Qwen3-4B, 4× RTX PRO 6000 Blackwell:
  - throughput 79–100 % of baseline (decode-heavy 79 %,
    very-long-prefill 100 %)
  - TPOT 135.2 ms vs baseline 138.1 ms on long-prefill (**faster**
    thanks to lower KV cache bandwidth)
  - slot size 196 B vs baseline 512 B → 2.6× cache compression
  - GSM8K 0.860 / 0.900 = 95.6 % of baseline (community verified)
- TurboQuant `3bit_nc` (4.9×): reported GSM8K 0.72; independent
  repros scored 0.009 (broken) on Qwen3-4B. Do NOT target this.

## Proposed algorithm for the vLLM backend

Compromise between v1.3 PPL (algorithm) and TurboQuant (engineering):

- **K path (Triton kernel)**:
    pre-computed Chol(Σ_q) L matrix (lookup per layer, per KV head)
      → K @ L           (in-kernel whitening)
      → WHT rotation    (in-kernel; already used by TurboQuant)
      → Lloyd-Max scalar quant, `b_K = 3`, calibrated centroids
        fit offline on Qwen3-4B post-WHT K residuals
      → bit-pack into paged cache slot
      → outlier side-buffer: coords with `|scaled| > T=2.0`
        stored as `(u16 idx, f16 val)` pairs appended to slot
- **V path (Triton kernel)**:
    WHT rotation → Lloyd-Max quant, `b_V = 3`, per-head → bit-pack
- **Decode path (Triton kernel)**:
    unpack K (undo bit-pack, restore WHT-rotated floats via centroid
      gather, apply outlier overrides, inverse WHT, apply L⁻¹ to
      restore un-whitened K) → compute Q · K.T scores in fp32 →
      softmax in fp32 → unpack V → score · V in bf16 → output
- **Prefill path**: same as TurboQuant — FA varlen over raw Q,K,V,
  then write compressed K,V into cache.
- **Boundary skip**: layers `[0, 1, L-2, L-1]` stay bf16 via
  `--kv-cache-dtype-skip-layers`.

Target compression: **3.5 – 4.0×**. Sits between TurboQuant `k8v4`
(2.6×, GSM8K 96 %) and `4bit_nc` (3.8×, claimed 93 %, community
not-yet-independently-verified). Target quality on Qwen3-4B GSM8K:
**≥ 96 %** (match or beat TurboQuant `k8v4` at higher compression).

## Differences from TurboQuant that matter

1. **Calibrated Lloyd-Max centroids** — TurboQuant uses Gaussian
   defaults; we fit on real post-WHT residuals of Qwen3-4B. The
   offline calibration itself is cheap (< 5 min, single GPU).
2. **Q-preconditioning on K** — TurboQuant rotates with WHT only.
   We pre-multiply by `L = chol(Σ_q)` before WHT so the quantizer
   minimises the Σ_q-weighted distortion that attention actually
   consumes. `L` is a per-(layer, KV-head) 128×128 constant matrix;
   in-kernel cost is one 128×128 matmul, amortisable.
3. **Outlier compensation on K** — 4-5 % of coords in a
   side-buffer at FP16 precision. Adds ~0.6 bits/coord of overhead
   on K, reducing compression by ~10 %, but recovers ~8 pp Δppl.

## Removal of research-only choices (PR #17 critique)

- ❌ no more CPU Rust subprocess
- ❌ no more KKTV disk I/O
- ❌ no more PCA rank reduction at runtime (only offline for
  calibration)
- ❌ no more `[tokens × num_kv_heads]` pooled PCA; codec is
  per-(token, head)
- ❌ no more dequant-back-to-bf16-write-to-cache; the compressed
  bytes stay compressed in the paged cache
- ❌ no more V0 `enforce_eager`; use V1 with CUDAGraph + compile

## Environment plan

1. On the Vast H200, **upgrade vLLM to the latest tag that
   contains PR #38479** (merged 2026-04-15, so any vLLM ≥ 0.10.x
   should have it; we'll target the most recent tagged release).
2. Download `Qwen/Qwen3-4B` (same model TurboQuant benchmarked on).
3. Install the `kakeya_v1_3_ppl` backend (Python package shipping a
   Triton kernel + backend registration hook).

## Measurement plan

Single script `benchmarks/vllm_backend_head2head.py`:

| Config | What |
|:---|:---|
| baseline | `--kv-cache-dtype auto` (bf16) |
| tq_k8v4 | `--kv-cache-dtype turboquant_k8v4` |
| tq_4bit_nc | `--kv-cache-dtype turboquant_4bit_nc` |
| **ours** | `--kv-cache-dtype kakeya_v1_3_ppl` |

Metrics, all on the SAME 4 prompts / 4 completions on Qwen3-4B:

- **Throughput**: output tok/s on {short-decode, long-prefill,
  decode-heavy, mixed} scenarios.
- **Latency**: TTFT, TPOT.
- **Peak GPU memory**: `torch.cuda.max_memory_allocated()` after
  steady state.
- **Quality**:
  - GSM8K 5-shot accuracy (200 questions, same script as TQ uses)
  - NIAH (512–32K context, ~77 probes)
  - WikiText-103 teacher-force Δppl (the metric we've been using,
    for continuity with earlier PRs)

## Milestones

- **M1** vLLM upgrade on H200 + reproduce TurboQuant k8v4 numbers
  on Qwen3-4B (our own measurement, not PR-description numbers).
- **M2** Fork `TurboQuantAttentionBackend` → `KakeyaV13PPLAttentionBackend`,
  replace centroid table, keep WHT + bit-pack + Triton scaffolding.
  First smoke test: simple generation works.
- **M3** Offline calibration of Σ_q + Lloyd-Max centroids on
  Qwen3-4B (reuse our `q_calibration.py` + `lloyd_max_calibration.py`
  with a new model path).
- **M4** Add pre-WHT `K @ L` whitening to the Triton store kernel
  and `K @ L^-1` to the decode kernel.
- **M5** Add outlier side-buffer to store + decode kernels.
- **M6** Head-to-head benchmark table; open PR #18 with real
  deployment numbers.

## Deliverables on this branch

- `vllm_backend/kakeya_v1_3_ppl/` — the Python package
- `benchmarks/vllm_backend_head2head.py` — benchmark script
- `benchmarks/qwen3_4b_calibration.py` — calibration for Qwen3-4B
- `reports/v1_3_ppl/vllm_backend/FINDINGS.md` — head-to-head table
- `reports/v1_3_ppl/vllm_backend/PLAN.md` — this file (kept for
  audit trail)

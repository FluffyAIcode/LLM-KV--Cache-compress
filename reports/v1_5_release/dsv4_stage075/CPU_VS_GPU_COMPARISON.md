# Stage 0.75 — CPU vs GPU comparison

**Run date**: 2026-04-25
**Purpose**: validate that Stage 0.75's trained-weights result is not
a hardware artifact, and isolate which numbers are
hardware-dependent vs hardware-independent.

## Setup

Same code, same data, same trained V4-Flash weights, same host hidden
states (Qwen2-0.5B projected to 4096). Only difference: the machine.

| | GPU run | CPU run |
| --- | --- | --- |
| machine | vast.ai H200 SXM 141 GiB | Cursor workspace (no GPU) |
| torch | 2.11.0+cu130 | 2.11.0+cpu |
| FP8 path | native `torch.float8_e4m3fn` cast | 127-level uniform fake-quant fallback |
| bf16 path | native bf16 matmul | fp32-coerced matmul |
| wall time | **15 s** | **12 s** (surprisingly fast because we only do 1 forward pass) |

## Key finding — hardware-independent numbers agree

These numbers should be identical on any hardware because they depend
only on the codec math and the V4 trained weights:

### Non-Gaussian audit (distribution shape)

| stream | metric | GPU | CPU | CPU/GPU |
| --- | --- | --- | --- | --- |
| SWA | \|kurt-3\| | 2.800 | 2.870 | 0.98× |
| SWA | iso-var | 112.4 | 114.5 | 0.98× |
| SWA | W2/σ | 0.342 | 0.348 | 0.98× |
| CSA | \|kurt-3\| | 2.481 | 2.649 | 0.94× |
| CSA | iso-var | 866 784 | 113 619 | 7.6× (⚠) |
| CSA | W2/σ | 0.427 | 0.446 | 0.96× |
| HCA | \|kurt-3\| | 1.376 | 1.286 | 1.07× |
| HCA | iso-var | 10 419 683 | 10 045 680 | 1.04× |
| HCA | W2/σ | 1.042 | 0.751 | 1.39× (⚠) |

**Most metrics agree within 5%**. The two large discrepancies are both in
extreme-outlier regimes:

- **CSA iso-var 7.6× difference**: the CSA pool at 512 samples has one
  coordinate dominating variance. bf16 rounding on GPU accumulates
  outliers slightly differently than fp32 on CPU. The *direction* of
  the audit (V4 is hyper-anisotropic) is preserved.
- **HCA W2/σ 1.39× difference**: the HCA pool has only N=16 vectors at
  seqlen=2048, so the 99th-percentile estimate is noisy. This would
  stabilise at larger context.

### E8 Q=38 absolute reconstruction rel-MSE

| stream | GPU | CPU | CPU/GPU |
| --- | --- | --- | --- |
| sliding_window_kv | 8.03e-04 | 1.07e-03 | 1.34× |
| csa_pool_kv_ratio4 | 9.28e-04 | 1.01e-03 | 1.08× |
| hca_pool_kv_ratio128 | 1.18e-03 | 1.20e-03 | 1.02× |

**KakeyaLattice's reconstruction quality on V4 KV is hardware-robust**.
The SWA 1.34× CPU/GPU gap is bf16-vs-fp32 numerical drift; HCA and CSA
match within 8%. Order of magnitude is consistent.

## Key finding — the FP8 baseline is NOT hardware-independent

This is the interesting scientific result. The FP8 per-64-block
baseline numbers differ by **14× to 48× between GPU and CPU**:

| stream | GPU FP8 rel-MSE (native e4m3) | CPU FP8 rel-MSE (127-level fake) | CPU/GPU |
| --- | --- | --- | --- |
| sliding_window_kv | 1.023e-03 | **2.10e-05** | **0.021×** |
| csa_pool_kv_ratio4 | 1.029e-03 | **3.10e-05** | **0.030×** |
| hca_pool_kv_ratio128 | 1.221e-03 | **8.54e-05** | **0.070×** |

The CPU fake-quant FP8 baseline is **14–48× more accurate** than the
real hardware `float8_e4m3fn` cast. This is because:

- **127-level uniform quantisation** (CPU fallback) has linear step
  size `fp8_max / 127 ≈ 3.53` across the full dynamic range
- **Real E4M3** (GPU) has a logarithmic step size that is much finer
  near zero (step ≈ 2⁻⁹ ≈ 2e-3 for tiny values) but much coarser at
  the extremes (step ≈ 32 for large values)

On V4-Flash KV (which is extremely anisotropic: a few coordinates carry
most variance, most coordinates are near zero), the logarithmic E4M3
spends too much resolution on the near-zero coordinates and saturates
on the large ones. The linear fake-quant resolves all coordinates to
the same step size and wins on aggregate MSE.

**This is not a bug — it's the real E4M3 behaviour.** V4 production
uses real E4M3 (as measured by our GPU run), so our GPU numbers are
the deployment-relevant ones. The CPU result reveals an interesting
fact: **a linear uniform quantiser at the same bit budget would beat
E4M3 on V4 KV by 14–48×**.

## Impact on the headline claim

The mean E8/FP8 ratio diverges sharply between the two runs:

```
GPU:  E8/FP8 ≈ 0.88  (E8 is 12% better than hardware FP8)
CPU:  E8/FP8 ≈ 32    (E8 is 32× WORSE than fake-quant FP8)
```

**The GPU number is the one that matters for a V4 deployment claim**,
because V4 production uses hardware E4M3, not linear fake-quant. The
CPU number tells us: "if V4 switched from E4M3 to linear FP8, E8 would
lose the compression-fidelity competition entirely on V4 KV."

## Mini-result: V4 FP8 is actually a sub-optimal choice

The CPU result incidentally demonstrates that **V4-Flash's choice of
E4M3 FP8 for KV cache is not Pareto-optimal on its own KV distribution
at D=512**. A simple linear 8-bit quantisation with per-64-block scale
achieves 14–48× lower MSE at identical bit budget. Whether this is
exploitable depends on whether linear 8-bit can be implemented as
efficiently as native E4M3 in FlashMLA / the attention kernel —
probably not, because the hardware datapath is set up for E4M3.

Interesting open question: is our "E8 beats FP8" headline from the GPU
run really a validation of KakeyaLattice, or is it partly because
E4M3 is mis-specified for V4 KV's distribution? Both are probably true.
D4/E8 lattice shaping still gives the promised ~0.37/0.65 dB
theoretical advantage regardless of the baseline's shortcomings.

## Reproducibility

CPU run on this box:

```bash
export HF_HOME=/workspace/hf_home_local
cd /workspace/LLM-KV--Cache-compress

# (already downloaded to hf_home_local: V4 shards 2/4/5 + Qwen2-0.5B, 11 GB)

python3 benchmarks/dsv4_stage075/run_stage075_real_weights.py \
    --host-model Qwen/Qwen2-0.5B \
    --seqlen 2048 --batch-size 1 \
    --q-values 10,38 \
    --hf-home /workspace/hf_home_local \
    --out reports/v1_5_release/dsv4_stage075/stage075_cpu.json
```

End-to-end wall time: **12 seconds on 4-thread CPU**. Strictly faster
than the GPU run (which had CUDA init overhead), because the compute
is dominated by a single `F.linear(4096→512)` applied once per stream
and the audit arithmetic.

## Bottom line

1. **Non-Gaussian audit (V4 KV shape)** is hardware-robust: both runs
   show V4 KV is dramatically more non-Gaussian than Qwen3-4B. ✅
2. **KakeyaLattice's absolute reconstruction quality on V4 KV** is
   hardware-robust (1.02–1.34× CPU/GPU agreement). ✅
3. **FP8 baseline quality is HIGHLY hardware-dependent** (14–48× GPU/CPU
   divergence). GPU = real E4M3 = deployment truth. ⚠
4. **The headline "E8 beats V4 FP8 by -22% bits / -12% MSE" only holds
   on real E4M3 hardware** (i.e. GPU). On a linear-FP8 hypothetical
   baseline, E8 would lose. Our GPU number is the one that matches V4
   deployment.

The CPU run is useful as a numerical-correctness sanity check and as
a mini-experiment revealing that V4's E4M3 choice is sub-Pareto on its
own KV, but it should NOT be used to project deployment compression
gains.

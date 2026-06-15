# Compression-quality sweep — KakeyaLattice on H200 + Qwen3-4B (2026-06-15)

Follow-up to `RESULTS.md` (which proved *real* 1.94× byte savings). This sweep
measures the codec's **reconstruction quality** across operating points, using
`benchmarks/hf_cache_demo/run_hf_cache_benchmark.py` with the
reconstruction-probe `KakeyaLatticeCache` against a bf16 `DynamicCache`
reference.

## Setup
- GPU **NVIDIA H200**, torch 2.12.0+cu130, transformers 5.12.0.
- Model **Qwen/Qwen3-4B** (36 layers, head_dim 128, 8 KV heads), bf16.
- ctx_len 2048, n_passages 4, n_eval 32, greedy decode.
- Variants: `e8_q38`, `e8_q37` (iso-bit to d4_q38), `e8_q10`, `d4_q38`.

## Part A — synthetic GPU smoke (`benchmarks/e8_parity_and_smoke.py`)
- **Frozen sha256 parity PASSES bit-for-bit on the H200** for v1.4 (D4) and v1.5
  (E8) at Q ∈ {4,10,38,152} → the deployed numbers reproduce exactly on this GPU.
- D4/E8 refactor bit-identity OK; all bit-rate formulas consistent.
- **E8 shaping gain over D4 at iso-bit** (random Gaussian KV, D=128):
  | match point | bits | D4 rel-MSE | E8 rel-MSE | Δ (dB) |
  |---|---:|---:|---:|---:|
  | Q≈4   | 416/400 | 5.49e-2 | 6.45e-2 | −0.70 |
  | Q≈10  | 576 | 8.77e-3 | 7.17e-3 | **+0.88** |
  | Q≈38  | 832 | 6.08e-4 | 4.24e-4 | **+1.57** |
  | Q≈152 | 1088 | 3.80e-5 | 2.55e-5 | **+1.74** |
  (E8 wins everywhere except the extreme Q≈4 budget, where iso-bit matching forces
  E8 down to Q=3.)

## Part B — real-model quality sweep (Qwen3-4B)

bf16 reference mean ppl = **1.1558** (note: the demo's sample passages are highly
repetitive, so ppl sits near 1 and |Δppl| is **not** a discriminating metric here;
**rel-MSE(K0)** is the meaningful quality signal).

| codec | codec bits/vec | bit-rate CR vs bf16 | ppl | \|Δppl\| | rel-MSE(K0) | decode× |
|---|---:|---:|---:|---:|---:|---:|
| bf16 baseline | 2048 | 1.00× | 1.156 | — | — | 1.00 |
| e8_q38 | 848 | 2.42× | 1.156 | 0.03% | **9.22e-05** | 1.04× |
| e8_q37 | 832 | 2.46× | 1.155 | 0.04% | 9.73e-05 | 1.05× |
| e8_q10 | 608 | 3.37× | 1.156 | 0.03% | 1.34e-03 | 1.10× |
| d4_q38 | 832 | 2.46× | 1.155 | 0.05% | 1.40e-04 | 0.80× |

### Findings
- **E8 beats D4 at iso-bit on real KV:** at 832 bits, E8 (Q=37) rel-MSE
  9.73e-05 vs D4 (Q=38) 1.40e-04 → **~1.44× lower error** (≈ +1.6 dB), matching
  the synthetic shaping-gain prediction.
- **Quality is essentially lossless at Q=38:** rel-MSE ~9e-5, |Δppl| ~0.03%.
- **Graceful degradation:** dropping to E8 Q=10 (3.37× *bit-rate* CR) raises
  rel-MSE to 1.3e-3 but |Δppl| stays ~0.03% on this corpus.
- These are **bit-rate** compression ratios (codec bits ÷ 2048). The *real*
  stored-byte ratio with int8 `KakeyaLatticeQuantizedCache` is **1.94×** (see
  `RESULTS.md`); the 2.4–3.4× bit-rate figures require the bit-packed v1.6
  storage to be realized as real HBM savings.

## Reproduce
```bash
python3 benchmarks/e8_parity_and_smoke.py
python3 benchmarks/hf_cache_demo/run_hf_cache_benchmark.py \
    --model Qwen/Qwen3-4B --ctx-len 2048 --n-passages 4 --n-eval 32 \
    --variants e8_q38 e8_q37 e8_q10 d4_q38 --device cuda
```
Raw output: `qwen3_4b_quality.json` (this directory).

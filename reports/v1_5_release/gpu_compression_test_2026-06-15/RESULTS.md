# Real-compression verification — KakeyaLattice on NVIDIA H200 (2026-06-15)

**Question:** does the current compression algorithm have a *real* compression
effect (fewer real bytes), or is it only a reconstruction-quality probe?

**Verdict: YES — real compression is confirmed.** `KakeyaLatticeQuantizedCache`
holds **1.94× fewer real bytes** than `transformers.DynamicCache` during a live
`model.generate()` on Qwen3-4B, with the compressed state genuinely resident on
the GPU and generated text remaining coherent.

## Environment
- GPU: **NVIDIA H200** (143 GB), driver 595.71.05
- torch **2.12.0+cu130** (CUDA 13.0), transformers **5.12.0**, Python 3.12.3
- Code: latest `main` (HEAD `be1586b`), synced to the box and `pip install -e`'d.

## Test 1 — codec/byte-accounting unit tests (`hf/test_quantized_cache.py`)
`13 passed`. Key facts proven:
- **Codec equivalence (bit-identical):** `encode_to_indices` → store int → 
  `decode_from_indices` equals `codec.roundtrip(x)` exactly
  (`max|Δ| == 0.0`) for D4 and E8 at Q ∈ {10, 22, 38}. ⇒ int storage adds **zero**
  extra error vs the reconstruction-only path.
- **Storage type selection:** D4 Q=38 → int8; E8 Q=38 → int8 (half-int doubled to
  ±76); E8 Q=76 → int16 fallback (above the Q≤63 int8 ceiling).
- **Byte counts:** head_dim=128 → **1.94×**, head_dim=64 → **1.88×**.

## Test 2 — end-to-end on a live model (`verify_real_compression.py`)
Identical 46-token prompt, greedy decode, 128 new tokens (final seq = 174), run
once with each cache:

| cache | persistent KV bytes | size | ratio |
|---|---:|---:|---:|
| `transformers.DynamicCache` (bf16) | 25,509,888 | 24.33 MiB | 1.00× |
| `KakeyaLatticeQuantizedCache` (E8, Q=38, int8) | 13,153,536 | 12.54 MiB | **1.939×** |

- **Bytes saved: 48.4%.** Measured ratio (1.939×) == theoretical per-vector
  ceiling at D=128 ((2·128)/(128+4)).
- `storage_devices = {cuda:0}` → the int8 indices + fp16 norm/qmax live **on the
  GPU**, not copied to host.
- `codec_fired = 4608` (36 layers × 128 reads), `skip_fired = 0` → the codec ran
  on every layer/step; no silent fallback.
- **Coherence:** the compressed-cache generation is fluent and tracks the bf16
  baseline almost verbatim for the first ~60 tokens before minor wording
  divergence — i.e. compression is *usable*, not degenerate.
- Decode time: bf16 8.53 s vs int8 10.82 s for 128 tokens (the reference Python
  codec is unoptimized; this is a quality/ò-bytes probe, not a latency-tuned path).

## Honest scope / caveat
The **real, measured** storage ratio at the int8 operating point is **1.94×**.
The README's higher **2.4×–2.8×** figure is the codec's *bit-rate* ceiling (a
Q=38 coordinate needs ~6.3 bits but int8 uses 8); closing that gap requires the
bit-packed v1.6 storage and is **not** realized in the current code. So: real
compression — **yes, 1.94×, lossless-relative to the codec's own
reconstruction**; the 2.4×+ headline is still aspirational.

## Reproduce
```bash
# on the GPU box, repo installed via: pip install -e kakeyalattice
python3 -m pytest kakeyalattice/python/kakeyalattice/hf/test_quantized_cache.py
python3 benchmarks/hf_cache_demo/verify_real_compression.py \
    --model Qwen/Qwen3-4B --variant e8 --q-range 38 --max-new 128
```
Raw machine output: `qwen3_4b_real_compression.json` (this directory).

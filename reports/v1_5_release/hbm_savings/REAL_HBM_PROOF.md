# Real HBM Savings — KakeyaLatticeQuantizedCache

**Date**: 2026-06-15
**Measurement**: pure tensor byte accounting via
`tensor.element_size() * tensor.numel()`, summed over all layers of
the cache after the same sequence of `update()` calls. This equals
on-device byte count regardless of CPU/GPU.

## Context — why this file exists

The published `KakeyaLatticeCache` (PyPI v1.5.0) round-trips K/V
through the codec but stores the **reconstructed bf16 tensor** in
DynamicCache's underlying storage. It therefore proves codec
**reconstruction quality**, not HBM savings. The 2026-06-15 audit
in `docs/GEO_AUDIT_2026-04-27.md`'s spirit pointed this out.

`KakeyaLatticeQuantizedCache` (this work) closes the gap by storing
**lattice indices (int8 / int16) + per-vector fp16 norm + fp16 qmax**
between calls, so the persistent KV state between attention reads
genuinely uses fewer bytes.

## Measurement protocol

```python
B, NKV, S, D = 1, 8, 2048, 128
NUM_LAYERS = 36  # Qwen3-4B-like
DTYPE = torch.bfloat16

dyn = DynamicCache()
qc  = KakeyaLatticeQuantizedCache(
    variant="e8", q_range=38,
    num_hidden_layers=NUM_LAYERS, head_dim=D,
    device="cpu", out_dtype=DTYPE,
)
for li in range(NUM_LAYERS):
    k, v = make_random_layer_kv()   # bf16 [1,8,2048,128]
    dyn.update(k, v, layer_idx=li)
    qc.update(k, v, layer_idx=li)

# Sum tensor bytes:
dyn_bytes = sum(t.element_size() * t.numel()
                for layer in dyn.layers
                for t in (layer.keys, layer.values))
qc_bytes  = qc.kv_storage_bytes()
```

Reproducer in `kakeyalattice/python/kakeyalattice/hf/test_quantized_cache.py`
(test class `TestRealHBMSavings`).

## Result

| cache | bytes held after 36 layers × 2048 tokens × bf16 | vs DynamicCache |
| --- | ---: | ---: |
| `transformers.DynamicCache` (bf16) | 301,989,888  (288.00 MiB) | baseline |
| `KakeyaLatticeCache` (roundtrip, bf16-stored) | 301,989,888  (288.00 MiB) | **0% savings** |
| `KakeyaLatticeQuantizedCache` (int8 indices, NEW) | 155,713,536  (148.50 MiB) | **−48.4% / 1.94× compression** |

Per-vector arithmetic (sanity check):

- bf16 per K- or V-vector at D=128:  `D * 2 = 256` bytes
- int8 per K- or V-vector:           `D * 1 + 2 (fp16 norm) + 2 (fp16 qmax) = 132` bytes
- Expected ratio:                    `256 / 132 = 1.939×`
- Measured ratio:                    `1.939×`  ✓ (agreement to 3 decimal places)

## How this differs from the codec's bit-rate ratio

The codec's bit-rate ratio at E8 Q=38 (head_dim=128) is **2.42×**
(see `kakeyalattice/python/kakeyalattice/v1_5_kakeya_zamir_e8_gpu.py`).
The real HBM ratio measured here is **1.94×**. The gap (1.94× < 2.42×)
is the int8-vs-6.3-bit overhead: a Q=38 value only needs
`log2(2·38+1) ≈ 6.27` bits, but int8 storage uses 8 bits per
coordinate.

Closing the gap requires bit-packing the int stream down to a packed
6-bit (or 6.3-bit) representation. This is deferred to v1.6 in
favour of shipping the simpler, dependency-free int8 implementation
now. The bit-packed v1.6 should reach the codec's full 2.42× HBM
ratio at the cost of one extra pack/unpack kernel per attention
read.

## Operating-point coverage

| variant | q_range | int storage | real HBM ratio at D=128 | usable? |
| --- | ---: | --- | ---: | --- |
| D4 | 10  | int8  | 1.94× | yes |
| D4 | 38  | int8  | 1.94× | yes (recommended) |
| D4 | 152 | int8  | 1.94× | yes |
| E8 | 10  | int8  | 1.94× | yes |
| E8 | 38  | int8  | 1.94× | yes (recommended) |
| E8 | 63  | int8  | 1.94× | yes (E8 int8 ceiling) |
| E8 | 76  | int16 | **0.98× — no win** | use `KakeyaLatticeCache` instead |
| E8 | 152 | int16 | **0.98× — no win** | use `KakeyaLatticeCache` for near-lossless |

E8's int8 ceiling is Q=63 (not 127) because E8 outputs half-integer
values: the encode path stores `int8(2 * q_lat)`, which doubles the
needed range. D4 is integer-valued so its int8 ceiling is the full
Q=127.

## Bit-identical codec equivalence

The encode→store-as-int→decode pipeline is **bit-identical** to the
existing `codec.roundtrip(x)` on both D4 and E8 across q_range ∈
{10, 22, 38}: `max(|x_rt - x_qc|) == 0.0` in all 6 cases. Therefore
swapping `KakeyaLatticeCache` for `KakeyaLatticeQuantizedCache`
preserves the codec's reconstruction error exactly — the |Δppl|
properties measured in
`reports/v1_4_release/kv_128k_isoppl_n8/V14_VS_TQ_ISOPPL_REPORT.md`
carry over without change.

## What the published claims should now say

| claim | before this PR | after this PR |
| --- | --- | --- |
| "2.4×–2.8× KV cache compression" | misleading — codec bit-rate, not measured HBM | **OK if using `KakeyaLatticeQuantizedCache`**: real ~1.94× at the int8 operating points; the headline 2.4× requires bit-packed v1.6. |
| "Drop-in DynamicCache subclass" | true | still true (both `KakeyaLatticeCache` and `KakeyaLatticeQuantizedCache` subclass `DynamicCache`) |
| "Real HBM savings" | false | **true** for `KakeyaLatticeQuantizedCache` at Q ≤ 63 (E8) / Q ≤ 127 (D4) |
| "Production-grade vLLM integration" | false | still false — vLLM-native paged-attention integration is the v1.6 work item |

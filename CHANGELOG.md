# Changelog

## v1.6.0 — 2026-06-15

**fix codec.roundtrip bug — contiguous, directly-SDPA-feedable K/V decode.**

### Fixed
- **`KakeyaLatticeQuantizedCache` storage layout.** The cache previously held a
  growing **Python list** of per-`update()` `(q_lat, norms, qmax)` chunks per
  layer and re-`torch.cat`-ed the *entire* history on every decode step before
  re-decoding it — **O(N²)** over a length-N generation, and not a single
  contiguous buffer. It now keeps **one contiguous tensor per layer** per
  component, grown by a single `cat` per `update()` (like
  `transformers.DynamicCache`), and the decoded K/V returned to attention are
  `.contiguous()` — i.e. a batched, contiguous, **directly SDPA-/FlashAttention-
  feedable** KV layout. Amortized-linear instead of quadratic.

### Added
- **Bit-packing (`kakeyalattice.hf.bitpack`).** GPU-native fixed-width packer plus
  exact-ceiling **D4 and E8 block codes** (D4 even-sum reduction; E8 doubled-int
  same-parity + sum-mod-4 reduction), with a small lossless **exception
  side-channel** for the ~1% of blocks the codec's defensive clamp pushes out of
  the lattice. Realises the codec's bit-rate ceiling as real bytes.
- **End-to-end packed caches (`kakeyalattice.hf.packed_cache`).**
  `KakeyaLatticePackedCache` (D4/E8) and `TurboQuantPackedCache` — contiguous,
  decode-on-read, with `kv_storage_bytes()` reporting the real bit-packed
  footprint. Measured on Qwen3-4B / H200: **D4 Q=38 = 2.46×, E8 Q=38 = 2.37×**
  real HBM (vs the int8 path's 1.94×); pack→unpack verified lossless.
- **`TurboQuantCodec` (`kakeyalattice.hf.turboquant`).** The scalar-quantise
  baseline, parameterised by bits/coordinate, for apples-to-apples comparison.
- **Real-byte iso-ppl comparison harness** (`benchmarks/bitpack_vs_tq/`) +
  report (`reports/v1_5_release/bitpack_vs_tq_2026-06-15/`). On Qwen3-4B, at the
  1–2% |Δppl| band the real-byte advantage over TurboQuant is **E8 +7.7%,
  D4 +5.0%** (vs the bit-rate table's +26.9%); E8 wins at every threshold tested.

### Notes
- Reconstruction is byte-for-byte identical to the int8 path (packing is a
  lossless re-encoding), so all previously reported quality numbers carry over.
- Single-model scope for the new comparison: Qwen3-4B, head_dim=128.

## v1.5.0
- E8 nested-lattice codec (`V15KakeyaZamirE8GPU`); `KakeyaLatticeCache` and
  `KakeyaLatticeQuantizedCache` (int8, real ~1.94× HBM). PyPI package.

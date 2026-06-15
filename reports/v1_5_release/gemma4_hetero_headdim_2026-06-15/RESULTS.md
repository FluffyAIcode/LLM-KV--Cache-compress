# Gemma-4-26B drop-in fix — per-layer head_dim + mask sizing (2026-06-15)

## Symptom
Using the caches drop-in on `google/gemma-4-26B-A4B-it` raised
`AssertionError: expected last dim 256, got 512`, and (after that was fixed) a
`CUDA error: device-side assert triggered` during multi-step decode.

## Root causes (two distinct bugs)
1. **Heterogeneous per-layer head_dim.** Gemma-4's text decoder mixes
   `sliding_attention` layers (`head_dim=256`) and `full_attention` layers
   (`global_head_dim=512`; layer idx 5,11,17,23,29 of 30). The caches built one
   codec from a single `config.head_dim=256`, so a 512-dim full-attention layer
   tripped `assert x.shape[-1] == codec.D_shape`.
2. **Wrong attention-mask sizes.** transformers-5's container
   `DynamicCache.get_mask_sizes(query_length, layer_idx)` delegates to
   `self.layers[layer_idx]`. The int-storage caches keep their compressed state
   *outside* `self.layers`, so the parent fell through to `(query_length, 0)` —
   reporting `kv_length=query_length` instead of the true cache length. During
   multi-step decode this corrupted Gemma-4's sliding-window + multimodal
   `blockwise_overlay` mask (`block_sequence_ids[batch, q_idx]` out of bounds) →
   device-side assert.

## Fixes
1. **Lazy per-layer codec keyed by observed head_dim** (`quantized_cache.py`,
   `cache.py`, `packed_cache.py`): each layer's codec is built on first
   `update()` from `key_states.shape[-1]`, validated (power-of-2 & divisible by
   the lattice block dim), cached; incompatible dims raise (strict) or fall back
   to raw bf16 (non-strict). Fully drop-in for any heterogeneous-head_dim model.
2. **`get_mask_sizes` override** (`quantized_cache.py`, `packed_cache.py`):
   reports `(get_seq_length(layer_idx) + query_length, 0)` from the cache's own
   buffers. Exact full-attention sizing; correct for sequences within the model's
   sliding window (Gemma-4 = 1024). (The roundtrip `KakeyaLatticeCache` stores
   into `self.layers` via the parent, so it was already correct.)

## Verification (NVIDIA H200)
- **Unit:** new `test_hetero_headdim.py` — two layers at 256 and 512 succeed for
  all int-storage caches; strict raise / non-strict raw-fallback; `get_mask_sizes`
  reports true length. Full hf suite: **90 passed**.
- **End-to-end on Gemma-4-26B** (`benchmarks/bitpack_vs_tq/gemma4_hetero_check.py`):

  | | result |
  |---|---|
  | per-layer K head_dim | `[256×5, 512, 256×5, 512, …]` (distinct {256, 512}) |
  | packed generate | **OK, no assertion** (E8 Q=38) |
  | per-layer codec D_shape | 256 for sliding, **512 for full** (5,11,17,23,29) |
  | real packed CR vs bf16 | **2.443×** |
  | pack→unpack lossless | **True** |
  | output | coherent, ~matches bf16 ("Lattice quantization is a non-perturbative numerical method used in quantum field theory…") |

## Caveat / follow-up
`get_mask_sizes` uses full-attention sizing. For sequences **longer than the
sliding window (1024)**, sliding-window layers would attend beyond their window
(a quality deviation, not a crash). Proper per-layer sliding-window eviction on
the compressed buffers (config-aware) is a clean follow-up if long-context
Gemma-4 fidelity is needed.

## Files
- Fix: `kakeyalattice/python/kakeyalattice/hf/{quantized_cache,cache,packed_cache}.py`
- Tests: `kakeyalattice/python/kakeyalattice/hf/test_hetero_headdim.py`
- Repro/verify: `benchmarks/bitpack_vs_tq/gemma4_hetero_check.py`; raw `gemma4_check.json`

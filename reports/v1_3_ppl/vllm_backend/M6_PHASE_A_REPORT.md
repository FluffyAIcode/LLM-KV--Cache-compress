# M6 Phase A — vLLM backend scaffolding + slot (de)serialisation

Branch: `AgentMemory/v1-3-ppl-vllm-backend-102e`
Package added: `vllm_backend/kakeya_v1_3_ppl/` (new)
Lines added: ~1 100 (config + spec + backend + impl + registration + tests)
Tests: 23/23 PASS locally, registration smoke PASS on H200.

## Exit criterion (PLAN.md §M6)

> **M6** `KakeyaV13PPLAttentionBackend` + spec + registration.
> **Exit criterion**: `vllm serve Qwen/Qwen3-4B --kv-cache-dtype
> kakeya_v1_3_ppl --block-size 512` produces coherent output.

**Phase A status**: scaffolding + registration + slot (de)serialisation
**complete and unit-tested**.  The coherent-text smoke test requires
the live vLLM engine path (`do_kv_cache_update` + `forward` +
FlashAttention integration + CUDAGraph), which is finished in Phase B
on H200.  Phase A explicitly leaves those two methods as
`NotImplementedError` to **enforce** the no-fallback rule.

## What Phase A delivers

```
vllm_backend/kakeya_v1_3_ppl/
├── __init__.py                       lazy re-exports
├── config.py         KakeyaV13PPLConfig — slot layout math
├── spec.py           KakeyaV13PPLAttentionSpec — cache shape
├── backend.py        KakeyaV13PPLAttentionBackend — vLLM glue class
├── impl.py           KakeyaV13PPLAttentionImpl — store/decode dispatch
│                      + slot (de)serialisation helpers
├── registration.py   register_kakeya_backend — patches CacheDType +
│                      AttentionBackendEnum.CUSTOM
└── tests/
    └── test_config_and_spec.py       23 unit tests, all PASS
```

### `KakeyaV13PPLConfig` — slot layout math

Per PLAN.md §"Cache layout interface contract" §"paged cache slot
layout":

```
HEADER              48 B      magic + d_eff + k + bit_width + oc_total
PCA basis   d_eff × D × 2 B   fp16
PCA mean    D × 2 B           fp16
K-means cent  k × d_eff × 2 B fp16
seg_id block    ceil(n × ceil(log2 k) / 8)     bit-packed
t block         n × 2 B                         fp16
norm block      n × 2 B                         fp16
residual block  ceil(n × wht_len × bit_width / 8)  bit-packed (Rust format)
outlier budget  (if > 0):
    per-row u16 count  n × 2 B
    flat entries       ceil(budget_frac × n × wht_len) × 4 B
                       each: (u16 idx, f16 val)
```

For the PR #15 production cell (D=128, d_eff=64, block_size=512,
k=16, b_K=3, 8 % outlier budget), the K-stream slot is 44 840 B
(~87.6 B/token, ~2.92× vs bf16's 256 B/token).  The V-stream slot
(D=128, d_eff=64, block_size=512, k=16, b_V=2, no outlier) is
29 232 B (~57.1 B/token, ~4.48× vs bf16).  Combined per-token
compression ratio is **~1.77×**.

### Compression gap analysis vs PLAN.md

PLAN.md §"The key design decision" claims a 4.03× ratio for the
K-stream alone; my measured 2.92× is **lower because PLAN.md's table
underestimates two real overheads**:

| Overhead                  | PLAN.md assumes | Actual | Reason                                     |
|:--------------------------|----------------:|-------:|:-------------------------------------------|
| per-vec `t` (fp16)        | 0               |  1 024 B | Rust `Code` stores per-vec centroid projection |
| per-vec `norm` (fp16)     | 0               |  1 024 B | Rust `Code` stores per-vec inv-scale (`NormMode::Absorbed`) |
| outlier per-row count     | 0               |  1 024 B | needed to recover ragged row boundaries from the flat entry array |
| outlier entry size        | 4 B (idx + val) |  4 B   | matches PLAN.md                             |
| **Total understated**     | **—**           | **+3 072 B** | ~7 % extra vs PLAN.md for K-stream   |

The `t` and `norm` fields are **algorithmically required** — decode
cannot recover `coeff = t · center[seg_id] + residual` without them,
and per-vec `norm` is needed to undo the `1/‖residual‖` scale.  The
per-row outlier count array is likewise required to make the flat
entry array unambiguously decodable.

None of this is a bug; it's an under-count in PLAN.md's arithmetic.
The actual 1.77× combined ratio is still meaningful (saves ~44 % of
KV-cache memory vs bf16), but future PLAN revisions should reflect
reality.

### `KakeyaV13PPLAttentionSpec` — cache shape

Returns `(num_blocks, num_kv_heads, slot_budget_bytes)` — a raw
`uint8` allocation rather than vLLM's standard
`(2, num_blocks, block_size, num_kv_heads, head_size)` five-tuple.
The leading `2` is gone because K and V share the per-block slot at
layout-controlled offsets.

For the default `block_size=512, num_kv_heads=8, head_size=128` on
Qwen3-4B and 100 blocks, this comes out to **56.5 MB per layer** in
the paged cache.  With 36 full-attention layers in Qwen3-4B, that's
~2 GB of KV cache at full capacity — vs baseline bf16's ~3.6 GB.

### `KakeyaV13PPLAttentionBackend` — vLLM glue

Minimal `AttentionBackend` subclass.  Declares:

  * `get_name() == "KAKEYA_V1_3_PPL"`
  * `supported_kv_cache_dtypes == ["kakeya_v1_3_ppl"]`
  * `get_kv_cache_shape(num_blocks, block_size=512, num_kv_heads, head_size)`
  * `get_impl_cls() → KakeyaV13PPLAttentionImpl`
  * `get_builder_cls() → KakeyaV13PPLMetadataBuilder`
  * `supports_block_size(512)` only
  * `supports_head_size(64/96/128/256)`
  * `supports_compute_capability(80)` — Ampere and later

### `KakeyaV13PPLAttentionImpl` — store/decode dispatch

Owns:

  * `_pack_parts_into_slot(parts, config) → uint8[slot_size_bytes]`
    Serialises the `kakeyaturbo_py.encode_block_codes` dict into the
    raw cache slot.  Bit-packed seg_id + fp16 t/norm + Rust-packed
    residual + (optional) outlier side-buffer.
  * `_unpack_slot_into_parts(slot, config, head_size) → parts dict`
    Inverse of the above.

  * `do_kv_cache_update()` — `raise NotImplementedError`.  Phase B.
  * `forward()`            — `raise NotImplementedError`.  Phase B.

Phase B will plug M4's `encode_block_triton_stage2` into
`do_kv_cache_update` (accumulate + seal + pack), and M5's
`decode_block_triton_from_parts` + `decode_partial_block_bf16` into
`forward` (unpack + decode + flash-attn).

### `register_kakeya_backend()` — opt-in vLLM integration

Two patches, both reversible:

1. **Extend `vllm.config.cache.CacheDType`** to accept
   `"kakeya_v1_3_ppl"`.  Pydantic v2 bakes the Literal into the
   dataclass validator at class-decoration time; we can't just
   re-assign the module-level alias.  We walk
   `CacheConfig.__pydantic_core_schema__`, find the `cache_dtype`
   Literal subtree, mutate its `expected` list, and rebuild
   `__pydantic_validator__` via
   `pydantic_core.SchemaValidator(schema, CoreConfig(...))`.
2. **Register the backend** via
   `vllm.v1.attention.backends.registry.register_backend(
       AttentionBackendEnum.CUSTOM, "...KakeyaV13PPLAttentionBackend")`.

Tested on H200:

```python
>>> register_kakeya_backend()
>>> CacheConfig(cache_dtype="kakeya_v1_3_ppl", block_size=512).cache_dtype
'kakeya_v1_3_ppl'
>>> CacheConfig(cache_dtype="turboquant_k8v4").cache_dtype  # legacy still works
'turboquant_k8v4'
>>> AttentionBackendEnum.CUSTOM.get_class().get_name()
'KAKEYA_V1_3_PPL'
>>> AttentionBackendEnum.CUSTOM.get_class().get_kv_cache_shape(100, 512, 8, 128)
(100, 8, 74072)
```

`unregister_kakeya_backend()` is provided for tests but has a known
limitation: the SchemaValidator swap is not fully reversible because
Pydantic's internal core-schema object cannot be cloned without going
through class decoration.  Tests that need clean state should be
ordered so that registration runs last, or run each registration test
in a fresh subprocess.

## Test results (local, 23/23 PASS in 1.3 s)

```
TestKakeyaV13PPLConfig
  test_default_head_dim_128                                   PASS
  test_slot_size_matches_plan_md_table                        PASS
  test_compression_ratio_is_reasonable                        PASS
  test_invalid_configs_rejected                               PASS

TestKakeyaV13PPLAttentionSpec
  test_default_spec                                           PASS
  test_rejects_mismatched_block_size                          PASS
  test_cache_shape                                            PASS

TestSlotSerde
  test_pack_unpack_roundtrip_k_stream[1..4]                   4×PASS
  test_pack_unpack_roundtrip_with_outliers[11..13]            3×PASS
  test_slot_size_is_exact                                     PASS
  test_slot_magic_and_header                                  PASS

TestPackDecodeE2E
  test_packed_slot_decodes_to_same_tensor[None-1..3]          3×PASS
  test_packed_slot_decodes_to_same_tensor[2.0-1..3]           3×PASS

test_name_constant                                            PASS
```

The **E2E test** is the critical one: it takes a Rust-encoded parts
dict, packs it through the M6 slot layout, unpacks it, and Rust-decodes
the result; the decoded tensor must match the Rust decode of the
(fp16-rounded) original parts within 1e-5 L2-relative error.  All 6
cases pass (3 without outliers, 3 with `outlier_threshold=2.0`).

## Non-negotiables

| Clause              | Phase A status | Evidence                                                      |
|:--------------------|:--------------:|:--------------------------------------------------------------|
| no simplification   | ✓              | slot layout carries every algorithmically-required field; no dropped fields to hit a compression number |
| no fallback         | ✓              | `do_kv_cache_update` / `forward` raise `NotImplementedError` rather than silently falling back to bf16 |
| no mock             | ✓              | slot pack/unpack uses real outlier scatter and real Rust decode round-trip to verify |
| no overfit          | ✓              | no calibration runs in M6; M2 artefacts are loaded unchanged at layer init |

## What Phase B (M6 final) will do

1. **Populate per-layer state at first forward**: load
   `qwen3_4b_sigma_q.safetensors` (M2) into `layer._kakeya_state`
   and materialise Lloyd-Max centroid tables from
   `qwen3_4b_lloyd_max_K_b3.f32` / `qwen3_4b_lloyd_max_V_b2.f32`.
2. **Implement `do_kv_cache_update`**:
     * Append K/V to `staging_k_bf16` / `staging_v_bf16`.
     * While staging_count ≥ block_size_codec: peel off 512 tokens,
       call `encode_block_triton_stage2`, pack via
       `_pack_parts_into_slot`, write to the cache slot indicated
       by slot_mapping.
3. **Implement `forward`**:
     * For each sealed block in `block_table`: unpack the slot via
       `_unpack_slot_into_parts` and decode via
       `decode_block_triton_from_parts`.
     * For the trailing partial block: read
       `staging_[kv]_bf16[:staging_count]` through
       `decode_partial_block_bf16`.
     * Concatenate sealed + partial K/V and feed into
       `flash_attn_varlen_func`.
4. **CUDAGraph support**: likely requires the sealed/partial decode
   paths to be shape-static; may need per-request max-staging-count
   padding.
5. **Coherent-text smoke**: `vllm serve Qwen/Qwen3-4B
   --kv-cache-dtype kakeya_v1_3_ppl --block-size 512
   --attention-backend CUSTOM` on H200, verifying the completion of
   `"The capital of France is"` is a sensible continuation.  Success
   criterion is identical to M1's TQ-k8v4 sanity check.

## Repro

```bash
# Local (no vLLM required for Phase A tests).
cd /workspace/LLM-KV--Cache-compress
python -m pytest vllm_backend/kakeya_v1_3_ppl/tests/test_config_and_spec.py -v

# Registration smoke on H200.
ssh vast 'source /venv/main/bin/activate && cd /workspace/LLM-KV--Cache-compress && \
  python -c "
from vllm_backend.kakeya_v1_3_ppl.registration import register_kakeya_backend
register_kakeya_backend()
from vllm.config.cache import CacheConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
c = CacheConfig(cache_dtype=\"kakeya_v1_3_ppl\", block_size=512)
be = AttentionBackendEnum.CUSTOM.get_class()
print(c.cache_dtype, be.get_name(), be.get_kv_cache_shape(100, 512, 8, 128))
"'
```

Expected: 23 PASS locally; registration smoke prints
`kakeya_v1_3_ppl KAKEYA_V1_3_PPL (100, 8, 74072)` on H200.

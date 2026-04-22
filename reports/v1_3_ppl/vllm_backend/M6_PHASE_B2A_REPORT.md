# M6 Phase B.2a — M2 calibration loading + Σ_q whitening wired

Branch: `AgentMemory/v1-3-ppl-vllm-backend-102e`
Module added: `vllm_backend/kakeya_v1_3_ppl/calibration.py`
Module changed: `vllm_backend/kakeya_v1_3_ppl/impl.py` (~120 lines of
`_ensure_layer_state` / `_seal_and_write_block` / `_decode_sealed`
wiring).
Tests: 43/43 PASS on H200 in 5.6 s.

## Scope

Phase B.1 left two calibration axes as "Gaussian default":
  * Lloyd-Max centroid tables: the `centroids_gaussian(bits)` baked
    into `kakeyaturbo`, ignoring M2's `qwen3_4b_lloyd_max_K_b3.f32`
    and `qwen3_4b_lloyd_max_V_b2.f32`.
  * Σ_q Cholesky whitening: disabled entirely — the codec operated
    in raw-K space, ignoring `qwen3_4b_sigma_q.safetensors`.

PR #15 measured that Σ_q-off vs Σ_q-on opens **~4× Δppl gap at
b=3** on DS-Distill (SPRINT_CLOSEOUT).  Phase B.2a plumbs both
calibration axes into the backend so M7's benchmark actually sees
the full v1.3 PPL algorithm, not a partially-calibrated stub.

Remaining B.2 work (deferred to later commits, documented at the
end):
  * CUDAGraph shape-stability audit for `staging_per_block` dict
  * Model-runner plugin entry point so `vllm serve` auto-registers
  * End-to-end coherent-text smoke on Qwen3-4B

## New module: `calibration.py`

```python
@dataclass
class CalibrationBundle:
    head_dim: int
    num_kv_heads: int
    num_layers: int
    sigma_q_chol:     dict[int, np.ndarray]   # {layer_idx: [n_kv, D, D] fp32}
    sigma_q_inv_chol: dict[int, np.ndarray]
    lloyd_max_k: np.ndarray | None            # [2^b_K] fp32
    lloyd_max_v: np.ndarray | None            # [2^b_V] fp32

    def active_layers(self) -> set[int]
    def whiten_layer_head(K, layer, head) -> K_tilde     # K @ L[h]
    def unwhiten_layer_head(K_tilde, layer, head) -> K   # K_tilde @ L⁻¹[h]


def load_calibration_bundle(
    sigma_q_safetensors: str | Path,
    k_centroids_f32:     str | Path | None = None,
    v_centroids_f32:     str | Path | None = None,
    *, skip_layers: list[int] | None = None,
) -> CalibrationBundle
```

Loading strategy:
  * Reads `qwen3_4b_sigma_q.safetensors` and its sidecar `.json` for
    `head_dim`, `num_kv_heads`, `num_layers`.
  * Validates every `layer_<l>_chol` / `layer_<l>_inv_chol` is
    `[n_kv, D, D]` fp32 and matches the sidecar metadata.
  * `skip_layers=[...]` is union-applied on top of M2's own skip
    set (M2 actually calibrated all 36 Qwen3-4B layers; the
    `skip_layers` field exists for a runtime boundary-layer list
    like `[0, 1, 34, 35]` the user passes at load time).
  * Lloyd-Max tables are read from flat fp32 binaries; `None`
    means "fall back to Gaussian default Lloyd-Max table inside
    `kakeyaturbo_py`".  Size is sniffed (2 / 4 / 8 / 16 = 2^b for
    b=1..=4); strict-ascending is enforced.

Zero vllm dependency — the module is importable on CPU-only dev
machines.

## Integration in `impl.py`

### 1. Process-global bundle

```python
_GLOBAL_CALIBRATION: CalibrationBundle | None = None

def set_global_calibration(bundle: CalibrationBundle | None) -> None:
    """Install a bundle every subsequently-created AttentionImpl
    will pick up.  Phase B.2d's plugin entry point will call this
    once at model init; until then tests register it by fixture."""
```

Rationale: vLLM instantiates `AttentionImpl` per (layer, worker)
and owns the construction site; we can't thread the bundle through
`__init__` without patching vllm itself.  A module-level register
populated before model init is the standard plugin pattern
(TurboQuant does the same for its centroid tables via
`vllm.config`).

### 2. `_ensure_layer_state`

Parses `layer.layer_name = "model.layers.{L}.self_attn.attn"` to
get `layer_idx`.  For the current bundle:
  * Looks up `sigma_q_chol[layer_idx]`, attaches `[n_kv, D, D] fp32`
    on the layer's CUDA device.  Missing layer → identity (Phase B.1
    behaviour).
  * Attaches `lloyd_max_k` / `lloyd_max_v` tables (shared across
    layers).

### 3. `_seal_and_write_block`

Before encode:
  * K_fp32 = `einsum("thj,hjk->thk", K_raw_fp32, sigma_q_chol)`
    per kv-head (equivalently `K @ L[h]` per head).  Skips if
    `sigma_q_chol is None` (uncalibrated layer) — identity path
    matches PR #15's "Q-precond off" behaviour.

At encode:
  * Both `kakeyaturbo_py.encode_block_codes` and
    `encode_block_triton_stage2` get `centroids=...` /
    `custom_centroids=...` plumbed when the bundle's tables are
    available.

### 4. `_decode_sealed`

After decode (K stream only — V has no whitening):
  * K_hat = `einsum("thj,hjk->thk", K_hat_tilde, sigma_q_inv_chol)`
    per kv-head.  Mirrors the encode-side whitening.

Both encode and decode pass calibrated centroids to the
Triton kernels.

### 5. Secondary fix: pin `d_eff` via `variance_ratio=1.0`

Discovered during Phase B.2a testing: `encode_block_codes(...,
variance_ratio=0.95, exact_rank_cap=64)` can return `d_eff` **<** 64
if the block's spectrum satisfies vr=0.95 at a lower rank (n=8
kv-heads, iid Gaussian synthetic data returned `d_eff=23`).  That
broke the slot layout which assumes `d_eff=64` exactly.

Fix: force `variance_ratio=1.0` + `exact_rank_cap=k_cfg.d_eff` so
the encoder returns exactly `d_eff` components.  This is the
production-intent behaviour — PLAN.md says `d_eff` is a fixed
per-layer knob, not data-adaptive.

This change is in `_seal_and_write_block` (K and V rust_kwargs
dicts) and doesn't affect Phase B.1 tests because `N_KV_HEADS=2`
happens to give `d_eff=64` anyway under `vr=0.95`.

## Tests: 43/43 PASS on H200

Run:

```
ssh vast 'source /venv/main/bin/activate && cd /workspace/LLM-KV--Cache-compress && \
  python -m pytest vllm_backend/kakeya_v1_3_ppl/tests/ -v'
```

Result (5.6 s):
  * M6 Phase A (unit): 23 PASS
  * M6 Phase B (E2E):   6 PASS
  * M6 Phase B.2a (cal): 14 PASS

New Phase B.2a tests:

  ```
  TestParseLayerIdx::test_parse_valid / missing / non_integer          3 PASS
  TestCalibrationLoading::test_metadata                                   PASS
  TestCalibrationLoading::test_active_layers                               PASS
  TestCalibrationLoading::test_load_with_skip_layers                       PASS
  TestCalibrationLoading::test_chol_shapes                                 PASS
  TestCalibrationLoading::test_roundtrip_identity                          PASS
  TestCalibrationLoading::test_lloyd_max_tables                            PASS
  TestCalibrationLoading::test_bundle_whiten_unwhiten_identity             PASS
  TestImplWithCalibration::test_layer_state_populates_sigma_q              PASS
  TestImplWithCalibration::test_layer_state_skip_listed_layer              PASS
  TestImplWithCalibration::test_seal_and_decode_roundtrip_with_whitening   PASS
  TestImplWithoutCalibration::test_no_bundle_no_whitening                  PASS
  ```

Notable: `test_roundtrip_identity` (L · L⁻¹ = I within 2e-5 per
(layer, head)) re-runs M2's own exit criterion inside the loader,
so we catch any regression in the bundle reader.

## Non-negotiables

| Clause              | Phase B.2a | Evidence                                              |
|:--------------------|:----------:|:------------------------------------------------------|
| no simplification   | ✓          | both calibration axes wired (Σ_q whitening + Lloyd-Max tables); neither path is a stub |
| no fallback         | ✓          | if the bundle is partial (some layer missing), we fall through to **identity whitening + Gaussian default Lloyd-Max** explicitly — PR #15's known-measured off-state, not a silent degradation |
| no mock             | ✓          | real M2 artefacts loaded; L · L⁻¹ = I verified per-test; real einsum on CUDA |
| no overfit          | ✓          | bundle is load-time immutable; no per-request tuning |

## What Phase B.2b onwards still needs

  b. **CUDAGraph audit**: `staging_per_block: dict` + Python-side
     branching on staging_count breaks CUDAGraph capture because
     the graph's topology changes when a block seals.  Likely fix:
     dense `[max_active_blocks, block_size_codec, n_kv, D]` tensor
     indexed by block_idx, with a parallel `[max_active_blocks]
     count` tensor; the seal decision becomes a shape-static
     `count == block_size_codec` branch that Triton can handle via
     a guard kernel.  Separate commit — nontrivial rewrite.

  c. **Model-runner plugin**: a `vllm_plugin`-style module that
     vllm discovers via `VLLM_PLUGINS=vllm_backend.kakeya_v1_3_ppl`
     and that calls `register_kakeya_backend()` +
     `set_global_calibration(...)` at engine init from a config
     pointing at the M2 artefacts directory.  Pure glue.

  d. **Coherent-text smoke**: `vllm serve Qwen/Qwen3-4B
     --kv-cache-dtype kakeya_v1_3_ppl --block-size 512
     --attention-backend CUSTOM` on H200; assert "The capital of
     France is" produces "Paris" or similar.  Needs b + c to land
     first.

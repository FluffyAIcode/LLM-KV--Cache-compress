# Stage 1 — DeepSeek-V4 (Flash/Pro) live-vLLM evaluation

**Status at 2026-04-25T01:10Z**: code scaffold + unit tests only.
Execution pending hardware provisioning; see
[HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md).

## Purpose

Measure `v1.5 E8` (and `v1.4 D4`, TurboQuant) compression on real
DeepSeek-V4-Flash or V4-Pro KV cache, using the live vLLM attention
kernel. Replaces the Stage 0.5 architectural probe's random-init
weights with **actual trained V4 attention + compressor + MoE
weights** producing real KV under real inference.

This is the companion to [`reports/v1_5_release/dsv4_stage0_5/`](../dsv4_stage0_5/),
which ran the pure-PyTorch port with random weights and produced
our first E8-vs-FP8 headline numbers (0.82–0.87× rel-MSE at 78% of
the bits).

## What's scaffolded

| file | lines | purpose |
| --- | --- | --- |
| `vllm_backend/kakeya_v1_4_snapshot/dsv4_snapshot_hook.py` | ~230 | monkey-patch `DeepseekV4Attention.forward`, intercepting the single 512-dim shared-KV latent between `fused_wqa_wkv` and the fused attention op |
| `benchmarks/dsv4_stage1/test_dsv4_snapshot_hook.py` | ~230 | 8 CPU-only unit tests covering three-phase semantics (capture / replace / inforward + skip), layer-id extraction, idempotency, MTP-layer id mapping |
| `benchmarks/dsv4_stage1/rigorous_eval_dsv4.py` | ~130 | thin wrapper that installs the DSV4 hook and delegates to `rigorous_eval.py` with one V4-specific flag `--kv-stream-filter` for per-stream (SWA / c4a / c128a) rollup |
| `reports/v1_5_release/dsv4_stage1/HARDWARE_REQUIREMENTS.md` | — | minimum hardware + exact vLLM commands + expected deliverables |
| `reports/v1_5_release/dsv4_stage1/README.md` | (this file) | — |

## Validation already in place

1. **Hook target verified** against vLLM PR #40760 at commit
   `3602f14f0e146b234be911d916e381b4e6a4dc0c`:
   - `vllm.model_executor.models.deepseek_v4.DeepseekV4Attention` —
     correct class, correct module path
   - `forward(self, positions, hidden_states, llama_4_scaling=None)` —
     correct 3-arg signature (different from V3)
   - Interception point: between lines 278–279
     (`qr_kv, _ = self.fused_wqa_wkv(hidden_states); qr, kv = qr_kv.split(...)`)
     and line 291 (`torch.ops.vllm.deepseek_v4_attention(...)`)
   - Self attributes read by our hook (`q_lora_rank`, `head_dim`,
     `n_local_heads`, `n_local_groups`, `nope_head_dim`,
     `rope_head_dim`, `padded_heads`, `rotary_emb.cos_sin_cache`,
     `wo_a.weight`, `wo_a.weight_scale_inv`, `wo_b`, `_einsum_recipe`,
     `_tma_aligned_scales`, `layer_name`) — all confirmed present in
     the `DeepseekV4Attention` class definition
2. **CPU unit tests pass** (8/8) without requiring vLLM V4 wheel,
   CUDA, or the 158 GB DeepSeek-V4-Flash checkpoint. They use a
   stand-in `_FakeDSV4Attention` to exercise the three-phase
   capture/replace/inforward logic on CPU tensors.
3. **Idempotency** verified: calling `install_dsv4_snapshot_patch()`
   twice is safe (the class-level `_kk_snapshot_patched` sentinel
   short-circuits).
4. **Graceful degradation**: on non-V4 vLLM installs,
   `install_all_snapshot_patches_dsv4_aware()` logs a warning and
   continues with only the existing Qwen3/Qwen2/Gemma4/GLM patches.

## What's NOT yet done

1. **Live wheel integration test**: the hook compiles but has not
   been exercised against a real `vllm==0.19.x + deepseekv4-cu130`
   wheel. The `test_patched_attribute_marker` test in the unit suite
   will validate live-wheel idempotency *once* a V4-capable vLLM is
   installed on any machine (test skips gracefully otherwise).
2. **End-to-end rigorous_eval run**: requires hardware listed in
   `HARDWARE_REQUIREMENTS.md`.
3. **FINDINGS.md report**: will be generated when Stage 1 executes.
4. **KV-stream-filter downstream aggregation**: we've added the
   flag and the HookState side-channel attribute but the actual
   per-stream MSE rollup in `rigorous_eval.py` is a TODO — it needs
   a small patch to read `HookState.dsv4_kv_stream_filter` and group
   `per_layer_K_mse` by the `compress_ratio` attribute of each layer
   before the final aggregate. The change is ≤ 20 lines; will be
   made at the time we actually have V4 data to aggregate.

## Decision log

**Why we hook `DeepseekV4Attention.forward` and not inside
`mla_attn.attention_impl`**: the latter is decorated as a custom op
via `torch.ops.vllm.deepseek_v4_attention`, which runs as compiled
CUDA code and is not Python-patchable without dropping into
`torch.library`. The outer forward is pure Python and is the one
place where the KV latent is still a PyTorch tensor before entering
the fused C++ path.

**Why shared-latent K=V**: V4 is a shared-KV MLA variant; the 512-dim
latent produced by `fused_wqa_wkv` is used as both key and value
downstream (after internal RoPE+FP8 projection and RMSNorm). We
feed the same tensor to both `K` and `V` slots of the shared
`_snapshot_capture_replace` helper; the codec result is applied
back to a single `kv` tensor. See the shared-latent test
(`test_capture_records_kv_latent`) for the invariant check.

**Why `--kv-stream-filter` is a post-hoc filter, not a pre-hoc
split**: the hook sees one latent per token per layer, not three.
The "stream" (SWA vs c4a-pool vs c128a-pool) is determined
downstream by each layer's `compress_ratio` attribute. We aggregate
per-layer MSE across all layers, then optionally filter by
`compress_ratio` at report time. This is consistent with how
Stage 0.5 exposed the three streams.

## Running the unit tests (any machine)

```bash
cd benchmarks/dsv4_stage1
pytest test_dsv4_snapshot_hook.py -v
```

Expected: `8 passed, 1 skipped` (the `test_patched_attribute_marker`
test skips when vLLM V4 isn't installed).

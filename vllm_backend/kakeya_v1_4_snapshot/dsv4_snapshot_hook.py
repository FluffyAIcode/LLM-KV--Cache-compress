"""Stage 1 snapshot hook for DeepSeek-V4 (Flash / Pro) attention layers.

Status: **scaffold only** — this code has been peer-reviewed against the
vLLM V4 implementation in PR #40760 (branch `zyongye/vllm/dsv4`,
commit `3602f14f`) but has NOT been executed on a real DeepSeek-V4
checkpoint. The vLLM V4 support is not yet merged to vLLM main (as of
2026-04-25T01:00Z) and the minimum recommended hardware is 4×B200 or
4×B300; our dev H200 instance (1 GPU, 141 GiB HBM, 7 GB free disk)
cannot host V4-Flash (158 GB weights). Live validation is pending
hardware.

Design
------
DeepSeek-V4 uses a *shared-KV* MLA variant: a single 512-dim latent
``kv`` tensor is produced per token by ``fused_wqa_wkv``, then fed
into the fused attention kernel which internally does RMSNorm + RoPE
+ FP8 quant + cache insert (for c4a / c128a / SWA paths). We hook
``DeepseekV4Attention.forward`` after ``fused_wqa_wkv`` and before
the attention custom op, so the codec operates on the pre-RoPE
latent exactly as in the Stage 0.5 DSV4KVGenerator reference port.

Three phases mirror the Qwen3 / Gemma-4 / GLM hooks in
``snapshot_hook.py``:

* ``capture``  — record the pre-attention latent per layer
* ``replace``  — splice a pre-computed latent back in
* ``inforward`` — roundtrip (encode+decode) through ``HookState.codec_fn``
  in-place, so downstream attention sees codec-reconstructed KV

Per-layer layout: ``kv`` has shape ``[num_tokens, head_dim=512]``
(single latent, NOT per-head). The ``compress_ratio`` attribute
determines the downstream attention path:

* ``compress_ratio == 1`` → pure sliding-window (SWA-only)
* ``compress_ratio == 4`` → c4a (compressed + indexer sparse attention)
* ``compress_ratio == 128`` → c128a (compressed, full-pool attention)

All three paths receive the same ``kv`` latent from ``fused_wqa_wkv``;
the downstream compressor and indexer consume separate hidden-states
paths. We do NOT hook the compressor or indexer directly in Stage 1:
the KV value that lands in the main attention cache IS our
pre-RoPE ``kv`` tensor, and that is the compression target of interest.

What this hook does NOT measure
-------------------------------
* Indexer KV cache (separate fp4 128-dim path; not compression-relevant
  since it's already 4x smaller and not used for reconstruction).
* Compressor's gated-pool output (its KV entries are created inside
  the fused compressor kernel and written directly to the compressed
  KV cache; Stage 1 does NOT intercept them). If we want to
  additionally probe the compressed-pool distribution, we would hook
  ``mla_attn.swa_cache_layer`` writes separately.
* FP4 MoE expert activations (out of scope; KV cache has no MoE
  dependency).

Integration
-----------
Call ``install_dsv4_snapshot_patch()`` from
``plugin.register()`` BEFORE model instantiation. The patch is
idempotent and a fire-count guard aborts the run if the hook does
not fire on every expected layer (43 non-MTP layers for V4-Flash,
61 for V4-Pro).

Validation status
-----------------
* Symbol targets verified against vLLM PR #40760 at commit
  `3602f14f0e146b234be911d916e381b4e6a4dc0c`
* Forward-signature verified: ``forward(self, positions, hidden_states,
  llama_4_scaling=None)`` — 3 args, different from V3's Q/K/V signature
* Hook interception point verified: between lines 278 and 291 of
  ``vllm/model_executor/models/deepseek_v4.py``, after
  ``qr_kv, _ = self.fused_wqa_wkv(hidden_states)`` and
  ``qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)``,
  and before ``torch.ops.vllm.deepseek_v4_attention(...)``
* Unit tests (``test_dsv4_snapshot_hook.py``) cover three-phase
  semantics on synthetic DSV4Attention-shaped inputs without
  requiring the real vLLM V4 wheel
"""
from __future__ import annotations

from typing import Any

import torch

# Re-use the shared HookState + _snapshot_capture_replace from the
# v1.4 snapshot hook — they are codec-family agnostic.
from .snapshot_hook import HookState, _snapshot_capture_replace


_DSV4_PATCHED = False


def _extract_layer_id_from_prefix(prefix: str | None) -> int:
    """Parse ``layers.<N>.self_attn`` style prefixes into the numeric
    layer id. MTP layers use prefixes like ``mtp.0.attn`` which we
    map to layer_id = ``num_hidden_layers + mtp_index``.

    Returns 0 on parse failure (the fire-count guard will then flag
    a silent passthrough).
    """
    if not prefix:
        return 0
    parts = prefix.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
        if p == "mtp" and i + 1 < len(parts):
            try:
                return int(parts[i + 1]) + 10_000  # reserved MTP id range
            except ValueError:
                pass
    return 0


def install_dsv4_snapshot_patch() -> None:
    """Monkey-patch ``DeepseekV4Attention.forward`` with our three-phase
    hook. Idempotent. Must be called BEFORE the model is instantiated
    (vLLM caches module attributes referenced inside forward).

    Imports are inside the function body so this module remains
    importable on machines where the vLLM V4 wheel is not yet
    installed (most of our dev machines, as of 2026-04-25).
    """
    global _DSV4_PATCHED
    if _DSV4_PATCHED:
        return

    try:
        from vllm.model_executor.models.deepseek_v4 import (  # type: ignore
            DeepseekV4Attention,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import vllm.model_executor.models.deepseek_v4."
            " DeepSeek-V4 support requires vLLM PR #40760 or later,"
            " e.g. the `vllm/vllm-openai:deepseekv4-cu130` docker image"
            " or an install from the `zyongye/vllm:dsv4` branch."
        ) from e

    if getattr(DeepseekV4Attention, "_kk_snapshot_patched", False):
        _DSV4_PATCHED = True
        return

    orig_forward = DeepseekV4Attention.forward

    def patched(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if HookState.phase == "off":
            return orig_forward(self, positions, hidden_states, llama_4_scaling)

        # Replay the first two lines of the real forward body verbatim
        # so we can intercept `kv` before the fused attention op.
        # Reference: vllm/model_executor/models/deepseek_v4.py:278-279
        # in PR #40760 (branch `zyongye/vllm/dsv4`, commit 3602f14f).
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split(
            [self.q_lora_rank, self.head_dim], dim=-1,
        )

        layer_id = _extract_layer_id_from_prefix(getattr(self, "prefix", None))
        # Single-latent shared-KV: the "K" and "V" slots of the shared
        # helper are fed the same tensor. Downstream replay uses only
        # ``K`` since the codec is symmetric; ``V`` is a marker that
        # the layer fired.
        HookState.head_size = self.head_dim
        HookState.num_kv_heads = 1
        HookState.num_heads = self.n_heads
        kv_as_k, _ = _snapshot_capture_replace(
            layer_id,
            kv,
            kv,
            nkv=1,
            hd=self.head_dim,
        )
        kv = kv_as_k

        # Re-splice the intercepted kv back into qr_kv for the
        # downstream custom op (which expects a single tensor).
        qr_kv = torch.cat([qr, kv], dim=-1)

        # Replay the rest of the forward body. We cannot call
        # orig_forward() here because we've already consumed
        # fused_wqa_wkv, so we inline the remaining lines.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Re-split qr_kv back to qr / kv before passing to attention op
        qr, kv = qr_kv.split(
            [self.q_lora_rank, self.head_dim], dim=-1,
        )

        torch.ops.vllm.deepseek_v4_attention(  # type: ignore[attr-defined]
            hidden_states,
            qr,
            kv,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse RoPE + FP8 + grouped output. Imported lazily so this
        # file doesn't break on non-V4 vLLM installs.
        from vllm.model_executor.layers.deepseek_v4_attention import (  # type: ignore
            fused_inv_rope_fp8_quant,
        )

        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(  # type: ignore[attr-defined]
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )
        return self.wo_b(z.flatten(1))

    DeepseekV4Attention.forward = patched
    DeepseekV4Attention._kk_snapshot_patched = True
    _DSV4_PATCHED = True
    print(
        "[snap-patch] DeepseekV4Attention.forward wrapped "
        "(capture / replace / inforward / off). "
        "See vllm_backend/kakeya_v1_4_snapshot/dsv4_snapshot_hook.py "
        "for the hook design and validation status.",
        flush=True,
    )


def install_all_snapshot_patches_dsv4_aware() -> None:
    """Call the existing v1.4 snapshot patches (Qwen3, Qwen2, Gemma4, GLM)
    AND the new DSV4 patch. Idempotent. If the vLLM V4 wheel is not
    installed, the DSV4 patch logs a warning and skips gracefully,
    keeping the existing 4-model patches active.
    """
    from .snapshot_hook import install_all_snapshot_patches

    install_all_snapshot_patches()
    try:
        install_dsv4_snapshot_patch()
    except ImportError as e:
        print(
            f"[snap-patch] DSV4 patch skipped (vLLM V4 not installed): {e}",
            flush=True,
        )


__all__ = [
    "install_dsv4_snapshot_patch",
    "install_all_snapshot_patches_dsv4_aware",
]

"""Three-phase Attention.forward hooks for the v1.4 KakeyaLattice
snapshot harness.

Lives in the `kakeya_v1_4_snapshot` package so that the
`vllm.general_plugins` entry point can install it in every vLLM
process (including the engine-core subprocess that v1 LLMEngine forks).

Activated by setting the env var `KAKEYA_SNAPSHOT_QWEN3=1` BEFORE
vLLM imports anything.  The harnesses
(`benchmarks/multimodel_v14_*.py`) do this in their main() before
they instantiate `LLM(...)`.

Installs four model-family patches: Qwen3, Qwen2, Gemma4, GLM (see
`install_all_snapshot_patches`).  Each patch is idempotent and fires
only when the corresponding model type is loaded.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class HookState:
    """Module-level state for the Qwen3Attention hook — three phases.

    * `phase == "capture"`: record POST-qk-norm, PRE-RoPE K / V for all
      tokens of the current prompt to CPU numpy.  The forward
      otherwise runs unmodified (so the captured K/V reflect a clean
      forward without any codec interference).

    * `phase == "replace"`: ignore the just-projected K / V and use
      pre-computed `replacements[layer_id]` instead.  This is what
      produces the HF two-pass semantic: Q still comes from the
      running residual, but K̂ and V̂ for past positions are the
      codec-round-tripped values.

    * `phase == "off"` (default): no-op; the hook is equivalent to
      the stock `Qwen3Attention.forward`.

    `replacements` is keyed by integer layer_id (0..num_layers-1).
    Each entry's "K" and "V" values are fp32 CUDA tensors shaped
    `[n_tokens, num_kv_heads, head_size]` — the harness populates
    them between pass 1 and pass 2.
    """

    phase: str = "off"
    captured: dict[int, dict[str, np.ndarray]] = {}
    replacements: dict[int, dict[str, torch.Tensor]] = {}
    capture_gpu: bool = False
    head_size: int = 0
    num_kv_heads: int = 0
    num_heads: int = 0
    replace_fired: dict[int, int] = {}
    replace_shape_mismatch: dict[int, list[tuple[int, int]]] = {}
    replace_missing: dict[int, int] = {}

    # ---- in-forward phase ----
    # `phase == "inforward"`: each layer's K and V are encode→decode'd
    # through `codec_fn` INSIDE the same forward pass, BEFORE RoPE.  The
    # reconstructed K/V then propagate through RoPE + attention + the
    # next layer's Q projection — so codec error accumulates across
    # layers, the honest "online" deployment semantics.
    #
    # The codec function must accept fp32 K/V of shape [N_tok, num_kv_heads,
    # head_dim] and return fp32 reconstructions of the same shape.  Boundary
    # layers (first/last few) can optionally be skipped by listing their
    # integer layer_ids in `inforward_skip_layers`.  Per-layer fire counts
    # are recorded in `inforward_fired` — the harness checks this > 0.
    codec_fn: "callable | None" = None
    inforward_skip_layers: set[int] = set()
    inforward_fired: dict[int, int] = {}


_PATCHED = False


def install_qwen3_snapshot_patch() -> None:
    """Monkey-patch `vllm.model_executor.models.qwen3.Qwen3Attention.forward`
    with a three-phase hook.  Idempotent.

    Must be called BEFORE the model is instantiated in the process
    (Qwen3 layer construction caches module attributes referenced
    inside forward).  The `vllm.general_plugins` entry point is the
    right place because vLLM loads plugins immediately after it
    imports its own modules and before it imports the model module.
    """
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.models.qwen3 import Qwen3Attention

    if getattr(Qwen3Attention, "_kk_snapshot_patched", False):
        _PATCHED = True
        return

    orig = Qwen3Attention.forward

    def patched(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        if HookState.phase == "off":
            return orig(self, positions, hidden_states)

        # Reimplement Qwen3Attention.forward's body so we can intercept
        # K / V between qk-norm and RoPE.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        # QK-norm (per-head RMSNorm) — Qwen3 only.
        q_by_head = q.view(
            *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim,
        )
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim,
        )
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # Derive the layer id from the Attention wrapper's `layer_name`.
        layer_id = 0
        name = getattr(self.attn, "layer_name", None)
        if name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        pass

        HookState.head_size = self.attn.head_size
        HookState.num_kv_heads = self.attn.num_kv_heads
        HookState.num_heads = self.attn.num_heads

        k, v = _snapshot_capture_replace(
            layer_id, k, v,
            nkv=self.attn.num_kv_heads,
            hd=self.attn.head_size,
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Qwen3Attention.forward = patched
    Qwen3Attention._kk_snapshot_patched = True
    _PATCHED = True
    # Print so the harness can see the install happened in the
    # subprocess (logging gets routed differently between parent and
    # engine-core).
    print("[snap-patch] Qwen3Attention.forward wrapped "
          "(capture / replace / off)", flush=True)


# ---------------------------------------------------------------------------
# Shared helper: the three-phase capture/replace logic for a (k, v)
# pair.  All four model-family patches below call into this after
# extracting k, v in model-specific ways.
# ---------------------------------------------------------------------------
def _snapshot_capture_replace(
    layer_id: int,
    k: torch.Tensor,
    v: torch.Tensor,
    nkv: int,
    hd: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply capture / replace logic to (k, v).  Returns possibly-
    replaced (k, v) of the same shape and dtype as input.

    * capture phase: stores k, v (reshaped to [N, nkv, hd]) to
      HookState.captured[layer_id] — as GPU tensor if capture_gpu
      is set, else as CPU numpy.  Returns original k, v unchanged.
    * replace phase: if HookState.replacements has an entry for
      layer_id and its K shape matches, substitute the flattened
      tensor back into k / v.  Otherwise record a mismatch /
      missing.
    """
    if HookState.phase == "capture":
        k_det = k.detach().to(torch.float32).reshape(-1, nkv, hd)
        v_det = v.detach().to(torch.float32).reshape(-1, nkv, hd)
        if HookState.capture_gpu:
            HookState.captured[layer_id] = {
                "K": k_det.clone(),
                "V": v_det.clone(),
            }
        else:
            HookState.captured[layer_id] = {
                "K": k_det.cpu().numpy(),
                "V": v_det.cpu().numpy(),
            }
        return k, v
    elif HookState.phase == "replace":
        repl = HookState.replacements.get(layer_id)
        if repl is not None:
            k_new = repl["K"]
            v_new = repl["V"]
            n_tokens = k.shape[0]
            if k_new.shape[0] == n_tokens:
                k = k_new.reshape(n_tokens, -1).to(k.dtype)
                v = v_new.reshape(n_tokens, -1).to(v.dtype)
                HookState.replace_fired.setdefault(layer_id, 0)
                HookState.replace_fired[layer_id] += 1
            else:
                HookState.replace_shape_mismatch.setdefault(
                    layer_id, [],
                ).append((k_new.shape[0], n_tokens))
        else:
            HookState.replace_missing.setdefault(layer_id, 0)
            HookState.replace_missing[layer_id] += 1
    elif HookState.phase == "inforward":
        # Honest online semantics: encode→decode K AND V here, before
        # RoPE + attention; the reconstructions then propagate to the
        # next layer.  Boundary layers (first/last N) optionally skip
        # the codec and pass K/V through unchanged.
        if layer_id in HookState.inforward_skip_layers:
            return k, v
        if HookState.codec_fn is None:
            raise RuntimeError(
                "HookState.phase=='inforward' requires HookState.codec_fn "
                "to be set by the harness before vLLM prefill starts. "
                "No fallback — refusing to silently pass K/V through."
            )
        N = k.shape[0]
        # Reshape to [N_tok, nkv, hd] for the codec, detach-cast to fp32.
        k_fp32 = k.detach().to(torch.float32).reshape(N, nkv, hd)
        v_fp32 = v.detach().to(torch.float32).reshape(N, nkv, hd)
        k_hat = HookState.codec_fn(k_fp32)
        v_hat = HookState.codec_fn(v_fp32)
        # Cast back to the layer's native dtype (bf16) and flatten to the
        # shape the downstream ops expect.
        k = k_hat.reshape(N, -1).to(k.dtype)
        v = v_hat.reshape(N, -1).to(v.dtype)
        HookState.inforward_fired.setdefault(layer_id, 0)
        HookState.inforward_fired[layer_id] += 1
    return k, v


def _extract_layer_id_from_attn_wrapper(attn) -> int:
    """Derive the integer layer id from `attn.layer_name` set by vLLM.

    vLLM attaches a `layer_name` like
    `model.layers.17.self_attn.attn` to the Attention wrapper.  We
    parse the number after "layers.".  Returns 0 if missing (safe
    fallback — the harness only uses the layer_id for keyed dict
    access so a missing one will just collide on layer 0 and fail
    the shape assertion downstream, loudly).
    """
    name = getattr(attn, "layer_name", None)
    if not name:
        return 0
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


# ---------------------------------------------------------------------------
# Qwen2Attention patch — for DeepSeek-R1-Distill-Qwen-1.5B and
# other Qwen2-family (no qk-norm) models.
# ---------------------------------------------------------------------------
_PATCHED_QWEN2 = False


def install_qwen2_snapshot_patch() -> None:
    """Patch Qwen2Attention.forward — used by DeepSeek-R1-Distill-Qwen-1.5B.

    Qwen2Attention has NO qk-norm by default (qk_norm parameter defaults
    to False; only BAGEL-style variants use it).  Capture point: after
    qkv_proj split, before RoPE.  Same three-phase capture/replace
    semantics as the Qwen3 patch.

    Idempotent.
    """
    global _PATCHED_QWEN2
    if _PATCHED_QWEN2:
        return

    from vllm.model_executor.models.qwen2 import Qwen2Attention
    if getattr(Qwen2Attention, "_kk_snapshot_patched", False):
        _PATCHED_QWEN2 = True
        return

    orig = Qwen2Attention.forward

    def patched(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        if HookState.phase == "off":
            return orig(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1,
        )

        # Qwen2's optional qk-norm branch (DeepSeek-1.5B does NOT enable it).
        if getattr(self, "qk_norm", False):
            total_tokens = q.shape[0]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)

        layer_id = _extract_layer_id_from_attn_wrapper(self.attn)
        HookState.head_size = self.attn.head_size
        HookState.num_kv_heads = self.attn.num_kv_heads
        HookState.num_heads = self.attn.num_heads

        k, v = _snapshot_capture_replace(
            layer_id, k, v,
            nkv=self.attn.num_kv_heads,
            hd=self.attn.head_size,
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Qwen2Attention.forward = patched
    Qwen2Attention._kk_snapshot_patched = True
    _PATCHED_QWEN2 = True
    print("[snap-patch] Qwen2Attention.forward wrapped "
          "(for DeepSeek-R1-Distill-Qwen-1.5B)", flush=True)


# ---------------------------------------------------------------------------
# Gemma4Attention patch — for Gemma 4 family (E2B, E4B, 26B-A4B, 31B).
# ---------------------------------------------------------------------------
_PATCHED_GEMMA4 = False


def install_gemma4_snapshot_patch() -> None:
    """Patch Gemma4Attention.forward.

    Gemma 4 structure (differs materially from Qwen3):
      * q_norm (learnable) applied to Q before RoPE.
      * k_norm (learnable) applied to K before RoPE.
      * v_norm (no learnable weight) applied to V.
      * Sliding-window layers (per-layer-type RoPE).
      * kv_sharing: some layers re-use another layer's KV cache.  When
        `is_kv_shared_layer`, K and V are NOT recomputed (they're
        None/never reach the capture point).  We skip capture/replace
        for those layers.

    Capture point: post-qk-norm and post-v-norm, pre-RoPE — same
    analytical role as the Qwen3 patch (canonical K/V in a form a
    codec can reconstruct).
    """
    global _PATCHED_GEMMA4
    if _PATCHED_GEMMA4:
        return

    from vllm.model_executor.models.gemma4 import Gemma4Attention
    if getattr(Gemma4Attention, "_kk_snapshot_patched", False):
        _PATCHED_GEMMA4 = True
        return

    orig = Gemma4Attention.forward

    def patched(self, positions: torch.Tensor,
                hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if HookState.phase == "off":
            return orig(self, positions, hidden_states, **kwargs)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1,
        )

        # Q norm (always applied, per Gemma4 arch).
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            # K norm.
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            # V norm (no learnable scale).
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)

            # Intercept HERE — post-qk/v norm, pre-RoPE.
            layer_id = _extract_layer_id_from_attn_wrapper(self.attn)
            HookState.head_size = self.attn.head_size
            HookState.num_kv_heads = self.attn.num_kv_heads
            HookState.num_heads = self.attn.num_heads

            k, v = _snapshot_capture_replace(
                layer_id, k, v,
                nkv=self.attn.num_kv_heads,
                hd=self.attn.head_size,
            )

            q, k = self.rotary_emb(positions, q, k)
        else:
            # kv-shared: only Q gets RoPE, K and V come from the
            # shared earlier layer's cache.  No capture/replace here
            # (would corrupt the shared cache if we tried).
            q = self.rotary_emb(positions, q, k)[0]

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Gemma4Attention.forward = patched
    Gemma4Attention._kk_snapshot_patched = True
    _PATCHED_GEMMA4 = True
    print("[snap-patch] Gemma4Attention.forward wrapped "
          "(for Gemma 4 E2B/E4B/26B-A4B/31B)", flush=True)


# ---------------------------------------------------------------------------
# GLMAttention patch — for GLM-4-9B-Chat and related ChatGLM-style models.
# ---------------------------------------------------------------------------
_PATCHED_GLM = False


def install_glm_snapshot_patch() -> None:
    """Patch GLMAttention.forward (GLM-4 / ChatGLM-family).

    GLM differs from Qwen3 in three relevant ways:
      1. No qk-norm — K/V captured at post-qkv-split, pre-RoPE.
      2. `query_key_value` (not `qkv_proj`) module name.
      3. forward signature is `(hidden_states, position_ids)` —
         NOTE the argument ORDER is (hidden, pos), NOT (pos, hidden)!
      4. RoPE uses partial_rotary_factor=0.5 (applied to half of each
         head_dim); this is internal to `rotary_emb` so doesn't affect
         the capture point.
    """
    global _PATCHED_GLM
    if _PATCHED_GLM:
        return

    from vllm.model_executor.models.chatglm import GLMAttention
    if getattr(GLMAttention, "_kk_snapshot_patched", False):
        _PATCHED_GLM = True
        return

    orig = GLMAttention.forward

    def patched(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        if HookState.phase == "off":
            return orig(self, hidden_states, position_ids)

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1,
        )

        layer_id = _extract_layer_id_from_attn_wrapper(self.attn)
        HookState.head_size = self.attn.head_size
        HookState.num_kv_heads = self.attn.num_kv_heads
        HookState.num_heads = self.attn.num_heads

        k, v = _snapshot_capture_replace(
            layer_id, k, v,
            nkv=self.attn.num_kv_heads,
            hd=self.attn.head_size,
        )

        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(q, k, v)
        attn_output, _ = self.dense(context_layer)
        return attn_output

    GLMAttention.forward = patched
    GLMAttention._kk_snapshot_patched = True
    _PATCHED_GLM = True
    print("[snap-patch] GLMAttention.forward wrapped "
          "(for GLM-4 / ChatGLM)", flush=True)


def install_all_snapshot_patches() -> None:
    """Install every model-family patch available.  Safe to call before
    vLLM picks a specific model — only the patch that matches the loaded
    model type will actually fire.  Idempotent per patch.
    """
    # Each try/except is deliberate: if a model module isn't importable
    # in this vLLM version (e.g. Gemma4 wasn't in older builds), skip it
    # but don't fail the others.
    for name, fn in [
        ("Qwen3",  install_qwen3_snapshot_patch),
        ("Qwen2",  install_qwen2_snapshot_patch),
        ("Gemma4", install_gemma4_snapshot_patch),
        ("GLM",    install_glm_snapshot_patch),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"[snap-patch] {name} install failed: {e}", flush=True)

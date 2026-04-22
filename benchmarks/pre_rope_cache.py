"""Pre-RoPE DynamicCache for Hugging Face Qwen2 / Qwen3 / Llama-family models.

Architecture:

    forward(prefill / decode):
        K_pre = k_proj(h)                     # never rotated
        V     = v_proj(h)
        cache.update(K_pre, V)                # cache stores PRE-RoPE K
        K_pre_all = cache.layers[l].keys
        (cos, sin) = rotary(seq_total)
        Q_post      = rotate(Q_pre,      cos[new tokens],    sin[new tokens])
        K_post_all  = rotate(K_pre_all,  cos[all positions], sin[all positions])
        attn(Q_post, K_post_all, V)

The codec is plugged in on the cache tensor directly; it sees K_pre, not
K_post. There is no inverse-RoPE anywhere — RoPE simply does not exist on the
data path the codec observes, matching vLLM / SGLang / TRT-LLM paged-attention
kernels that store pre-RoPE K and apply RoPE inside the attention kernel.

This module exposes `install(model)` which monkey-patches every Qwen2/Qwen3
attention module's `forward`.
"""

from __future__ import annotations

import types
from typing import Callable

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotate: returns [-x2, x1] concatenated along last dim."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply half-split RoPE: x is [bsz, heads, seq, hd], cos/sin are [bsz, seq, hd].

    Forward RoPE in Hugging Face's convention is:
        out = x * cos + rotate_half(x) * sin
    with cos/sin broadcast over the head dim.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def _build_pre_rope_forward(
    eager_attention_forward: Callable,
    *,
    apply_rope: Callable | None = None,
    rotary_adapter: Callable | None = None,
    uses_qk_norm: bool = False,
    pass_sliding_window: bool = True,
):
    """Factory: return a forward() that stores K_pre in cache and applies
    RoPE on the full stacked K at read time.

    Q-calibration hook
    ------------------
    If `self.config._q_recorder` is set to a dict of lists
    (one per layer_idx), the patched forward appends every `q_pre`
    tensor seen by this attention module.  This is cheap because it's
    just a list append; no extra compute.  The calibration driver
    installs the recorder, runs forward on calibration data, and
    consumes the lists to build Σ_q per (layer, kv-head).

    Parameters
    ----------
    apply_rope
        If provided, used instead of the built-in `_rotate_half_split` —
        allows family-specific RoPE conventions (e.g., GLM's
        interleaved rotation).  Signature: `apply_rope(q, k, cos, sin)
        -> (q_rot, k_rot)`.  If None, falls back to the Qwen/Llama
        split-half convention.
    rotary_adapter
        Optional adapter `(rotary, dummy, positions) -> (cos, sin)` for
        families whose rotary expects a different call signature.  If
        None, uses `rotary(dummy, positions)` (Qwen-style).
    uses_qk_norm
        If True, calls `self.q_norm` and `self.k_norm` after q_proj/k_proj
        (Qwen3 style).
    pass_sliding_window
        Whether to pass `sliding_window=getattr(self, 'sliding_window',
        None)` to `eager_attention_forward` (Qwen2) vs. omit it (GLM).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_pre = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k_pre_new = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if uses_qk_norm:
            # Qwen3-style: RMSNorm applied BEFORE RoPE, so what we cache is
            # post-norm pre-RoPE K.  We rewrap: q_pre/k_pre must go through
            # q_norm/k_norm before the rotary so downstream attention matches
            # the unpatched model.
            q_pre_shape = q_pre.transpose(1, 2).contiguous()  # (B, S, H, D)
            q_pre = self.q_norm(q_pre_shape).transpose(1, 2)
            k_pre_new_shape = k_pre_new.transpose(1, 2).contiguous()
            k_pre_new = self.k_norm(k_pre_new_shape).transpose(1, 2)

        recorder = getattr(self.config, "_q_recorder", None)
        if recorder is not None:
            recorder.setdefault(self.layer_idx, []).append(
                q_pre.detach().to(torch.float32).cpu()
            )

        if past_key_values is not None:
            k_pre_all, v_all = past_key_values.update(
                k_pre_new, v_new, self.layer_idx
            )
        else:
            k_pre_all, v_all = k_pre_new, v_new

        bsz, n_kv, seq_total, hd = k_pre_all.shape
        seq_new = k_pre_new.shape[-2]
        start_pos = seq_total - seq_new

        rotary = self.config._rotary_emb
        device = hidden_states.device
        full_positions = torch.arange(seq_total, device=device).unsqueeze(0).expand(bsz, -1)
        dummy = torch.zeros(bsz, seq_total, hd, device=device, dtype=hidden_states.dtype)
        if rotary_adapter is not None:
            cos_full, sin_full = rotary_adapter(rotary, dummy, full_positions)
        else:
            cos_full, sin_full = rotary(dummy, full_positions)
        cos_new = cos_full[:, start_pos:]
        sin_new = sin_full[:, start_pos:]

        if apply_rope is not None:
            q_post, _ = apply_rope(q_pre, q_pre, cos_new, sin_new)
            _, k_post_all = apply_rope(k_pre_all, k_pre_all, cos_full, sin_full)
        else:
            q_post = _rotate_half_split(q_pre, cos_new, sin_new)
            k_post_all = _rotate_half_split(k_pre_all, cos_full, sin_full)

        attn_kwargs = {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
        }
        if pass_sliding_window:
            attn_kwargs["sliding_window"] = getattr(self, "sliding_window", None)
        attn_kwargs.update(kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self,
            q_post,
            k_post_all,
            v_all,
            attention_mask,
            **attn_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _find_first(model: nn.Module, cls):
    for m in model.modules():
        if isinstance(m, cls):
            return m
    return None


def install(model: nn.Module) -> dict:
    """Monkey-patch every attention module so that cache stores pre-RoPE K.

    Supported families (auto-detected by architecture name):
    - Qwen2 (DeepSeek-R1-Distill, Qwen2.5)
    - Qwen3 (has q_norm/k_norm pre-RoPE)
    - GLM / GLM-edge (interleaved RoPE + partial rotation)
    - Gemma3 (q_norm/k_norm, alternating full/sliding attention layers)

    Returns a dict with metadata about the patch, for diagnostics.
    """
    cfg = model.config.get_text_config(decoder=True)
    arch = getattr(model.config, "architectures", ["?"])[0].lower()

    if "qwen2" in arch:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention, Qwen2RotaryEmbedding,
            eager_attention_forward as eager,
        )
        cls, rot_cls, uses_qk_norm = Qwen2Attention, Qwen2RotaryEmbedding, False
        apply_rope = None
        pass_sw = True
        family = "qwen2"
    elif "qwen3" in arch:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention, Qwen3RotaryEmbedding,
            eager_attention_forward as eager,
        )
        cls, rot_cls, uses_qk_norm = Qwen3Attention, Qwen3RotaryEmbedding, True
        apply_rope = None
        pass_sw = True
        family = "qwen3"
    elif "glm" in arch:
        from transformers.models.glm.modeling_glm import (
            GlmAttention, GlmRotaryEmbedding,
            eager_attention_forward as eager,
            apply_rotary_pos_emb as glm_apply_rope,
        )
        cls, rot_cls, uses_qk_norm = GlmAttention, GlmRotaryEmbedding, False
        apply_rope = glm_apply_rope
        pass_sw = False
        family = "glm"
    else:
        raise NotImplementedError(
            f"pre_rope_cache.install: architecture '{arch}' not supported. "
            f"Supported: qwen2, qwen3, glm."
        )

    rotary = _find_first(model, rot_cls)
    if rotary is None:
        rotary = rot_cls(config=cfg).to(next(model.parameters()).device)
    cfg._rotary_emb = rotary

    patched = 0
    fwd = _build_pre_rope_forward(
        eager, apply_rope=apply_rope, uses_qk_norm=uses_qk_norm,
        pass_sliding_window=pass_sw,
    )
    for m in model.modules():
        if isinstance(m, cls):
            m.config = cfg
            m.forward = types.MethodType(fwd, m)
            patched += 1

    assert patched > 0, f"no {cls.__name__} found — model family mismatch"
    return {"patched_layers": patched, "family": family}

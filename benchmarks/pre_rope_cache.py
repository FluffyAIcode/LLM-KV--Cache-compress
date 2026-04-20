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


def _build_pre_rope_forward(eager_attention_forward: Callable):
    """Factory: return a forward() that stores K_pre in cache and applies
    RoPE on the full stacked K at read time.
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
        cos_full, sin_full = rotary(dummy, full_positions)
        cos_new = cos_full[:, start_pos:]
        sin_new = sin_full[:, start_pos:]

        q_post = _rotate_half_split(q_pre, cos_new, sin_new)
        k_post_all = _rotate_half_split(k_pre_all, cos_full, sin_full)

        attn_output, attn_weights = eager_attention_forward(
            self,
            q_post,
            k_post_all,
            v_all,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def install(model: nn.Module) -> dict:
    """Monkey-patch every attention module so that cache stores pre-RoPE K.

    Returns a dict with metadata about the patch, for diagnostics.
    """
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2RotaryEmbedding,
        eager_attention_forward as qwen2_eager,
    )

    cfg = model.config.get_text_config(decoder=True)
    rotary = None
    for m in model.modules():
        if isinstance(m, Qwen2RotaryEmbedding):
            rotary = m
            break
    if rotary is None:
        rotary = Qwen2RotaryEmbedding(config=cfg).to(next(model.parameters()).device)

    cfg._rotary_emb = rotary

    patched = 0
    fwd = _build_pre_rope_forward(qwen2_eager)
    for m in model.modules():
        if isinstance(m, Qwen2Attention):
            m.config = cfg
            m.forward = types.MethodType(fwd, m)
            patched += 1

    assert patched > 0, "no Qwen2Attention found — model family not yet supported"
    return {"patched_layers": patched, "family": "qwen2"}

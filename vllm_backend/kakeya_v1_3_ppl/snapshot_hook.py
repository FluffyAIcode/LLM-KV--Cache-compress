"""Three-phase Qwen3Attention.forward hook for the snapshot harness.

Lives in the `kakeya_v1_3_ppl` package so that the `vllm.general_plugins`
entry point can install it in every vLLM process (including the
engine-core subprocess that v1 LLMEngine forks).

Activated by setting the env var `KAKEYA_SNAPSHOT_QWEN3=1` BEFORE
vLLM imports anything.  The harness
(`benchmarks/e2e_ppl_validation_vllm_snapshot_qwen3.py`) does this
in its main() before it instantiates `LLM(...)`.

Design mirrors the Qwen2 hook from PR #17 verbatim; only the body of
the patched `forward` is adapted to Qwen3Attention's signature and
internal structure (qk-norm is a Qwen3-specific addition that the
Qwen2 hook didn't need to worry about).
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
    # Populated by the hook itself so the harness can consult them
    # post-capture without introspecting the model.
    head_size: int = 0
    num_kv_heads: int = 0
    num_heads: int = 0
    # Per-layer diagnostics — reset by the harness at the start of
    # each passage.  Help localise silent "fall-through-with-live-K/V"
    # failures in the replace phase.
    replace_fired: dict[int, int] = {}
    replace_shape_mismatch: dict[int, list[tuple[int, int]]] = {}
    replace_missing: dict[int, int] = {}


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

        nkv = self.attn.num_kv_heads
        hd = self.attn.head_size

        if HookState.phase == "capture":
            k_np = (
                k.detach().to(torch.float32).cpu().numpy()
                .reshape(-1, nkv, hd)
            )
            v_np = (
                v.detach().to(torch.float32).cpu().numpy()
                .reshape(-1, nkv, hd)
            )
            HookState.captured[layer_id] = {"K": k_np, "V": v_np}
            # Unmodified k / v continue to RoPE + attention.
        elif HookState.phase == "replace":
            repl = HookState.replacements.get(layer_id)
            if repl is not None:
                k_new = repl["K"]
                v_new = repl["V"]
                n_tokens = k.shape[0]
                if k_new.shape[0] == n_tokens:
                    k = k_new.reshape(n_tokens, -1).to(k.dtype)
                    v = v_new.reshape(n_tokens, -1).to(v.dtype)
                    HookState.replace_fired.setdefault(
                        layer_id, 0,
                    )
                    HookState.replace_fired[layer_id] += 1
                else:
                    HookState.replace_shape_mismatch.setdefault(
                        layer_id, []
                    ).append((k_new.shape[0], n_tokens))
            else:
                HookState.replace_missing.setdefault(layer_id, 0)
                HookState.replace_missing[layer_id] += 1

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

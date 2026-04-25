r"""Stage 0.75 — load trained DeepSeek-V4-Flash attention + Compressor weights
into the Stage 0.5 DSV4KVGenerator.

Goal: replace the random-Gaussian init weights with real trained weights
for THREE representative layers (0 = SWA, 2 = c4a, 3 = c128a), so the
non-Gaussian audit on V4 KV streams is measured against actual learned
distributions instead of architectural-defaults.

No MoE experts, no shared experts, no Indexer's weights-projection for
downstream sparse attention — we only need the projection + compressor
sub-path that produces the KV tensors.

Weight storage format (V4-Flash inference/model.py:123-152):
  - `.weight`  shape [out, in] dtype float8_e4m3fn
  - `.scale`   shape [ceil(out/128), ceil(in/128)] dtype float8_e8m0fnu
               (FP8 weights are block-scaled per 128x128 tile on (out, in))
  - For each 128x128 tile, the dequantized bf16 value is
    ``fp8_weight_tile * fp8_e8m0_scale_value``.
  - Some weights (RMSNorm.weight, attn_sink, compressor.ape, wgate) are
    stored directly in bf16/fp32 and have no `.scale`.

Our dequantization: load once into fp32, then feed into the Stage 0.5
``DSV4MainKVProjection`` / ``DSV4Compressor`` which already uses fp32
arithmetic internally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open


_FP8_E8M0_BIAS = 127
"""IEEE754 FP8 E8M0 (unsigned) exponent bias — same as standard float32's
exponent bias."""


def _dequant_fp8_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Convert a torch.float8_e8m0fnu scale tensor to float32.

    E8M0 encodes 2^(e - 127) where e is the stored uint8 byte. Some
    PyTorch builds don't have a direct .to(torch.float32) for
    float8_e8m0fnu; we fall back to bitcast + exponent conversion.
    """
    if x.dtype == torch.float32:
        return x
    # Fast path: if PyTorch supports direct cast, use it
    try:
        return x.to(torch.float32)
    except (RuntimeError, TypeError):
        pass
    # Bitcast fallback
    e = x.view(torch.uint8).to(torch.int32)
    # 2^(e - 127)
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - _FP8_E8M0_BIAS)


def _dequant_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Dequantize an FP8-E4M3 weight tensor using an E8M0 block scale.

    weight: [out, in] float8_e4m3fn
    scale:  [ceil(out/block), ceil(in/block)] float8_e8m0fnu
    returns: [out, in] float32
    """
    out_dim, in_dim = weight.shape
    try:
        w_fp32 = weight.to(torch.float32)
    except RuntimeError:
        # Bitcast path for older torch
        w_fp32 = weight.view(torch.uint8).to(torch.float32)

    s_fp32 = _dequant_fp8_e8m0(scale)
    # Expand scale to per-element using repeat_interleave
    s_expanded_out = s_fp32.repeat_interleave(block_size, dim=0)[:out_dim]
    s_expanded = s_expanded_out.repeat_interleave(block_size, dim=1)[:, :in_dim]
    return w_fp32 * s_expanded


def load_single_layer_weights(
    safetensors_path: str,
    layer_id: int,
) -> Dict[str, torch.Tensor]:
    """Return a dict of dequantized (fp32) weight tensors for the
    ``layers.<layer_id>.attn.*`` sub-tree in the given safetensors shard.

    Keys in the returned dict follow the source naming, with suffixed
    ``.weight`` (dequant to fp32 if FP8) and ``.scale`` omitted.

    Example:
        out = load_single_layer_weights(".../shard-2.safetensors", layer_id=0)
        out["layers.0.attn.wkv.weight"]    # [head_dim, hidden] fp32
        out["layers.0.attn.kv_norm.weight"] # [head_dim] fp32
    """
    want_prefix = f"layers.{layer_id}.attn."
    out: Dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith(want_prefix)]
        # Group by basename (drop .weight / .scale)
        wanted = {}
        for k in keys:
            if k.endswith(".scale"):
                wanted.setdefault(k[:-len(".scale")], {})["scale"] = k
            else:
                # .weight, or bare param (ape, attn_sink, norm.weight)
                base = k
                if k.endswith(".weight"):
                    base = k[:-len(".weight")]
                wanted.setdefault(base, {})["weight"] = k
        for base, parts in wanted.items():
            wk = parts.get("weight")
            sk = parts.get("scale")
            if wk is None:
                continue
            w = f.get_tensor(wk)
            if sk is not None:
                s = f.get_tensor(sk)
                w_fp32 = _dequant_fp8_weight(w, s, block_size=128)
            else:
                try:
                    w_fp32 = w.to(torch.float32)
                except RuntimeError:
                    w_fp32 = w.view(torch.uint8).to(torch.float32)
            # Put back under `.weight` naming so callers see the same
            # interface as raw PyTorch state dicts
            out_key = wk
            out[out_key] = w_fp32
    return out


def inject_weights_into_main_kv(
    proj: "DSV4MainKVProjection",  # type: ignore[name-defined]
    params: Dict[str, torch.Tensor],
    layer_id: int,
    device: str = "cpu",
) -> None:
    """Replace random-init weights in a DSV4MainKVProjection with
    trained weights from ``params``. Expected keys:

      layers.<L>.attn.wkv.weight    — [head_dim, hidden]
      layers.<L>.attn.kv_norm.weight — [head_dim]
    """
    wkv_key = f"layers.{layer_id}.attn.wkv.weight"
    norm_key = f"layers.{layer_id}.attn.kv_norm.weight"
    if wkv_key not in params:
        raise KeyError(
            f"Expected {wkv_key!r} in loaded params; available keys: "
            f"{list(params.keys())[:5]}..."
        )
    with torch.no_grad():
        proj.wkv.weight.data.copy_(params[wkv_key].to(device))
        proj.kv_norm.weight.data.copy_(params[norm_key].to(proj.kv_norm.weight.dtype).to(device))


def inject_weights_into_compressor(
    comp: "DSV4Compressor",  # type: ignore[name-defined]
    params: Dict[str, torch.Tensor],
    layer_id: int,
    device: str = "cpu",
) -> None:
    """Replace random-init weights in a DSV4Compressor with trained
    weights. Expected keys:

      layers.<L>.attn.compressor.wkv.weight   [head_dim, hidden]  (c128a)
                                              [2*head_dim, hidden] (c4a with overlap)
      layers.<L>.attn.compressor.wgate.weight  same shape
      layers.<L>.attn.compressor.ape           [ratio, (1+overlap)*head_dim]
      layers.<L>.attn.compressor.norm.weight   [head_dim]
    """
    prefix = f"layers.{layer_id}.attn.compressor."
    with torch.no_grad():
        comp.wkv.weight.data.copy_(params[f"{prefix}wkv.weight"].to(device))
        comp.wgate.weight.data.copy_(params[f"{prefix}wgate.weight"].to(device))
        comp.ape.data.copy_(params[f"{prefix}ape"].to(device))
        comp.norm.weight.data.copy_(params[f"{prefix}norm.weight"].to(comp.norm.weight.dtype).to(device))


def load_v4_shard_paths(hf_cache_dir: str, model_id: str) -> Dict[int, str]:
    """Scan the HF cache for DeepSeek-V4-Flash and return a mapping
    from shard number (1..46) to absolute file path.
    """
    # Cache layout: HF_HOME/hub/models--<org>--<model>/snapshots/<rev>/<file>
    # or HF_HOME/models--<org>--<model>/snapshots/<rev>/<file> depending on
    # how hf_hub_download was invoked (cache_dir vs HF_HOME).
    org, _, name = model_id.replace("/", "--").partition("--")
    candidates = [
        Path(hf_cache_dir) / "hub" / f"models--{org}--{name}" / "snapshots",
        Path(hf_cache_dir) / f"models--{org}--{name}" / "snapshots",
    ]
    base = None
    for c in candidates:
        if c.exists():
            base = c
            break
    if base is None:
        raise FileNotFoundError(
            f"HF cache dir not found for {model_id}.  Tried: "
            f"{[str(c) for c in candidates]}"
        )
    # Pick the most recent snapshot
    snaps = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {base}")
    rev_dir = snaps[-1]
    shard_paths: Dict[int, str] = {}
    for p in rev_dir.glob("model-*-of-*.safetensors"):
        # e.g. model-00002-of-00046.safetensors
        parts = p.stem.split("-")
        if len(parts) >= 2:
            try:
                shard_num = int(parts[1])
                shard_paths[shard_num] = str(p.resolve())
            except ValueError:
                pass
    return shard_paths


__all__ = [
    "load_single_layer_weights",
    "inject_weights_into_main_kv",
    "inject_weights_into_compressor",
    "load_v4_shard_paths",
    "_dequant_fp8_weight",
    "_dequant_fp8_e8m0",
]

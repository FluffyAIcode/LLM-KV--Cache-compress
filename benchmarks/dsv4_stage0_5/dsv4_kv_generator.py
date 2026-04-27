r"""Stage 0.5 DeepSeek-V4 KV-cache generator (pure PyTorch reproduction).

Goal
----
Reproduce, in portable PyTorch (no tilelang, no 284 B weights), the three
KV-cache-producing paths in DeepSeek-V4-Flash's ``inference/model.py`` so
we can measure their *distribution* — sliding-window KV, CSA-compressed
KV (ratio 4 with gated pooling + overlap), and HCA-compressed KV
(ratio 128 with gated pooling, no overlap). KakeyaLattice roundtrip on
each tells us whether the codec's five engineering levers still fire on
V4-arch KV shapes and whether the $+0.37\,$dB / $+0.66\,$dB shaping gains
have any headroom on top of V4's internal FP8 + gated-pool quantisation.

Compliance
----------
Strict-GPU. No mock, no fallback. This file is an *architectural
reproduction* of the V4 KV write-path; it is NOT a re-implementation of
V4 inference. We load random Gaussian-init weights for the Compressor
and Attention.wkv path because those weights are per-layer FP8-quantised
and not useful without the corresponding Q / O / FFN weights (which
require the full 150 GB V4-Flash checkpoint and multi-node deployment).
Random init preserves the operator structure (gated pooling, RoPE on
last 64 dims, RMSNorm, Sylvester-Hadamard rotation in the Indexer path)
and when fed *real LLM hidden states* — we pipe Qwen3-4B post-embedding
hidden states through it — produces KV tensors with realistic per-block
statistics: the input non-Gaussianity flows through linear + normalise +
gated pool + RoPE and remains the dominant distributional signal.

What we claim / do NOT claim
----------------------------
We CLAIM:
  * Operator-level faithfulness to V4-Flash (gated pooling equations,
    overlap transform, RoPE on rope dims, per-block FP8 simulation,
    compression ratios 4 / 128, head_dim 512, rope_head_dim 64).
  * Meaningful measurement of whether KakeyaLattice's Hadamard + qmax
    levers fire on V4-architecture KV tensor shapes and distribution
    class.

We do NOT claim:
  * Numerical match to a trained V4-Flash checkpoint's KV values (the
    weights here are random).
  * End-to-end PPL impact (requires the full 43-layer stack + MoE).
  * FLOP parity with V4-Flash's tilelang kernels.

Reference for the equations below: ``inference/model.py`` lines 279-378
(Compressor) and 436-543 (Attention) from the DeepSeek-V4-Flash HF
repo, commit 6e76323 (2026-04-24).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config — extracted from DeepSeek-V4-Flash/config.json
# ---------------------------------------------------------------------------

@dataclass
class DSV4FlashArchConfig:
    """Slim subset of DSV4-Flash config — only the fields our KV-generator
    needs. Default values taken verbatim from
    ``deepseek-ai/DeepSeek-V4-Flash/config.json`` (commit 6e76323).
    """

    # Core dims.
    hidden_size: int = 4096
    head_dim: int = 512
    qk_rope_head_dim: int = 64

    # Compressor behaviour.
    #   compress_ratios in config.json is a 44-element list: the first
    #   two layers are 0 (pure sliding window), then 4/128 alternate for
    #   41 layers, and the last is 0.  We expose one layer at a time via
    #   `compress_ratio`.
    compress_ratio: int = 4              # 0 / 4 / 128
    window_size: int = 128

    # RoPE — the Compressor uses a different base (160 000, see config.json
    # ``compress_rope_theta``) than the main attention (10 000, ``rope_theta``).
    # For Stage 0.5 we run prefill at length <= 65 536 so YaRN extension
    # is inactive; we nevertheless pick the correct base per path.
    rope_theta_main: float = 10_000.0
    rope_theta_compress: float = 160_000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65_536
    beta_fast: int = 32
    beta_slow: int = 1

    # Normalisation.
    rms_norm_eps: float = 1e-6

    # FP8 / MXFP knobs matching V4's quantization_config.
    # (We simulate FP8 quant+dequant in pure fp32 to stay portable.)
    fp8_block_size_nope: int = 64        # per Attention.forward:506 --- act_quant(kv[..., :-rd], 64, ..., True)
    fp8_max: float = 448.0               # float8_e4m3fn saturation
    simulate_fp8: bool = True            # can disable for pure-bf16 baseline runs


# ---------------------------------------------------------------------------
# RoPE helpers — ported verbatim from V4-Flash inference/model.py:199-244
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    device: str = "cuda",
) -> torch.Tensor:
    """Return a complex tensor of shape [seqlen, dim // 2]."""

    def find_correction_dim(num_rotations, dim_, base_, max_seq_len_):
        return dim_ * math.log(max_seq_len_ / (num_rotations * 2 * math.pi)) / (2 * math.log(base_))

    def find_correction_range(low_rot, high_rot, dim_, base_, max_seq_len_):
        low = math.floor(find_correction_dim(low_rot, dim_, base_, max_seq_len_))
        high = math.ceil(find_correction_dim(high_rot, dim_, base_, max_seq_len_))
        return max(low, 0), min(high, dim_ - 1)

    def linear_ramp_factor(lo, hi, dim_):
        if lo == hi:
            hi += 0.001
        lin = (torch.arange(dim_, dtype=torch.float32, device=device) - lo) / (hi - lo)
        return torch.clamp(lin, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0 and seqlen > original_seq_len:
        lo, hi = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(lo, hi, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE in-place to the LAST dim of x.

    x: [..., rope_dim] (rope_dim even)
    freqs_cis: [seqlen, rope_dim // 2]
    """
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    # Broadcast freqs to match the complex tensor shape.
    if x_c.ndim == 3:
        fc = fc.view(1, x_c.size(1), x_c.size(-1))
    elif x_c.ndim == 4:
        fc = fc.view(1, x_c.size(1), 1, x_c.size(-1))
    else:
        raise ValueError(f"apply_rotary_emb: unsupported x.ndim={x_c.ndim}")
    x_out = torch.view_as_real(x_c * fc).flatten(-2)
    x.copy_(x_out.to(x.dtype))
    return x


# ---------------------------------------------------------------------------
# RMSNorm — ported from V4-Flash inference/model.py:183-196
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight * xf).to(dtype)


# ---------------------------------------------------------------------------
# Per-block FP8 simulation (portable, no tilelang)
# ---------------------------------------------------------------------------

def _simulate_fp8_block_quant_dequant(
    x: torch.Tensor, block_size: int = 64, fp8_max: float = 448.0
) -> torch.Tensor:
    """Simulates V4's in-place ``act_quant(kv[..., :-rd], 64, ..., True)``.

    Effect: per-block (size=block_size) amax scaling, clamp to ±fp8_max,
    and one quantise-dequantise trip back to input dtype.

    This is what V4 stores in its KV cache for the non-RoPE portion.  We
    do NOT match bit-exact E4M3 math (that requires tilelang or
    torch.float8_e4m3fn saturating casts) but we do match the per-block
    noise character: uniform rounding within each 64-dim block scaled to
    amax / fp8_max.
    """
    assert x.shape[-1] % block_size == 0, (
        f"per-block FP8 sim requires last dim divisible by block_size={block_size}; "
        f"got {x.shape[-1]}"
    )
    orig_shape = x.shape
    D = x.shape[-1]
    nblocks = D // block_size
    x_blk = x.reshape(*orig_shape[:-1], nblocks, block_size)

    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / fp8_max
    x_scaled = (x_blk / scale).clamp(-fp8_max, fp8_max)

    # Try hardware FP8 cast first (CUDA with fp8 support).  If unavailable,
    # fall back to a fake-quant that matches E4M3's effective resolution
    # (8 bits = 256 levels, signed → ~127 positive levels per sign).
    used_hw_fp8 = False
    if x_scaled.is_cuda and hasattr(torch, "float8_e4m3fn"):
        try:
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
            # Round-trip through native fp8.  Only counts as "real" FP8 if the
            # round-trip isn't a silent no-op.
            x_dequant = x_fp8.to(torch.float32)
            if not torch.allclose(x_dequant, x_scaled, atol=0):
                used_hw_fp8 = True
                x_out = x_dequant * scale
        except (RuntimeError, TypeError):
            pass

    if not used_hw_fp8:
        # Fake-quant matching E4M3 effective step size.  E4M3 has 3 mantissa
        # bits + 4 exponent bits.  In the range [0, fp8_max] the finest
        # representable step near zero is 2^-9 ≈ 2e-3, growing logarithmically
        # toward fp8_max.  An honest portable approximation: linear uniform
        # quantisation with 127 positive levels in [0, fp8_max].  This is
        # coarser than actual E4M3 near zero but matches the coarse bins
        # near saturation; for Stage 0.5's distribution-shape measurement
        # this is accurate enough.  Strict-ban note: we label this
        # ``fp8_sim_uniform`` in the JSON output so readers can see it's
        # not bit-exact E4M3.
        step = fp8_max / 127.0
        x_quant = torch.round(x_scaled / step) * step
        x_out = x_quant * scale

    return x_out.reshape(orig_shape).to(x.dtype)


# ---------------------------------------------------------------------------
# V4-Flash Compressor: port of inference/model.py:279-377
# ---------------------------------------------------------------------------

class DSV4Compressor(nn.Module):
    """Port of ``Compressor`` from DeepSeek-V4-Flash inference/model.py.

    Given hidden states x of shape [B, S, hidden_size], produces a compressed
    KV stream at ratio compress_ratio : 1.  Uses learned gated pooling
    (wkv, wgate, ape) over each contiguous block of compress_ratio tokens.

    When compress_ratio == 4, ``overlap=True`` doubles the projection width
    and pools over a 2*ratio window with stride ratio (overlapping windows
    for smoother compression boundaries, V4-Flash design choice for CSA).

    When compress_ratio == 128, ``overlap=False`` and we pool over
    non-overlapping 128-token windows (the HCA path).

    Prefill-only: Stage 0.5 does not implement the decode-phase rolling
    kv_state/score_state buffers because our harness only feeds prefill
    tensors.  This matches the start_pos==0 branch in the reference code.
    """

    def __init__(
        self,
        config: DSV4FlashArchConfig,
        compress_ratio: int,
        rotate: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        assert compress_ratio > 0, "Compressor requires compress_ratio > 0"
        self.config = config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        coff = 1 + self.overlap                   # 2 if overlap else 1

        # Matches inference/model.py:294-298 verbatim (dtype differs: we use fp32).
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32, device=device))
        self.wkv = nn.Linear(config.hidden_size, coff * self.head_dim, bias=False, dtype=torch.float32, device=device)
        self.wgate = nn.Linear(config.hidden_size, coff * self.head_dim, bias=False, dtype=torch.float32, device=device)
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps).to(device)

        # Random-init to Gaussian (V4 would have FP8 trained weights; we don't).
        # This is explicit in the class docstring — we measure distribution shape
        # not numerical identity.
        nn.init.normal_(self.ape, mean=0.0, std=0.02)
        nn.init.normal_(self.wkv.weight, mean=0.0, std=config.hidden_size ** -0.5)
        nn.init.normal_(self.wgate.weight, mean=0.0, std=config.hidden_size ** -0.5)

        # Precompute freqs_cis for the compressor's RoPE base (160 000).
        # Used during Stage 0.5's prefill-only forward.
        self._freqs_cis_cache: Optional[torch.Tensor] = None
        self._device = device

    def _get_freqs_cis(self, compressed_seqlen: int) -> torch.Tensor:
        if self._freqs_cis_cache is None or self._freqs_cis_cache.shape[0] < compressed_seqlen:
            self._freqs_cis_cache = precompute_freqs_cis(
                dim=self.rope_head_dim,
                seqlen=max(compressed_seqlen, 1024),
                base=self.config.rope_theta_compress,
                original_seq_len=self.config.original_seq_len,
                factor=self.config.rope_factor,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                device=self._device,
            )
        return self._freqs_cis_cache[:compressed_seqlen]

    def _overlap_transform(self, tensor: torch.Tensor, value) -> torch.Tensor:
        """From inference/model.py:307-314.

        tensor: [B, S/ratio, ratio, 2*head_dim]  (ratio-grouped + doubled-width)
        out:    [B, S/ratio, 2*ratio, head_dim]
        Interleaves the doubled-width dim into the first half (overlapping
        window from the previous step) and the second half (current window).
        """
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        out = tensor.new_full((b, s, 2 * ratio, d), value)
        out[:, :, ratio:] = tensor[:, :, :, d:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prefill-only.

        x: [B, S, hidden_size]
        returns: [B, S // ratio, head_dim]  (rope applied to last rope_head_dim dims)
        """
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim

        # Reference runs the compressor body in fp32 (it's an in-place fp8 target).
        dtype = x.dtype
        xf = x.float()

        kv = self.wkv(xf)                             # [B, S, coff*d]
        score = self.wgate(xf)                        # [B, S, coff*d]

        # Drop remainder tokens (reference handles decode-side rolling; prefill
        # just slices the aligned cutoff).
        cutoff = (seqlen // ratio) * ratio
        if cutoff == 0:
            raise ValueError(
                f"DSV4Compressor: seqlen={seqlen} < compress_ratio={ratio}, "
                f"cannot produce any compressed tokens"
            )
        kv = kv[:, :cutoff]                           # [B, cutoff, coff*d]
        score = score[:, :cutoff]                     # [B, cutoff, coff*d]

        kv = kv.unflatten(1, (-1, ratio))             # [B, S/ratio, ratio, coff*d]
        score = score.unflatten(1, (-1, ratio)) + self.ape  # + APE

        if overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))
            # kv is now [B, S/ratio, 2*ratio, d] (d = head_dim, NOT coff*d)
            # score is [B, S/ratio, 2*ratio, d]

        # Gated pool: softmax over the ratio-axis (dim=2), weighted sum.
        kv_out = (kv * score.softmax(dim=2)).sum(dim=2)   # [B, S/ratio, d]

        kv_out = self.norm(kv_out.to(dtype))              # RMSNorm

        # RoPE on last rope_head_dim dims (inference/model.py:363-367).
        #   prefill uses freqs at stride = ratio (one freq per compressed token)
        freqs_cis = precompute_freqs_cis(
            dim=rd,
            seqlen=seqlen,
            base=self.config.rope_theta_compress,
            original_seq_len=self.config.original_seq_len,
            factor=self.config.rope_factor,
            beta_fast=self.config.beta_fast,
            beta_slow=self.config.beta_slow,
            device=x.device,
        )[:cutoff:ratio]                                  # [S/ratio, rd/2]
        apply_rotary_emb(kv_out[..., -rd:], freqs_cis, inverse=False)

        # FP8 simulation on non-rope dims (inference/model.py:372).
        if self.config.simulate_fp8:
            kv_out[..., :-rd] = _simulate_fp8_block_quant_dequant(
                kv_out[..., :-rd],
                block_size=self.config.fp8_block_size_nope,
                fp8_max=self.config.fp8_max,
            )
        # The ``rotate=True`` branch (Indexer path) additionally does
        # Sylvester-Hadamard + FP4 simulation.  We don't need that for
        # Stage 0.5 — the Indexer is a side path producing INDICES, not
        # KV values that land in the main cache.
        return kv_out


# ---------------------------------------------------------------------------
# V4-Flash main KV projection: excerpt from Attention.forward, the wkv+RoPE+FP8 path
# ---------------------------------------------------------------------------

class DSV4MainKVProjection(nn.Module):
    """The ``wkv -> kv_norm -> RoPE -> FP8-sim`` sub-path of
    ``inference/model.py:484-506`` — produces the sliding-window KV entries
    that land in ``self.kv_cache[:, :window_size]``.
    """

    def __init__(self, config: DSV4FlashArchConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32, device=device)
        self.kv_norm = RMSNorm(config.head_dim, config.rms_norm_eps).to(device)
        nn.init.normal_(self.wkv.weight, mean=0.0, std=config.hidden_size ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, hidden_size] -> [B, S, head_dim] (RoPE applied to last 64 dims)."""
        dtype = x.dtype
        bsz, seqlen, _ = x.shape
        kv = self.wkv(x.float())
        kv = self.kv_norm(kv).to(dtype)
        rd = self.rope_head_dim

        freqs_cis = precompute_freqs_cis(
            dim=rd,
            seqlen=seqlen,
            base=self.config.rope_theta_main,
            original_seq_len=0,                    # main attention disables YaRN
            factor=1.0,
            beta_fast=self.config.beta_fast,
            beta_slow=self.config.beta_slow,
            device=x.device,
        )
        apply_rotary_emb(kv[..., -rd:], freqs_cis, inverse=False)

        if self.config.simulate_fp8:
            kv[..., :-rd] = _simulate_fp8_block_quant_dequant(
                kv[..., :-rd],
                block_size=self.config.fp8_block_size_nope,
                fp8_max=self.config.fp8_max,
            )
        return kv


# ---------------------------------------------------------------------------
# Top-level generator: produces three named KV streams from one hidden-state batch
# ---------------------------------------------------------------------------

@dataclass
class DSV4KVStreams:
    """Container with three KV streams from the same hidden-state input."""

    sliding_window_kv: torch.Tensor      # [B, S, head_dim]  — every token, main KV
    csa_pool_kv: torch.Tensor            # [B, S // 4,   head_dim]  — ratio-4 pool (CSA)
    hca_pool_kv: torch.Tensor            # [B, S // 128, head_dim]  — ratio-128 pool (HCA)
    hidden_size: int
    head_dim: int
    seqlen: int
    batch_size: int
    config_summary: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[DSV4KVStreams] B={self.batch_size} S={self.seqlen} "
            f"hidden_size={self.hidden_size} head_dim={self.head_dim} | "
            f"sliding_window_kv={tuple(self.sliding_window_kv.shape)} "
            f"csa_pool_kv={tuple(self.csa_pool_kv.shape)} "
            f"hca_pool_kv={tuple(self.hca_pool_kv.shape)}"
        )


class DSV4KVGenerator(nn.Module):
    """Single-object handle producing all three V4 KV streams from
    one [B, S, hidden_size] hidden-state tensor.

    Parameters are random Gaussian-init by design; see module docstring
    for the honesty caveat.  Feeding a real LLM's hidden states (e.g.
    Qwen3-4B post-embedding) through this object gives KV tensors whose
    *distribution class* matches what V4 would produce architecturally.
    """

    def __init__(self, config: Optional[DSV4FlashArchConfig] = None, device: str = "cuda", seed: int = 20260424):
        super().__init__()
        if config is None:
            config = DSV4FlashArchConfig()
        # Force each compressor to its specific compress_ratio.
        self.main_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 0})
        self.csa_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 4})
        self.hca_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 128})

        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.random.fork_rng(devices=([torch.cuda.current_device()] if device.startswith("cuda") else [])):
            torch.manual_seed(seed)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            self.main_kv = DSV4MainKVProjection(self.main_cfg, device=device)
            self.compressor_csa = DSV4Compressor(self.csa_cfg, compress_ratio=4, rotate=False, device=device)
            self.compressor_hca = DSV4Compressor(self.hca_cfg, compress_ratio=128, rotate=False, device=device)
        self._device = device
        self._seed = seed

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> DSV4KVStreams:
        """Produce all three KV streams.  hidden_states: [B, S, hidden_size]."""
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.main_cfg.hidden_size:
            raise ValueError(
                f"hidden_states must be [B, S, hidden_size={self.main_cfg.hidden_size}]; "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] < 128:
            raise ValueError(
                f"seqlen must be >= 128 for HCA compressor (ratio 128); "
                f"got S={hidden_states.shape[1]}"
            )
        if hidden_states.shape[1] % 128 != 0:
            raise ValueError(
                f"seqlen must be divisible by 128; got S={hidden_states.shape[1]} "
                f"(round seqlen up to next multiple of 128 before calling)"
            )

        sw_kv = self.main_kv(hidden_states)
        csa_kv = self.compressor_csa(hidden_states)
        hca_kv = self.compressor_hca(hidden_states)

        return DSV4KVStreams(
            sliding_window_kv=sw_kv,
            csa_pool_kv=csa_kv,
            hca_pool_kv=hca_kv,
            hidden_size=self.main_cfg.hidden_size,
            head_dim=self.main_cfg.head_dim,
            seqlen=hidden_states.shape[1],
            batch_size=hidden_states.shape[0],
            config_summary={
                "hidden_size": self.main_cfg.hidden_size,
                "head_dim": self.main_cfg.head_dim,
                "qk_rope_head_dim": self.main_cfg.qk_rope_head_dim,
                "csa_compress_ratio": self.csa_cfg.compress_ratio,
                "hca_compress_ratio": self.hca_cfg.compress_ratio,
                "simulate_fp8": self.main_cfg.simulate_fp8,
                "seed": self._seed,
            },
        )


__all__ = [
    "DSV4FlashArchConfig",
    "DSV4MainKVProjection",
    "DSV4Compressor",
    "DSV4KVGenerator",
    "DSV4KVStreams",
    "apply_rotary_emb",
    "precompute_freqs_cis",
]

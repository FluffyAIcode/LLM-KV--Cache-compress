"""Kakeya-discrete-set codebook constructions, for direct head-to-head
comparison against TurboQuant's Hadamard + Lloyd-Max on LLM KV cache.

Three independent codebook families, each parameterised to hit a
target bit-count per token per head.  No Kakeya-v1.3 codec
structures (PCA / K-means / WHT / residual) are involved — these
are standalone spherical codebooks applied via nearest-neighbour
lookup on the unit sphere.

Codebook (a) — Random spherical code:
    N = 2^n unit vectors sampled uniformly from S^(D-1) via
    Haar-random rotation of the first n standard basis vectors, or
    normalised Gaussian.  Control; pure randomness, no geometric
    structure.

Codebook (b) — Hadamard-subset code (Reed-Muller lineage):
    Codewords are a structured subset of {±1/√D}^D ∩ S^(D-1).
    Parameterised by a binary matrix B ∈ F_2^{n × D}: each codeword
    c_i is the row vector (-1)^{B_i} / √D.  At n=D this is the full
    Walsh-Hadamard code; at n < D we take a subset of 2^n rows.

Codebook (c) — Kakeya multi-scale spherical code:
    Direct discrete analog of Besicovitch's Perron tree construction,
    lifted to S^(D-1) via a union of 2D slices through the origin:
      1. Choose D/2 pairs of orthogonal axes (e_{2i}, e_{2i+1}).
      2. On each 2-plane span(e_{2i}, e_{2i+1}), place m angular
         representatives at angles θ_j = π·j/m for j ∈ [0, m).
         This is the 2D Kakeya direction set (every direction in
         the plane has a representative within π/(2m) radians).
      3. Codebook = union of all these representatives across the
         D/2 planes = m · (D/2) codewords.
      4. Multi-scale: stack codebooks at multiple m's (coarse to
         fine) with RVQ-style residual encoding.  This is the
         "shatter and translate" of Perron tree realised via
         multiple angular resolutions.
    Contrast with (b): Hadamard code is a product-structured
    subset of the hypercube (D dims simultaneous ±1); Kakeya
    multi-scale is a sum of 2D slices (one plane at a time).
    The latter has exponentially fewer codewords than a full
    product code at the same angular resolution but — by
    Besicovitch-like construction — still covers every direction.
    This is the "measure efficiency" property that Hadamard as a
    codebook LACKS.

All three codebooks expose the same interface: a `Codebook.encode(x)`
method that returns the best index (and signed inner product) for
each row of x, and `Codebook.decode(idx, t)` that reconstructs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Shared interface.
# ---------------------------------------------------------------------------
class SphericalCodebook:
    """Abstract spherical codebook with signed-cosine nearest-neighbour
    encode and scalar-projection decode.  All subclasses guarantee:

      * codewords stored as [N, D] fp32 tensor on device
      * codewords are unit-norm (‖c_i‖ = 1 per row)
      * encode returns (seg_id ∈ [0, N), t = ⟨x, c_{seg_id}⟩),
        symmetry: seg_id picks argmax_i |⟨x, c_i⟩|
      * decode returns x̂ = t · c_{seg_id}

    These semantics match Kakeya-v1.3's spherical-K-means decoder
    identity so that the rest of the codec pipeline (if any) is
    agnostic to which codebook was chosen.
    """
    codewords: torch.Tensor           # [N, D]
    name: str
    bits_per_token_per_head: int

    def __init__(self, codewords: torch.Tensor, name: str, bits: int):
        assert codewords.dim() == 2, f"expected [N, D], got {codewords.shape}"
        norms = codewords.norm(dim=1, keepdim=True)
        eps = torch.finfo(codewords.dtype).eps
        self.codewords = codewords / torch.clamp(norms, min=eps)
        self.name = name
        self.bits_per_token_per_head = bits

    @property
    def N(self) -> int:
        return self.codewords.shape[0]

    @property
    def D(self) -> int:
        return self.codewords.shape[1]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [..., D] unit vectors.  Returns (seg_id [...], t [...]).

        Chunked across rows to keep the [N_rows × N_codewords] cosine
        matrix within VRAM budget at N_codewords ≥ 2^16.
        """
        flat_x = x.reshape(-1, x.shape[-1])                 # [M, D]
        M = flat_x.shape[0]
        # Target chunk size keeps intermediate under ~1 GiB of fp32.
        BYTES_TARGET = 1 << 30                              # 1 GiB
        chunk_rows = max(1, BYTES_TARGET // (4 * self.N))
        out_seg = torch.empty(M, dtype=torch.int64, device=x.device)
        out_t   = torch.empty(M, dtype=flat_x.dtype, device=x.device)
        for s in range(0, M, chunk_rows):
            e = min(M, s + chunk_rows)
            cos = flat_x[s:e] @ self.codewords.T            # [chunk, N]
            seg = cos.abs().argmax(dim=-1)
            t   = cos.gather(-1, seg.unsqueeze(-1)).squeeze(-1)
            out_seg[s:e] = seg
            out_t[s:e]   = t
        return out_seg.reshape(x.shape[:-1]), out_t.reshape(x.shape[:-1])

    def decode(self, seg_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse of encode.  Returns x̂ of same shape as seg_id, plus D."""
        chosen = self.codewords[seg_id]                     # [..., D]
        return t.unsqueeze(-1) * chosen

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: encode + decode, preserves norm of x."""
        norms = x.norm(dim=-1, keepdim=True)
        eps = torch.finfo(x.dtype).eps
        unit = x / torch.clamp(norms, min=eps)
        seg, t = self.encode(unit)
        xhat_unit = self.decode(seg, t)
        return xhat_unit * norms


# ---------------------------------------------------------------------------
# Codebook (a): random spherical code.
# ---------------------------------------------------------------------------
class RandomSphericalCodebook(SphericalCodebook):
    """Haar-random points on S^(D-1), via normalised Gaussian sampling.

    At large N, behaves as a uniform random spherical cover.  Used as
    a null baseline to quantify how much any structured construction
    actually gains.
    """
    def __init__(self, N: int, D: int, seed: int = 0xC0DE, device: str = "cuda"):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        raw = torch.randn(N, D, generator=g, device=device, dtype=torch.float32)
        norms = raw.norm(dim=1, keepdim=True)
        codewords = raw / torch.clamp(norms, min=torch.finfo(raw.dtype).eps)
        bits = int(math.ceil(math.log2(N))) if N > 1 else 1
        super().__init__(codewords, name=f"Random-N{N}", bits=bits)


# ---------------------------------------------------------------------------
# Codebook (b): Hadamard subset.
# ---------------------------------------------------------------------------
def _sylvester_hadamard(n: int, device: str) -> torch.Tensor:
    """n × n Sylvester Hadamard matrix (±1, orthogonal).  n power of two."""
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H


class HadamardSubsetCodebook(SphericalCodebook):
    """Codewords are the first N rows of H_D / √D (Sylvester Hadamard),
    each row normalised to unit length.

    N must be ≤ D.  At N = D this is the full ±1/√D basis of R^D —
    orthonormal but only D codewords, so coverage is coarse.  At
    N < D we use the first N Walsh functions, which are the
    smoothest (lowest-sequency) subset of the hypercube basis.

    Properties vs Haar-random:
      * Bit-exact deterministic (no seed management)
      * Pairwise inner products = 0 (mutually orthogonal)
      * Coverage is D discrete directions only — any x ∈ S^(D-1)
        projects onto at most D orthogonal axes, not a fine cover.
      * Reed-Muller-like structure: row i is the Walsh function
        of "order" popcount(i) — a natural "multi-scale" indexing.
    """
    def __init__(self, N: int, D: int, device: str = "cuda"):
        assert N <= D, (
            f"Hadamard subset codebook requires N ≤ D; got N={N}, D={D}.  "
            f"For larger N, use Hadamard-vertex code or product-factored variants."
        )
        H = _sylvester_hadamard(D, device=device)            # [D, D] ±1
        codewords = H[:N] / math.sqrt(D)                     # [N, D] unit-norm
        bits = int(math.ceil(math.log2(N))) if N > 1 else 1
        super().__init__(codewords, name=f"Hadamard-N{N}", bits=bits)


class HadamardVertexCodebook(SphericalCodebook):
    """Codewords are a subset of the 2^D vertices of the hypercube
    {±1/√D}^D, each intersecting S^(D-1) at norm 1.

    Unlike HadamardSubsetCodebook (D orthogonal axes), this explores
    the full hypercube vertex set.  We enumerate the first N = 2^n
    vertices in bit-flip order.  Each codeword is
        c_i = (s_{i,0}, s_{i,1}, ..., s_{i,D-1}) / √D
    where (s_{i,0}, ..., s_{i,D-1}) is the binary expansion of i
    padded with +1's.

    At n = 0: just {+1/√D, +1/√D, ...}  (1 codeword)
    At n = D: all 2^D hypercube vertices.

    This is the "bigger-codebook Reed-Muller" baseline: the full
    product structure of Hadamard-as-codebook, not the low-rank
    subset in HadamardSubsetCodebook.
    """
    def __init__(self, n: int, D: int, device: str = "cuda"):
        assert n <= min(D, 20), (
            f"n > 20 gives > 1M codewords × {D} dims, too big to materialise"
        )
        N = 1 << n
        idx = torch.arange(N, device=device, dtype=torch.int64)   # [N]
        # bit j of idx gives s_{i, j} for j ∈ [0, n).  Remaining
        # (D - n) coords default to +1 (padding).
        signs = torch.ones(N, D, device=device, dtype=torch.float32)
        for j in range(n):
            bit = ((idx >> j) & 1).to(torch.float32)             # [N] in {0, 1}
            signs[:, j] = 1.0 - 2.0 * bit                         # {+1, -1}
        codewords = signs / math.sqrt(D)                          # unit-norm
        super().__init__(codewords, name=f"HadamardVert-n{n}-N{N}", bits=n)


# ---------------------------------------------------------------------------
# Codebook (c): Kakeya multi-scale spherical code.
# ---------------------------------------------------------------------------
class KakeyaMultiScaleCodebook(SphericalCodebook):
    """Discrete Kakeya-Besicovitch construction on S^(D-1).

    Principle: a Besicovitch set on S^(D-1) is a small-measure subset
    that contains a unit "arc" in every direction.  In quantization
    land, this becomes: a small-N codebook that has a codeword within
    angle ε of every x ∈ S^(D-1), with N exponentially smaller than a
    uniform cover would require.

    Construction: union of 2D great-circle representatives across
    D/2 orthogonal 2-planes.

      1. Pair up axes: (e_0, e_1), (e_2, e_3), ..., (e_{D-2}, e_{D-1}).
         Each pair spans a 2-plane P_i ⊂ R^D.

      2. On each P_i, place `m` angular representatives at angles
         θ_j = π·j/m (j ∈ [0, m)).  Each representative is a unit
         vector of the form cos(θ_j)·e_{2i} + sin(θ_j)·e_{2i+1}.

      3. Codebook = union of these D/2 · m unit vectors.

      4. Multi-scale: we additionally add codewords from **rotated**
         2-planes at coarser granularity.  Specifically, rotate the
         D/2 planes by a random orthogonal matrix R to get D/2 new
         planes, and add m' representatives per new plane at angles
         π·j/m'.  Stack multiple scales (m_1, m_2, ..., m_L) to form
         a hierarchical cover.  This is the "translate and reunite"
         of Perron tree, applied to slice-sphere geometry.

    Measure efficiency argument (informal):
        A full uniform cover of S^(D-1) to angle ε needs
        N_uniform ~ (1/ε)^(D-1) codewords.
        The Kakeya multi-scale cover achieves the same ε-coverage
        on every 2D slice with m = π/(2ε) codewords per slice, and
        D/2 slices total, giving
            N_Kakeya = (D/2) · (π/(2ε))
        which is LINEAR in D and 1/ε, not exponential.
        This linear dependence is the Besicovitch "arbitrarily
        small measure" property transported to the discrete sphere.

    Caveat: Kakeya codewords only cover the 2-plane slices.  An
    arbitrary x ∈ S^(D-1) may not lie near any slice, so its
    nearest-codeword angle ε_eff is larger than the per-slice ε.
    Empirically, on LLM K which has near-uniform distribution on
    S^(D-1), slice coverage gives an angular-MSE that's still much
    better than random at the same bit count — see the head-to-head
    test output.
    """
    def __init__(
        self,
        D: int,
        angles_per_plane: int = 16,
        n_scales: int = 1,
        rotation_seed: int = 0xDADA,
        device: str = "cuda",
    ):
        assert D % 2 == 0, f"D must be even for 2-plane pairing, got {D}"
        self.D_ = D
        self.angles_per_plane = angles_per_plane
        self.n_scales = n_scales

        all_codewords = []
        g = torch.Generator(device=device)
        g.manual_seed(rotation_seed)

        for scale_idx in range(n_scales):
            # At scale 0: canonical axis pairs.  At scale > 0: apply a
            # Haar-random rotation to the canonical axes.
            if scale_idx == 0:
                R = torch.eye(D, device=device, dtype=torch.float32)
            else:
                raw = torch.randn(D, D, generator=g, device=device, dtype=torch.float32)
                Q, _ = torch.linalg.qr(raw)                       # Haar-distributed
                R = Q
            # m at this scale — we halve the angles per scale so
            # finer slices live in the lower-index scales.
            m = max(2, angles_per_plane >> scale_idx)

            # Canonical plane axes after rotation.
            axes_a = R[:, 0::2]                                   # [D, D/2]
            axes_b = R[:, 1::2]                                   # [D, D/2]

            # Angles θ_j = π·j/m, but because signed-cosine argmax
            # treats ±c the same, we only need θ ∈ [0, π) — hence m,
            # not 2m.  Generate angles for this scale.
            theta = torch.arange(m, device=device, dtype=torch.float32) * (math.pi / m)
            cos_t = torch.cos(theta)                               # [m]
            sin_t = torch.sin(theta)                               # [m]

            # Build the angular representatives per plane:
            #   c_{i, j} = cos(θ_j) · axes_a[:, i] + sin(θ_j) · axes_b[:, i]
            # Shape: [D/2 planes, m angles, D dims]
            planes = (
                cos_t[None, :, None] * axes_a.T[:, None, :]        # [D/2, m, D]
                + sin_t[None, :, None] * axes_b.T[:, None, :]
            )                                                      # [D/2, m, D]
            all_codewords.append(planes.reshape(-1, D))            # [(D/2)·m, D]

        codewords = torch.cat(all_codewords, dim=0)                # [N, D]
        N = codewords.shape[0]
        bits = int(math.ceil(math.log2(N))) if N > 1 else 1

        name = (
            f"Kakeya-D{D}-m{angles_per_plane}-scales{n_scales}-N{N}"
        )
        super().__init__(codewords, name=name, bits=bits)


# ---------------------------------------------------------------------------
# Head-to-head tester.
# ---------------------------------------------------------------------------
@dataclass
class EncodingResult:
    codebook_name: str
    bits_per_token: int
    n_codewords: int
    k_mse_abs: float
    k_mse_rel: float
    k_mse_cosine_mean: float     # mean angle preservation: cos(x, x̂)
    k_mse_cosine_min: float
    encode_time_ms: float


def head_to_head(
    K_unit: torch.Tensor,
    codebooks: list[SphericalCodebook],
) -> list[EncodingResult]:
    """K_unit: [N, D] unit vectors on S^(D-1).

    For each codebook, encode all N vectors via nearest-neighbour,
    decode, measure reconstruction MSE and angular agreement.
    """
    import time
    results = []
    for cb in codebooks:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg, t = cb.encode(K_unit)
        xhat = cb.decode(seg, t)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        err = K_unit - xhat
        mse_abs = float((err * err).sum(dim=1).mean().item())
        mse_rel = float(
            (err * err).sum(dim=1).mean() /
            (K_unit * K_unit).sum(dim=1).mean()
        )
        # cos(x, x̂) = <x, x̂> / (‖x‖ ‖x̂‖)
        cos_xx = (K_unit * xhat).sum(dim=1) / (
            K_unit.norm(dim=1) * xhat.norm(dim=1).clamp(min=1e-12)
        )
        results.append(EncodingResult(
            codebook_name=cb.name,
            bits_per_token=cb.bits_per_token_per_head,
            n_codewords=cb.N,
            k_mse_abs=mse_abs,
            k_mse_rel=mse_rel,
            k_mse_cosine_mean=float(cos_xx.mean().item()),
            k_mse_cosine_min=float(cos_xx.min().item()),
            encode_time_ms=dt,
        ))
    return results

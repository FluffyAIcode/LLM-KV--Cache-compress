#!/usr/bin/env python3
"""Compute compression ratio for v1.3 + guardrails configs with and
without V Besi on DS-Distill (D=128, ctx=2048, 6 boundary layers,
b=3 K + optional outlier).

Byte model is the SAME one used in reports/v1_3_revival/FINDINGS.md
and reports/v1_3_riemann_b2/FINDINGS.md, re-implemented here so the
NB3/NR3 numbers are apples-to-apples with the original B3/R3.

Model params (deepseek-r1-distill-qwen-1.5b):
  * num_hidden_layers = 28
  * num_kv_heads      = 2   (GQA)
  * head_dim          = 128
  * ctx_len           = 2048
  * n_compressed_rows_per_layer = n_kv * ctx_len = 2*2048 = 4096
  * block_size        = 512   -> 8 blocks per layer
  * bf16 raw bytes / layer = 2 * ctx_len * D * 2 * 2 (K+V) = 2,097,152
                             = 2 (bf16) * 2 (heads) * 2048 * 128 * 2 (K+V)

Boundary layers (6): [0, 1, 7, 14, 26, 27] -- kept bf16 (no compression)
Middle layers       : 22 -- compressed

Byte breakdown per middle layer (per vector):
  * K (kakeyaturbo RSVD b=3 + outlier):
      skeleton (mean 256B + basis 64*128*f16 = 16384B) = 16640 B/block
      codes = bit_width * D / 8 = 3 * 128 / 8 = 48 B/v
      outlier list: T=2.0 -> 4.5% of coords * 4B(u16+f16) = 23.04 B/v
                    T=1.5 -> 13.4% of coords * 4B         = 68.61 B/v
  * V kakeyaturbo RSVD b=3:
      skeleton: same as K = 16640 B/block
      codes   : 48 B/v  (b=3), or 32 B/v (b=2)
      no outlier (v_outlier_threshold=None)
  * V besicovitch d=3 m=4:
      direction_bits=3, magnitude_bits=4 per group-of-2
      per vector: (3 + 4) * (128/2) / 8 = 56 B/v  (codes only)
      plus subtract_mean: 128*f16 = 256B per block = 0.5 B/v
      --> ~56.5 B/v
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def layer_bytes(
    *,
    block_size: int,
    block_vectors: int,  # = block_size; kept explicit for clarity
    num_blocks: int,     # per layer per stream
    head_dim: int,
    n_kv_heads: int,
    seq: int,
    # K config
    k_bit_width: int,
    k_rsvd_rank: int,       # RSVD target_rank
    k_outlier_frac: float,  # e.g. 0.045 for T=2.0, 0.134 for T=1.5
    # V config
    v_codec: str,           # 'kakeyaturbo' or 'besicovitch'
    v_bit_width: int | None,
    v_rsvd_rank: int | None,
    v_share_basis: bool = False,
    besi_g: int = 2,
    besi_db: int = 3,
    besi_mb: int = 4,
    besi_subtract_mean: bool = True,
) -> dict:
    """Compute compressed bytes for ONE non-boundary layer (K stream + V stream)."""
    n_vec = n_kv_heads * seq  # 2 * 2048 = 4096
    # K: kakeyaturbo, shared across K stream
    k_skel_bytes = (head_dim * 2) + (k_rsvd_rank * head_dim * 2)  # mean f16 + basis f16
    k_code_bytes_per_v = k_bit_width * head_dim // 8
    # u16 (idx) + f16 (val) = 4 B per outlier coord
    k_outlier_bytes_per_v = int(round(k_outlier_frac * head_dim * 4))
    k_bytes = num_blocks * k_skel_bytes + n_vec * (k_code_bytes_per_v + k_outlier_bytes_per_v)

    # V:
    if v_codec == "kakeyaturbo":
        v_skel_bytes = (head_dim * 2) + (v_rsvd_rank * head_dim * 2)
        v_code_bytes_per_v = v_bit_width * head_dim // 8
        # share_basis_v: one V skeleton per LAYER instead of per BLOCK
        #   -> skel cost / n_vec  instead of skel cost * num_blocks / n_vec
        v_skel_blocks = 1 if v_share_basis else num_blocks
        v_bytes = v_skel_blocks * v_skel_bytes + n_vec * v_code_bytes_per_v
    elif v_codec == "besicovitch":
        # per vector: (direction_bits + magnitude_bits) bits per group
        # groups = head_dim / g, bits per vector = (db+mb) * (D/g)
        bits_per_vec = (besi_db + besi_mb) * (head_dim // besi_g)
        v_code_bytes_per_v = bits_per_vec / 8
        # subtract_mean: 128 f16 per block
        v_skel_bytes = (head_dim * 2) if besi_subtract_mean else 0
        v_bytes = int(round(num_blocks * v_skel_bytes + n_vec * v_code_bytes_per_v))
    else:
        raise ValueError(v_codec)

    # raw bf16 cost for this layer (K + V)
    raw_bytes_per_stream = n_vec * head_dim * 2  # bf16
    raw_bytes = 2 * raw_bytes_per_stream
    return {
        "raw": raw_bytes,
        "k": k_bytes,
        "v": v_bytes,
        "compressed": k_bytes + v_bytes,
        "ratio_layer": raw_bytes / (k_bytes + v_bytes),
    }


def model_ratio(
    *,
    n_layers: int = 28,
    boundary_layers: tuple[int, ...] = (0, 1, 7, 14, 26, 27),
    seq: int = 2048,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    block_size: int = 512,
    k_bit_width: int = 3,
    k_outlier_frac: float = 0.045,
    v_codec: str = "kakeyaturbo",
    v_bit_width: int | None = 3,
    boundary_bit_width: int = 4,
    boundary_rsvd_rank: int = 64,
    k_rsvd_rank: int = 64,
    v_rsvd_rank: int = 64,
    v_share_basis: bool = False,
    besi_db: int = 3,
    besi_mb: int = 4,
    besi_g: int = 2,
) -> dict:
    """Compute total-layer ratio for the full model.

    Boundary layers are kept BF16 (no compression).
    """
    n_vec = n_kv_heads * seq
    num_blocks = n_vec // block_size
    raw_per_layer = 2 * n_vec * head_dim * 2  # K + V bf16

    middle_layers = [i for i in range(n_layers) if i not in boundary_layers]
    n_middle = len(middle_layers)
    n_bdry = len(boundary_layers)

    mid = layer_bytes(
        block_size=block_size, block_vectors=block_size, num_blocks=num_blocks,
        head_dim=head_dim, n_kv_heads=n_kv_heads, seq=seq,
        k_bit_width=k_bit_width, k_rsvd_rank=k_rsvd_rank,
        k_outlier_frac=k_outlier_frac,
        v_codec=v_codec, v_bit_width=v_bit_width, v_rsvd_rank=v_rsvd_rank,
        v_share_basis=v_share_basis,
        besi_g=besi_g, besi_db=besi_db, besi_mb=besi_mb,
    )

    # Boundary: bf16 (raw)
    bdry_compressed = raw_per_layer

    total_raw = n_layers * raw_per_layer
    total_compressed = n_middle * mid["compressed"] + n_bdry * bdry_compressed
    ratio = total_raw / total_compressed

    return {
        "total_raw_bytes": total_raw,
        "total_compressed_bytes": total_compressed,
        "middle_layer_bytes": mid["compressed"],
        "middle_k_bytes": mid["k"],
        "middle_v_bytes": mid["v"],
        "middle_ratio": mid["ratio_layer"],
        "model_ratio": ratio,
        "n_middle": n_middle,
        "n_bdry": n_bdry,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=[
        "B3", "R3", "NB3", "NR3", "NB3v2", "NR3v2",
        "NB3sv3", "NR3sv3", "NB3sv2", "NR3sv2",
    ], required=True)
    args = ap.parse_args()

    # Original configs with V Besi
    presets = {
        # B3 = RSVD b=3 + K cal + outlier T=2.0 + V Besi d=3 m=4 + 6 bdry
        "B3":  dict(k_bit_width=3, k_outlier_frac=0.045,
                    v_codec="besicovitch", v_bit_width=None, v_rsvd_rank=None),
        "R3":  dict(k_bit_width=3, k_outlier_frac=0.134,
                    v_codec="besicovitch", v_bit_width=None, v_rsvd_rank=None),
        # NB3/NR3: drop V Besi -> V kakeyaturbo RSVD b=3
        "NB3": dict(k_bit_width=3, k_outlier_frac=0.045,
                    v_codec="kakeyaturbo", v_bit_width=3, v_rsvd_rank=64),
        "NR3": dict(k_bit_width=3, k_outlier_frac=0.134,
                    v_codec="kakeyaturbo", v_bit_width=3, v_rsvd_rank=64),
        # v2: V at b=2 (with V RSVD b=2 centroid calibration)
        "NB3v2": dict(k_bit_width=3, k_outlier_frac=0.045,
                      v_codec="kakeyaturbo", v_bit_width=2, v_rsvd_rank=64),
        "NR3v2": dict(k_bit_width=3, k_outlier_frac=0.134,
                      v_codec="kakeyaturbo", v_bit_width=2, v_rsvd_rank=64),
        # share-basis-v: V skeleton amortised across whole layer (v1.3 default)
        "NB3sv3": dict(k_bit_width=3, k_outlier_frac=0.045,
                       v_codec="kakeyaturbo", v_bit_width=3, v_rsvd_rank=64,
                       v_share_basis=True),
        "NR3sv3": dict(k_bit_width=3, k_outlier_frac=0.134,
                       v_codec="kakeyaturbo", v_bit_width=3, v_rsvd_rank=64,
                       v_share_basis=True),
        "NB3sv2": dict(k_bit_width=3, k_outlier_frac=0.045,
                       v_codec="kakeyaturbo", v_bit_width=2, v_rsvd_rank=64,
                       v_share_basis=True),
        "NR3sv2": dict(k_bit_width=3, k_outlier_frac=0.134,
                       v_codec="kakeyaturbo", v_bit_width=2, v_rsvd_rank=64,
                       v_share_basis=True),
    }
    cfg = presets[args.config]
    res = model_ratio(**cfg)
    print(f"=== {args.config} ===")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:>12.3f}")
        else:
            print(f"  {k:30s} {v:>12}")


if __name__ == "__main__":
    main()

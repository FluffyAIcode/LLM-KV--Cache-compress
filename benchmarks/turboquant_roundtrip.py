"""TurboQuant roundtrip adapter for the PPL harness.

Uses the reference Python implementation at `turboquant_plus/turboquant/
turboquant.py` (PolarQuant + QJL, Algorithm 2 from the paper).

Public entry points:
  - turboquant_k_roundtrip(arr, bit_width, seed=42)
        In: [N, D] fp32, Out: [N, D] fp32 after quantize→dequantize with
        TurboQuant (K stream: InnerProduct variant = PolarQuant + QJL).
  - turboquant_v_roundtrip(arr, bit_width, seed=42)
        Same, but TurboQuantMSE (V stream: MSE-only PolarQuant).

Both return the roundtripped tensor plus a cheap report dict (bytes,
MSE) that matches the shape the Rust codec reporter emits, so the rest
of the e2e_ppl_pre_rope pipeline can stay metric-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TQ_PATH = REPO / "turboquant_plus"
if str(TQ_PATH) not in sys.path:
    sys.path.insert(0, str(TQ_PATH))

from turboquant.turboquant import TurboQuant, TurboQuantMSE  # noqa: E402


def turboquant_k_roundtrip(arr: np.ndarray, bit_width: int = 3,
                            seed: int = 42) -> tuple[np.ndarray, dict]:
    """K-stream roundtrip: TurboQuant (PolarQuant + QJL) at `bit_width` bits."""
    assert arr.ndim == 2, arr.shape
    n, d = arr.shape
    tq = TurboQuant(d=d, bit_width=bit_width, seed=seed)
    compressed = tq.quantize(arr.astype(np.float32, copy=False))
    arr_hat = tq.dequantize(compressed).astype(np.float32, copy=False)
    mse = float(np.mean((arr - arr_hat) ** 2))
    bytes_compressed = int(tq.compressed_size_bits(n) / 8)
    return arr_hat, {
        "mean_block_mse": mse,
        "compressed_bytes": bytes_compressed,
        "ratio_vs_bf16": (n * d * 2) / max(bytes_compressed, 1),
        "codec": "turboquant",
        "bit_width": bit_width,
    }


def turboquant_v_roundtrip(arr: np.ndarray, bit_width: int = 3,
                            seed: int = 42) -> tuple[np.ndarray, dict]:
    """V-stream roundtrip: TurboQuantMSE (PolarQuant only) at `bit_width` bits."""
    assert arr.ndim == 2, arr.shape
    n, d = arr.shape
    tq = TurboQuantMSE(d=d, bit_width=bit_width, seed=seed)
    indices, norms = tq.quantize(arr.astype(np.float32, copy=False))
    arr_hat = tq.dequantize(indices, norms).astype(np.float32, copy=False)
    mse = float(np.mean((arr - arr_hat) ** 2))
    # MSE variant: no QJL bytes, b bits/coord + 32-bit norm
    bytes_compressed = int(n * (d * bit_width + 32) / 8)
    return arr_hat, {
        "mean_block_mse": mse,
        "compressed_bytes": bytes_compressed,
        "ratio_vs_bf16": (n * d * 2) / max(bytes_compressed, 1),
        "codec": "turboquant_mse",
        "bit_width": bit_width,
    }


def turboquant_ratio(d: int, bit_width: int, variant: str = "k") -> float:
    """Byte ratio relative to bf16 for a TurboQuant config.
    `variant='k'` uses PolarQuant+QJL; `variant='v'` uses PolarQuant only."""
    if variant == "k":
        return TurboQuant(d=d, bit_width=bit_width).compression_ratio(16)
    elif variant == "v":
        # TurboQuantMSE: d*b + 32 bits per vector, bf16 = d*16 per vector
        return (d * 16) / (d * bit_width + 32)
    raise ValueError(f"variant must be 'k' or 'v', got {variant}")

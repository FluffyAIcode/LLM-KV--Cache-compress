"""Q-preconditioning utility for the K-stream codec.

This module owns the math

    K_tilde = K @ L                (whiten)
    K_hat   = K_hat_tilde @ L^{-1} (un-whiten)

where L is the lower-triangular Cholesky factor of the per-(layer, kv_head)
query Gram matrix Sigma_q produced by `benchmarks/q_calibration.py`.

Minimising standard MSE on K_tilde is mathematically equivalent to
minimising the Sigma_q-weighted distortion on K, which is exactly the
"InnerProduct on K" failure metric that v1.3 paper §2 claims but the
current codec (per-coordinate MSE on K) does not actually enforce.

The Rust codec is not touched.  Whitening is done in Python, right
before the tensor is serialised to the KKTV format and sent into the
bench binary.  Unwhitening is done right after the decoded tensor is
read back.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class QPrecond:
    """Per-layer, per-kv-head Cholesky factors L and L^{-1} held in fp32 cpu.

    Shapes:
        chol[l]     : [n_kv, D, D]  lower triangular
        inv_chol[l] : [n_kv, D, D]  lower triangular (L^{-1})
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        cfg = json.loads(path.with_suffix(".json").read_text())
        self.head_dim = cfg["head_dim"]
        self.n_kv = cfg["num_kv_heads"]
        self.n_layers = cfg["num_layers"]
        self.layer_types = cfg["layer_types"]
        from safetensors.torch import load_file
        tensors = load_file(str(path))
        self.chol: dict[int, np.ndarray] = {}
        self.inv_chol: dict[int, np.ndarray] = {}
        for l in range(self.n_layers):
            k_chol = f"layer_{l}_chol"
            k_inv = f"layer_{l}_inv_chol"
            if k_chol in tensors:
                self.chol[l]     = tensors[k_chol].numpy().astype(np.float32)
                self.inv_chol[l] = tensors[k_inv].numpy().astype(np.float32)

    @property
    def n_calibrated_layers(self) -> int:
        return len(self.chol)

    def whiten(self, k_per_head: np.ndarray, layer: int) -> np.ndarray:
        """Input: K with shape [seq, n_kv, D] (pre-RoPE, one passage, one layer).
        Output: same shape, whitened per kv-head."""
        assert k_per_head.ndim == 3
        assert k_per_head.shape[1] == self.n_kv
        assert k_per_head.shape[2] == self.head_dim
        L = self.chol[layer]                # [n_kv, D, D]
        # K[t, h, :] @ L[h, :, :]  → K_tilde[t, h, :]
        return np.einsum("thj,hjk->thk", k_per_head, L, optimize=True).astype(
            np.float32, copy=False
        )

    def unwhiten(self, k_tilde_per_head: np.ndarray, layer: int) -> np.ndarray:
        """Inverse of `whiten`.  Applies L^{-1} on the right per kv-head."""
        assert k_tilde_per_head.ndim == 3
        Linv = self.inv_chol[layer]
        return np.einsum("thj,hjk->thk", k_tilde_per_head, Linv, optimize=True).astype(
            np.float32, copy=False
        )


def sanity_check(qp: QPrecond) -> dict:
    """Verify whiten ∘ unwhiten ≈ identity on random data for every layer.

    Reports max absolute error and max relative error (per layer).  A
    correctly-built QPrecond will have errors ≲ 1e-5 (fp32 round-off).
    """
    rng = np.random.default_rng(0)
    results = []
    for l, L in qp.chol.items():
        x = rng.normal(size=(128, qp.n_kv, qp.head_dim)).astype(np.float32)
        x_tilde = qp.whiten(x, l)
        x_back = qp.unwhiten(x_tilde, l)
        abs_err = float(np.abs(x - x_back).max())
        rel_err = float(np.abs(x - x_back).max() / max(np.abs(x).max(), 1e-30))
        results.append({"layer": l, "max_abs_err": abs_err, "max_rel_err": rel_err})
    return {
        "max_abs_err": max(r["max_abs_err"] for r in results),
        "max_rel_err": max(r["max_rel_err"] for r in results),
        "per_layer": results,
    }


def load(path: Optional[str | Path]) -> Optional[QPrecond]:
    if path is None:
        return None
    return QPrecond(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    args = ap.parse_args()
    qp = QPrecond(args.calib)
    print(f"loaded {qp.n_calibrated_layers} layers, n_kv={qp.n_kv}, D={qp.head_dim}")
    san = sanity_check(qp)
    print(f"sanity: max_abs_err={san['max_abs_err']:.3e}  "
          f"max_rel_err={san['max_rel_err']:.3e}")
    # Also show anisotropy of L directly (not Sigma)
    for l in sorted(qp.chol)[:3]:
        L = qp.chol[l][0]  # kv-head 0
        print(f"  layer {l} kv0 L diag[:5]: {np.diag(L)[:5]}")

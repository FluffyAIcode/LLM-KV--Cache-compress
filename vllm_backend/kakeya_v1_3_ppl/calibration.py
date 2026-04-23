"""Load M2 calibration artefacts into per-layer state.

M2 (PLAN.md §M2) produced three files per model:

    reports/v1_3_ppl/vllm_backend/calibration/
      qwen3_4b_sigma_q.safetensors      per-(layer, kv-head) L, L^{-1}
      qwen3_4b_lloyd_max_K_b3.f32       K-stream Lloyd-Max table (8 fp32)
      qwen3_4b_lloyd_max_V_b2.f32       V-stream Lloyd-Max table (4 fp32)

Every layer's whitening / unwhitening is a right-multiplication by a
`[n_kv_heads, D, D]` Cholesky factor; the calibrated tables replace
the Gaussian-default Lloyd-Max codebook used in Phase B.1.

Contract:

  * L, L^{-1} shapes: `[n_kv, D, D]` fp32, one per full-attention layer.
    Layers not in the safetensors (skipped during calibration) are
    treated as identity — consistent with M2's RESUME.md §"Known
    decisions" (layer 0 and boundary layers are skip-listed).

  * Lloyd-Max tables: flat fp32 binary, length `2^b` with strictly
    ascending entries.  Phase B-2a loads them once at
    `_ensure_layer_state` and pins them on every layer so encode /
    decode can pass them as `custom_centroids`.

This module has zero dependency on vllm internals, so it can be
unit-tested on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CalibrationBundle:
    """Layer-level Σ_q factors + stream-level Lloyd-Max tables.

    `sigma_q_chol[layer] = L` is the lower-triangular Cholesky
    factor such that `Σ_q = L · Lᵀ`.  `sigma_q_inv_chol[layer] =
    L^{-1}`.  Both are shape `[n_kv, D, D]`, fp32.  Layers not
    calibrated (e.g. layer 0) are **absent** from the dicts — the
    backend treats absence as identity, matching the old PR #15
    harness.
    """

    head_dim: int
    num_kv_heads: int
    num_layers: int
    # {layer_index: fp32 tensor [n_kv, D, D]}
    sigma_q_chol: dict[int, np.ndarray]
    sigma_q_inv_chol: dict[int, np.ndarray]
    # Per-stream Lloyd-Max tables.
    lloyd_max_k: Optional[np.ndarray] = None   # fp32, length 2^b_K
    lloyd_max_v: Optional[np.ndarray] = None   # fp32, length 2^b_V

    def active_layers(self) -> set[int]:
        """Layers for which whitening / unwhitening is non-trivial."""
        return set(self.sigma_q_chol.keys())

    def whiten_layer_head(self, K: np.ndarray, layer: int, head: int
                          ) -> np.ndarray:
        """Whiten one kv-head's K tensor: `K_tilde = K @ L[head]`.

        Input  K shape: `[seq, D]` fp32, C-contig.
        Output   shape: `[seq, D]` fp32, C-contig.
        """
        if layer not in self.sigma_q_chol:
            return np.ascontiguousarray(K, dtype=np.float32)
        L = self.sigma_q_chol[layer][head]  # [D, D]
        return np.ascontiguousarray(K @ L, dtype=np.float32)

    def unwhiten_layer_head(self, K_hat_tilde: np.ndarray, layer: int,
                            head: int) -> np.ndarray:
        """Inverse of `whiten_layer_head`: `K_hat = K_hat_tilde @ L^{-1}`."""
        if layer not in self.sigma_q_inv_chol:
            return np.ascontiguousarray(K_hat_tilde, dtype=np.float32)
        Linv = self.sigma_q_inv_chol[layer][head]
        return np.ascontiguousarray(K_hat_tilde @ Linv, dtype=np.float32)


def load_calibration_bundle(
    sigma_q_safetensors: str | Path,
    k_centroids_f32: str | Path | None = None,
    v_centroids_f32: str | Path | None = None,
    *,
    skip_layers: Optional[list[int]] = None,
) -> CalibrationBundle:
    """Load M2 artefacts.

    Loading strategy:
      * `sigma_q_safetensors` is required; its sidecar `.json` tells
        us `head_dim`, `num_kv_heads`, `num_layers`, `layer_types`.
      * `k_centroids_f32` / `v_centroids_f32` are optional — if None,
        the backend falls back to the Gaussian default Lloyd-Max
        table at runtime.  This is explicit in the result dataclass
        via `lloyd_max_k is None` / `lloyd_max_v is None`.
      * `skip_layers` is applied on top of the calibration's own
        skip set; the final set is union (a layer skipped either
        at calibration OR at load time gets identity whitening).

    Raises:
      FileNotFoundError — safetensors or sidecar missing.
      ValueError — shape mismatch between sidecar config and loaded
                    tensors.
    """
    sigma_path = Path(sigma_q_safetensors)
    if not sigma_path.exists():
        raise FileNotFoundError(f"M2 Σ_q safetensors missing: {sigma_path}")
    sidecar = sigma_path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"M2 Σ_q sidecar JSON missing: {sidecar}")

    import json
    cfg = json.loads(sidecar.read_text())
    head_dim = int(cfg["head_dim"])
    num_kv_heads = int(cfg["num_kv_heads"])
    num_layers = int(cfg["num_layers"])

    # safetensors is torch-only here because that's what M2 writes.
    from safetensors.torch import load_file
    tensors = load_file(str(sigma_path))

    skip = set(skip_layers or [])
    chol: dict[int, np.ndarray] = {}
    inv_chol: dict[int, np.ndarray] = {}
    for l in range(num_layers):
        if l in skip:
            continue
        kch = f"layer_{l}_chol"
        kinv = f"layer_{l}_inv_chol"
        if kch not in tensors or kinv not in tensors:
            # layer was skipped during calibration
            continue
        L = tensors[kch].cpu().numpy()
        Linv = tensors[kinv].cpu().numpy()
        if L.shape != (num_kv_heads, head_dim, head_dim):
            raise ValueError(
                f"layer {l} chol shape {L.shape}, expected "
                f"({num_kv_heads}, {head_dim}, {head_dim})"
            )
        if Linv.shape != L.shape:
            raise ValueError(
                f"layer {l} inv_chol shape {Linv.shape} != chol shape {L.shape}"
            )
        chol[l] = L.astype(np.float32, copy=False)
        inv_chol[l] = Linv.astype(np.float32, copy=False)

    def _load_centroids(path: str | Path | None, expected: int
                        ) -> Optional[np.ndarray]:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Lloyd-Max table missing: {p}")
        vals = np.fromfile(p, dtype=np.float32)
        if vals.size != expected:
            raise ValueError(
                f"{p}: {vals.size} fp32 entries, expected {expected}"
            )
        # Enforce strict-ascending (M2 writes them this way; defend
        # against corrupted files).
        if not np.all(np.diff(vals) > 0):
            raise ValueError(f"{p}: centroids not strictly ascending")
        return vals

    # We don't know b_K / b_V from the sidecar; sniff from file sizes.
    k_tbl = None
    v_tbl = None
    if k_centroids_f32 is not None:
        kp = Path(k_centroids_f32)
        nbytes = kp.stat().st_size if kp.exists() else 0
        if nbytes % 4 != 0:
            raise ValueError(f"{kp}: size {nbytes} not a multiple of 4")
        n = nbytes // 4
        if n not in (2, 4, 8, 16):
            raise ValueError(f"{kp}: {n} centroids, must be 2/4/8/16")
        k_tbl = _load_centroids(kp, n)
    if v_centroids_f32 is not None:
        vp = Path(v_centroids_f32)
        nbytes = vp.stat().st_size if vp.exists() else 0
        if nbytes % 4 != 0:
            raise ValueError(f"{vp}: size {nbytes} not a multiple of 4")
        n = nbytes // 4
        if n not in (2, 4, 8, 16):
            raise ValueError(f"{vp}: {n} centroids, must be 2/4/8/16")
        v_tbl = _load_centroids(vp, n)

    return CalibrationBundle(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        sigma_q_chol=chol,
        sigma_q_inv_chol=inv_chol,
        lloyd_max_k=k_tbl,
        lloyd_max_v=v_tbl,
    )

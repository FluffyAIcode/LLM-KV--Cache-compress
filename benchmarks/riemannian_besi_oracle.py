#!/usr/bin/env python3
"""Riemannian Besicovitch oracle — attention-weighted Kakeya-set on K.

Differs from Euclidean Q-precond path by moving the magnitude scale
out of per-vector (`max_k |α_k|`) into per-(layer, group)
offline calibration. This breaks the trilemma that sank the previous
K-Besi + Q-precond + quantized-magnitude attempt.

Pipeline (encode):
    K ∈ R^D, split into G = D/g groups x_k ∈ R^g
    For each group k:
        # Riemannian direction assignment: argmax over |<x_k, d_i>_Σq|
        # where Σq is the block-diagonal 2x2 projection of Σ_q onto group k
        M_k = Σ_q^(k,k)                       # 2x2 block (offline-calibrated)
        d_i ∈ Haar codebook on R^g
        id = argmax_i |x_k^T M_k d_i|         # Riemannian inner product
        α = x_k^T M_k d_{id}                  # raw magnitude in Σq metric
        s_k = offline-calibrated per-(layer, group) scale
        α_q = Lloyd_Max( α / s_k )            # quantize in normalized units

Pipeline (decode):
    α_dequant = s_k · α_q
    x_hat_k = α_dequant · (M_k^{-1} d_{id})   # Riemannian "raise index" to get vector

This is mathematically equivalent to "encode in Σq-weighted inner product,
decode by raising to vector form via M_k^-1", which is the right
Riemannian structure.

Equivalence note: For g=2 and M_k SPD, this reduces to Euclidean Besi
on y = L_k^T x_k where L_k L_k^T = M_k, but with:
  (1) per-(layer, group) fixed L_k (instead of per-layer full Cholesky)
  (2) per-group offline scale s_k (instead of per-vector max|α_k|)
  (3) direct decode via M_k^-1 (instead of unwhiten via L^-T globally)

The three together break the previous trilemma.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import benchmarks.pre_rope_cache as prc
from benchmarks.q_precondition import load as load_qp
from scipy.special import erfinv


def haar_codebook(M: int) -> np.ndarray:
    theta = np.pi * np.arange(M) / M
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def lloyd_max_centroids(bits: int) -> np.ndarray:
    """Unit-Gaussian Lloyd-Max centroids for the given bit width."""
    n = 1 << bits
    u = (np.arange(n) + 0.5) / n
    return np.sqrt(2) * erfinv(2 * u - 1).astype(np.float32)


def calibrate_group_blocks(k_whitened_flat: np.ndarray, D: int, g: int) -> np.ndarray:
    """Compute per-group per-block-diagonal M_k (2x2) from whitened data.

    Since k_whitened was produced by full Σ_q^-1/2 whitening, in whitened
    space the per-group 2x2 covariance is what the Riemannian structure
    sees. We compute M_k from the data (closed-loop calibration).

    Returns: M_k array of shape (D//g, g, g).
    """
    n_groups = D // g
    M_per_group = np.zeros((n_groups, g, g), dtype=np.float32)
    for k in range(n_groups):
        vec = k_whitened_flat[:, k*g:(k+1)*g]
        M_per_group[k] = np.cov(vec.T)
    return M_per_group


def calibrate_group_scales(k_whitened_flat: np.ndarray, M_per_group: np.ndarray,
                           D: int, g: int) -> np.ndarray:
    """Per-group scale s_k for Lloyd-Max normalization.

    s_k = sqrt(trace(M_k))  gives expected magnitude of α in Σq metric.
    (Could alternatively use 99th percentile of |α|; test both.)
    """
    n_groups = D // g
    scales = np.zeros(n_groups, dtype=np.float32)
    for k in range(n_groups):
        # sqrt(trace) is the expected magnitude in this metric
        scales[k] = np.sqrt(max(M_per_group[k].trace(), 1e-12))
    return scales


def riemannian_besi_encode_decode(
    block: np.ndarray, g: int, M_dir: int,
    M_per_group: np.ndarray, scales: np.ndarray,
    magnitude_bits: int, subtract_mean: bool = True,
) -> tuple[np.ndarray, float]:
    """Riemannian Besi round-trip on a block IN THE SAME SPACE block lives in.
    
    M_per_group[k] is the 2x2 metric for group k.
    scales[k] is the offline-calibrated magnitude scale for group k.
    """
    N, D = block.shape
    n_groups = D // g
    cb = haar_codebook(M_dir)
    centroids = lloyd_max_centroids(magnitude_bits) if magnitude_bits > 0 else None

    if subtract_mean:
        bm = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    else:
        bm = np.zeros((1, D), dtype=np.float32)
    centered = block - bm

    rec = np.zeros_like(centered)
    for k in range(n_groups):
        x_k = centered[:, k*g:(k+1)*g]                  # (N, 2)
        Mk = M_per_group[k]                              # (2, 2)
        # Direction assignment: argmax_i |<x_k, Mk d_i>|
        Md = Mk @ cb.T                                   # (2, M)
        scores = x_k @ Md                                # (N, M)
        ids = np.abs(scores).argmax(axis=1)
        # Magnitude: α = <x_k, Mk d_id>
        alphas = scores[np.arange(N), ids]               # (N,)
        # Quantize in normalized units
        s_k = scales[k]
        if magnitude_bits == 0:
            alphas_q = alphas.astype(np.float16).astype(np.float32)
        else:
            u = alphas / s_k
            idx = np.abs(u[:, None] - centroids[None, :]).argmin(axis=1)
            alphas_q = centroids[idx] * s_k
        # Decode: x_hat_k = alpha_q · (Mk^-1 d_id)
        # Precompute Mk^-1 cb once
        try:
            Mk_inv = np.linalg.inv(Mk)
        except np.linalg.LinAlgError:
            Mk_inv = np.linalg.pinv(Mk)
        # For each sample: x_hat = alphas_q * Mk_inv @ d_id
        d_selected = cb[ids]                             # (N, 2)
        Mk_inv_d = d_selected @ Mk_inv.T                 # (N, 2) = Mk_inv @ d_selected per row
        rec[:, k*g:(k+1)*g] = alphas_q[:, None] * Mk_inv_d

    rec_final = rec + bm
    # MSE in original space — but what matters for attention is
    # MSE weighted by Σ_q (which is what we're encoding against)
    mse = ((block - rec_final) ** 2).mean()
    return rec_final, float(mse)


def euclidean_besi_encode_decode(
    block: np.ndarray, g: int, M_dir: int, scales: np.ndarray,
    magnitude_bits: int, subtract_mean: bool = True,
) -> tuple[np.ndarray, float]:
    """Baseline: same per-group scale but Euclidean (no Σq metric)."""
    N, D = block.shape
    n_groups = D // g
    cb = haar_codebook(M_dir)
    centroids = lloyd_max_centroids(magnitude_bits) if magnitude_bits > 0 else None
    if subtract_mean:
        bm = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    else:
        bm = np.zeros((1, D), dtype=np.float32)
    centered = block - bm
    rec = np.zeros_like(centered)
    for k in range(n_groups):
        x_k = centered[:, k*g:(k+1)*g]
        proj = x_k @ cb.T
        ids = np.abs(proj).argmax(axis=1)
        alphas = proj[np.arange(N), ids]
        s_k = scales[k]
        if magnitude_bits == 0:
            alphas_q = alphas.astype(np.float16).astype(np.float32)
        else:
            u = alphas / s_k
            idx = np.abs(u[:, None] - centroids[None, :]).argmin(axis=1)
            alphas_q = centroids[idx] * s_k
        rec[:, k*g:(k+1)*g] = alphas_q[:, None] * cb[ids]
    rec_final = rec + bm
    mse = ((block - rec_final) ** 2).mean()
    return rec_final, float(mse)


def load_ds_kv_cache(model_path: str, ctx_len: int = 2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True).eval()
    prc.install(model)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(ds["text"][:200])
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx_len]
    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--qp-path",
                    default="reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--direction-bits", type=int, default=4)
    ap.add_argument("--magnitude-bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--skip-layers", type=int, nargs="*", default=[0, 1, 26, 27])
    args = ap.parse_args()

    cache = load_ds_kv_cache(args.model_path, args.ctx_len)
    D = cache.layers[0].keys.shape[-1]
    g = 2; M = 1 << args.direction_bits
    qp = load_qp(args.qp_path, skip_layers=args.skip_layers)

    print(f"Riemannian K-Besi oracle: D={D}, g={g}, M={M}, m_bits={args.magnitude_bits}")
    print()
    print("Comparing three K codecs in Σq-weighted MSE (attention-quality proxy):")
    print("  A. PCA d=D/2 f16 on raw K   (upper bound for v1.4 Kakeya-PCA)")
    print("  B. Euclidean Besi on whitened K + per-(layer, group) offline scale")
    print("  C. Riemannian Besi on whitened K + per-(layer, group) offline scale")
    print()
    print(f"{'L':>3} {'A (PCA)':>11} {'B (Euc)':>11} {'C (Riem)':>11} "
          f"{'B/A':>6} {'C/A':>6} {'C/B':>6}")
    print("-"*68)

    A_mses = []; B_mses = []; C_mses = []
    for l in range(len(cache.layers)):
        if not qp.is_active(l):
            continue
        k_tensor = cache.layers[l].keys.to(torch.float32).cpu().numpy()
        k_swap = k_tensor.squeeze(0).transpose(1, 0, 2)
        k_white = qp.whiten(k_swap, layer=l)
        k_white_flat = k_white.reshape(-1, D)
        k_flat = k_tensor.reshape(-1, D)

        # Offline calibration: M_per_group (in whitened space) + scales
        M_per_group = calibrate_group_blocks(k_white_flat, D, g)
        scales = calibrate_group_scales(k_white_flat, M_per_group, D, g)

        bs = args.block_size
        mA, mB, mC = [], [], []
        i = 0
        while i + bs <= k_flat.shape[0]:
            blk_raw = k_flat[i:i+bs]
            blk_white = k_white_flat[i:i+bs]

            # A. PCA d=D/2 f16 on raw K, Σq-MSE
            d_eff = D // 2
            mean_pca = blk_raw.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
            U, s, Vt = np.linalg.svd(blk_raw - mean_pca, full_matrices=False)
            coeffs_q = (U[:, :d_eff] * s[:d_eff]).astype(np.float16).astype(np.float32)
            rec_pca = coeffs_q @ Vt[:d_eff] + mean_pca
            # Whiten the error to get Σq-MSE
            err_raw = blk_raw - rec_pca
            err_reshape = err_raw.reshape(-1, k_swap.shape[1], D)
            err_white = qp.whiten(err_reshape, layer=l).reshape(-1, D)
            mA.append((err_white ** 2).mean())

            # B. Euclidean Besi on whitened K (per-group scale, no metric adjustment)
            _, mse_B = euclidean_besi_encode_decode(
                blk_white, g, M, scales, args.magnitude_bits)
            # mse_B is in whitened space (= Σq-MSE in original space)
            mB.append(mse_B)

            # C. Riemannian Besi on whitened K (with M_k metric in assignment)
            _, mse_C = riemannian_besi_encode_decode(
                blk_white, g, M, M_per_group, scales, args.magnitude_bits)
            mC.append(mse_C)

            i += bs
        A_m, B_m, C_m = np.mean(mA), np.mean(mB), np.mean(mC)
        print(f"{l:>3} {A_m:>10.3e} {B_m:>10.3e} {C_m:>10.3e} "
              f"{B_m/A_m:>5.2f}x {C_m/A_m:>5.2f}x {C_m/B_m:>5.2f}x")
        A_mses.append(A_m); B_mses.append(B_m); C_mses.append(C_m)

    print("-"*68)
    A_g, B_g, C_g = np.mean(A_mses), np.mean(B_mses), np.mean(C_mses)
    print(f"{'Σ':>3} {A_g:>10.3e} {B_g:>10.3e} {C_g:>10.3e} "
          f"{B_g/A_g:>5.2f}x {C_g/A_g:>5.2f}x {C_g/B_g:>5.2f}x")
    print()
    print(f"Key question: does Riemannian C beat Euclidean B?")
    print(f"  C/B = {C_g/B_g:.3f}  (< 1 means Riemannian helps)")
    print(f"Secondary: does either beat PCA baseline at skeleton savings?")
    print(f"  B/A = {B_g/A_g:.3f}, C/A = {C_g/A_g:.3f}")
    print()
    # Byte accounting
    bs = args.block_size
    pca_bv = (D*2 + D*(D//2)*2 + bs*(D//2)*2) / bs  # skeleton + coeffs
    besi_bv = ((bs * (D//g) * args.direction_bits) / 8 +
               (bs * (D//g) * args.magnitude_bits) / 8 + D*2) / bs
    offline_scale_cost = (D//g) * 4 * 28 * 2 / (28 * bs * 64)  # per-(layer, group, head) 4-byte / total vectors
    print(f"Byte accounting (D={D}, d_eff={D//2}, bs={bs}, b_dir={args.direction_bits}, b_mag={args.magnitude_bits}):")
    print(f"  PCA per block: ~{pca_bv:.1f} B/v")
    print(f"  Besi per block: ~{besi_bv:.1f} B/v")
    print(f"  Offline scale amortized: ~{offline_scale_cost:.3f} B/v (negligible)")


if __name__ == "__main__":
    main()

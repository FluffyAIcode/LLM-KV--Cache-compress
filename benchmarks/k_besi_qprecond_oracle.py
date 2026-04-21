#!/usr/bin/env python3
"""Oracle: replace K-stream Kakeya-PCA with Q-preconditioned Besi.

Motivation: user's Perron-tree / attention-weighted Kakeya proposal
really means: use Σ_q-based weighted Kakeya codebook to replace per-
block PCA skeleton, saving 16 B/v of skeleton overhead.

The correct form of "attention-weighted Kakeya" in this context:

    1. Offline calibration: store one L (Cholesky of Σ_q) per
       (layer, kv-head) — same as current Q-precond.
    2. Runtime: K_whitened = K @ L.  This IS the Σ_q weighting,
       expressed as a coordinate transform.  MSE in whitened space
       = Σ_q-weighted MSE in original space (attention-aware).
    3. Encode K_whitened with Besi (Haar codebook — now optimal
       because whitened K IS isotropic under Σ_q metric).
    4. Decode + unwhiten: K_hat = K_hat_whitened @ L^{-T}.

Key claims to test:
    (a) Besi on whitened K has MSE competitive with Kakeya-PCA on
        whitened K — at MATCHED bit budget, ignoring skeleton.
    (b) If (a) holds, Besi wins at the TOTAL byte level because no
        per-block PCA skeleton.

This is the Rust implementation of Σ_q L-matrix preprocessing already
exists (Q-precond applies L before codec).  So if we just set
--codec kakeyaturbo --bit-width X on whitened K, OR --codec besi on
whitened K, we get a direct comparison.  Python harness already
handles this via --q-precondition flag.

This script is ORACLE-only: simulate both paths in numpy on real K
data and measure per-block MSE + byte budget.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import benchmarks.pre_rope_cache as prc
from benchmarks.q_precondition import load as load_qp


def haar_codebook(M: int) -> np.ndarray:
    theta = np.pi * np.arange(M) / M
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def besi_encode_decode(block: np.ndarray, g: int, M: int,
                        magnitude_bits: int,
                        subtract_mean: bool = True) -> float:
    """Vanilla Besi round-trip → MSE."""
    N, D = block.shape
    n_groups = D // g
    if subtract_mean:
        bm = block.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
    else:
        bm = np.zeros((1, D), dtype=np.float32)
    centered = block - bm
    cb = haar_codebook(M)
    rec = np.zeros_like(centered)
    for k in range(n_groups):
        x_k = centered[:, k*g:(k+1)*g]
        proj = x_k @ cb.T
        ids = np.abs(proj).argmax(axis=1)
        alphas = proj[np.arange(N), ids]
        if magnitude_bits == 0:
            alphas_q = alphas.astype(np.float16).astype(np.float32)
        else:
            # Per-vector shared scale (over all groups)
            pass  # stub — do it below after collecting all alphas
        # For simplicity: treat each group's magnitude as f16 here
        # (m=4 quantized would need joint per-vector scale across all
        # groups; use f16 as upper-bound oracle).
        alphas_q = alphas.astype(np.float16).astype(np.float32)
        rec[:, k*g:(k+1)*g] = alphas_q[:, None] * cb[ids]
    rec_final = rec + bm
    return ((block - rec_final) ** 2).mean()


def load_ds_kv_cache(model_path: str, ctx_len: int = 2048) -> tuple[DynamicCache, dict]:
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
    return cache, {"D": cache.layers[0].keys.shape[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--q-precond-path",
                    default="reports/v1_4_q_pca/flagship/deepseek_distill_q_calib.safetensors")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--direction-bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--skip-layers", type=int, nargs="*", default=[0, 1, 26, 27])
    args = ap.parse_args()

    cache, meta = load_ds_kv_cache(args.model_path, args.ctx_len)
    D = meta["D"]; g = 2; M = 1 << args.direction_bits
    qp = load_qp(args.q_precond_path, skip_layers=args.skip_layers)
    L_total = len(cache.layers)

    print(f"Model: L={L_total}, D={D}, g={g}, M={M}")
    print(f"Q-precond loaded: {qp.n_calibrated_layers} layers calibrated")
    print(f"Skip layers (boundary): {args.skip_layers}\n")

    print("Compare MSE IN WHITENED SPACE (which = Σ_q-weighted MSE in original,")
    print("i.e., the quantity that correlates with attention quality):")
    print("  A. Besi(Haar) on whitened K  (encode/decode IN whitened space)")
    print("  B. PCA d_eff=D/2 on whitened K (encode/decode IN whitened space)")
    print("  C. Kakeya-PCA baseline on raw K (what v1.4 actually does — includes Q-precond)")
    print()
    print(f"{'L':>3} {'A: Besi':>10} {'B: PCA':>10} {'C: v1.4 ref':>12} "
          f"{'A/B':>6} {'A/C':>6}  {'winner':>10}")
    print("-"*68)

    besi_white_mses = []; pca_white_mses = []; pca_raw_ref_mses = []
    for l in range(L_total):
        if not qp.is_active(l):
            continue  # skip boundary
        k_tensor = cache.layers[l].keys.to(torch.float32).cpu().numpy()
        k_swap = k_tensor.squeeze(0).transpose(1, 0, 2)  # (seq, n_kv, D)
        k_whitened = qp.whiten(k_swap, layer=l)
        k_flat = k_tensor.reshape(-1, D)
        k_white_flat = k_whitened.reshape(-1, D)

        bs = args.block_size
        mA, mB, mC = [], [], []
        i = 0
        while i + bs <= k_flat.shape[0]:
            blk_raw = k_flat[i:i+bs]
            blk_white = k_white_flat[i:i+bs]

            # A. Besi encode/decode IN whitened space (MSE in whitened space)
            N = blk_white.shape[0]
            bm = blk_white.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
            centered = blk_white - bm
            cb = haar_codebook(M)
            rec_white = np.zeros_like(centered)
            for k in range(D // g):
                x_k = centered[:, k*g:(k+1)*g]
                proj = x_k @ cb.T
                ids = np.abs(proj).argmax(axis=1)
                alphas = proj[np.arange(N), ids]
                alphas_q = alphas.astype(np.float16).astype(np.float32)
                rec_white[:, k*g:(k+1)*g] = alphas_q[:, None] * cb[ids]
            rec_white_final = rec_white + bm
            mA.append(((blk_white - rec_white_final) ** 2).mean())

            # B. PCA on whitened K (d_eff=D/2 + f16 coefficients, IN whitened space)
            d_eff = D // 2
            mean_pca_w = blk_white.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
            centered_pca_w = blk_white - mean_pca_w
            U, s, Vt = np.linalg.svd(centered_pca_w, full_matrices=False)
            coeffs = U[:, :d_eff] * s[:d_eff]
            coeffs_q = coeffs.astype(np.float16).astype(np.float32)
            rec_pca_w = coeffs_q @ Vt[:d_eff] + mean_pca_w
            mB.append(((blk_white - rec_pca_w) ** 2).mean())

            # C. PCA on RAW K (d_eff=D/2 + f16 coefficients, MSE in RAW space)
            # This is the actual v1.4 K-codec baseline in raw space.
            # Convert to whitened-space MSE via L-matrix.
            mean_pca_r = blk_raw.mean(axis=0, keepdims=True).astype(np.float16).astype(np.float32)
            centered_pca_r = blk_raw - mean_pca_r
            U, s, Vt = np.linalg.svd(centered_pca_r, full_matrices=False)
            coeffs = U[:, :d_eff] * s[:d_eff]
            coeffs_q = coeffs.astype(np.float16).astype(np.float32)
            rec_pca_r = coeffs_q @ Vt[:d_eff] + mean_pca_r
            # Σ_q-weighted MSE: MSE(whiten(rec) - whiten(raw))
            diff_raw = blk_raw - rec_pca_r
            diff_reshape = diff_raw.reshape(-1, k_swap.shape[1], D)
            diff_white = qp.whiten(diff_reshape, layer=l).reshape(-1, D)
            mC.append((diff_white ** 2).mean())

            i += bs

        mA_m = np.mean(mA); mB_m = np.mean(mB); mC_m = np.mean(mC)
        A_over_B = mA_m / mB_m if mB_m > 0 else 1.0
        A_over_C = mA_m / mC_m if mC_m > 0 else 1.0
        winner = "Besi" if mA_m < min(mB_m, mC_m) else ("PCA-wht" if mB_m < mC_m else "PCA-raw(C)")
        print(f"{l:>3} {mA_m:>9.3e} {mB_m:>9.3e} {mC_m:>11.3e} "
              f"{A_over_B:>5.2f}x {A_over_C:>5.2f}x  {winner:>10}")
        besi_white_mses.append(mA_m); pca_white_mses.append(mB_m); pca_raw_ref_mses.append(mC_m)

    print("-"*68)
    print(f"Summary across {len(besi_white_mses)} non-boundary layers:")
    print(f"  MSE in whitened space (Σ_q-weighted, correlates with PPL):")
    print(f"    A. Besi on whitened K:      {np.mean(besi_white_mses):.3e}")
    print(f"    B. PCA on whitened K:       {np.mean(pca_white_mses):.3e}")
    print(f"    C. PCA on RAW K (v1.4 ref): {np.mean(pca_raw_ref_mses):.3e}  (→ whitened-space)")
    print()
    ratio_AB = np.mean(besi_white_mses) / np.mean(pca_white_mses)
    ratio_AC = np.mean(besi_white_mses) / np.mean(pca_raw_ref_mses)
    print(f"  A/B (Besi vs PCA in same whitened space): {ratio_AB:.2f}x")
    print(f"  A/C (Besi on whitened vs v1.4 baseline) : {ratio_AC:.2f}x")
    print()
    if ratio_AC < 1.5:
        print(f"  → Besi-on-whitened IS competitive; skeleton savings may tip the balance")
        print(f"     ({ratio_AC:.2f}× MSE penalty for 16 B/v skeleton savings).")
    elif ratio_AC < 3:
        print(f"  → Besi-on-whitened is {ratio_AC:.1f}× worse Σ_q-MSE; marginal case.")
    else:
        print(f"  → Besi-on-whitened is >{ratio_AC:.1f}× worse. Drop this path.")

    # Compute the skeleton-savings upper bound
    print()
    print("Byte accounting (DS-Distill D=128, d_eff=64, block_size=1024):")
    D_val = 128; d_eff = 64; bs = 1024
    pca_skeleton = (D_val * 2 + D_val * d_eff * 2)  # mean (D f16) + basis (D×d_eff f16)
    pca_coeffs = bs * d_eff * 2  # (N × d_eff) f16 — upper bound
    besi_coeffs_m4 = bs * (D_val // g) * (args.direction_bits + 4)  // 8 + bs * 2  # dir + m=4 + scale
    # More realistic with direction_bits=M_log2, mag=4 quantized
    dir_bytes = (bs * (D_val // g) * args.direction_bits) // 8
    mag_bytes_quant = (bs * (D_val // g) * 4) // 8 + bs * 2  # m=4 + per-vector scale
    mag_bytes_f16 = bs * (D_val // g) * 2  # f16 per group magnitude
    besi_total_m4 = dir_bytes + mag_bytes_quant + D_val * 2  # + block mean
    besi_total_f16 = dir_bytes + mag_bytes_f16 + D_val * 2
    print(f"  PCA per block: skeleton={pca_skeleton} + coeffs={pca_coeffs} = {pca_skeleton+pca_coeffs} B "
          f"= {(pca_skeleton+pca_coeffs)/bs:.2f} B/v")
    print(f"  Besi per block (d={args.direction_bits}, m=4 q): skeleton=0 + codes={besi_total_m4} = {besi_total_m4} B "
          f"= {besi_total_m4/bs:.2f} B/v")
    print(f"  Besi per block (d={args.direction_bits}, f16):  skeleton=0 + codes={besi_total_f16} = {besi_total_f16} B "
          f"= {besi_total_f16/bs:.2f} B/v")
    print(f"  Skeleton overhead savings if Besi wins: {pca_skeleton} B/block = {pca_skeleton/bs:.2f} B/v")


if __name__ == "__main__":
    main()

"""Parity check: ablation v14_full == canonical V14KakeyaZamirLatticeGPU.

Also smoke-tests all 6 ablation variants on a toy tensor and prints the
rel-MSE comparison so we can see each factor's contribution isolated.
"""
from __future__ import annotations

import torch
from kakeyalattice import V14KakeyaZamirLatticeGPU
from kakeyalattice.ablation_codecs import make_ablation_codec, ABLATION_VARIANTS


def main() -> None:
    torch.manual_seed(42)
    D = 128
    N_TOK = 2048
    H_HEADS = 8
    Q = 38

    X = torch.randn(N_TOK, H_HEADS, D, device="cuda", dtype=torch.float32) * 0.3

    # 1. Parity: ablation v14_full should match canonical class bit-for-bit
    #    (both use the same closest-lattice-point + Hadamard + qmax stack).
    canonical = V14KakeyaZamirLatticeGPU(D=D, q_range=Q, device="cuda")
    ablation_full = make_ablation_codec("v14_full", D, Q)

    X_canon = canonical.roundtrip(X)
    X_ablat = ablation_full(X)
    max_diff = float((X_canon - X_ablat).abs().max().item())
    is_bit_identical = bool(torch.equal(X_canon, X_ablat))
    print(f"Parity (v14_full vs canonical V14KakeyaZamirLatticeGPU):")
    print(f"  bit-identical:  {is_bit_identical}")
    print(f"  max_abs_diff:   {max_diff:.3e}")
    # Sub-ULP differences can happen from GEMM dispatch; require < 1e-5.
    assert max_diff < 1e-5, f"Parity check FAILED: max_diff={max_diff:.3e}"
    print(f"  PARITY OK")
    print()

    # 2. Ablation sweep: rel-MSE for each variant.
    print(f"Ablation rel-MSE sweep at Q={Q}, D={D}, {N_TOK} tokens × {H_HEADS} heads:")
    print(f"{'variant':<22} {'bits/tok/head':>14} {'rel-MSE':>12} {'vs full':>10}")
    print('-' * 62)
    full_mse = None
    for variant in ABLATION_VARIANTS:
        fn = make_ablation_codec(variant, D, Q)
        X_hat = fn(X)
        diff = X_hat - X
        rel_mse = float(
            ((diff ** 2).sum(-1).mean() /
             (X ** 2).sum(-1).mean().clamp(min=1e-12)).item()
        )
        if variant == "v14_full":
            full_mse = rel_mse
            ratio = "1.000×"
        else:
            ratio = f"{rel_mse / full_mse:6.3f}×"
        print(f'{variant:<22} {fn.bits_per_token_per_head:>14d} '
              f'{rel_mse:12.4e} {ratio:>10}')


if __name__ == "__main__":
    main()

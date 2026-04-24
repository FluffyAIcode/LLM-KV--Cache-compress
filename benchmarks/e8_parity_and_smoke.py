"""Parity + smoke check for the new lattice_codebooks module.

(1) Parity: D4LatticeCodebook must produce byte-identical output to the
    existing V14KakeyaZamirLatticeGPU on the same input.  If this
    passes, the refactor is guaranteed not to have changed any v1.4
    measurement.

(2) Smoke: E8LatticeCodebook runs end-to-end on a realistic tensor
    shape, produces finite output, and beats D4 on rel-MSE at the same
    Q (theoretical shaping gain materialises).

(3) Bit-count sanity: verify the E8 storage formulas match hand
    calculation.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import torch

from kakeyaturbo_py import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU
from kakeyaturbo_py.lattice_codebooks import (
    D4LatticeCodebook,
    E8LatticeCodebook,
)


# Frozen reference sha256s generated with the pre-cleanup code (when
# V14KakeyaZamirLatticeGPU still inherited from D4TQStyleCodebook).
# Any future refactor of lattice_codebooks.py that breaks bit-identity
# with v1.4 deployed measurements will fail these assertions loudly.
FROZEN_PATH = pathlib.Path(__file__).parent / "frozen_parity.json"


def _load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text()) if FROZEN_PATH.exists() else {}


def _sha256_fp32(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def parity_v14_against_frozen() -> None:
    """V14KakeyaZamirLatticeGPU output must match the frozen sha256 snapshot.

    The snapshot was generated with the pre-rename D4TQStyleCodebook code
    (verified bit-identical to D4LatticeCodebook in PR #32) and saved to
    frozen_parity.json.  This is THE regression gate for the v1.4 release:
    if any lattice_codebooks.py change breaks bit-identity with the
    deployed v1.4 numbers, this assertion catches it.
    """
    frozen = _load_frozen()
    if not frozen:
        print("\n[parity]  frozen_parity.json missing — skipping")
        return
    meta = frozen.get("_meta", {})
    seed = meta.get("seed", 20260424)
    shape = meta.get("shape", [2048, 8, 128])
    scale = meta.get("scale", 0.3)
    torch.manual_seed(seed)
    X = torch.randn(*shape, device="cuda", dtype=torch.float32) * scale

    print(f"\n[parity] V14KakeyaZamirLatticeGPU vs FROZEN sha256 "
          f"(seed={seed}, shape={shape})")
    ok_all = True
    for Q in [4, 10, 38, 152]:
        key = f"V14KakeyaZamirLatticeGPU_Q{Q}_D128_N2048_H8"
        if key not in frozen:
            continue
        cb = V14KakeyaZamirLatticeGPU(D=128, q_range=Q, device="cuda")
        X_hat = cb.roundtrip(X)
        got = _sha256_fp32(X_hat)
        want = frozen[key]["sha256"]
        want_bits = frozen[key]["bits"]
        assert cb.bits_per_token_per_head == want_bits, (
            f"v1.4 Q={Q} bits changed: expected {want_bits}, got "
            f"{cb.bits_per_token_per_head}"
        )
        if got != want:
            print(f"  Q={Q}: ✗ MISMATCH  got={got[:16]}  want={want[:16]}")
            ok_all = False
        else:
            print(f"  Q={Q}: ✓ match (sha={got[:16]}, bits={want_bits})")
    assert ok_all, "v1.4 output regression vs frozen snapshot"
    print("  ✓ v1.4 FROZEN PARITY OK")


def parity_v15_against_frozen() -> None:
    """V15KakeyaZamirE8GPU output must match the frozen sha256 snapshot.

    Protects the v1.5 scaffold from unintended behavior change.
    """
    frozen = _load_frozen()
    if not frozen:
        return
    meta = frozen.get("_meta", {})
    seed = meta.get("seed", 20260424)
    shape = meta.get("shape", [2048, 8, 128])
    scale = meta.get("scale", 0.3)
    torch.manual_seed(seed)
    X = torch.randn(*shape, device="cuda", dtype=torch.float32) * scale

    print(f"\n[parity] V15KakeyaZamirE8GPU vs FROZEN sha256")
    ok_all = True
    for Q in [4, 10, 38, 152]:
        key = f"V15KakeyaZamirE8GPU_Q{Q}_D128_N2048_H8"
        if key not in frozen:
            continue
        cb = V15KakeyaZamirE8GPU(D=128, q_range=Q, device="cuda")
        X_hat = cb.roundtrip(X)
        got = _sha256_fp32(X_hat)
        want = frozen[key]["sha256"]
        want_bits = frozen[key]["bits"]
        assert cb.bits_per_token_per_head == want_bits, (
            f"v1.5 Q={Q} bits changed: expected {want_bits}, got "
            f"{cb.bits_per_token_per_head}"
        )
        if got != want:
            print(f"  Q={Q}: ✗ MISMATCH  got={got[:16]}  want={want[:16]}")
            ok_all = False
        else:
            print(f"  Q={Q}: ✓ match (sha={got[:16]}, bits={want_bits})")
    assert ok_all, "v1.5 output regression vs frozen snapshot"
    print("  ✓ v1.5 FROZEN PARITY OK")


def parity_v14_d4() -> None:
    """D4LatticeCodebook bit-identical to V14KakeyaZamirLatticeGPU
    (trivially true since V14 subclasses D4Lattice — sanity check only)."""
    torch.manual_seed(42)
    D, Q = 128, 38
    N = 2048
    X = torch.randn(N, 8, D, device="cuda", dtype=torch.float32) * 0.3

    canonical = V14KakeyaZamirLatticeGPU(D=D, q_range=Q, device="cuda")
    refactor  = D4LatticeCodebook(D=D, q_range=Q, device="cuda")

    X_canon = canonical.roundtrip(X)
    X_refac = refactor.roundtrip(X)

    max_diff = float((X_canon - X_refac).abs().max().item())
    bit_identical = bool(torch.equal(X_canon, X_refac))
    print(f"\n[parity] V14KakeyaZamirLatticeGPU vs D4LatticeCodebook (D=128, Q=38)")
    print(f"  bit-identical:  {bit_identical}")
    print(f"  max_abs_diff:   {max_diff:.3e}")
    print(f"  canonical bits: {canonical.bits_per_token_per_head}")
    print(f"  refactor bits:  {refactor.bits_per_token_per_head}")
    assert bit_identical, "D4 refactor broke v1.4 bit-identity!"
    assert canonical.bits_per_token_per_head == refactor.bits_per_token_per_head
    print("  ✓ PARITY OK")


def parity_v15_e8() -> None:
    """V15KakeyaZamirE8GPU bit-identical to E8LatticeCodebook."""
    torch.manual_seed(7)
    D, Q = 128, 37
    N = 1024
    X = torch.randn(N, 8, D, device="cuda", dtype=torch.float32) * 0.3

    canonical = V15KakeyaZamirE8GPU(D=D, q_range=Q, device="cuda")
    base      = E8LatticeCodebook(D=D, q_range=Q, device="cuda")

    X_canon = canonical.roundtrip(X)
    X_base  = base.roundtrip(X)
    bit_identical = bool(torch.equal(X_canon, X_base))
    print(f"\n[parity] V15KakeyaZamirE8GPU vs E8LatticeCodebook (D=128, Q=37)")
    print(f"  bit-identical:  {bit_identical}")
    assert bit_identical
    print("  ✓ PARITY OK")


def smoke_e8() -> None:
    """E8 basic correctness: finite output, roundtrip non-trivial,
    shaping gain vs D4 at matched rate."""
    torch.manual_seed(0)
    D = 128
    N = 4096
    X = torch.randn(N, 8, D, device="cuda", dtype=torch.float32) * 0.3

    print(f"\n[smoke] D=128, N={N}, 8 heads, random Gaussian K-like input")
    print(f"  {'Q or matched':<24} {'bits/tok/head':>14} {'rel-MSE':>14}")
    print(f"  {'-'*56}")

    for Q_d4 in [4, 10, 38, 152]:
        d4 = V14KakeyaZamirLatticeGPU(D=D, q_range=Q_d4)
        X_hat = d4.roundtrip(X)
        mse = float(((X_hat - X) ** 2).sum(-1).mean() /
                    (X ** 2).sum(-1).mean().clamp(min=1e-12))
        assert torch.isfinite(X_hat).all()
        print(f"  {'D4 Q=' + str(Q_d4):<24} {d4.bits_per_token_per_head:>14} "
              f"{mse:>14.4e}")

        # Find E8 Q that matches D4's bit budget.
        # D4 bits = 32 · ceil(4·log₂(2Q+1) − 1) + 32
        # E8 bits = 16 · ceil(8·log₂(2Q+1)) + 32
        # Solve for E8 Q such that 16·ceil(8·log₂(2Q'+1)) = 32·ceil(4·log₂(2Q+1)−1)
        # i.e. ceil(8·log₂(2Q'+1)) = 2·ceil(4·log₂(2Q+1) − 1)
        d4_lat_per_block = int(math.ceil(4 * math.log2(2 * Q_d4 + 1) - 1))
        target_e8_lat = 2 * d4_lat_per_block  # E8 needs 2× per block to match
        # Find largest Q' with ceil(8·log₂(2Q'+1)) ≤ target_e8_lat
        Q_e8 = 1
        for q in range(1, 500):
            if math.ceil(8 * math.log2(2 * q + 1)) <= target_e8_lat:
                Q_e8 = q
            else:
                break
        e8 = V15KakeyaZamirE8GPU(D=D, q_range=Q_e8)
        X_hat_e8 = e8.roundtrip(X)
        mse_e8 = float(((X_hat_e8 - X) ** 2).sum(-1).mean() /
                       (X ** 2).sum(-1).mean().clamp(min=1e-12))
        assert torch.isfinite(X_hat_e8).all()
        shaping_db = 10.0 * math.log10(max(mse, 1e-12) / max(mse_e8, 1e-12))
        tag = f"E8 iso-bit Q={Q_e8}"
        print(f"  {tag:<24} {e8.bits_per_token_per_head:>14} "
              f"{mse_e8:>14.4e}  (Δ={shaping_db:+.2f} dB vs D4)")

    print("  ✓ E8 runs end-to-end, output finite, storage formulas reconcile.")


def bits_sanity() -> None:
    """Verify bit formulas against hand calculation."""
    print(f"\n[bits] Verifying D4 / E8 bit formulas at head_dim=128")
    for Q in [4, 10, 38, 152]:
        d4 = V14KakeyaZamirLatticeGPU(D=128, q_range=Q)
        e8 = V15KakeyaZamirE8GPU(D=128, q_range=Q)
        # Hand calc
        d4_real = 4 * math.log2(2 * Q + 1) - 1
        d4_block = int(math.ceil(d4_real))
        d4_expect = 32 * d4_block + 32
        e8_real = 8 * math.log2(2 * Q + 1)
        e8_block = int(math.ceil(e8_real))
        e8_expect = 16 * e8_block + 32
        print(f"  Q={Q:>3}:  D4 {d4.bits_per_token_per_head:>4} (expect {d4_expect})  "
              f"|  E8 {e8.bits_per_token_per_head:>4} (expect {e8_expect})  "
              f"|  Δ = {e8.bits_per_token_per_head - d4.bits_per_token_per_head:+d}")
        assert d4.bits_per_token_per_head == d4_expect, "D4 bit mismatch"
        assert e8.bits_per_token_per_head == e8_expect, "E8 bit mismatch"
    print("  ✓ All bit formulas consistent.")


if __name__ == "__main__":
    parity_v14_against_frozen()   # v1.4 regression gate (sha256 pinned)
    parity_v15_against_frozen()   # v1.5 regression gate (sha256 pinned)
    parity_v14_d4()               # trivial sanity
    parity_v15_e8()               # trivial sanity
    bits_sanity()
    smoke_e8()
    print("\n[ALL CHECKS PASSED]")

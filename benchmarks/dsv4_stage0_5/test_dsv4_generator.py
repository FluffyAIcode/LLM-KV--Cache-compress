"""Stage 0.5: shape + sanity tests for DSV4KVGenerator.

Runs on CPU if no CUDA — the generator itself forces fp32 arithmetic so
device choice only affects speed, not correctness.

Compliance: no mock, no fallback, strict-shape-checking.  These tests
verify the architectural port (shapes, RoPE application, FP8 simulation
no-op on zero input, overlap-pool stride) without needing any real
Qwen3 hidden states or the full KakeyaLattice install.
"""
from __future__ import annotations

import math
import sys

import torch

from dsv4_kv_generator import (
    DSV4FlashArchConfig,
    DSV4KVGenerator,
    DSV4Compressor,
    DSV4MainKVProjection,
    apply_rotary_emb,
    precompute_freqs_cis,
    _simulate_fp8_block_quant_dequant,
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_shapes_at_S_256():
    dev = _device()
    gen = DSV4KVGenerator(device=dev)
    B, S = 2, 256
    H = 4096
    x = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
    out = gen(x)
    assert out.sliding_window_kv.shape == (B, S, 512), out.sliding_window_kv.shape
    assert out.csa_pool_kv.shape == (B, S // 4, 512), out.csa_pool_kv.shape
    assert out.hca_pool_kv.shape == (B, S // 128, 512), out.hca_pool_kv.shape
    print(f"[OK] shapes at S={S}: {out.summary()}")


def test_shapes_at_S_2048():
    dev = _device()
    gen = DSV4KVGenerator(device=dev)
    B, S = 1, 2048
    H = 4096
    x = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
    out = gen(x)
    assert out.sliding_window_kv.shape == (B, 2048, 512)
    assert out.csa_pool_kv.shape == (B, 512, 512)
    assert out.hca_pool_kv.shape == (B, 16, 512)
    print(f"[OK] shapes at S={S}: {out.summary()}")


def test_rope_only_touches_last_64_dims():
    dev = _device()
    cfg = DSV4FlashArchConfig(simulate_fp8=False)  # isolate RoPE effect
    proj = DSV4MainKVProjection(cfg, device=dev)
    B, S, H = 1, 128, 4096
    x = torch.randn(B, S, H, device=dev, dtype=torch.float32)

    # Run normal forward.
    kv = proj(x)
    # Run forward without RoPE: monkey-patch a no-op.
    _orig = apply_rotary_emb.__wrapped__ if hasattr(apply_rotary_emb, "__wrapped__") else apply_rotary_emb

    import dsv4_kv_generator as gmod
    saved = gmod.apply_rotary_emb
    gmod.apply_rotary_emb = lambda tensor, freqs, inverse=False: tensor
    try:
        kv_no_rope = proj(x)
    finally:
        gmod.apply_rotary_emb = saved

    # Non-RoPE dims must be byte-identical between the two paths.
    diff_nope = (kv[..., :-64] - kv_no_rope[..., :-64]).abs().max().item()
    assert diff_nope < 1e-5, f"RoPE leaked into non-rope dims: max diff {diff_nope}"
    # RoPE dims MUST differ (otherwise RoPE is a no-op).
    diff_rope = (kv[..., -64:] - kv_no_rope[..., -64:]).abs().max().item()
    assert diff_rope > 1e-3, f"RoPE did nothing: max diff {diff_rope} (expected > 1e-3)"
    print(f"[OK] RoPE isolated to last 64 dims (nope diff={diff_nope:.2e}, rope diff={diff_rope:.2e})")


def test_fp8_simulation_is_noop_on_zeros():
    dev = _device()
    x = torch.zeros(4, 128, device=dev, dtype=torch.float32)
    y = _simulate_fp8_block_quant_dequant(x, block_size=64, fp8_max=448.0)
    assert torch.allclose(y, x, atol=0), "FP8 sim should be exact on zeros"
    print("[OK] FP8 simulation is no-op on zero input")


def test_fp8_simulation_preserves_amax():
    """FP8 per-block round-trip should keep the per-block amax close to the
    input amax (within the fp8_max/127 quantisation floor).  If not, the
    kernel is saturating wrong."""
    dev = _device()
    torch.manual_seed(0)
    x = torch.randn(4, 256, device=dev, dtype=torch.float32) * 5.0
    y = _simulate_fp8_block_quant_dequant(x, block_size=64, fp8_max=448.0)
    # Per-64-dim-block amax comparison.
    x_amax = x.reshape(4, 4, 64).abs().amax(dim=-1)
    y_amax = y.reshape(4, 4, 64).abs().amax(dim=-1)
    rel_diff = ((y_amax - x_amax).abs() / x_amax.clamp(min=1e-3))
    assert rel_diff.max().item() < 0.1, f"FP8 amax drift too large: {rel_diff.max().item()}"
    print(f"[OK] FP8 sim preserves per-block amax (max rel drift {rel_diff.max().item():.3e})")


def test_overlap_transform_stride_2():
    """CSA Compressor with ratio=4 uses overlap=True, producing 2*ratio=8
    slots whose interleaving matches inference/model.py:307-314.  The test:
    feed a known indicator input and verify the output slots.
    """
    dev = _device()
    cfg = DSV4FlashArchConfig(simulate_fp8=False)
    c = DSV4Compressor(cfg, compress_ratio=4, device=dev)

    # Construct a kv-shaped tensor [B, S/ratio=2, ratio=4, 2*d=1024] with an
    # indicator: first half of last dim = step "a", second half = step "b".
    B, S_over_r, r, d = 1, 2, 4, cfg.head_dim
    t = torch.zeros(B, S_over_r, r, 2 * d, device=dev, dtype=torch.float32)
    # Mark step 0's second half (a's "main" region)
    t[:, 0, :, d:] = 1.0
    # Mark step 1's first half (b's "overlap" region)
    t[:, 1, :, :d] = 2.0

    out = c._overlap_transform(t, value=-99.0)
    # Expected:
    #   out[:, 0, 0:4, :] = -99  (no prior step for index 0)
    #   out[:, 0, 4:8, :] = 1.0  (from step 0's second half)
    #   out[:, 1, 0:4, :] = 1.0  (from step 0's first half — wait, t[:, 0, :, :d]==0 since only second half was marked)
    # Re-read model.py:307-314:
    #    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]        → fills slot[ratio:] with "main" side
    #    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]     → fills slot[:ratio] for step>=1 with prior step's "overlap" side
    # With our input:
    #   tensor[:, 0, :, :d] = 0.0     (not set)
    #   tensor[:, 0, :, d:] = 1.0
    #   tensor[:, 1, :, :d] = 2.0
    #   tensor[:, 1, :, d:] = 0.0
    # Therefore:
    #   out[:, 0, 0:4, :] = -99.0     (value; step 0 has no prior)
    #   out[:, 0, 4:8, :] = 1.0       (step 0's main)
    #   out[:, 1, 0:4, :] = 0.0       (step 0's overlap side, which was zero)
    #   out[:, 1, 4:8, :] = 0.0       (step 1's main, which was zero)
    assert (out[:, 0, 0:4, :] == -99.0).all(), "step 0 prefix not filled with default"
    assert (out[:, 0, 4:8, :] == 1.0).all(), "step 0 main region wrong"
    assert (out[:, 1, 0:4, :] == 0.0).all(), "step 0->1 overlap region wrong"
    assert (out[:, 1, 4:8, :] == 0.0).all(), "step 1 main region wrong"
    print("[OK] overlap_transform matches inference/model.py:307-314")


def test_determinism():
    dev = _device()
    gen_a = DSV4KVGenerator(device=dev, seed=42)
    gen_b = DSV4KVGenerator(device=dev, seed=42)
    B, S, H = 1, 256, 4096
    x = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
    out_a = gen_a(x)
    out_b = gen_b(x)
    for name in ["sliding_window_kv", "csa_pool_kv", "hca_pool_kv"]:
        a = getattr(out_a, name)
        b = getattr(out_b, name)
        assert torch.equal(a, b), f"seed-same outputs differ on {name}: max diff {(a-b).abs().max()}"
    print("[OK] determinism: same seed -> identical KV streams")


def test_different_seed_gives_different_output():
    dev = _device()
    gen_a = DSV4KVGenerator(device=dev, seed=1)
    gen_b = DSV4KVGenerator(device=dev, seed=2)
    B, S, H = 1, 256, 4096
    x = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
    out_a = gen_a(x)
    out_b = gen_b(x)
    # Sliding window KV still depends on wkv weights so seed=1 vs seed=2 must differ.
    diff = (out_a.sliding_window_kv.float() - out_b.sliding_window_kv.float()).abs().max().item()
    assert diff > 1e-3, f"different seeds gave identical sliding_window_kv (max diff {diff})"
    print(f"[OK] different seeds produce different KV (max diff {diff:.3e})")


def main():
    print(f"device = {_device()}")
    test_shapes_at_S_256()
    test_shapes_at_S_2048()
    test_rope_only_touches_last_64_dims()
    test_fp8_simulation_is_noop_on_zeros()
    test_fp8_simulation_preserves_amax()
    test_overlap_transform_stride_2()
    test_determinism()
    test_different_seed_gives_different_output()
    print("\n[PASS] all Stage 0.5 DSV4KVGenerator unit tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())

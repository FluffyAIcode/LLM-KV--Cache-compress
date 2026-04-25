"""Unit tests for ``kakeyalattice.hf.KakeyaLatticeCache``.

Tests run on CPU, do NOT require a GPU or a real LLM. They use
synthetic K/V tensors of the shape ``[batch, num_kv_heads, seq, head_dim]``
that transformers' ``DynamicCache`` expects.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers", minversion="4.45")

from kakeyalattice.hf import KakeyaLatticeCache  # noqa: E402


def _make_kv(batch=1, nkv=8, seq=16, hd=128, dtype=torch.float32):
    """Synthesise realistic-ish K and V: Gaussian + per-head gain."""
    torch.manual_seed(42)
    k = torch.randn(batch, nkv, seq, hd, dtype=dtype)
    v = torch.randn(batch, nkv, seq, hd, dtype=dtype)
    # Give each head a different energy level (realistic LLM pattern).
    k = k * torch.linspace(0.5, 2.0, nkv).view(1, nkv, 1, 1)
    v = v * torch.linspace(0.5, 2.0, nkv).view(1, nkv, 1, 1)
    return k, v


class TestKakeyaLatticeCacheAPI:
    """API-level contract checks."""

    def test_requires_num_hidden_layers_and_head_dim(self):
        with pytest.raises(ValueError, match="num_hidden_layers"):
            KakeyaLatticeCache(variant="e8", q_range=38)

    def test_rejects_invalid_variant(self):
        with pytest.raises(ValueError, match="variant"):
            KakeyaLatticeCache(
                variant="d16", q_range=38,
                num_hidden_layers=4, head_dim=128, device="cpu",
            )

    def test_e8_rejects_non_power_of_2_head_dim_strict(self):
        """head_dim=96 is not power-of-2; E8 requires both power-of-2 AND divisible-by-8."""
        with pytest.raises(ValueError, match="not a power of 2"):
            KakeyaLatticeCache(
                variant="e8", q_range=38,
                num_hidden_layers=4, head_dim=96, device="cpu",
                strict=True,
            )

    def test_d4_rejects_non_power_of_2_head_dim(self):
        """head_dim=96 is divisible by 4 but not power-of-2, D4 should still reject."""
        with pytest.raises(ValueError, match="not a power of 2"):
            KakeyaLatticeCache(
                variant="d4", q_range=38,
                num_hidden_layers=4, head_dim=96, device="cpu",
                strict=True,
            )

    def test_d4_accepts_head_dim_64(self):
        """head_dim=64 is power-of-2 and divisible by 4, D4 should accept."""
        cache = KakeyaLatticeCache(
            variant="d4", q_range=38,
            num_hidden_layers=4, head_dim=64, device="cpu",
        )
        assert cache._supports_lattice is True

    def test_e8_rejects_head_dim_32_multiple_of_8_but_too_small(self):
        """head_dim=32 is power-of-2 and divisible by 8 - should work."""
        cache = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=32, device="cpu",
        )
        assert cache._supports_lattice is True

    def test_non_strict_warns_and_falls_back(self, recwarn):
        """head_dim=96 with strict=False: warn, fall back."""
        cache = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=96, device="cpu",
            strict=False,
        )
        assert cache._supports_lattice is False
        assert any("not a power of 2" in str(w.message) for w in recwarn.list)


class TestKakeyaLatticeCacheRoundtrip:
    """Correctness of the update path against a plain DynamicCache."""

    def test_update_shapes_match_dynamic_cache(self):
        """After update, cached K/V should have the same shape as a
        plain DynamicCache would produce."""
        from transformers import DynamicCache

        dc = DynamicCache()
        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        k, v = _make_kv(batch=1, nkv=8, seq=16, hd=128)

        k_out_plain, v_out_plain = dc.update(k, v, layer_idx=0)
        k_out_klc, v_out_klc = klc.update(k, v, layer_idx=0)

        assert k_out_plain.shape == k_out_klc.shape
        assert v_out_plain.shape == v_out_klc.shape
        assert k_out_plain.dtype == k_out_klc.dtype

    def test_update_applies_codec_not_identity(self):
        """The codec MUST change the K/V tensor (otherwise it's a no-op)."""
        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        k, v = _make_kv(batch=1, nkv=4, seq=8, hd=128)
        k_out, v_out = klc.update(k, v, layer_idx=0)
        # Output should NOT be byte-identical to input (codec is lossy).
        assert not torch.equal(k_out, k)
        assert not torch.equal(v_out, v)
        # But should be close (rel-MSE < 5% at Q=38 on random input).
        rel_mse = ((k_out - k).pow(2).sum() / k.pow(2).sum()).item()
        assert rel_mse < 0.05

    def test_codec_fire_counter(self):
        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        k, v = _make_kv()
        for li in range(4):
            klc.update(k, v, layer_idx=li)
        assert klc.codec_fired_per_layer == {0: 1, 1: 1, 2: 1, 3: 1}
        assert klc.skip_fired_per_layer == {}

    def test_boundary_layer_skip(self):
        """Layers in the boundary range pass through without codec."""
        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=6, head_dim=128, device="cpu",
            boundary=2,  # Layers 0,1,4,5 skip; layers 2,3 run codec
        )
        k, v = _make_kv()

        # Layer 0: boundary, should skip codec → output byte-identical to input
        k_out, v_out = klc.update(k, v, layer_idx=0)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

        # Layer 2: non-boundary, should run codec → output differs
        k_out, v_out = klc.update(k, v, layer_idx=2)
        assert not torch.equal(k_out, k)

        # Layer 5: boundary (last-2 of 6), should skip codec
        k_out, v_out = klc.update(k, v, layer_idx=5)
        assert torch.equal(k_out, k)

        assert klc.codec_fired_per_layer == {2: 1}
        assert klc.skip_fired_per_layer == {0: 1, 5: 1}

    def test_seq_length_is_dynamic_cache_behaviour(self):
        """Basic DynamicCache behaviour passes through correctly."""
        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        k, v = _make_kv(seq=8)
        klc.update(k, v, layer_idx=0)
        assert klc.get_seq_length(0) == 8

        # Decode-step update (new single token)
        k_new, v_new = _make_kv(seq=1)
        klc.update(k_new, v_new, layer_idx=0)
        assert klc.get_seq_length(0) == 9

    def test_isinstance_checks_pass(self):
        """transformers uses isinstance(cache, Cache) in generation.
        KakeyaLatticeCache must pass these checks."""
        from transformers import Cache, DynamicCache

        klc = KakeyaLatticeCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        assert isinstance(klc, Cache)
        assert isinstance(klc, DynamicCache)

    def test_strict_false_non_power_of_2_passes_through(self):
        """strict=False on incompatible head_dim: codec disabled,
        cache still works as a plain DynamicCache."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            klc = KakeyaLatticeCache(
                variant="e8", q_range=38,
                num_hidden_layers=4, head_dim=96, device="cpu",
                strict=False,
            )

        k = torch.randn(1, 4, 8, 96)
        v = torch.randn(1, 4, 8, 96)
        k_out, v_out = klc.update(k, v, layer_idx=0)
        # Codec should not have fired
        assert torch.equal(k_out, k)
        assert klc.skip_fired_per_layer == {0: 1}
        assert klc.codec_fired_per_layer == {}


class TestKakeyaLatticeCacheVariants:
    @pytest.mark.parametrize("variant,q_range", [
        ("d4", 10),
        ("d4", 38),
        ("d4", 152),
        ("e8", 10),
        ("e8", 38),
        ("e8", 152),
    ])
    def test_variant_q_combinations_work(self, variant, q_range):
        klc = KakeyaLatticeCache(
            variant=variant, q_range=q_range,
            num_hidden_layers=2, head_dim=128, device="cpu",
        )
        k, v = _make_kv(seq=4)
        k_out, v_out = klc.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 8, 4, 128)  # [b, nkv, seq, hd]

        # Sanity: higher Q should give lower rel-MSE
        rel_mse = ((k_out - k).pow(2).sum() / k.pow(2).sum()).item()
        if q_range == 152:
            assert rel_mse < 1e-3, f"Q=152 near-lossless; got rel-MSE {rel_mse}"
        elif q_range == 38:
            assert rel_mse < 1e-2, f"Q=38 balanced; got rel-MSE {rel_mse}"
        # Q=10 aggressive: just needs to be bounded
        assert rel_mse < 0.2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

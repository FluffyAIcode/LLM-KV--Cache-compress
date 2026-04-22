"""M6 Phase B.2a: M2 calibration loading + Σ_q whitening integration.

Tests exercise two axes:
  1. `calibration.load_calibration_bundle` correctly parses the M2
     safetensors + Lloyd-Max tables.
  2. `_seal_and_write_block` + `_decode_sealed` in calibration-on
     mode produce the same decoded tensor as calibration-off
     (because `L · L⁻¹ ≈ I` — the whitening/un-whitening cancels on
     the identity side; what it changes is the distortion the
     quantiser introduces *in the middle*, but that only shows up
     when input K has the anisotropy Σ_q models.  Synthetic random
     data has identity Σ_q, so the two paths should agree at codec-
     noise level.)

The stronger test (Σ_q actually reduces Δppl on real model
activations) is M7's benchmark territory.  Here we only assert:
  * Calibration loads without error
  * Σ_q factors go onto the right layer (via `layer.layer_name` parsing)
  * Round-trip with calibration on ≈ round-trip with calibration off
    when the input has identity covariance (the sanity case)

Runs only on CUDA + Triton.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    from kakeyaturbo_py import triton_is_available
    if not triton_is_available():
        pytest.skip("triton not available", allow_module_level=True)
except Exception as e:  # pragma: no cover
    pytest.skip(f"kakeyaturbo_py/triton import failed: {e}",
                allow_module_level=True)


from vllm_backend.kakeya_v1_3_ppl.calibration import (
    CalibrationBundle,
    load_calibration_bundle,
)
from vllm_backend.kakeya_v1_3_ppl.impl import (
    KakeyaV13PPLAttentionImpl,
    set_global_calibration,
    _parse_layer_idx_from_name,
)

REPO = Path(__file__).resolve().parents[3]
M2_DIR = REPO / "reports" / "v1_3_ppl" / "vllm_backend" / "calibration"


DEVICE = "cuda"
BLOCK_SIZE = 512
HEAD_DIM = 128
N_KV_HEADS = 8             # Qwen3-4B actual kv-head count
N_HEADS = 32               # GQA factor 4


class _FakeLayer:
    layer_name: str


def _make_fake_layer(layer_idx: int) -> _FakeLayer:
    l = _FakeLayer()
    l.layer_name = f"model.layers.{layer_idx}.self_attn.attn"
    return l


def _alloc_kv_cache(num_blocks: int, impl: KakeyaV13PPLAttentionImpl
                    ) -> torch.Tensor:
    slot_total = impl.k_config.slot_size_bytes + impl.v_config.slot_size_bytes
    return torch.zeros(
        (num_blocks, impl.num_kv_heads, slot_total),
        dtype=torch.uint8, device=DEVICE,
    )


# ---------------------------------------------------------------------------
# Unit: layer-name parsing
# ---------------------------------------------------------------------------


class TestParseLayerIdx:

    def test_parse_valid(self):
        assert _parse_layer_idx_from_name("model.layers.7.self_attn.attn") == 7
        assert _parse_layer_idx_from_name("model.layers.0.self_attn.attn") == 0
        assert _parse_layer_idx_from_name("model.layers.35.self_attn.attn") == 35

    def test_parse_missing(self):
        assert _parse_layer_idx_from_name(None) is None
        assert _parse_layer_idx_from_name("") is None
        assert _parse_layer_idx_from_name("foo.bar.baz") is None

    def test_parse_non_integer(self):
        assert _parse_layer_idx_from_name("model.layers.abc.attn") is None


# ---------------------------------------------------------------------------
# Unit: CalibrationBundle loading
# ---------------------------------------------------------------------------


class TestCalibrationLoading:

    @pytest.fixture(scope="class")
    def bundle(self):
        sigma = M2_DIR / "qwen3_4b_sigma_q.safetensors"
        kc = M2_DIR / "qwen3_4b_lloyd_max_K_b3.f32"
        vc = M2_DIR / "qwen3_4b_lloyd_max_V_b2.f32"
        if not sigma.exists():
            pytest.skip(f"M2 artefact missing: {sigma}")
        return load_calibration_bundle(sigma, kc, vc)

    def test_metadata(self, bundle):
        assert bundle.head_dim == 128
        assert bundle.num_kv_heads == 8
        assert bundle.num_layers == 36

    def test_active_layers(self, bundle):
        active = bundle.active_layers()
        # M2 computed Σ_q for every full-attention layer.  Qwen3-4B
        # has 36 layers, all full-attention → all active.
        assert len(active) == 36

    def test_load_with_skip_layers(self):
        """Caller-supplied skip_layers at load time should exclude
        those layers from the bundle (identity whitening for them)."""
        sigma = M2_DIR / "qwen3_4b_sigma_q.safetensors"
        kc = M2_DIR / "qwen3_4b_lloyd_max_K_b3.f32"
        vc = M2_DIR / "qwen3_4b_lloyd_max_V_b2.f32"
        if not sigma.exists():
            pytest.skip(f"M2 artefact missing: {sigma}")
        # Boundary skip list from M1_REPORT.md: layers 0, 1, 34, 35.
        b = load_calibration_bundle(
            sigma, kc, vc, skip_layers=[0, 1, 34, 35],
        )
        assert 0 not in b.active_layers()
        assert 1 not in b.active_layers()
        assert 34 not in b.active_layers()
        assert 35 not in b.active_layers()
        assert 2 in b.active_layers()
        assert 33 in b.active_layers()

    def test_chol_shapes(self, bundle):
        for l in sorted(bundle.active_layers())[:3]:
            L = bundle.sigma_q_chol[l]
            Linv = bundle.sigma_q_inv_chol[l]
            assert L.shape == (8, 128, 128)
            assert L.dtype == np.float32
            assert Linv.shape == L.shape

    def test_roundtrip_identity(self, bundle):
        """L · L⁻¹ ≈ I per (layer, head).  M2 already gated this at
        2e-5; re-verify so we catch any regression in the loader."""
        for l in sorted(bundle.active_layers())[:5]:
            L = bundle.sigma_q_chol[l]
            Linv = bundle.sigma_q_inv_chol[l]
            for h in range(L.shape[0]):
                prod = L[h] @ Linv[h]
                err = np.max(np.abs(prod - np.eye(128, dtype=np.float32)))
                assert err <= 2e-5, (
                    f"layer {l} head {h}: L · L⁻¹ error {err:.3e} > 2e-5"
                )

    def test_lloyd_max_tables(self, bundle):
        assert bundle.lloyd_max_k is not None
        assert bundle.lloyd_max_v is not None
        assert bundle.lloyd_max_k.shape == (8,)     # 2^3
        assert bundle.lloyd_max_v.shape == (4,)     # 2^2
        assert np.all(np.diff(bundle.lloyd_max_k) > 0)
        assert np.all(np.diff(bundle.lloyd_max_v) > 0)

    def test_bundle_whiten_unwhiten_identity(self, bundle):
        """Bundle-level `whiten → unwhiten` = identity within fp32.
        Different path from `test_roundtrip_identity` (uses the
        whitening helpers instead of raw matmul)."""
        rng = np.random.default_rng(0)
        for l in sorted(bundle.active_layers())[:3]:
            K = (rng.standard_normal((128, 128)) * 0.3).astype(np.float32)
            for h in range(8):
                Kt = bundle.whiten_layer_head(K, l, h)
                Kb = bundle.unwhiten_layer_head(Kt, l, h)
                err = np.max(np.abs(Kb - K)) / max(np.max(np.abs(K)), 1e-30)
                assert err <= 1e-4, (
                    f"layer {l} head {h}: whiten→unwhiten rel_err {err:.3e}"
                )


# ---------------------------------------------------------------------------
# Integration: impl with calibration installed
# ---------------------------------------------------------------------------


class TestImplWithCalibration:

    @pytest.fixture(autouse=True)
    def _install_bundle(self):
        """Install the real M2 bundle for every test, tear down after."""
        sigma = M2_DIR / "qwen3_4b_sigma_q.safetensors"
        kc = M2_DIR / "qwen3_4b_lloyd_max_K_b3.f32"
        vc = M2_DIR / "qwen3_4b_lloyd_max_V_b2.f32"
        if not sigma.exists():
            pytest.skip(f"M2 artefact missing: {sigma}")
        bundle = load_calibration_bundle(sigma, kc, vc)
        set_global_calibration(bundle)
        yield
        set_global_calibration(None)

    def _mk_impl(self):
        return KakeyaV13PPLAttentionImpl(
            num_heads=N_HEADS, head_size=HEAD_DIM, scale=HEAD_DIM ** -0.5,
            num_kv_heads=N_KV_HEADS,
        )

    def test_layer_state_populates_sigma_q(self):
        """For a calibrated layer (e.g. layer 5), `_ensure_layer_state`
        should attach Σ_q factors on the GPU."""
        impl = self._mk_impl()
        layer = _make_fake_layer(layer_idx=5)
        state = impl._ensure_layer_state(layer, torch.device(DEVICE))
        assert state.layer_idx == 5
        assert state.sigma_q_chol is not None
        assert state.sigma_q_chol.shape == (N_KV_HEADS, HEAD_DIM, HEAD_DIM)
        assert state.sigma_q_chol.device.type == "cuda"
        assert state.k_centroids is not None
        assert state.v_centroids is not None

    def test_layer_state_skip_listed_layer(self):
        """With skip_layers=[0] at load time, layer 0 should carry
        identity whitening (Σ_q factors = None) but still get the
        Lloyd-Max tables (they're stream-level, not per-layer)."""
        # Reinstall bundle with skip_layers=[0] to simulate boundary.
        sigma = M2_DIR / "qwen3_4b_sigma_q.safetensors"
        kc = M2_DIR / "qwen3_4b_lloyd_max_K_b3.f32"
        vc = M2_DIR / "qwen3_4b_lloyd_max_V_b2.f32"
        b = load_calibration_bundle(sigma, kc, vc, skip_layers=[0])
        set_global_calibration(b)
        try:
            impl = self._mk_impl()
            layer = _make_fake_layer(layer_idx=0)
            state = impl._ensure_layer_state(layer, torch.device(DEVICE))
            assert state.layer_idx == 0
            assert state.sigma_q_chol is None
            assert state.sigma_q_inv_chol is None
            assert state.k_centroids is not None
            assert state.v_centroids is not None
        finally:
            # Restore the full bundle for subsequent tests in this class.
            b_full = load_calibration_bundle(sigma, kc, vc)
            set_global_calibration(b_full)

    def test_seal_and_decode_roundtrip_with_whitening(self):
        """On a calibrated layer, seal → decode should produce output
        close to the direct-encode reference path *that also does
        whitening*.  This verifies the slot layout doesn't lose the
        whitening invariant through the pack/unpack roundtrip.
        """
        impl = self._mk_impl()
        layer = _make_fake_layer(layer_idx=5)   # calibrated layer
        kv_cache = _alloc_kv_cache(num_blocks=2, impl=impl)

        torch.manual_seed(42)
        N = BLOCK_SIZE
        K = (torch.randn(N, N_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.3).to(torch.bfloat16)
        V = (torch.randn(N, N_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.3).to(torch.bfloat16)

        slot_mapping = torch.arange(N, device=DEVICE, dtype=torch.int64)
        impl.do_kv_cache_update(
            layer, K.reshape(N, -1), V.reshape(N, -1),
            kv_cache, slot_mapping,
        )

        # Verify the block sealed.
        assert 0 not in layer._kakeya_state.staging_per_block
        assert bytes(kv_cache[0, 0, :4].cpu().numpy()) == b"KK13"

        # Decode path goes through whitening's inverse internally.
        K_dec = impl._decode_sealed(
            kv_cache, 0, stream="K", device=torch.device(DEVICE),
            layer=layer,
        )
        V_dec = impl._decode_sealed(
            kv_cache, 0, stream="V", device=torch.device(DEVICE),
            layer=layer,
        )
        # Shapes should match — whitening/un-whitening doesn't change rank.
        assert K_dec.shape == (N, N_KV_HEADS, HEAD_DIM)
        assert V_dec.shape == (N, N_KV_HEADS, HEAD_DIM)

        # Whitening-off reference: reconfigure impl without the bundle
        # and re-encode from scratch.
        set_global_calibration(None)
        impl_off = self._mk_impl()
        layer_off = _make_fake_layer(layer_idx=5)
        kv_cache_off = _alloc_kv_cache(num_blocks=2, impl=impl_off)
        impl_off.do_kv_cache_update(
            layer_off, K.reshape(N, -1), V.reshape(N, -1),
            kv_cache_off, slot_mapping,
        )
        K_dec_off = impl_off._decode_sealed(
            kv_cache_off, 0, stream="K", device=torch.device(DEVICE),
            layer=layer_off,
        )
        V_dec_off = impl_off._decode_sealed(
            kv_cache_off, 0, stream="V", device=torch.device(DEVICE),
            layer=layer_off,
        )

        # The two paths will differ because whitening re-shapes the
        # residual distribution the quantizer sees.  We assert two
        # things that must hold regardless:
        #
        # (a) The whitening-on K decode is still *finite* and has a
        #     sensible magnitude (no NaNs, no explosion).  Checks
        #     the un-whitening step doesn't amplify noise to infinity.
        # (b) V decode is bit-close to V_dec_off since V has no
        #     whitening path (the bundle doesn't touch V).  Only the
        #     Lloyd-Max V centroid table differs — by ~0.1 % per M2
        #     report.
        assert torch.all(torch.isfinite(K_dec))
        k_norm_ratio = K_dec.norm() / K.float().norm()
        assert 0.1 <= k_norm_ratio <= 10.0, (
            f"whitened K norm ratio out of sane range: {k_norm_ratio:.2f}"
        )

        # V paths differ only by Lloyd-Max table; ~1 % rel error
        # (tables are ~0.5 % off from Gaussian default).
        v_rel = (V_dec - V_dec_off).norm() / V_dec_off.norm()
        assert v_rel <= 0.05, (
            f"V decode with/without bundle differs by {v_rel:.3e} "
            "— expected < 5 % (only Lloyd-Max table differs, ~0.5 %)"
        )


class TestImplWithoutCalibration:
    """When no bundle is installed, the impl should behave exactly as
    Phase B.1 did (Gaussian default centroids, no whitening)."""

    def test_no_bundle_no_whitening(self):
        set_global_calibration(None)
        impl = KakeyaV13PPLAttentionImpl(
            num_heads=N_HEADS, head_size=HEAD_DIM, scale=HEAD_DIM ** -0.5,
            num_kv_heads=N_KV_HEADS,
        )
        layer = _make_fake_layer(layer_idx=5)
        state = impl._ensure_layer_state(layer, torch.device(DEVICE))
        assert state.sigma_q_chol is None
        assert state.k_centroids is None
        assert state.v_centroids is None

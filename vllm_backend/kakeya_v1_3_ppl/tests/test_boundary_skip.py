"""Guardrail #3 (in-backend): boundary-layer skip.

Scenario: `KAKEYA_BOUNDARY_SKIP_LAYERS="<list>"` must cause
`_seal_and_write_block` to bypass the codec for the listed layers
and stash bf16 K/V in the per-layer shadow.  `_decode_sealed` must
then return those bf16 tensors verbatim (bit-exact) on the matched
decode path — i.e. boundary layers see ZERO codec distortion, while
non-boundary layers still run through the normal codec pipeline.

This is the CPU-reference PR #17 recipe for DS-1.5B
(`[0, 1, 7, 14, 26, 27]`) and the Qwen3-4B natural-boundary recipe
(`[0, 1, 34, 35]`).  The env-var string drives both via the same
parser.

The test runs on CUDA (H200 in our CI).  No calibration files are
needed — the shadow path is data-independent.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
import torch


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for boundary-skip test "
                    "(seal/decode run Triton kernels on cuda).")


def _make_impl(num_layers: int = 4, num_kv_heads: int = 4, head_size: int = 128):
    """Build a `KakeyaV13PPLAttentionImpl` with a fake layer object
    sufficient for `_ensure_layer_state` to populate `layer_idx`.

    We don't need real calibration files — the bf16-shadow path
    doesn't touch Σ_q / centroids / outlier.
    """
    from kakeya_v1_3_ppl.impl import KakeyaV13PPLAttentionImpl
    impl = KakeyaV13PPLAttentionImpl(
        num_heads=num_kv_heads, head_size=head_size, scale=1.0,
        num_kv_heads=num_kv_heads, kv_cache_dtype="kakeya_v1_3_ppl",
    )
    return impl


class _FakeLayer:
    """Minimal stand-in for `vllm.v1.attention.backend.AttentionLayer`
    — only exposes the attributes the impl touches inside
    `_ensure_layer_state` + `_seal_and_write_block` +
    `_decode_sealed`.
    """
    def __init__(self, layer_idx: int):
        self.layer_name = f"model.layers.{layer_idx}.self_attn.attn"


class BoundarySkipParseTest(unittest.TestCase):
    """Pure-CPU tests for the env-var parser."""

    def setUp(self) -> None:
        # Snapshot env so we don't leak across tests.
        self._prev = os.environ.pop("KAKEYA_BOUNDARY_SKIP_LAYERS", None)

    def tearDown(self) -> None:
        if self._prev is not None:
            os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = self._prev
        else:
            os.environ.pop("KAKEYA_BOUNDARY_SKIP_LAYERS", None)

    def test_unset_means_empty(self):
        impl = _make_impl()
        self.assertEqual(impl._boundary_skip_set(), frozenset())
        self.assertFalse(impl._is_boundary_layer(0))
        self.assertFalse(impl._is_boundary_layer(99))
        self.assertFalse(impl._is_boundary_layer(None))

    def test_single_layer(self):
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "5"
        impl = _make_impl()
        self.assertEqual(impl._boundary_skip_set(), frozenset({5}))
        self.assertTrue(impl._is_boundary_layer(5))
        self.assertFalse(impl._is_boundary_layer(4))

    def test_ds15b_recipe(self):
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "0,1,7,14,26,27"
        impl = _make_impl()
        self.assertEqual(
            impl._boundary_skip_set(),
            frozenset({0, 1, 7, 14, 26, 27}),
        )
        # Spot-check interior.
        self.assertFalse(impl._is_boundary_layer(15))

    def test_whitespace_tolerant(self):
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = " 0, 1 , 34,35 "
        impl = _make_impl()
        self.assertEqual(impl._boundary_skip_set(), frozenset({0, 1, 34, 35}))

    def test_malformed_raises(self):
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "0,one,2"
        impl = _make_impl()
        with self.assertRaises(RuntimeError) as ctx:
            impl._boundary_skip_set()
        self.assertIn("one", str(ctx.exception))

    def test_empty_string_no_skip(self):
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = ""
        impl = _make_impl()
        self.assertEqual(impl._boundary_skip_set(), frozenset())

    def test_cached(self):
        """Parser runs once per impl; the cache is populated
        eagerly on first call and reused."""
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "2,3"
        impl = _make_impl()
        first = impl._boundary_skip_set()
        # Flip env — cached result should not change.
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "99"
        self.assertIs(impl._boundary_skip_set(), first)


class BoundarySkipSealDecodeTest(unittest.TestCase):
    """End-to-end seal → decode roundtrip on CUDA.

    Boundary layer: bf16-exact roundtrip (lossless except bf16 ULP).
    Non-boundary layer: runs full codec; not asserted lossless (codec
    has RSVD/Lloyd-Max quantisation) — we only assert the output
    shape is right and the bf16-shadow was NOT populated.
    """

    def setUp(self) -> None:
        _cuda_or_skip()
        self._prev = os.environ.pop("KAKEYA_BOUNDARY_SKIP_LAYERS", None)
        os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = "0,2"

    def tearDown(self) -> None:
        if self._prev is not None:
            os.environ["KAKEYA_BOUNDARY_SKIP_LAYERS"] = self._prev
        else:
            os.environ.pop("KAKEYA_BOUNDARY_SKIP_LAYERS", None)

    def _make_fake_cache_and_st(self, impl, device="cuda"):
        """Allocate a raw-byte paged cache large enough for one block,
        and a `_BlockStaging` filled with random bf16 K/V data."""
        from kakeya_v1_3_ppl.impl import _BlockStaging
        n = impl.block_size_codec
        H = impl.num_kv_heads
        D = impl.head_size
        k = torch.randn(n, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(n, H, D, device=device, dtype=torch.bfloat16)
        st = _BlockStaging(k_bf16=k, v_bf16=v, count=n)
        total_slot = (
            impl.k_config.slot_size_bytes + impl.v_config.slot_size_bytes
        )
        kv_cache = torch.zeros(
            (4, H, total_slot), device=device, dtype=torch.uint8,
        )
        return st, kv_cache

    def test_boundary_layer_roundtrip_is_bf16_exact(self):
        """Layer 0 is on the boundary list → seal stashes bf16, decode
        returns it verbatim.  We assert bit-exact: bf16_shadow stores
        the same tensor we fed in, and _decode_sealed converts bf16
        → fp32 losslessly (all representable bf16 values have exact
        fp32 counterparts)."""
        impl = _make_impl()
        layer0 = _FakeLayer(layer_idx=0)
        state = impl._ensure_layer_state(layer0, device=torch.device("cuda"))
        st, kv_cache = self._make_fake_cache_and_st(impl)
        block_idx = 0

        impl._seal_and_write_block(st, block_idx, kv_cache, layer0)
        # Shadow populated, cache slot zeroed.
        self.assertIn(block_idx, state.bf16_shadow)
        k_shadow, v_shadow = state.bf16_shadow[block_idx]
        self.assertTrue(torch.equal(k_shadow, st.k_bf16))
        self.assertTrue(torch.equal(v_shadow, st.v_bf16))
        v_end = impl.k_config.slot_size_bytes + impl.v_config.slot_size_bytes
        self.assertTrue(
            torch.all(kv_cache[block_idx, :, :v_end] == 0).item(),
            "boundary skip should zero the paged-cache slot",
        )

        # Decode returns bf16 values cast to fp32, bit-exact.
        k_dec = impl._decode_sealed(
            kv_cache, block_idx, stream="K",
            device=torch.device("cuda"), layer=layer0,
        )
        v_dec = impl._decode_sealed(
            kv_cache, block_idx, stream="V",
            device=torch.device("cuda"), layer=layer0,
        )
        self.assertTrue(torch.equal(k_dec, st.k_bf16.to(torch.float32)))
        self.assertTrue(torch.equal(v_dec, st.v_bf16.to(torch.float32)))

    def test_non_boundary_layer_runs_codec(self):
        """Layer 1 is NOT on the boundary list → seal runs the full
        codec; bf16_shadow is untouched."""
        impl = _make_impl()
        layer1 = _FakeLayer(layer_idx=1)
        state = impl._ensure_layer_state(layer1, device=torch.device("cuda"))
        st, kv_cache = self._make_fake_cache_and_st(impl)
        block_idx = 3
        impl._seal_and_write_block(st, block_idx, kv_cache, layer1)
        # Shadow empty for this block — codec ran.
        self.assertNotIn(block_idx, state.bf16_shadow)
        # Paged-cache slot has codec data — first 4 bytes are the
        # 'KK13' magic of _pack_parts_into_slot's K-stream slot.
        magic = kv_cache[block_idx, 0, :4].cpu().numpy().tobytes()
        self.assertEqual(magic, b"KK13")

    def test_mixed_request_boundary_and_codec_layers(self):
        """Two layers in one test: a boundary layer seals as bf16, a
        non-boundary layer seals with codec.  Ensures their shadow
        dicts don't cross-contaminate."""
        impl = _make_impl()
        layer0 = _FakeLayer(layer_idx=0)      # boundary
        layer1 = _FakeLayer(layer_idx=1)      # codec

        state0 = impl._ensure_layer_state(layer0, device=torch.device("cuda"))
        state1 = impl._ensure_layer_state(layer1, device=torch.device("cuda"))
        # Distinct state objects.
        self.assertIsNot(state0, state1)

        st0, kv0 = self._make_fake_cache_and_st(impl)
        st1, kv1 = self._make_fake_cache_and_st(impl)
        impl._seal_and_write_block(st0, 0, kv0, layer0)
        impl._seal_and_write_block(st1, 0, kv1, layer1)

        self.assertIn(0, state0.bf16_shadow)
        self.assertNotIn(0, state1.bf16_shadow)


if __name__ == "__main__":
    unittest.main()

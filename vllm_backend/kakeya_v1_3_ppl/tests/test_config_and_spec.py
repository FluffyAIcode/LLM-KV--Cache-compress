"""Phase-A unit tests: config sizing, spec layout, slot (de)serialisation.

Run locally (no vLLM install required for these tests):

    python -m pytest vllm_backend/kakeya_v1_3_ppl/tests/test_config_and_spec.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

# kakeyaturbo_py is required because _pack_parts_into_slot consumes
# an encode_block_codes dict.
pytest.importorskip("kakeyaturbo_py")

from kakeyaturbo_py import encode_block_codes

from vllm_backend.kakeya_v1_3_ppl.config import (
    KAKEYA_V1_3_PPL_NAME,
    KakeyaV13PPLConfig,
)
from vllm_backend.kakeya_v1_3_ppl.impl import KakeyaV13PPLAttentionImpl
from vllm_backend.kakeya_v1_3_ppl.spec import KakeyaV13PPLAttentionSpec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestKakeyaV13PPLConfig:

    def test_default_head_dim_128(self):
        cfg = KakeyaV13PPLConfig.default_for_head_dim(128)
        assert cfg.head_dim == 128
        assert cfg.d_eff == 64
        assert cfg.block_size_codec == 512
        assert cfg.k_centers == 16
        assert cfg.bit_width == 3
        assert cfg.outlier_budget_frac == 0.08
        assert cfg.wht_len == 64

    def test_slot_size_matches_plan_md_table(self):
        """PLAN.md §The key design decision table says the K-stream
        for D=128, d_eff=64, block_size=512, K=16, b_K=3 has:

            HEADER            48 B
            PCA basis   16 384 B
            PCA mean       256 B
            K-means cent  2 048 B
            K-means idx + residual  sum to ~12 544 B
            Outlier budget  ~5 242 B

        **Known gap vs PLAN.md**: PLAN.md forgot the per-vec `t` (fp16
        projection onto the chosen K-means centre) and `norm` (fp16
        inv-scale) fields, which add 2 × 512 × 2 B = 2 048 B.  Plus
        the u16-per-row outlier-count array (1 024 B) which PLAN.md's
        "4 B per outlier" formula also omitted.  These are both real
        algorithmic requirements, not slack — see M6_PHASE_A_REPORT.md
        §"Compression gap analysis" for the full derivation.
        """
        cfg = KakeyaV13PPLConfig.default_for_head_dim(128)
        assert cfg.pca_basis_bytes == 64 * 128 * 2
        assert cfg.pca_mean_bytes == 128 * 2
        assert cfg.kmeans_centroids_bytes == 16 * 64 * 2
        # Slot total in a reasonable range (36 KB … 50 KB).  Lower bound is
        # PLAN.md's under-count; upper bound is the current byte-aligned
        # layout.
        assert 36_000 <= cfg.slot_size_bytes <= 50_000, (
            f"unexpected slot_size: {cfg.slot_size_bytes} B"
        )

    def test_compression_ratio_is_reasonable(self):
        """K-stream with b=3, 8 % outlier budget, per-vec t+norm fields
        delivers ~2.92× compression vs bf16.  V-stream at b=2 without
        outlier delivers ~4.48×.  Combined is ~1.77× — less than
        PLAN.md's optimistic 4.03× because PLAN.md underestimates the
        per-vec codec metadata (see `test_slot_size_matches_plan_md_table`
        for the breakdown).

        This test nails the K-stream lower bound at 2.5× to catch any
        future regression; the combined target is checked in the spec
        test below.
        """
        cfg = KakeyaV13PPLConfig.default_for_head_dim(128)
        assert cfg.compression_ratio_vs_bf16 >= 2.5, (
            f"K-stream ratio below 2.5×: {cfg.compression_ratio_vs_bf16:.3f}×"
        )

    def test_invalid_configs_rejected(self):
        with pytest.raises(ValueError, match="bit_width"):
            KakeyaV13PPLConfig(head_dim=128, bit_width=5)
        with pytest.raises(ValueError, match="k_centers"):
            KakeyaV13PPLConfig(head_dim=128, k_centers=7)   # not pow-of-2
        with pytest.raises(ValueError, match="d_eff"):
            KakeyaV13PPLConfig(head_dim=128, d_eff=256)     # > head_dim
        with pytest.raises(ValueError, match="block_size_codec"):
            KakeyaV13PPLConfig(head_dim=128, block_size_codec=500)   # not pow-of-2
        with pytest.raises(ValueError, match="outlier_budget_frac"):
            KakeyaV13PPLConfig(head_dim=128, outlier_budget_frac=1.5)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


class TestKakeyaV13PPLAttentionSpec:

    def test_default_spec(self):
        spec = KakeyaV13PPLAttentionSpec(
            block_size=512, num_kv_heads=8, head_size=128,
        )
        assert spec.k_config.bit_width == 3
        assert spec.v_config.bit_width == 2
        assert spec.v_config.outlier_budget_frac == 0.0
        s = spec.summary()
        # Combined K+V ratio lands at ~1.77× given the per-vec t/norm
        # overhead (see TestKakeyaV13PPLConfig.test_compression_ratio).
        # Regression bar: ≥ 1.5× (catches gross layout inflation).
        assert s["compression_ratio_vs_bf16"] >= 1.5, s

    def test_rejects_mismatched_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            KakeyaV13PPLAttentionSpec(
                block_size=256, num_kv_heads=8, head_size=128,
            )

    def test_cache_shape(self):
        spec = KakeyaV13PPLAttentionSpec(
            block_size=512, num_kv_heads=8, head_size=128,
        )
        shape = spec.get_kv_cache_shape(num_blocks=100)
        assert shape[0] == 100
        assert shape[1] == 8
        assert shape[2] == spec.slot_budget_bytes
        # sanity: per-layer bytes at 100 blocks, 8 heads
        expected = 100 * 8 * spec.slot_budget_bytes
        assert shape[0] * shape[1] * shape[2] == expected


# ---------------------------------------------------------------------------
# Slot (de)serialisation
# ---------------------------------------------------------------------------


class TestSlotSerde:

    @pytest.fixture
    def impl(self):
        return KakeyaV13PPLAttentionImpl(
            num_heads=16, head_size=128, scale=1.0,
            num_kv_heads=8,
        )

    def _rust_parts(self, n: int, d: int, bit_width: int,
                    seed: int, outlier_threshold: float | None,
                    d_eff: int = 64):
        """Produce a kakeyaturbo_py encode_block_codes dict with
        `exact_rank_cap=d_eff` so the slot layout matches the
        production config (which pins d_eff for deterministic
        allocation sizes).
        """
        rng = np.random.default_rng(seed)
        X = (rng.standard_normal((n, d)) * 0.3).astype(np.float32)
        kwargs = dict(
            metric="mse", block_size=n, bit_width=bit_width,
            variance_ratio=1.0, k=16, rotation_seed=3405691582,
            pca_method="exact", skeleton_dtype="fp16", share_basis=False,
            exact_rank_cap=d_eff,
        )
        if outlier_threshold is not None:
            kwargs["outlier_threshold"] = float(outlier_threshold)
        parts = encode_block_codes(X, **kwargs)
        return {k: (np.asarray(v) if hasattr(v, "shape") else v)
                for k, v in parts.items()}

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_pack_unpack_roundtrip_k_stream(self, impl, seed):
        n = impl.block_size_codec
        d = impl.head_size
        parts = self._rust_parts(n, d, bit_width=3, seed=seed,
                                 outlier_threshold=None)
        cfg = impl.k_config
        slot = impl._pack_parts_into_slot(parts, cfg)
        recovered = impl._unpack_slot_into_parts(slot, cfg, head_size=d)

        # Skeleton fields: fp16 round-trip is lossy vs the fp32 value
        # kakeyaturbo_py returns, but the slot store *is* fp16, so the
        # correct comparison is against the fp16-rounded version of
        # the Rust output.
        def f16_rt(x):
            return np.asarray(x).astype(np.float16).astype(np.float32)

        np.testing.assert_array_equal(f16_rt(parts["mean"]), recovered["mean"])
        np.testing.assert_array_equal(f16_rt(parts["basis"]), recovered["basis"])
        np.testing.assert_array_equal(f16_rt(parts["centers"]), recovered["centers"])

        # Code fields must round-trip bit-exactly.
        np.testing.assert_array_equal(parts["seg_id"], recovered["seg_id"])
        np.testing.assert_array_equal(parts["t"], recovered["t"])
        np.testing.assert_array_equal(parts["norm"], recovered["norm"])
        np.testing.assert_array_equal(
            parts["residual_packed"], recovered["residual_packed"],
        )
        # Header-derived values.
        assert recovered["d_eff"] == parts["d_eff"]
        assert recovered["k"] == parts["k"]
        assert recovered["bit_width"] == parts["bit_width"]
        assert recovered["wht_len"] == cfg.wht_len

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_pack_unpack_roundtrip_with_outliers(self, impl, seed):
        n = impl.block_size_codec
        d = impl.head_size
        parts = self._rust_parts(n, d, bit_width=3, seed=seed,
                                 outlier_threshold=2.0)
        cfg = impl.k_config
        slot = impl._pack_parts_into_slot(parts, cfg)
        recovered = impl._unpack_slot_into_parts(slot, cfg, head_size=d)

        np.testing.assert_array_equal(
            parts["outlier_count"], recovered["outlier_count"],
        )
        # Per-row outlier (idx, val) sets must agree (order within a row
        # may vary because the pack/unpack round-trip re-groups entries
        # by row cursor; we compare as sorted sets).
        for i in range(n):
            cnt_rust = int(parts["outlier_count"][i])
            cnt_rec = int(recovered["outlier_count"][i])
            assert cnt_rec == cnt_rust, f"row {i}: {cnt_rec} vs {cnt_rust}"
            if cnt_rust == 0:
                continue
            rust_pairs = sorted(zip(
                parts["outlier_idx"][i, :cnt_rust].tolist(),
                parts["outlier_val"][i, :cnt_rust].tolist(),
            ))
            rec_pairs = sorted(zip(
                recovered["outlier_idx"][i, :cnt_rec].tolist(),
                recovered["outlier_val"][i, :cnt_rec].tolist(),
            ))
            assert rust_pairs == rec_pairs, f"row {i}: {rust_pairs} vs {rec_pairs}"

    def test_slot_size_is_exact(self, impl):
        """The slot byte count must equal `config.slot_size_bytes`
        exactly; vLLM's allocator will barf otherwise."""
        n = impl.block_size_codec
        d = impl.head_size
        parts = self._rust_parts(n, d, bit_width=3, seed=0,
                                 outlier_threshold=None)
        slot = impl._pack_parts_into_slot(parts, impl.k_config)
        assert slot.shape == (impl.k_config.slot_size_bytes,)

    def test_slot_magic_and_header(self, impl):
        n = impl.block_size_codec
        d = impl.head_size
        parts = self._rust_parts(n, d, bit_width=3, seed=0,
                                 outlier_threshold=None)
        slot = impl._pack_parts_into_slot(parts, impl.k_config)
        assert bytes(slot[:4]) == b"KK13"
        d_eff = int(np.frombuffer(bytes(slot[4:8]), dtype="<u4")[0])
        assert d_eff == impl.k_config.d_eff


# ---------------------------------------------------------------------------
# Name constants
# ---------------------------------------------------------------------------


def test_name_constant():
    assert KAKEYA_V1_3_PPL_NAME == "kakeya_v1_3_ppl"


# ---------------------------------------------------------------------------
# End-to-end: pack → Rust decode must equal the un-packed-parts Rust
# decode.  This is the real contract the M6 backend relies on: the
# paged-cache bytes we write must be recoverable into a parts dict
# that decodes to the correct tensor.
# ---------------------------------------------------------------------------


from kakeyaturbo_py import decode_block_from_parts


class TestPackDecodeE2E:

    @pytest.fixture
    def impl(self):
        return KakeyaV13PPLAttentionImpl(
            num_heads=16, head_size=128, scale=1.0,
            num_kv_heads=8,
        )

    def _parts(self, n, d, bit_width, seed, d_eff, outlier_threshold=None):
        rng = np.random.default_rng(seed)
        X = (rng.standard_normal((n, d)) * 0.3).astype(np.float32)
        kwargs = dict(
            metric="mse", block_size=n, bit_width=bit_width,
            variance_ratio=1.0, k=16, rotation_seed=3405691582,
            pca_method="exact", skeleton_dtype="fp16", share_basis=False,
            exact_rank_cap=d_eff,
        )
        if outlier_threshold is not None:
            kwargs["outlier_threshold"] = float(outlier_threshold)
        parts = encode_block_codes(X, **kwargs)
        return {k: (np.asarray(v) if hasattr(v, "shape") else v)
                for k, v in parts.items()}

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("outlier", [None, 2.0])
    def test_packed_slot_decodes_to_same_tensor(self, impl, seed, outlier):
        n, d = impl.block_size_codec, impl.head_size
        parts = self._parts(
            n=n, d=d, bit_width=3, seed=seed,
            d_eff=impl.k_config.d_eff, outlier_threshold=outlier,
        )
        # Serialise through the paged-cache slot, then unpack and decode.
        slot = impl._pack_parts_into_slot(parts, impl.k_config)
        recovered = impl._unpack_slot_into_parts(slot, impl.k_config, head_size=d)

        # Rust-decode both the original Rust parts AND the
        # slot-round-tripped parts; they must agree bit-exactly (the
        # slot stores fp16-rounded skeletons already, so decoding the
        # original parts through fp16 → fp32 gives the same tensor).
        # Force fp16 roundtrip on the original skeleton to match what
        # the slot layout captures.
        parts_f16 = dict(parts)
        for k in ("mean", "basis", "centers"):
            parts_f16[k] = np.asarray(parts[k]).astype(np.float16).astype(np.float32)

        dec_orig = np.asarray(decode_block_from_parts(parts_f16))
        dec_rec = np.asarray(decode_block_from_parts(recovered))

        # The slot-round-tripped decode must equal the fp16-rounded
        # Rust decode within fp32 precision.
        max_abs = float(np.max(np.abs(dec_orig - dec_rec)))
        rel = float(np.linalg.norm(dec_orig - dec_rec)
                    / (np.linalg.norm(dec_orig) + 1e-30))
        assert rel <= 1e-5, (
            f"slot decode diverges: rel={rel:.3e}, max_abs={max_abs:.3e}"
        )

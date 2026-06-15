"""Per-layer heterogeneous head_dim support (e.g. Gemma-4 sliding=256 / full=512).

Regression tests for the fix to "expected last dim 256, got 512": each cache must
build its codec per layer from the head_dim actually observed at that layer, so a
model whose layers present different K/V head dims works drop-in.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", minversion="4.45")

from kakeyalattice.hf import (  # noqa: E402
    KakeyaLatticeCache,
    KakeyaLatticeQuantizedCache,
    KakeyaLatticePackedCache,
    TurboQuantPackedCache,
)


def _bf16(*shape):
    torch.manual_seed(0)
    return torch.randn(*shape, dtype=torch.bfloat16)


# (B, NKV, S) fixed; only head_dim varies per layer, like Gemma-4.
B, NKV, S = 1, 4, 32
DIM0, DIM1 = 256, 512   # sliding vs full


class TestQuantizedHetero:
    def test_two_layers_different_head_dim(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38, num_hidden_layers=2, head_dim=DIM0, device="cpu")
        k0, v0 = _bf16(B, NKV, S, DIM0), _bf16(B, NKV, S, DIM0)
        k1, v1 = _bf16(B, NKV, S, DIM1), _bf16(B, NKV, S, DIM1)
        ok0 = qc.update(k0, v0, layer_idx=0)            # 256 — declared default
        ok1 = qc.update(k1, v1, layer_idx=1)            # 512 — full-attention layer
        assert ok0[0].shape[-1] == DIM0 and ok1[0].shape[-1] == DIM1
        assert ok0[0].is_contiguous() and ok1[0].is_contiguous()
        assert qc._codecs[0].D_shape == DIM0
        assert qc._codecs[1].D_shape == DIM1
        assert qc.get_seq_length(0) == S and qc.get_seq_length(1) == S
        assert qc.kv_storage_bytes() > 0

    def test_strict_raises_on_incompatible_dim(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38, num_hidden_layers=1, head_dim=256,
            device="cpu", strict=True)
        with pytest.raises(ValueError):
            qc.update(_bf16(B, NKV, S, 320), _bf16(B, NKV, S, 320), layer_idx=0)  # 320 not pow2

    def test_get_mask_sizes_reports_true_length(self):
        # Regression for the Gemma-4 sliding/blockwise mask device-assert:
        # get_mask_sizes must report the real cache length, not (query_length,0).
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38, num_hidden_layers=2, head_dim=DIM0, device="cpu")
        assert qc.get_mask_sizes(query_length=5, layer_idx=0) == (5, 0)   # empty layer
        qc.update(_bf16(B, NKV, S, DIM0), _bf16(B, NKV, S, DIM0), layer_idx=0)
        # after S stored tokens, a 1-token query sees kv_length = S + 1
        assert qc.get_mask_sizes(query_length=1, layer_idx=0) == (S + 1, 0)

    def test_nonstrict_falls_back_to_raw(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38, num_hidden_layers=1, head_dim=256,
            device="cpu", strict=False)
        k = _bf16(B, NKV, S, 320)
        out_k, out_v = qc.update(k, k.clone(), layer_idx=0)
        assert out_k.shape == (B, NKV, S, 320)          # raw passthrough
        assert qc.skip_fired_per_layer.get(0, 0) == 1
        assert 0 in qc._raw_layers


class TestRoundtripHetero:
    def test_two_layers_different_head_dim(self):
        c = KakeyaLatticeCache(
            variant="e8", q_range=38, num_hidden_layers=2, head_dim=DIM0, device="cpu")
        c.update(_bf16(B, NKV, S, DIM0), _bf16(B, NKV, S, DIM0), layer_idx=0)
        c.update(_bf16(B, NKV, S, DIM1), _bf16(B, NKV, S, DIM1), layer_idx=1)
        assert c._codecs[0].D_shape == DIM0 and c._codecs[1].D_shape == DIM1


class TestPackedHetero:
    def test_kakeya_packed_two_dims(self):
        qc = KakeyaLatticePackedCache(
            variant="e8", q_range=38, num_hidden_layers=2, head_dim=DIM0, device="cpu")
        qc.update(_bf16(B, NKV, S, DIM0), _bf16(B, NKV, S, DIM0), layer_idx=0)
        qc.update(_bf16(B, NKV, S, DIM1), _bf16(B, NKV, S, DIM1), layer_idx=1)
        assert qc._codecs[0].D_shape == DIM0 and qc._codecs[1].D_shape == DIM1
        assert qc.kv_storage_bytes() > 0
        assert qc.packed_pack_unpack_ok()              # lossless across both dims

    def test_turboquant_packed_two_dims(self):
        tq = TurboQuantPackedCache(
            bits_b=4, num_hidden_layers=2, head_dim=DIM0, device="cpu")
        o0 = tq.update(_bf16(B, NKV, S, DIM0), _bf16(B, NKV, S, DIM0), layer_idx=0)
        o1 = tq.update(_bf16(B, NKV, S, DIM1), _bf16(B, NKV, S, DIM1), layer_idx=1)
        assert o0[0].shape[-1] == DIM0 and o1[0].shape[-1] == DIM1
        assert tq._codecs[0].D_shape == DIM0 and tq._codecs[1].D_shape == DIM1
        assert tq.get_seq_length(1) == S
        assert tq.kv_storage_bytes() > 0

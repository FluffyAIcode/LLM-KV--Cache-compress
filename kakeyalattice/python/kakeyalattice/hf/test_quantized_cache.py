"""Unit tests for ``kakeyalattice.hf.KakeyaLatticeQuantizedCache``.

These tests run on CPU and do not require a GPU. They verify the two
properties that matter for the published claim of real HBM savings:

1. **Byte accounting**: a cache filled with N tokens of K and V across
   L layers holds strictly fewer tensor bytes than the equivalent
   DynamicCache. The expected ratio is bf16_bytes / (head_dim + 4)
   per K- or V-vector, which is 1.94x at head_dim=128.

2. **Codec equivalence**: encode-to-int8 followed by decode-from-int8
   is bit-identical to ``codec.roundtrip(x)``. This proves the
   storage savings come without any additional reconstruction loss
   relative to the existing roundtrip-only KakeyaLatticeCache.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers", minversion="4.45")

from kakeyalattice.hf import (  # noqa: E402
    KakeyaLatticeQuantizedCache,
    encode_to_indices,
    decode_from_indices,
)
from kakeyalattice import (  # noqa: E402
    V14KakeyaZamirLatticeGPU,
    V15KakeyaZamirE8GPU,
)


def _bf16(*shape):
    torch.manual_seed(42)
    return torch.randn(*shape, dtype=torch.bfloat16)


class TestCodecEquivalence:
    """encode_to_indices + decode_from_indices == codec.roundtrip,
    bit-identical."""

    @pytest.mark.parametrize("codec_cls", [V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU])
    @pytest.mark.parametrize("q_range", [10, 22, 38])
    def test_roundtrip_equivalence(self, codec_cls, q_range):
        torch.manual_seed(7)
        D = 128
        codec = codec_cls(D=D, q_range=q_range, device="cpu")
        x = torch.randn(16, D)
        x_rt = codec.roundtrip(x)
        q, n, m = encode_to_indices(codec, x)
        x_qc = decode_from_indices(codec, q, n, m, out_dtype=torch.float32)
        assert (x_rt - x_qc).abs().max().item() == 0.0, (
            f"encode/decode diverged from roundtrip for "
            f"{codec_cls.__name__} Q={q_range}"
        )

    def test_d4_uses_int8(self):
        codec = V14KakeyaZamirLatticeGPU(D=128, q_range=38, device="cpu")
        x = torch.randn(4, 128)
        q, _, _ = encode_to_indices(codec, x)
        assert q.dtype == torch.int8, f"D4 Q=38 should use int8 storage; got {q.dtype}"
        # D4 outputs integers in [-Q, Q]
        assert q.min().item() >= -38 and q.max().item() <= 38

    def test_e8_uses_int8_at_low_q(self):
        codec = V15KakeyaZamirE8GPU(D=128, q_range=38, device="cpu")
        x = torch.randn(4, 128)
        q, _, _ = encode_to_indices(codec, x)
        assert q.dtype == torch.int8, f"E8 Q=38 should use int8 storage; got {q.dtype}"
        # E8 half-integer doubling -> range [-2Q, 2Q] = [-76, 76]
        assert q.min().item() >= -76 and q.max().item() <= 76

    def test_e8_falls_back_to_int16_above_ceiling(self):
        # E8 ceiling for int8 is Q=63 (since values are doubled).
        codec = V15KakeyaZamirE8GPU(D=128, q_range=76, device="cpu")
        x = torch.randn(4, 128)
        q, _, _ = encode_to_indices(codec, x)
        assert q.dtype == torch.int16, (
            f"E8 Q=76 doubled -> {2*76} exceeds int8 [-128,127]; "
            f"should use int16; got {q.dtype}"
        )


class TestRealHBMSavings:
    """Byte-count proofs of real storage savings."""

    def test_byte_count_qwen3_like_layer(self):
        """1 layer, bs=1, 8 KV heads, 2048 seq, head_dim=128, bf16."""
        from transformers import DynamicCache

        B, NKV, S, D = 1, 8, 2048, 128
        # Baseline: DynamicCache stores raw bf16 K + V.
        dyn = DynamicCache()
        k, v = _bf16(B, NKV, S, D), _bf16(B, NKV, S, D)
        dyn.update(k, v, layer_idx=0)
        # Quantized: KakeyaLatticeQuantizedCache stores int8 indices +
        # fp16 norms + fp16 qmaxes.
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=1, head_dim=D, device="cpu",
        )
        qc.update(k, v, layer_idx=0)

        # Baseline bytes
        baseline_bytes = 0
        if hasattr(dyn, "layers"):
            for layer in dyn.layers:
                baseline_bytes += layer.keys.element_size() * layer.keys.numel()
                baseline_bytes += layer.values.element_size() * layer.values.numel()
        else:
            for layer in dyn.key_cache + dyn.value_cache:
                baseline_bytes += layer.element_size() * layer.numel()

        qc_bytes = qc.kv_storage_bytes()

        # Expected ratio: per-vector bf16=2D, int8-stored=D+4 (1 byte
        # per coord + 2 bytes for norm + 2 bytes for qmax).
        per_vec_bf16 = 2 * D
        per_vec_q = D + 4
        expected_ratio = per_vec_bf16 / per_vec_q
        measured_ratio = baseline_bytes / qc_bytes

        # Allow 1% tolerance for any future overhead.
        assert abs(measured_ratio - expected_ratio) < 0.02, (
            f"HBM ratio mismatch. Expected {expected_ratio:.3f}x, "
            f"got {measured_ratio:.3f}x (baseline={baseline_bytes}, "
            f"quantized={qc_bytes})"
        )
        # And confirm we actually saved bytes (not just matched).
        assert qc_bytes < baseline_bytes
        assert measured_ratio >= 1.9  # 1.94 expected at head_dim=128

    def test_byte_count_qwen2_05b_like_layer(self):
        """head_dim=64 -> ratio = 128/68 = 1.88x."""
        from transformers import DynamicCache

        B, NKV, S, D = 1, 8, 512, 64
        dyn = DynamicCache()
        k, v = _bf16(B, NKV, S, D), _bf16(B, NKV, S, D)
        dyn.update(k, v, layer_idx=0)
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=1, head_dim=D, device="cpu",
        )
        qc.update(k, v, layer_idx=0)

        baseline_bytes = 0
        if hasattr(dyn, "layers"):
            for layer in dyn.layers:
                baseline_bytes += layer.keys.element_size() * layer.keys.numel()
                baseline_bytes += layer.values.element_size() * layer.values.numel()
        qc_bytes = qc.kv_storage_bytes()
        ratio = baseline_bytes / qc_bytes
        # head_dim=64: 128/(64+4) = 1.882
        assert 1.85 <= ratio <= 1.92, f"Expected ~1.88x at D=64, got {ratio:.3f}x"


class TestGenerationCompatibility:
    """The cache must satisfy transformers' DynamicCache interface
    contract so that model.generate(past_key_values=cache) works."""

    def test_get_seq_length_matches_token_count(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
        )
        assert qc.get_seq_length(0) == 0
        # Push 16 tokens, then 32 more.
        k1, v1 = _bf16(1, 8, 16, 128), _bf16(1, 8, 16, 128)
        qc.update(k1, v1, layer_idx=0)
        assert qc.get_seq_length(0) == 16
        k2, v2 = _bf16(1, 8, 32, 128), _bf16(1, 8, 32, 128)
        qc.update(k2, v2, layer_idx=0)
        assert qc.get_seq_length(0) == 48

    def test_update_returns_concatenated_bf16(self):
        """update() must return bf16 K, V of shape [B, NKV, total_seq, D]
        for downstream attention to consume."""
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=4, head_dim=128, device="cpu",
            out_dtype=torch.bfloat16,
        )
        k1 = _bf16(1, 8, 16, 128)
        v1 = _bf16(1, 8, 16, 128)
        out_k, out_v = qc.update(k1, v1, layer_idx=0)
        assert out_k.dtype == torch.bfloat16
        assert out_k.shape == (1, 8, 16, 128)

        # Second update concats.
        k2 = _bf16(1, 8, 32, 128)
        v2 = _bf16(1, 8, 32, 128)
        out_k2, out_v2 = qc.update(k2, v2, layer_idx=0)
        assert out_k2.shape == (1, 8, 48, 128)


class TestContiguousBufferLayout:
    """The persistent KV state must be a single contiguous tensor per
    layer (not a Python list of per-call chunks), and the K/V returned
    to attention must be contiguous + SDPA-ready."""

    def test_storage_is_single_tensor_per_layer(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=2, head_dim=128, device="cpu",
        )
        # Two updates should grow ONE buffer, not append two list entries.
        qc.update(_bf16(1, 8, 16, 128), _bf16(1, 8, 16, 128), layer_idx=0)
        qc.update(_bf16(1, 8, 32, 128), _bf16(1, 8, 32, 128), layer_idx=0)
        codes = qc._k_codes[0]
        assert isinstance(codes, torch.Tensor), "K codes must be a single Tensor"
        assert codes.shape[-2] == 48, "buffer must grow along seq dim"
        assert codes.is_contiguous(), "persistent codes buffer must be contiguous"
        assert qc._k_norms[0].shape[-2] == 48
        assert qc._k_qmax[0].shape[-2] == 48
        # No stale list attributes should remain.
        assert not hasattr(qc, "_k_quant_entries")

    def test_returned_kv_is_contiguous(self):
        qc = KakeyaLatticeQuantizedCache(
            variant="e8", q_range=38,
            num_hidden_layers=2, head_dim=128, device="cpu",
            out_dtype=torch.bfloat16,
        )
        out_k, out_v = qc.update(_bf16(1, 8, 16, 128), _bf16(1, 8, 16, 128), layer_idx=0)
        assert out_k.is_contiguous() and out_v.is_contiguous()
        out_k2, out_v2 = qc.update(_bf16(1, 8, 8, 128), _bf16(1, 8, 8, 128), layer_idx=0)
        assert out_k2.is_contiguous() and out_v2.is_contiguous()
        assert out_k2.shape == (1, 8, 24, 128)

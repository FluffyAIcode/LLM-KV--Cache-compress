"""Tests for hf.bitpack (D4/E8 block codes + packer) and hf.turboquant.

All run on CPU. They establish that:
  * the fixed-width packer is lossless,
  * D4/E8 block encode/decode is a lossless bijection on real codec outputs,
  * block widths equal the codec's per-block bit budget (so packed bytes hit
    the bit-rate ceiling), and
  * TurboQuantCodec is the scalar-quantise baseline and is lossless.
"""
from __future__ import annotations

import math

import pytest
import torch

from kakeyalattice.hf.bitpack import (
    pack_codes, unpack_codes,
    d4_block_bits, e8_block_bits,
    d4_encode_blocks, d4_decode_blocks,
    e8_encode_blocks, e8_decode_blocks,
    is_regular_blocks,
    pack_lattice_codes, unpack_lattice_codes, packed_storage_bytes,
    real_packed_bytes, total_bits_per_vector,
    block_dim_for,
)
from kakeyalattice.hf import encode_to_indices
from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU
from kakeyalattice.hf.turboquant import TurboQuantCodec


class TestPacker:
    @pytest.mark.parametrize("width", [1, 2, 3, 5, 7, 8, 11, 16])
    def test_pack_unpack_identity(self, width):
        torch.manual_seed(width)
        n = 1000
        codes = torch.randint(0, 1 << width, (n,), dtype=torch.int64)
        buf = pack_codes(codes, width)
        assert buf.dtype == torch.uint8
        assert buf.numel() == math.ceil(n * width / 8)
        back = unpack_codes(buf, width, n)
        assert torch.equal(back, codes)

    def test_empty(self):
        buf = pack_codes(torch.zeros(0, dtype=torch.int64), 5)
        assert buf.numel() == 0


class TestD4BlockCode:
    @pytest.mark.parametrize("Q", [4, 10, 22, 38])
    def test_width_matches_ceiling(self, Q):
        expected = int(math.ceil(4 * math.log2(2 * Q + 1) - 1))
        assert d4_block_bits(Q) == expected

    @pytest.mark.parametrize("Q", [4, 10, 38, 152])
    def test_roundtrip_on_regular_codec_codes(self, Q):
        torch.manual_seed(Q)
        codec = V14KakeyaZamirLatticeGPU(D=128, q_range=Q, device="cpu")
        x = torch.randn(64, 128)
        q_lat, _, _ = encode_to_indices(codec, x)        # int8, even-sum D4
        blocks = q_lat.reshape(-1, 4).to(torch.int64)
        reg = is_regular_blocks(blocks, "d4", Q)
        rb = blocks[reg]
        codes = d4_encode_blocks(rb, Q)
        if d4_block_bits(Q) <= 62:
            assert int(codes.max().item()) < (1 << d4_block_bits(Q))
        assert torch.equal(d4_decode_blocks(codes, Q), rb)


class TestE8BlockCode:
    @pytest.mark.parametrize("Q", [4, 10, 22, 38])
    def test_width_matches_ceiling(self, Q):
        expected = int(math.ceil(8 * math.log2(2 * Q + 1)))
        assert e8_block_bits(Q) == expected

    @pytest.mark.parametrize("Q", [4, 10, 38])
    def test_roundtrip_on_regular_codec_codes(self, Q):
        torch.manual_seed(Q + 100)
        codec = V15KakeyaZamirE8GPU(D=128, q_range=Q, device="cpu")
        x = torch.randn(64, 128)
        q_lat, _, _ = encode_to_indices(codec, x)        # int8, DOUBLED E8
        blocks = q_lat.reshape(-1, 8).to(torch.int64)
        reg = is_regular_blocks(blocks, "e8", Q)
        rb = blocks[reg]
        codes = e8_encode_blocks(rb, Q)
        assert int(codes.max().item()) < (1 << e8_block_bits(Q))
        assert torch.equal(e8_decode_blocks(codes, Q), rb)


class TestPackLatticeRoundtrip:
    """Full pack -> unpack must be lossless on the WHOLE code tensor,
    including the rare irregular blocks (exception side-channel)."""

    @pytest.mark.parametrize("variant,Q", [
        ("d4", 4), ("d4", 10), ("d4", 38), ("d4", 152),
        ("e8", 4), ("e8", 10), ("e8", 38), ("e8", 152),
    ])
    def test_pack_unpack_lossless(self, variant, Q):
        torch.manual_seed(hash((variant, Q)) % 2**31)
        Cls = V14KakeyaZamirLatticeGPU if variant == "d4" else V15KakeyaZamirE8GPU
        codec = Cls(D=128, q_range=Q, device="cpu")
        x = torch.randn(80, 128)
        q_lat, _, _ = encode_to_indices(codec, x)
        packed = pack_lattice_codes(q_lat, variant, Q)
        back = unpack_lattice_codes(packed)
        assert back.shape == q_lat.shape
        assert torch.equal(back.to(torch.int64), q_lat.to(torch.int64)), (
            f"{variant} Q={Q} pack/unpack not lossless "
            f"(mode={packed['mode']}, exceptions={packed['exc_idx'].numel()})"
        )


class TestRealPackedBytes:
    @pytest.mark.parametrize("variant,Q", [("d4", 38), ("e8", 38), ("d4", 10), ("e8", 10)])
    def test_packed_bytes_hit_bit_rate_ceiling(self, variant, Q):
        torch.manual_seed(7)
        Cls = V14KakeyaZamirLatticeGPU if variant == "d4" else V15KakeyaZamirE8GPU
        codec = Cls(D=128, q_range=Q, device="cpu")
        x = torch.randn(40, 128)
        q_lat, norms, qmax = encode_to_indices(codec, x)
        bd = block_dim_for(variant)
        blk = d4_block_bits(Q) if variant == "d4" else e8_block_bits(Q)
        nvec = q_lat.numel() // 128
        packed = pack_lattice_codes(q_lat, variant, Q)
        assert packed["mode"] == "block"
        assert packed["width"] == blk
        expected_lat_bytes = math.ceil(nvec * (128 // bd) * blk / 8)
        assert packed["buf"].numel() == expected_lat_bytes
        # Real bytes per vector (incl. fp16 overhead + ~1% exceptions) < int8
        # (D+4) at Q=38.
        rb = real_packed_bytes(q_lat, norms, qmax, variant, Q)
        per_vec = rb / nvec
        assert per_vec < (128 + 4), f"{variant} Q={Q}: packed {per_vec:.1f}B not < int8 132B"


class TestTurboQuantCodec:
    @pytest.mark.parametrize("b", [3, 4, 6, 8])
    def test_matches_inline_scalar_quantise(self, b):
        """TurboQuantCodec.roundtrip == manual Hadamard+qmax+scalar-round."""
        torch.manual_seed(b)
        D = 128
        codec = TurboQuantCodec(D=D, bits_b=b, device="cpu")
        x = torch.randn(32, D)
        # inline reference
        flat = x.to(torch.float32)
        eps = torch.finfo(torch.float32).eps
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        n16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms
        y = unit @ codec.H
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        q16 = qmax.to(torch.float16).to(torch.float32)
        scale = q16 / float(codec.q_range)
        q = torch.round(y / scale).clamp(-codec.q_range, codec.q_range)
        ref = ((q * scale) @ codec.H) * n16
        got = codec.roundtrip(x.to(torch.float32))
        assert torch.allclose(got, ref, atol=0, rtol=0)

    @pytest.mark.parametrize("b", [3, 4, 8])
    def test_symbols_in_range_and_lossless(self, b):
        torch.manual_seed(b + 1)
        codec = TurboQuantCodec(D=128, bits_b=b, device="cpu")
        x = torch.randn(16, 128)
        sym, n, m = codec.encode_to_symbols(x)
        assert int(sym.min()) >= 0 and int(sym.max()) < (1 << b)
        dec = codec.decode_from_symbols(sym, n, m, out_dtype=torch.float32)
        rt = codec.roundtrip(x.to(torch.float32))
        assert torch.equal(dec, rt)

    def test_bit_budget(self):
        codec = TurboQuantCodec(D=128, bits_b=4, device="cpu")
        assert codec.bits_per_token_per_head == 128 * 4 + 32


class TestPackedCaches:
    """End-to-end packed caches: lossless packing + real packed-byte ratios."""

    def _bf16(self, *shape):
        torch.manual_seed(0)
        return torch.randn(*shape, dtype=torch.bfloat16)

    @pytest.mark.parametrize("variant,Q,lo,hi", [
        ("d4", 38, 2.3, 2.6), ("e8", 38, 2.3, 2.55), ("d4", 10, 3.0, 3.7),
    ])
    def test_kakeya_packed_ratio_and_lossless(self, variant, Q, lo, hi):
        from kakeyalattice.hf import KakeyaLatticePackedCache
        from transformers import DynamicCache
        B, NKV, S, D = 1, 8, 256, 128
        k, v = self._bf16(B, NKV, S, D), self._bf16(B, NKV, S, D)
        dyn = DynamicCache(); dyn.update(k, v, layer_idx=0)
        base = 0
        if hasattr(dyn, "layers"):
            for layer in dyn.layers:
                base += layer.keys.element_size() * layer.keys.numel()
                base += layer.values.element_size() * layer.values.numel()
        qc = KakeyaLatticePackedCache(variant=variant, q_range=Q,
                                      num_hidden_layers=1, head_dim=D, device="cpu")
        qc.update(k, v, layer_idx=0)
        ratio = base / qc.kv_storage_bytes()
        assert lo <= ratio <= hi, f"{variant} Q={Q}: packed ratio {ratio:.3f} not in [{lo},{hi}]"
        assert qc.packed_pack_unpack_ok()

    def test_turboquant_packed_ratio(self):
        from kakeyalattice.hf import TurboQuantPackedCache
        from transformers import DynamicCache
        B, NKV, S, D = 1, 8, 256, 128
        k, v = self._bf16(B, NKV, S, D), self._bf16(B, NKV, S, D)
        dyn = DynamicCache(); dyn.update(k, v, layer_idx=0)
        base = 0
        for layer in dyn.layers:
            base += layer.keys.element_size() * layer.keys.numel()
            base += layer.values.element_size() * layer.values.numel()
        tq = TurboQuantPackedCache(bits_b=4, num_hidden_layers=1, head_dim=D, device="cpu")
        tq.update(k, v, layer_idx=0)
        ratio = base / tq.kv_storage_bytes()
        # b=4: per-vec 128*4+32 = 544 bits = 68 bytes -> 256/68 = 3.76x
        assert 3.6 <= ratio <= 3.9, f"TQ b=4 packed ratio {ratio:.3f} unexpected"
        assert tq.get_seq_length(0) == S

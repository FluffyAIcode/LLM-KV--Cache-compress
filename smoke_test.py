#!/usr/bin/env python3
"""Minimal smoke test for Kakeya Gemma 4 KV compression."""

import torch

from kakeya_kv_codec import KakeyaCompressedLayer


def main():
    layer = KakeyaCompressedLayer(
        residual_length=8,
        block_size=4,
        min_rows_to_build=4,
    )

    for _ in range(5):
        key_states = torch.randn(1, 2, 3, 16)
        value_states = torch.randn(1, 2, 3, 16)
        full_keys, full_values = layer.update(key_states, value_states)

        assert full_keys.shape == full_values.shape
        assert full_keys.shape[:3] == (1, 2, layer.get_seq_length())

    print(
        "ok",
        {
            "seq_len": layer.get_seq_length(),
            "compressed_blocks": len(layer._compressed_key_blocks),
            "exact_tail_shape": tuple(layer.keys.shape),
        },
    )


if __name__ == "__main__":
    main()

# kakeyalattice-mlx

MLX (Apple Silicon) port of the KakeyaLattice v1.5 E8 nested-lattice
KV-cache codec. Bit-identical to the canonical PyTorch reference at
`kakeyalattice.V15KakeyaZamirE8GPU` when run in `float32`.

## What ships

- `hadamard.build_hadamard(D)` — Sylvester–Hadamard matrix normalised by `1/√D`.
- `closest_point.closest_e8(y)` — Conway–Sloane Alg 5 closest E8 point.
- `codec.E8LatticeCodebookMLX(D, q_range, dtype=mx.float32)` — the top-level
  codec, mirrors the PyTorch class's `.roundtrip(x)` signature.
- `kv_cache.KakeyaLatticeMLXCache` — mlx-lm compatible KV cache wrapper
  (per-layer codec, boundary-layer skip, fire counters).

## Parity guarantee

`tests/test_codec_parity.py` runs on Apple Silicon Macs and asserts
`max_abs_diff(reference.roundtrip(x), mlx.roundtrip(x).numpy()) == 0.0`
across random seeds at `D ∈ {64, 128, 256}` and `Q ∈ {4, 10, 38, 152}`.

## Install

```bash
pip install -e ".[mlx,parity,dev]"
pytest tests/ -v
```

On Linux CI (no MLX) the platform-agnostic tests still run; the MLX tests
skip cleanly via `pytest.importorskip("mlx.core")`.

## License

Apache-2.0.

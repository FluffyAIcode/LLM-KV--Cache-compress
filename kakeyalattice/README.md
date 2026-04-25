# kakeyalattice

**Nested-lattice KV-cache compression for LLM inference.**

```
pip install kakeyalattice
```

Ready-to-use Python codec that reduces the KV-cache memory footprint of
transformer-based LLM inference via a Zamir--Feder nested-lattice
quantiser. Two variants are exposed: $D_4$ (block dim 4, $+0.37\,$dB
shaping gain over $\mathbb{Z}^4$) and $E_8$ (block dim 8, $+0.65\,$dB
over $\mathbb{Z}^8$, $+0.29\,$dB over $D_4$).

## Quickstart — raw codec

```python
import torch
from kakeyalattice import V15KakeyaZamirE8GPU

codec = V15KakeyaZamirE8GPU(D=128, q_range=38, device="cuda")

# x: any tensor whose last dim equals D
x = torch.randn(16, 128, device="cuda")
x_reconstructed = codec.roundtrip(x)

# Compression info
codec.bits_per_token_per_head   # 832 for D=128 Q=38
codec.shaping_gain_db           # 0.65 (E8 vs Z^8)
```

## HuggingFace `transformers` integration

```
pip install kakeyalattice[hf]
```

Drop-in replacement for `DynamicCache` on any HF causal LM whose
`head_dim` is divisible by 4 (D4) or 8 (E8):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeyalattice.hf import KakeyaLatticeCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B", torch_dtype="bfloat16", device_map="cuda",
)

cache = KakeyaLatticeCache(
    variant="e8", q_range=38,                 # balanced operating point
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,           # 128 for Qwen2-1.5B
    device="cuda",
)

inputs = tok("Explain nested-lattice quantisation:", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200, past_key_values=cache)
print(tok.decode(out[0], skip_special_tokens=True))
```

Typical per-token KV memory vs bf16 baseline:

- `variant="e8" q_range=10` (aggressive): ~3.4× compression
- `variant="e8" q_range=38` (balanced, recommended): ~2.5× compression
- `variant="e8" q_range=152` (near-lossless): ~1.9× compression

## What it is (and what it is not)

### What it IS
- A **Python, GPU-first** reference implementation of the $D_4$/$E_8$
  nested-lattice codec from the paper
  [*KakeyaLattice: Nested-Lattice KV-Cache Compression with
  Kakeya-Style Discrete Codebooks*](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper).
- Five engineering levers (unit-norm factorisation, Sylvester--Hadamard
  rotation, per-vector adaptive $q_\mathrm{max}$, joint scaling, clamp)
  + one lattice (D4 or E8) closest-point. All on GPU via PyTorch
  tensor ops; no hand-written CUDA required.
- Bit-level reproducibility: `benchmarks/e8_parity_and_smoke.py` pins
  SHA256 hashes of codec output at fixed seeds, so regressions are
  caught in CI.

### What it is NOT
- **Not a fused decode kernel**: each call runs PyTorch-level ops. A
  Triton-fused E8 closest-point kernel would reduce decode latency by
  ~3× on H200 but is out of scope for this release.
- **Not a drop-in replacement for FP8 on arbitrary attention kernels**:
  FlashAttention / FlashMLA / paged-attention kernels expect FP8 or
  BF16 KV. `KakeyaLatticeCache` stores the codec-roundtripped tensor
  at the same dtype as the model's KV, which means the memory saving
  is currently **nominal (bytes saved in storage format)** unless you
  also change the cache dtype. A follow-up vLLM integration (see
  `vllm_backend/kakeya_v1_4_snapshot/`) changes the cache storage
  dtype for real HBM savings.
- **Not claiming to beat every quantisation baseline**: it beats FP8
  per-64-block on DeepSeek-V4-Flash's highly anisotropic KV by ~12%
  rel-MSE at ~22% fewer bits (see
  [Stage 0.75 findings](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_5_release/dsv4_stage075)).
  On near-Gaussian KV from small dense models it is comparable to or
  slightly behind well-tuned scalar quantisers.

## Head-dim compatibility

| model family | head_dim | D4 compatible | E8 compatible |
| --- | --- | --- | --- |
| LLaMA-3.x | 128 | ✅ | ✅ |
| Qwen2/Qwen3 (hidden-size scale) | 64–256 (must be divisible by 8 for E8) | ✅ (all) | ✅ (128, 256); ✗ (96, 176) |
| Mistral / Mixtral | 128 | ✅ | ✅ |
| DeepSeek-V2/V3 (MLA) | 128 + 64 rope | ✅ | ✅ |
| DeepSeek-V4-Flash (MLA + CSA/HCA) | 512 shared-latent | ✅ | ✅ |
| Gemma-3 / Gemma-4 | 256 (full attn), 256 (sliding) | ✅ | ✅ |

If `head_dim % 4 != 0`, the codec raises `ValueError` by design (no
silent fallback). Models like legacy GPT-NeoX variants with
`head_dim = 96` need a different block structure.

## Reproducibility

```bash
pip install kakeyalattice[hf,dev]
git clone https://github.com/FluffyAIcode/LLM-KV--Cache-compress
cd LLM-KV--Cache-compress
pytest benchmarks/e8_parity_and_smoke.py -v   # frozen SHA256 parity
pytest benchmarks/ablation_parity_check.py -v  # 6-variant ablation
```

`benchmarks/frozen_parity.json` contains 8 pinned SHA256 hashes. Any
change to the codec that breaks bit-level parity fails CI.

## Citation

If you use this package, please cite:

```bibtex
@misc{li2026kakeyalattice,
  author = {Li, Allen},
  title  = {KakeyaLattice: Nested-Lattice KV-Cache Compression with
            Kakeya-Style Discrete Codebooks},
  year   = {2026},
  url    = {https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper},
}
```

## License

Apache-2.0. See `LICENSE`.

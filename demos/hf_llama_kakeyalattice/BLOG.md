# Adding KakeyaLattice as a HuggingFace `DynamicCache` drop-in

**TL;DR**: [`kakeyalattice`](https://pypi.org/project/kakeyalattice/) is a
pip-installable nested-lattice KV-cache compression codec that plugs
into `transformers` via a `DynamicCache` subclass. Install, swap
`past_key_values=KakeyaLatticeCache(...)`, ship.

```python
pip install 'kakeyalattice[hf]'
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kakeyalattice.hf import KakeyaLatticeCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype="bfloat16", device_map="cuda",
)

cache = KakeyaLatticeCache(
    variant="e8", q_range=38,
    num_hidden_layers=model.config.num_hidden_layers,
    head_dim=model.config.head_dim,
    device="cuda",
)

inputs = tok("Write a haiku about nested lattices:", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=40, past_key_values=cache)
print(tok.decode(out[0]))
```

## What it does

`KakeyaLatticeCache` intercepts every K/V write into the cache and
applies a **Zamir-Feder nested-lattice roundtrip** (encode + decode)
before storing. Two lattice variants are available:

- **`variant="d4"`**: 4-dim blocks, Conway-Sloane Alg 4, +0.37 dB shaping gain vs Z⁴
- **`variant="e8"`**: 8-dim blocks, Conway-Sloane Alg 5, +0.65 dB vs Z⁸, +0.29 dB vs D4

The codec is **five engineering levers + one lattice**:

1. **Unit-norm factorisation** (stores ‖x‖ as one fp16)
2. **Sylvester-Hadamard rotation** (whitens coordinate-wise variance)
3. **Per-vector adaptive qmax** (one fp16 scalar per vector)
4. **Joint scaling** across all blocks of a vector
5. **Clamp** to lattice range
6. **Closest-point search** on D4 or E8

For the underlying math and our benchmark results across 4 transformer
families see the
[paper](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper).

## Who is it for

| Workload | Benefit |
| --- | --- |
| Long-context inference on highly non-Gaussian KV (DeepSeek-V4, MoE models) | KakeyaLattice beats FP8 per-64-block by ~12% rel-MSE at ~22% fewer bits (measured on V4-Flash, [Stage 0.75](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_5_release/dsv4_stage075)) |
| Aggressive compression (3× and beyond) where FP4/INT4 KV is considered | D4/E8 Q=10 gives 3.4× compression at bounded quality loss |
| Research on KV-cache distribution geometry | Built-in non-Gaussian audit (kurtosis, isotropy, Hadamard-whitened variance ratio) |
| Reproducibility-sensitive experiments | SHA256 frozen-parity regression gate on every release |

| Not well-suited for | Why |
| --- | --- |
| Latency-critical decode at small context | Pure PyTorch ops; ~1.3–2× slower than bf16 decode without a fused kernel |
| Models with `head_dim ∉ {power-of-2} ∩ {multiple-of-block-dim}` | D4 needs `head_dim % 4 == 0`; E8 needs `% 8 == 0`; both need power-of-2 (for Hadamard). GPT-NeoX-class models with `head_dim=96` need a different block structure. |
| Strict iso-rate comparisons with FP8 | KakeyaLattice's `ceil(4·log₂(2Q+1)-1)` block rate adds ~3-11% packed-rate premium over scalar int; the comparison is near-matched rate, not strictly iso-rate |

## A small benchmark

On Qwen3-4B (H200, ctx=2048, n=4 passages from WikiText-style passages),
our [paper §5.2](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper)
reports:

| codec | bits/vec | K-MSE ratio | \|Δppl\| | top-1 |
| --- | --- | --- | --- | --- |
| TurboQuant b=4 (reference) | 544 | 1.00× | 4.22% | 97.7% |
| **KakeyaLattice D4 Q=10** | **576 (+6%)** | **0.64×** | **1.86%** | **97.3%** |
| TurboQuant b=8 (reference) | 1056 | 1.00× | 0.66% | 98.8% |
| **KakeyaLattice D4 Q=152** | **1088 (+3%)** | **0.91×** | **0.37%** | **99.6%** |

**D4 at Q=10** wins on both K-MSE (-36%) and Δppl (-56%) at a near-matched
rate. **E8 at Q=38** gives another +1.78 dB K-MSE advantage over D4 on
the same four-model benchmark, though at a 1.6–1.8× encode-latency cost.

On **DeepSeek-V4-Flash** (the most non-Gaussian KV distribution we've
measured to date), E8 Q=38 beats V4's internal FP8 per-64-block by
~12% rel-MSE at ~22% fewer bits. See
[Stage 0.75 findings](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_5_release/dsv4_stage075).

## Honest caveats

### 1. Memory savings are nominal in the transformers path

`KakeyaLatticeCache` roundtrips K/V but stores the reconstructed tensor
in the model's KV dtype (bf16/fp16/fp32). **Actual HBM bytes stored do
not decrease** unless you also change the cache storage dtype.

To realize real HBM savings you need a custom attention kernel that
can read lattice-encoded KV and dequantise at attention time. Our
`vllm_backend/kakeya_v1_4_snapshot/` and the Stage 1 scaffold
(`benchmarks/dsv4_stage1/`) target this; upstream transformers
integration via `QuantizedCache` requires a fused decode kernel that
is on our Stage 2 roadmap.

**What you get today from `KakeyaLatticeCache`**:
- Reconstruction-accuracy benefit (K/V closer to unquantised than
  after FP8 storage)
- Distribution audit infrastructure (non-Gaussian gates)
- A reference implementation to validate on your own models

**What you don't get today**: bytes saved in the GPU KV cache.

### 2. Decode latency

The codec runs as pure PyTorch ops per layer per token. On H200 this
adds ~0.57 ms per layer at D=512 (measured Stage 0.75), totalling
~25 ms for V4-Flash's 43 layers. A fused Triton implementation would
bring this to ~5 ms.

### 3. Head-dim constraints

Both lattice variants require:

- **Power-of-2** `head_dim` (for Sylvester-Hadamard rotation)
- Divisible by block dim: **4 for D4**, **8 for E8**

Common model head_dims:

| head_dim | D4 | E8 |
| --- | --- | --- |
| 64 | ✅ | ✅ |
| 96 | ❌ (not power-of-2) | ❌ (not power-of-2) |
| 128 | ✅ | ✅ |
| 176 | ❌ (not power-of-2) | ❌ (not power-of-2) |
| 256 | ✅ | ✅ |
| 512 | ✅ | ✅ |

Pass `strict=False` to the cache constructor to silently fall back to
plain `DynamicCache` on incompatible models.

## What's next

1. **Try it on your model**: `pip install 'kakeyalattice[hf]'`, run
   our [benchmark script](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/benchmarks/hf_cache_demo/run_hf_cache_benchmark.py)
   on a few of your workloads, tell us what you find
2. **Space demo**: the [HF Space](https://huggingface.co/spaces) code
   is in `demos/hf_llama_kakeyalattice/`. Clone, push, serve
3. **Fused kernel**: Triton implementation of the Conway-Sloane
   closest-point algorithms. PRs welcome — this is the biggest
   unlock on the roadmap
4. **Upstream transformers integration**: once we have a fused
   kernel and multi-model evidence at scale, we'll propose
   `KakeyaLatticeQuantizedCache` as a `QuantizedCache` backend
   (see the project RFC)

## Citation

```bibtex
@misc{li2026kakeyalattice,
  author = {Li, Allen},
  title  = {KakeyaLattice: Nested-Lattice KV-Cache Compression with
            Kakeya-Style Discrete Codebooks},
  year   = {2026},
  url    = {https://github.com/FluffyAIcode/LLM-KV--Cache-compress},
}
```

## Links

- **PyPI**: https://pypi.org/project/kakeyalattice/
- **GitHub**: https://github.com/FluffyAIcode/LLM-KV--Cache-compress
- **Paper**: [reports/paper/](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/paper)
- **Benchmarks**: [reports/v1_5_release/](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/reports/v1_5_release)
- **HF Space demo source**: [`demos/hf_llama_kakeyalattice/`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/tree/main/demos/hf_llama_kakeyalattice)

License: Apache-2.0.

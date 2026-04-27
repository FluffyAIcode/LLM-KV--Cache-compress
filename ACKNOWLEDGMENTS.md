# Acknowledgments

KakeyaLattice stands on a stack of prior work — theoretical, algorithmic,
and infrastructural. This file names that stack explicitly, both to give
credit and to make the provenance of our design decisions reproducible
for reviewers.

## Theoretical foundations

The nested-lattice quantisation framework that KakeyaLattice specialises
to LLM KV caches originates with Zamir and Feder's long line of work on
nested lattice codes and shaping gain, and with Conway and Sloane's
reference text on lattices and closest-point decoders.

- **Zamir, R., & Feder, M.** (1996). *On lattice quantization noise.*
  IEEE Transactions on Information Theory 42(4), 1152–1159.
  [doi:10.1109/18.508838](https://doi.org/10.1109/18.508838). — The
  nested-lattice quantisation model and the shaping-gain bound that
  motivates our Sylvester–Hadamard + L² scaling pre-processing.
- **Conway, J. H., & Sloane, N. J. A.** (1999). *Sphere Packings,
  Lattices and Groups* (3rd ed.). Springer.
  [doi:10.1007/978-1-4757-6568-7](https://doi.org/10.1007/978-1-4757-6568-7).
  — The closest-point algorithms for D4 and E8 that implement the
  "snap" step of the codec (`kakeyalattice/python/kakeyalattice/lattice_codebooks.py`).
- **Sylvester, J. J.** (1867). *Thoughts on inverse orthogonal matrices,
  simultaneous sign-successions, and tessellated pavements in two or
  more colours, with applications to Newton's rule, ornamental
  tile-work, and the theory of numbers.* Philosophical Magazine
  34(232), 461–475. — The 1867 construction of the ±1 Hadamard matrix
  family we use as the rotation basis.

## Peer methods we compare against

We benchmark KakeyaLattice head-to-head against the strongest published
KV-cache compression methods. The comparison would not be possible
without the authors' open-source reference implementations and clear
writeups — we thank them for making direct comparability the community
norm.

- **TurboQuant**. Zandieh, A., Han, I., Karbasi, A., & Mirrokni, V.
  (2024). *TurboQuant: Fast, High-Fidelity Quantization for Large
  Language Models.* arXiv:2406.17005.
  [https://arxiv.org/abs/2406.17005](https://arxiv.org/abs/2406.17005).
  — Our primary scalar-quantiser baseline across all four iso-PPL
  benchmark models.
- **KIVI**. Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V.,
  Chen, B., & Hu, X. (2024). *KIVI: A Tuning-Free Asymmetric 2bit
  Quantization for KV Cache.* arXiv:2402.02750.
  [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750).
  — The leading 2-bit KV quantiser; referenced in
  [`docs/faq.md`](docs/faq.md#comparisons) as our low-bit baseline.
- **SmoothQuant**. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J.,
  & Han, S. (2023). *SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models.* ICML 2023.
  arXiv:2211.10438.
  [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438).
  — The migration-based per-channel scalar quantiser that established
  the "difficulty migrates from activations to weights" framing.
- **HQQ**. Badri, H., & Shaji, A. (2023). *Half-Quadratic Quantization
  of Large Machine Learning Models.*
  [https://mobiusml.github.io/hqq_blog/](https://mobiusml.github.io/hqq_blog/).
  — Weight quantiser; orthogonal to KV-cache compression. Documented in
  [`docs/faq.md`](docs/faq.md#comparisons) as a composable partner.
- **Quanto** (`QuantoQuantizedCache`). Hugging Face transformers team
  (2024). In-tree KV-cache quantiser.
  [transformers/src/transformers/cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py).
  — The reference per-channel scalar KV quantiser inside the
  `transformers` library; users will compare us to this first.
- **SnapKV**. Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A.,
  Ye, H., Cai, T., Lewis, P., & Chen, D. (2024). *SnapKV: LLM Knows
  What You are Looking for Before Generation.* arXiv:2404.14469.
  [https://arxiv.org/abs/2404.14469](https://arxiv.org/abs/2404.14469).
  — Eviction-based, orthogonal to quantisation, composes with us.
- **H2O**. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R.,
  Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., & Chen, B.
  (2023). *H2O: Heavy-Hitter Oracle for Efficient Generative Inference
  of Large Language Models.* NeurIPS 2023. arXiv:2306.14048.
  [https://arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048).
  — Eviction-based; composes with us.
- **Scissorhands**. Liu, Z., Desai, A., Liao, F., Wang, W., Xie, V.,
  Xu, Z., Kyrillidis, A., & Shrivastava, A. (2023). *Scissorhands:
  Exploiting the Persistence of Importance Hypothesis for LLM KV
  Cache Compression at Test Time.* NeurIPS 2023. arXiv:2305.17118.
  [https://arxiv.org/abs/2305.17118](https://arxiv.org/abs/2305.17118).
  — Eviction-based; composes with us.

## Infrastructure

- **vLLM** (Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H.,
  Gonzalez, J. E., Zhang, H., & Stoica, I., 2023. *Efficient Memory
  Management for Large Language Model Serving with PagedAttention.*
  SOSP 2023. arXiv:2309.06180.
  [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))
  — paged-attention reference implementation and KV cache
  instrumentation hooks. All iso-PPL benchmark numbers in the README
  come from vLLM prefill on real weights; the `vllm_backend/` plugin
  is a `vllm.general_plugins` entry point.
- **FlashAttention** (Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C.,
  2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
  [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135);
  Dao, T., 2023. *FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.* arXiv:2307.08691.
  [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691))
  — the bfloat16 attention kernel that every reported perplexity
  number was measured against.
- **Hugging Face transformers** (Wolf, T., Debut, L., Sanh, V., Chaumond,
  J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz,
  M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y.,
  Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., &
  Rush, A. M., 2020. *Transformers: State-of-the-Art Natural Language
  Processing.* EMNLP 2020 Systems Demonstrations.
  arXiv:1910.03771.
  [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771))
  — `DynamicCache`, the interface our `KakeyaLatticeCache` subclasses.

## Model weights we evaluate on

All reported numbers depend on specific model checkpoints released by
their respective teams. We thank them for the open weights that make
direct comparability possible.

- **Qwen3 team**, Alibaba (2026).
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
  [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
- **Qwen2 team**, Alibaba (2024).
  [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B).
- **DeepSeek team** (2025).
  [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
  [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).
- **GLM team**, Zhipu AI (2024).
  [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat).
- **Gemma team**, Google DeepMind (2025).
  [google/gemma-4-e4b](https://huggingface.co/google/gemma-4-e4b).
- **Meta** (2024).
  [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

## Evaluation datasets

- **WikiText-103**. Merity, S., Xiong, C., Bradbury, J., & Socher, R.
  (2017). *Pointer Sentinel Mixture Models.* ICLR 2017.
  arXiv:1609.07843.
  [https://arxiv.org/abs/1609.07843](https://arxiv.org/abs/1609.07843).
  — All iso-PPL numbers in the README and in
  `reports/v1_4_release/kv_128k_isoppl_n8/` are measured on
  WikiText-103.

## Compute

The n=8 iso-PPL benchmark sweeps, the DeepSeek-V4-Flash Stage 0.75
audit, and the streaming-latency measurements were run on rented
NVIDIA H200 SXM 141 GiB instances on [vast.ai](https://vast.ai).
Total H200 compute budget for the data that backs the public README
tables is under USD 50.

## Early users and contributors

This section tracks named early users and external contributors as
they arrive. If you use KakeyaLattice in a deployment and would like
to be listed here, see
[`DEPLOYMENTS.md`](DEPLOYMENTS.md).

(empty at time of release 1.5.0 — 2026-04-24)

## Corrections and reviewers

If you spot a missed citation, an incorrect attribution, or a method
we should benchmark against, please open an issue titled
`Acknowledgment: <what is missing>` at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/issues>. We
take this file as seriously as the paper itself.

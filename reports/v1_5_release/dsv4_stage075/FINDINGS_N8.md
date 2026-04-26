# Stage 0.75 Findings (n=8) — DeepSeek-V4-Flash with **trained** weights

**Run date**: 2026-04-26
**Hardware**: NVIDIA H200 SXM 141 GiB × 2 (run uses only GPU 0), vast.ai
**V4 weights**: `deepseek-ai/DeepSeek-V4-Flash` safetensors shards 2, 4, 5 (layers 0/SWA, 2/c4a, 3/c128a; FP8-E4M3 dequantised via E8M0 block scales to FP32)
**Host hidden states**: `Qwen/Qwen2-0.5B` post-embedding, projected 896→4096 via fixed-seed linear
**Protocol**: **n=8** semantically diverse WikiText-style passages × 1 forward each, `seqlen=2048`, `batch=1`, FP8-simulated nope path
**Aggregation**: Student-t 95% CI half-width over n=8 independent passage runs

## Purpose — closing Caveat 1 of `FINDINGS.md`

`reports/v1_5_release/dsv4_stage075/FINDINGS.md` Caveat 1:

> One passage, one layer of each type. V4-Flash has 21 c4a layers + 20 c128a layers + 3 SWA/MTP layers; we tested one of each. Per-layer statistics can vary across layers; for a paper-grade claim we'd need to audit all 43 layers (scaling this script is cheap on H200 once shards are pre-fetched).

This companion file does **half the Caveat 1 expansion** — n=1 passage → n=8 semantically diverse passages on the **same three** representative V4 layers (0/SWA, 2/c4a, 3/c128a). The full per-layer audit (all 43 layers) remains a separate follow-up.

**Bottom line change vs n=1**: SWA and CSA wins are **statistically confirmed to ±1% tolerance**. The HCA headline from n=1 flips from a "marginal win" (0.966×) to **statistically neutral / slight loss** (1.043× ± 0.051). The paper-ready statement must be softened accordingly — see §Impact on headline claim.

## TL;DR — n=8 aggregates

| stream | rel-MSE (E8 Q=38) | rel-MSE (FP8) | **E8/FP8 ratio (95% CI)** | n=1 original | verdict |
| --- | --- | --- | --- | --- | --- |
| `sliding_window_kv` | $8.30\times10^{-4}\ ({\pm}3.2\!\times\!10^{-5})$ | $1.051\times10^{-3}\ ({\pm}3.7\!\times\!10^{-5})$ | **0.790 ± 0.005** | 0.786 | **confirmed strong win** (−21% MSE, −22% bits) |
| `csa_pool_kv_ratio4` | $9.60\times10^{-4}\ ({\pm}3.7\!\times\!10^{-5})$ | $1.066\times10^{-3}\ ({\pm}3.5\!\times\!10^{-5})$ | **0.900 ± 0.006** | 0.902 | **confirmed moderate win** (−10% MSE, −22% bits) |
| `hca_pool_kv_ratio128` | $1.375\times10^{-3}\ ({\pm}1.2\!\times\!10^{-4})$ | $1.317\times10^{-3}\ ({\pm}8.3\!\times\!10^{-5})$ | **1.043 ± 0.051** | 0.966 | **statistically neutral / slight loss** at matched Q; bits still −22% |

- Bits saved: unchanged at **−22% by codec arithmetic** (3296 bit/vec E8-Q38 vs 4224 bit/vec FP8 per-64-block). This is a construction property of the codec, not a measured value, and is identical across all n=8 passages.
- MSE: n=8 with CI tightens the SWA/CSA claims and corrects the HCA claim.
- HCA n=1 was an underestimate (outlier passage 0 at 0.966 was 1.6 σ below the n=8 mean — reproducibly; our unique random seed in 2026-04-25's original run put us on the lucky tail). With n=8 the true HCA E8/FP8 ratio is ~1.04 (slightly *worse* than hardware FP8 at matched Q=38).

## Per-passage detail — E8 Q=38 / FP8 ratio

| passage | topic | SWA | CSA | HCA |
| --- | --- | --- | --- | --- |
| 0 | algebraic topology | 0.786 | 0.902 | 0.966 |
| 1 | Italian Renaissance | 0.791 | 0.901 | 1.060 |
| 2 | molecular biology | 0.793 | 0.890 | 1.072 |
| 3 | macroeconomics | 0.800 | 0.909 | 1.011 |
| 4 | quantum mechanics | 0.787 | 0.890 | 1.123 |
| 5 | generative grammar | 0.788 | 0.911 | 0.952 |
| 6 | tonal harmony | 0.781 | 0.898 | 1.065 |
| 7 | reinforced concrete | 0.793 | 0.902 | 1.096 |
| **mean** | | **0.790** | **0.900** | **1.043** |
| **std** | | 0.006 | 0.008 | 0.061 |
| **95% CI hw** | | 0.005 | 0.006 | 0.051 |

**Observations**

1. `sliding_window_kv` is remarkably stable (std/mean = 0.7%). The E8 Q=38 win on SWA is a property of the V4 SWA projection's trained distribution, not of any particular passage.
2. `csa_pool_kv_ratio4` has std/mean = 0.9%. Same stability story — the c4a compressor's 512-dim output is passage-agnostic at the distribution level.
3. `hca_pool_kv_ratio128` has std/mean = 5.8% — 6–8× more variance than the other two streams. This is expected: the c128a compressor pools 128 tokens → 1 vector, giving only `seqlen/128 = 16` vectors per passage. Tail statistics on N=16 vectors are noisy; the per-passage ratio oscillates from 0.95 to 1.12 across topics. The **n=8 mean is the first statistically supported value**.

## Non-Gaussian audit — stability across n=8

| stream | metric | mean | 95% CI hw | paper gate |
| --- | --- | --- | --- | --- |
| SWA | \|kurt-3\| | 3.112 | ±0.352 | >0.5 ✓ (6.2σ above gate) |
| SWA | iso-var | 109.7 | ±9.6 | >1.5 ✓ |
| SWA | had-var | 11.61 | ±1.25 | >1.5 ✓ |
| SWA | W2/σ | 0.358 | ±0.018 | >0.05 ✓ |
| CSA | \|kurt-3\| | 2.822 | ±0.305 | >0.5 ✓ |
| CSA | iso-var | 732 400 | ±136 800 | >1.5 ✓ |
| CSA | had-var | 17.22 | ±2.61 | >1.5 ✓ |
| CSA | W2/σ | 0.459 | ±0.034 | >0.05 ✓ |
| HCA | \|kurt-3\| | 1.212 | ±0.135 | >0.5 ✓ |
| HCA | iso-var | 1.125e7 | ±6.43e6 | >1.5 ✓ |
| HCA | had-var | 434.2 | ±165.8 | >1.5 ✓ |
| HCA | W2/σ | 0.912 | ±0.124 | >0.05 ✓ |

**All four non-Gaussian gates fire on all three streams across all 8 passages.** The audit verdict "V4-Flash trained KV is far more non-Gaussian than Qwen3-4B post-QK-norm K" from `FINDINGS.md` is **confirmed with tight CI** for SWA and CSA, and **confirmed with looser CI** for HCA (pool-size limited).

Notes:
- The n=1 single-passage `iso-var` for CSA was 866 784; the n=8 mean is 732 400 ± 136 800. The n=1 value sits inside the CI — the n=1 number was an atypically high sample but still within the distribution.
- The n=1 HCA `iso-var` was 10 419 683; the n=8 mean is 11 250 000 ± 6 426 000. Also consistent.

## Layer-weighted deployment forecast — revised

V4-Flash layer mix: 3 SWA/MTP + 20 c4a + 20 c128a = 43 attention layers.

### MSE change (E8 Q=38 vs FP8, layer-weighted)

| aggregation | ratio | MSE change |
| --- | --- | --- |
| simple 3-stream mean (original FINDINGS.md) | (0.790 + 0.900 + 1.043) / 3 = **0.911** | −8.9% MSE |
| layer-weighted (3·0.790 + 20·0.900 + 20·1.043) / 43 | **0.959** | **−4.1% MSE** |

Previous `FINDINGS.md` reported a **−12% MSE** simple-mean estimate from n=1. The n=8 corrected estimate is **−9% (simple) / −4% (layer-weighted)**. The direction (E8 still wins on average) is preserved; the magnitude is roughly halved.

### Bit savings (unchanged)

- E8 Q=38 = 3296 bits/vector, FP8 per-64-block = 4224 bits/vector → **−22% bits**, identical in all 8 runs by codec construction.

### Revised end-to-end forecast

| metric | n=1 forecast | n=8 forecast |
| --- | --- | --- |
| Attention-KV bits saved | −22% | **−22%** (unchanged) |
| Attention-KV rel-MSE change, simple mean | −11.6% | **−8.9% ± 1.7%** |
| Attention-KV rel-MSE change, layer-weighted | −7% | **−4.1% ± 2.3%** |
| Deployment gain (per-user, 1M ctx) | ~18% saving | ~17–20% saving (bit budget is the dominant factor) |
| 4×H200 concurrent-user lift | 126 → 153 (+21%) | 126 → ~148–156 (+18–24%) |

The per-user / node-users numbers are nearly unchanged because they are driven by the bit saving, not the MSE change.

## Impact on the headline claim

`FINDINGS.md` Bottom line:

> **If the goal is a paper addendum with "KakeyaLattice on DeepSeek-V4"**: this Stage 0.75 data is sufficient. It's measured, reproducible, and shows a clean 22% bit saving with ~12% MSE improvement.

Revised for n=8:

> **If the goal is a paper addendum with "KakeyaLattice on DeepSeek-V4"**: use the n=8 numbers. **22% bit saving** (unchanged, by construction), **~4–9% layer-weighted MSE improvement** (halved and tightened with ±CI), and **stream-differentiated**: strong win on SWA (−21% MSE), moderate win on CSA (−10%), statistically neutral on HCA (+4% ± 5%).

**The "beats FP8 on all three streams" claim from n=1 does NOT hold for HCA once CI is computed on n=8.** The conservative paper statement is "beats FP8 on SWA and CSA streams with tight CI; neutral on HCA". The deployment claim (22% bit saving + non-regressive quality) survives.

## Reproducibility

Any NVIDIA H200 or equivalent with 12 GB local SSD:

```bash
export HF_HOME=/workspace/hf_home
export HF_TOKEN=...            # for DeepSeek-V4-Flash gated repo

# 1) Fetch V4-Flash shards 2/4/5 + tokenizer (~11 GB one-time)
python3 -c "
from huggingface_hub import hf_hub_download
import os
for f in ['config.json','tokenizer.json','tokenizer_config.json',
          'model.safetensors.index.json',
          'model-00002-of-00046.safetensors',
          'model-00004-of-00046.safetensors',
          'model-00005-of-00046.safetensors']:
    hf_hub_download('deepseek-ai/DeepSeek-V4-Flash', f,
                    cache_dir=os.environ['HF_HOME'])
"

# 2) Fetch host model (~1 GB)
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('Qwen/Qwen2-0.5B', cache_dir=os.environ['HF_HOME'])
"

# 3) Run the n=8 audit (this PR's new entry point)
python3 benchmarks/dsv4_stage075/run_stage075_n8.py \
    --host-model Qwen/Qwen2-0.5B \
    --seqlen 2048 --batch-size 1 \
    --n-passages 8 \
    --q-values 10,38 \
    --hf-home $HF_HOME \
    --out reports/v1_5_release/dsv4_stage075/stage075_n8.json
```

End-to-end wall time (H200 with warm cache): **~20 seconds** (V4 blocks load once, host model loads once, codecs build once; per-passage iteration is ~0.02–0.5 s — the first passage pays all warm-up cost).

Total cost: <\$0.05 of H200 time.

## Caveats still open (for future PRs)

1. **One layer per stream-type, not all 43** — we still test layers 0, 2, 3 only. Per-layer expansion requires loading shards 2..46 (~158 GB total) and is not yet done. This is the larger half of `FINDINGS.md` Caveat 1.
2. **One host model** (Qwen2-0.5B). The post-embedding hidden-state distribution flowing into V4's attention layers would differ if propagated through V4's own 43 layers (which would need MoE experts loaded). Our hidden-state → V4-attn projection is a fixed linear; n=8 holds the projection constant and varies the text.
3. **No Hyper-Connections** — V4's 4-copy residual rebalancing is bypassed.
4. **No end-to-end Δppl**. For that we need Stage 1 (full V4-Flash + vLLM, scaffold already merged in PR #47, execution still gated on Blackwell hardware per `reports/v1_5_release/dsv4_stage1/HARDWARE_REQUIREMENTS.md`).
5. **Passages are English-only WikiText-style prose**. A multilingual or code-mixed corpus may shift the distribution further; not expected to flip SWA/CSA wins given the ~0.5% std/mean ratio seen here.

## Relation to sibling reports

- `FINDINGS.md` — the original n=1 writeup. This file supersedes its numerical tables; the prose analysis (why gains are stream-dependent, shaping-gain bounds, FP8 behaviour) remains valid.
- `CPU_VS_GPU_COMPARISON.md` — hardware-independence study. Numbers there are n=1 CPU vs n=1 GPU; n=8 was not redone on CPU (the FP8 baseline is hardware-dependent per that report, so there's no scientific value in n=8 CPU).
- `stage075_trained.json` — the n=1 JSON (preserved unchanged).
- `stage075_n8.json` (new) — full per-passage + aggregate JSON from this run.
- `stage075_n8_run.log` (new) — console log captured from the H200 run for audit trail.

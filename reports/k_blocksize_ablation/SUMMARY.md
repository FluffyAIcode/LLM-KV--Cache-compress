# K-stream block_size ablation — results

Runs the full v1.2 monomorphic MSE codec at block_size ∈ {512, 1024, 2048}
on every full-attention K stream of 7 open-source models (real HF
forward passes, bf16, eager attention, ctx=4096, same prompt as the
d_eff/outlier ablation).

Verdict thresholds (same as the deff/outlier ablation):
- ≤ 10% MSE inflation → **ACCEPT**
- 10–30% → **MARGINAL**
- > 30% → **REJECT**

## Per-model aggregate (mean inflation / byte ratio across all full-attn layers)

| model | hd | layers | bs=512 (baseline) | bs=1024 mse / bytes | bs=2048 mse / bytes | bs=1024 verdict | bs=2048 verdict |
|---|---:|---:|---|---|---|---|---|
| `qwen2_5_0_5b` | 64 | 24 | mse=1.034e+00 | 1.20× / 0.94× | 1.37× / 0.88× | **MARGINAL** | **REJECT** |
| `qwen3_0_6b` | 128 | 28 | mse=1.135e+00 | 1.13× / 0.84× | 1.27× / 0.80× | **MARGINAL** | **MARGINAL** |
| `gemma4_e2b` | 256 | 3 | mse=1.154e-03 | 1.07× / 0.77× | 1.13× / 0.62× | **ACCEPT** | **MARGINAL** |
| `deepseek_r1_distill_qwen_1_5b` | 128 | 28 | mse=5.575e-01 | 1.16× / 0.80× | 1.34× / 0.69× | **MARGINAL** | **REJECT** |
| `glm_edge_1_5b` | 128 | 28 | mse=7.476e-01 | 1.13× / 0.83× | 1.24× / 0.72× | **MARGINAL** | **MARGINAL** |
| `smollm2_1_7b` | 64 | 24 | mse=6.468e-01 | 1.16× / 0.88× | 1.31× / 0.82× | **MARGINAL** | **REJECT** |
| `glm_edge_4b` | 128 | 40 | mse=6.801e-01 | 1.13× / 0.82× | 1.24× / 0.72× | **MARGINAL** | **MARGINAL** |

## Cross-model means

| block_size | mean MSE inflation | mean byte ratio | max inflation | verdict |
|---:|---:|---:|---:|---|
| 512 | 1.000× | 1.000× | 1.000× | **ACCEPT** |
| 1024 | 1.138× | 0.839× | 1.198× | **MARGINAL** |
| 2048 | 1.271× | 0.751× | 1.370× | **MARGINAL** |

## Per-model per-layer worst case (max inflation at bs=1024 and bs=2048)

| model | bs=1024 max layer inflation | bs=2048 max layer inflation |
|---|---:|---:|
| `qwen2_5_0_5b` | 1.38× | 1.61× |
| `qwen3_0_6b` | 1.18× | 1.38× |
| `gemma4_e2b` | 1.13× | 1.13× |
| `deepseek_r1_distill_qwen_1_5b` | 1.46× | 2.37× |
| `glm_edge_1_5b` | 1.19× | 1.36× |
| `smollm2_1_7b` | 1.20× | 1.39× |
| `glm_edge_4b` | 1.19× | 1.37× |

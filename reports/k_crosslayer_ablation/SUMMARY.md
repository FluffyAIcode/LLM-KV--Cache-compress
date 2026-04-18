# K-stream cross-layer basis-sharing ablation — results

Runs the full v1.2 monomorphic MSE codec on every full-attention
K stream of 7 open-source models (real HF forward passes, bf16,
eager attention, ctx=4096, same prompt as the d_eff/outlier ablation)
under three basis-sharing strategies:

- `per_block` — one PCA per block (v1.2 K default)
- `per_layer_pooled` — one PCA per layer (share_basis=true)
- `per_type_pooled` — one PCA for ALL full-attn layers of the model

Thresholds (consistent with the d_eff/outlier ablation):
- ≤ 10% MSE inflation → **ACCEPT**
- 10–30% → **MARGINAL**
- > 30% → **REJECT**

## Per-model results

| model | hd | layers | per_layer_pooled (MSE × / byte ×) | per_type_pooled (MSE × / byte ×) | pooled d_eff | per_layer verdict | per_type verdict |
|---|---:|---:|---|---|---:|---|---|
| `qwen2_5_0_5b` | 64 | 24 | 2.23× / 0.81× | 4.65× / 0.41× | 12 | **REJECT** | **REJECT** |
| `qwen3_0_6b` | 128 | 28 | 1.95× / 0.67× | 2.23× / 0.62× | 46 | **REJECT** | **REJECT** |
| `gemma4_e2b` | 256 | 3 | 1.09× / 0.56× | 1.11× / 0.50× | 244 | **ACCEPT** | **MARGINAL** |
| `deepseek_r1_distill_qwen_1_5b` | 128 | 28 | 8.68× / 0.66× | 15.18× / 0.13× | 3 | **REJECT** | **REJECT** |
| `glm_edge_1_5b` | 128 | 28 | 1.06× / 0.67× | 1.06× / 0.66× | 109 | **ACCEPT** | **ACCEPT** |
| `smollm2_1_7b` | 64 | 24 | 1.20× / 0.82× | 1.17× / 0.83× | 56 | **MARGINAL** | **MARGINAL** |
| `glm_edge_4b` | 128 | 40 | 1.07× / 0.66× | 1.06× / 0.65× | 108 | **ACCEPT** | **ACCEPT** |

## Cross-model aggregates

| strategy | mean MSE inflation | median | max | mean byte ratio | verdict (mean) |
|---|---:|---:|---:|---:|---|
| `per_layer_pooled` | 2.47× | 1.20× | 8.68× | 0.69× | **REJECT** |
| `per_type_pooled` | 3.78× | 1.17× | 15.18× | 0.54× | **REJECT** |

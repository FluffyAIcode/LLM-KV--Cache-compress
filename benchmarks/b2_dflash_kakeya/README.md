# B2 — DFlash × KakeyaLattice acceptance-rate benchmark

**目标**：量化 KakeyaLattice E8 KV-cache 压缩对 DFlash block-diffusion
speculative decoding 的影响 — 具体是 acceptance length 的变化（是否因
target 分布扰动而掉速）以及端到端 tok/s 的实际收益。

B2 (`integrations/atomic-chat-b2/`) 的所有工程都是围绕"能不能叠加"的
理论推演 + 骨架 + 单测。M5 负责把推演变成数字。

## 实验设计

### Target × Draft × KV channel

| Target | Draft (DFlash) | KV channel |
|:-|:-|:-|
| `Qwen/Qwen3-8B` (non-thinking) | `z-lab/Qwen3-8B-DFlash-b16` | bf16 baseline |
| 同上 | 同上 | Kakeya E8 Q=38 (near-lossless) |
| 同上 | 同上 | Kakeya E8 Q=10 (balanced) |
| 同上 | 同上 | Kakeya E8 Q=4 (aggressive) |

共 **4 组 (1 target × 1 draft × 4 KV channel)**。bf16 是对照组，不走
KakeyaLatticeMLXCache，直接用 mlx-lm 原生 KVCache。

### 数据集

- **`gsm8k`** (GSM8K test split)：数学推理；DFlash 论文主 benchmark。
- **`humaneval`** (HumanEval openai)：代码生成；DFlash 论文次 benchmark。

对每个数据集，随机抽 `n_samples=32` prompt；若要快速烟测可调低到 8。
种子固定 (`seed=42`) 以便两次跑的 prompt 集一致。

### 指标

| 指标 | 含义 | 来源 |
|:-|:-|:-|
| `acceptance_length_mean` | 平均每 verify step 接受 token 数；DFlash 的核心指标 | `dflash.model_mlx.stream_generate` 每步返回 |
| `acceptance_length_p50 / p95` | 分位数 | 同上 |
| `generation_tps` | 端到端 tok/s | dflash 或 mlx_lm 的 timer |
| `total_tokens` | 生成 token 总数 | tokenizer 统计 |
| `first_token_latency_s` | 首 token 延迟 | wall clock |
| `kakeya_codec_fired` | codec 触发次数（非 boundary 层） | `KakeyaLatticeMLXCache.fire_count` |
| `correctness_proxy` (可选) | answer 是否含 expected | 仅对 gsm8k / humaneval 有简单匹配 |

### 预期结果（按 PR #57 §12.2 的理论分析）

| channel | acceptance_length | tps 相对 baseline | 结论 |
|:-|:-:|:-:|:-|
| bf16 baseline | ~14-16 (DFlash 官方报数) | 1.00× | ✓ |
| Kakeya Q=38 | ~13-15 (降 <1pp) | ~0.95-1.00× | 近无损可用 |
| Kakeya Q=10 | ~11-13 (降 1-3pp) | ~0.80-0.90× | 用 KV 节省换速度 |
| Kakeya Q=4 | ~7-10 (显著下降) | ~0.50-0.70× | 不进默认档位 |

**如果 Q=38 acceptance 掉得超过 2pp**，该档位不作为 B2 默认；回退到
Q=76 或 Q=152。**如果 Q=10 acceptance 掉得超过 5pp**，B2 的"加速 + 压缩
双赢"叙事需要修正。

## 运行

### 真实运行（需要 Apple Silicon + MLX + dflash）

```bash
# 1. 装依赖
pip install -e integrations/atomic-chat-b2/kakeyalattice_mlx[mlx]
pip install dflash                # z-lab/dflash 官方包 (MLX backend)
pip install "mlx-lm>=0.20"

# 2. 跑基线 (bf16) + 三个 Kakeya 档
python -m benchmarks.b2_dflash_kakeya.runner \
    --target Qwen/Qwen3-8B \
    --draft  z-lab/Qwen3-8B-DFlash-b16 \
    --datasets gsm8k humaneval \
    --n-samples 32 \
    --channels bf16 e8-q38 e8-q10 e8-q4 \
    --out-dir reports/b2_release

# 3. 结果在 reports/b2_release/b2_dflash_kakeya_{dataset}_{channel}.json
```

### Dry-run（Linux CI 可跑；不下模型、不加载 dflash）

```bash
python -m benchmarks.b2_dflash_kakeya.runner --dry-run
```

Dry-run 会走完参数解析 + dataset 加载 + 指标聚合路径；推理步骤由
`--mock-engine` 自动注入的 FakeEngine 替身提供，用来 CI 验证 runner
工程完整度。

## 文件

```
benchmarks/b2_dflash_kakeya/
├── README.md         (本文件)
├── __init__.py
├── runner.py         主入口 + 参数解析 + 顶层流程
├── datasets.py       gsm8k / humaneval 加载器 (本地 jsonl + 可选 HF datasets)
├── engines.py        RealEngine (DFlash+Kakeya) + MockEngine (CI)
├── metrics.py        acceptance_length 分布, tps, Δppl 估算
├── schema.py         输出 JSON 的 TypedDict + 版本号
└── tests/
    ├── __init__.py
    ├── test_metrics.py        (Linux CI green)
    ├── test_datasets.py       (Linux CI green)
    └── test_runner_mock.py    (Linux CI green, 用 MockEngine)
```

## 输出 schema

```json
{
  "schema_version": "b2-dflash-kakeya-v1",
  "target_model": "Qwen/Qwen3-8B",
  "draft_model":  "z-lab/Qwen3-8B-DFlash-b16",
  "dataset":      "gsm8k",
  "channel":      "e8-q10",
  "n_samples":    32,
  "samples": [ { prompt..., metrics... }, ... ],
  "aggregate": {
    "acceptance_length_mean": 12.3,
    "acceptance_length_p50":  12,
    "acceptance_length_p95":  18,
    "generation_tps_mean":    210.5,
    "first_token_latency_s":  0.142,
    "total_tokens_sum":       8192,
    "codec_fired_mean":       35.2
  },
  "hardware": { "device": "mlx:metal", "chip": "Apple M3 Pro", ... },
  "software": { "mlx": "...", "dflash": "...", "kakeyalattice_mlx": "..." }
}
```

## 对比 atomic.chat 首页宣传

atomic.chat 首页声称 *"Google TurboQuant built-in"* 与 *"Compressed down
to just 3 bits"*。v1.5 报告里 TQ b=2 在 4 模型上结构性不可用、b=3 被
E8 Q=4 全面压过 3-6×。M5 的 b2 报告会补上：**同样 (CR, |Δppl|) 前提下,
DFlash 加速损失是多少** — 也就是宣传里未兑现的"速度 + 压缩双赢"那一栏
的真实数字。

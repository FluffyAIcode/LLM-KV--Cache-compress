# Atomic-Chat × KakeyaLattice v1.5 — B2 方案 (MLX + DFlash + KakeyaLattice-MLX)

此目录交付 **B2 方案**：Mac 上 MLX 原生推理栈 × DFlash block-diffusion
speculative decoding × KakeyaLattice E8 KV-cache 压缩，作为对
[`integrations/atomic-chat/` (B1 PR #57)](../atomic-chat/) 的性能升级。

## B1 vs B2

| | B1 (PR #57) | B2 (本 PR) |
|:-|:-|:-|
| 推理引擎 | HF transformers + torch MPS | **MLX** (Apple 官方, Apple Silicon 原生) |
| KV 压缩 | `KakeyaLatticeCache` (PyTorch) | `KakeyaLatticeMLXCache` (MLX port) |
| 推测解码 | 无 | **DFlash 3-6×** (z-lab/Qwen3-8B-DFlash-b16 等) |
| decode 速度 (Qwen3-8B @ M3 Pro) | ~50 tok/s | **~200-280 tok/s effective** |
| 长上下文 (Mac 16GB) | ~32k | **~48-64k** (KV 3.37× 压缩) |
| 平台 | Mac + Win + Linux (MPS/CUDA/CPU) | **Mac 专属** |
| 工程成本 | 低 (PyTorch 生态成熟) | 中 (需 MLX port + DFlash 集成) |

完整对比见 [`docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md`](../../docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md) §14。

## 目录

```
integrations/atomic-chat-b2/
├── README.md                         (本文件)
├── kakeyalattice_mlx/                ★ E8 codec 的 MLX 实现
│   ├── pyproject.toml
│   ├── kakeyalattice_mlx/
│   │   ├── __init__.py
│   │   ├── hadamard.py               Sylvester Hadamard matrix (MLX)
│   │   ├── closest_point.py          D8 + E8 closest-lattice-point (MLX)
│   │   ├── codec.py                  E8LatticeCodebookMLX (与 PyTorch parity)
│   │   └── kv_cache.py               KakeyaLatticeMLXCache (mlx-lm 风格)
│   └── tests/
│       ├── test_hadamard.py
│       ├── test_closest_point.py
│       └── test_codec_parity.py      ★ bit-level parity vs PyTorch reference
├── kakeya_sidecar_mlx/               ★ B2 OpenAI 兼容 sidecar (骨架)
│   ├── pyproject.toml
│   ├── kakeya_sidecar_mlx/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── engine_mlx.py             MLX 推理引擎
│   │   ├── engine_dflash.py          DFlash drafter wrapper (可选)
│   │   ├── model_registry_mlx.py     MLX 版部署档位 + DFlash draft 映射
│   │   └── server.py                 复用 B1 路由格式
│   └── tests/
│       └── test_model_registry_mlx.py
└── ROADMAP.md                        M1-M6 里程碑清单
```

## M1-M6 里程碑

本 PR 交付 **M1 + M2 的骨架 + 单测 + parity harness**（不需真硬件）；
M3-M6 作为 follow-up PR 推进。

| M | 内容 | 本 PR 状态 |
|:-:|:--|:-:|
| M1 | `kakeyalattice_mlx/` — E8 codec MLX 版 + PyTorch parity 测试 | ✅ |
| M2 | `KakeyaLatticeMLXCache` — mlx-lm KV cache wrapper | ✅ 骨架 |
| M3 | `kakeya_sidecar_mlx/` — OpenAI 兼容 MLX sidecar | ✅ 骨架 |
| M4 | 接 DFlash: `dflash.model_mlx.stream_generate` + target KV 压缩 | ⏳ 下一 PR |
| M5 | acceptance-rate benchmark (Qwen3-8B × DFlash × KakeyaLattice) | ⏳ 下一 PR |
| M6 | Atomic-Chat extension 追加 backend 选项 | ⏳ 下一 PR |

## 开发环境要求

MLX 只在 Apple Silicon Mac 上能跑；Linux CI 只能跑**逻辑单测**（不依赖 MLX），
`test_codec_parity.py` 需 Mac 本机运行。本 PR 的所有测试都做了分层：

- **Platform-agnostic**（registry / parser / Hadamard 结构验证）— Linux CI 可跑
- **MLX-gated**（用 `@pytest.mark.skipif(not mx.metal.is_available())`）— Mac 本机跑

## License

Apache-2.0，与主仓库一致。

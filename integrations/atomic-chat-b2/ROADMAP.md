# B2 Roadmap — MLX + DFlash + KakeyaLattice-MLX

**Branch**: `AgentMemory/atomic-chat-b2-mlx-dflash-kakeya-04ae`
**Parent PR**: #57 (B1 — HF + MPS sidecar, in review)
**Status**: M1-M4 已落 (PRs #58, M4); M5-M6 后续

## 动机

B1 用 HF transformers + `KakeyaLatticeCache` + MPS 在 Mac 上跑通 KV 压缩，
但 MPS 的 codec overhead 占 decode step 的 20-30%，单轨 decode 速度
~50 tok/s 也远不及 Apple Silicon 原生 MLX 的潜力。DFlash (z-lab,
arXiv:2602.06036, 2026-02) 的 block-diffusion speculative decoding 已在
MLX 上原生跑通，在 Qwen3 系列上报 3-6× 无损加速。两者叠加理论上能把
Mac 本地推理从 "~50 tok/s" 推到 "~200+ tok/s effective"，同时保留
KakeyaLattice 的 KV 3.37× 压缩能力。

## 里程碑

### M1 — `kakeyalattice_mlx/` (本 PR 交付)

MLX 版 E8 codec，与 PyTorch 参考实现 bit-level parity。

**文件**:
- `hadamard.py` — Sylvester-Hadamard matrix 生成 (MLX)
- `closest_point.py` — D8 + E8 Conway-Sloane closest-point (MLX)
- `codec.py` — `E8LatticeCodebookMLX` with `.roundtrip(x)` 接口
- `tests/test_codec_parity.py` — **在 Mac 本机上** 验证
  max_abs_diff vs `kakeyalattice.V15KakeyaZamirE8GPU` 为 0

**验收**:
- `pytest tests/ -v` 在 Mac M 系 Apple Silicon 上全绿
- Linux CI 跑 platform-agnostic 子集 (Hadamard 结构 / bit 计数 / 输入校验) 全绿

### M2 — `KakeyaLatticeMLXCache` (本 PR 骨架)

包装 `mlx_lm.models.cache.KVCache`，在每次 `update_and_fetch(keys, values)`
调用中对新写入的 K/V 做 codec roundtrip。

**要点**:
- mlx-lm 的 KVCache 是 per-layer 实例，与 HF DynamicCache 的单实例多层存储模型不同
- 每层一个 `E8LatticeCodebookMLX` (与 `KakeyaLatticeCache` 对齐)
- `boundary` 层跳过 codec (DeepSeek-R1-Distill 小模型强制 `boundary=2`)
- 支持 `mlx_lm.models.cache.make_prompt_cache(model)` 工厂风格

### M3 — `kakeya_sidecar_mlx/` (本 PR 骨架)

OpenAI 兼容 MLX sidecar，接口与 B1 完全一致 (`/v1/models`,
`/v1/chat/completions`, `/v1/kakeya/stats`)，但默认端口 `1339`
(sit between B1 的 1338 和 Atomic-Chat 前门 1337)。

**本 PR 只给骨架 + 纯逻辑单测**: model_registry_mlx, channel parsing,
routing mock。真正的 MLX 模型加载 + generate 需要 M4 接入 DFlash。

### M4 — DFlash 集成 (✅ 本 PR)

接 `dflash.model_mlx.stream_generate`，把 target LLM 的 KV 替换为
`KakeyaLatticeMLXCache`，draft LLM 保留默认 `RotatingKVCache`（Phase 2
再压缩 draft KV）。

**本 PR 交付**:
- `cache_injection.py` — 三种注入策略（kwarg / model.make_cache /
  module-level make_prompt_cache）+ 特性检测 + `FALLBACK_NATIVE_MLX`
  兜底，适配 dflash API 在 2026 年多次变动的实际情况
- `engine_mlx.py` — `chat()` / `chat_stream()` 打通，两条路径：
  DFlash + Kakeya KV，以及 native MLX + Kakeya KV 兜底
- `server.py` — `/v1/chat/completions` 正式打开，stream + non-stream
  两模式；`x_kakeya` 响应字段带 `dflash_used` /
  `injection_strategy` / `acceptance_length_mean`
- **32 sidecar 单测 全绿**（含 8 条 cache_injection 策略测试 + 4 条
  engine routing 测试，均用 stub 替身模拟 MLX / dflash）

**阻碍**:
- `dflash.model_mlx` 的 target / draft KV 接口需要 dflash patch 或我方 wrapper
- `draft_sliding_window_size` 与 target `boundary` 的联动需重测
- acceptance-rate 可能随 target |Δppl| 下降，需 M5 实测

### M5 — Acceptance-rate benchmark (follow-up PR)

实验设计:
- Target: `Qwen/Qwen3-8B` (non-thinking)
- Draft: `z-lab/Qwen3-8B-DFlash-b16`
- KV 通道: `bf16 baseline`, `KakeyaLattice e8 Q=38`, `Q=10`, `Q=4`
- 数据集: gsm8k + math500 + humanoeval (DFlash 默认 benchmark set)
- 指标: acceptance length 分布, tokens/s, Δppl on WikiText-103

**预期**:
- Q=38: acceptance 掉 <1pp, effective throughput 保持 3× (DFlash 本身)
- Q=10: acceptance 掉 1-3pp, effective throughput ~2.5×，用 KV 长 ctx 换速度
- Q=4: 不进入默认档位，仅保留做上限压测

### M6 — Atomic-Chat extension backend 选项

在 B1 的 `kakeyalattice-extension` 基础上:
- 扩展 `KakeyaBackend` 增加 `"variant": "mlx-dflash"` 模式
- UI 层多一个 backend 选择器: "KakeyaLattice (MPS)" / "KakeyaLattice (MLX+DFlash) ★ Pro"
- Tauri plugin supervisor 同时托管两个 sidecar (`:1338` B1, `:1339` B2)，按
  backend 选择路由

## 非目标 (本 PR 不做)

- 不改任何 vLLM 路径 (C / C2 方案走完全独立 PR)
- 不 port v1.4 D4 codec (B2 只需要 E8; D4 留在 B1)
- 不做 Metal Performance Shaders 级别的 fused E8 kernel (MLX 内置算子已够用;
  融合 kernel 是 B2 merge 后的优化项)
- 不处理 Windows / Linux 用户 (B2 明确 Mac-only; 非 Mac 用户走 B1 或 llama.cpp)

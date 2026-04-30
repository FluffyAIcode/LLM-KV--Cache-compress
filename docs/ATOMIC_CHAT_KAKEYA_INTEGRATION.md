# Atomic-Chat × KakeyaLattice v1.5 — 本地 Mac 部署集成架构

**Date**: 2026-04-30
**Scope**: 把 KakeyaLattice v1.5 (E8 nested-lattice KV-cache codec) 作为核心
压缩层嵌进 [`AtomicBot-ai/Atomic-Chat`](https://github.com/AtomicBot-ai/Atomic-Chat)
的本地推理栈，支撑 Qwen3 / Llama-3.x / Gemma-3/4 / DeepSeek-R1-Distill /
GLM-4 / Mistral 等多个开源模型在 **Mac (Apple Silicon, Metal)** 上的
离线部署。

本文档是"分析 + 设计"一体的规划稿。代码骨架落在
[`integrations/atomic-chat/`](../integrations/atomic-chat/)。

---

## 1. Atomic-Chat 现状架构（先分析，再集成）

从官方 README / CONTRIBUTING 与配套项目 [atomic.chat](https://atomic.chat/)
可以还原出下面这张图（简化，聚焦"推理后端接入点"）。

```
┌────────────────────────────────────────────────────────────┐
│  Web App (React + Redux + React Router + Vite + TS)        │
│    - 对话 UI、模型管理 UI、Assistant 编排                  │
└────────────┬───────────────────────────┬───────────────────┘
             │ imports                    │ imports
             ▼                            ▼
   ┌──────────────────────┐     ┌──────────────────────────┐
   │ Core SDK (core/)     │     │ Extensions (extensions/) │
   │ - TypeScript APIs    │◄────│ - assistant-extension    │
   │ - Extension System   │ uses│ - conversation-extension │
   │ - Event Bus          │     │ - download-extension     │
   │ - Type Definitions   │     │ - llamacpp-extension  ◄──┼── 本地推理
   └──────────┬───────────┘     └───────────┬──────────────┘
              │                              │
              │        Tauri IPC (invoke)    │
              └──────────────┬───────────────┘
                             ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Tauri Backend (Rust, src-tauri/)                         │
   │  - Window / IPC / 安全 / 文件系统                          │
   └──────────────────────────────────────────────────────────┘
                             ▲
                             │ 提供能力
   ┌──────────────────────────────────────────────────────────┐
   │  Tauri Plugins (src-tauri/plugins/)                       │
   │  - hardware  (CPU/GPU/RAM 探测)                            │
   │  - llamacpp  (llama.cpp 进程管理、Metal/CUDA 后端、推理)   │
   └──────────────────────────────────────────────────────────┘
                             │
                             ▼
          OpenAI 兼容本地 HTTP 服务 @ localhost:1337
                    (/v1/models, /v1/chat/completions …)
```

关键事实:

| 维度 | 现状 |
|:-----|:-----|
| Shell | **Tauri**（不是 Electron）|
| 推理后端 | 只有一个：`llamacpp-extension` + `plugins/llamacpp` 驱动 **llama.cpp** 进程 |
| 模型格式 | GGUF（主力），宣传支持 MLX / ONNX（由 llama.cpp 生态/路线图覆盖）|
| 模型来源 | 直连 HuggingFace 下载 |
| 硬件加速 | Mac 走 Metal（`xcodebuild -downloadComponent MetalToolchain`），Windows x64 走 CPU / CUDA |
| 对外 API | OpenAI 兼容的 `localhost:1337`，其他程序可直接打 |
| Agent 能力 | MCP (Model Context Protocol) 集成，跑 agentic workflow |
| 营销口径 | 站点声称 *"Google TurboQuant built-in"* — 指的是 KV 压缩 |

模型下载到推理的主流程(文件系统级):

```
User clicks download
  → web-app 发请求
  → download-extension 决定源
  → Tauri backend 落盘
  → llamacpp-extension 注册模型
  → plugins/llamacpp 起 llama.cpp 子进程
  → OpenAI 兼容 /v1 路由暴露
```

---

## 2. 硬冲突：为什么不能把 KakeyaLattice 塞进 llama.cpp

要"把 KakeyaLattice 作为核心架构"集成，先认清一个工程事实。

**KakeyaLattice v1.5 是 PyTorch-first 的 GPU 原生实现。** 其核心算子
(`V15KakeyaZamirE8GPU.roundtrip`) 依赖:

- Sylvester–Hadamard rotation（任意长度 2^k，PyTorch `matmul`）
- E8 Conway–Sloane closest-point（两 coset 评估 + `argmin`）
- Per-vector adaptive `q_max`（`amax` + `clamp`）

最干净的宿主是 HuggingFace `transformers`:`kakeyalattice.hf.KakeyaLatticeCache`
已经是 `DynamicCache` 的一级子类，`model.generate(past_key_values=cache)` 一行接入。

反之 **llama.cpp 没有可插拔的"KV-cache 量化策略"的对外接口**:
1. `kv_cache` 是 C++ 内部结构，量化类型 (`q4_0`, `q8_0`, `f16`) 写死在 `ggml_type`。
2. 没有 Python hook，改造需要实现一套 E8 的 C++/Metal 内核并 patch `llama_kv_cache_unified`。
3. 即便改造出来，也无法复用 KakeyaLattice 现有的 bit-parity 测试、ablation
   benchmark、`rigorous_eval.py` 全套回归。

所以"核心架构"集成的工程现实是 — **把 KakeyaLattice 做成与 llama.cpp
平级的第二个本地推理后端**，由 Atomic-Chat 的 Extension System 负责路由。

---

## 3. 集成目标架构

在 Atomic-Chat 既有架构上加"右臂"，与 llama.cpp 的"左臂"并列:

```
┌────────────────────────────────────────────────────────────┐
│  Web App (unchanged)                                        │
└────────────────────┬──────────────────────────────────────┘
                     ▼
          Core SDK (unchanged) + Extensions
            ├─ llamacpp-extension          (既有, GGUF)
            └─ kakeyalattice-extension     ★ 新增
                      │
                      │ Tauri IPC
                      ▼
          Tauri Plugins
            ├─ plugins/llamacpp            (既有)
            └─ plugins/kakeyalattice       ★ 新增
                      │
                      │ 管理 Python sidecar 进程生命周期
                      ▼
          Kakeya Sidecar  (localhost:1338, OpenAI 兼容)
            ├─ HuggingFace transformers + torch MPS
            ├─ KakeyaLatticeCache (E8, per-model Q)
            └─ /v1/models, /v1/chat/completions (stream)
                      │
                      ▼
          本地模型仓库 (HF safetensors)
            Qwen3-4B / Llama-3.2-3B / Gemma-4-E4B /
            DeepSeek-R1-Distill-1.5B / GLM-4-9B-Chat / Mistral-7B …
```

在用户视角:

- 既有的 Atomic-Chat UI 不需要重画。模型选择 UI 多一个 "Backend" 过滤器
  (`llama.cpp` / `KakeyaLattice (E8)`)。
- 原 `localhost:1337` 继续由 Atomic-Chat 前门承担，内部按选择的模型
  将 OpenAI 请求路由到 llama.cpp 或 Kakeya sidecar。
- **用户只感觉到一件事**：选了 `KakeyaLattice` 后端后，长上下文占用的
  内存少一截（E8 Q=10 ~3.37×，Q=4 ~4.57×），质量损失可控（|Δppl|<7% 在
  Qwen3/Gemma/GLM），能塞进 Mac 16/32GB 的上限里。

---

## 4. 关键设计决策

### 4.1 为什么 Python sidecar，而不是 Rust FFI

方案对比:

| 方案 | 优点 | 致命缺点 |
|:-----|:-----|:---------|
| Rust 直接调 libtorch / tch-rs | 无进程边界 | torch MPS ABI 不稳定；transformers 的 cache / generate 逻辑在 Python，不能照搬；维护成本爆炸 |
| **Python sidecar**（本方案）| 直接复用 `kakeyalattice.hf.KakeyaLatticeCache`；transformers 自带所有模型支持；Metal 通过 torch MPS 零改动 | 多一个进程（由 Tauri plugin 托管，用户无感）|
| 改 llama.cpp | 与 Atomic-Chat 现架构对齐 | 前述 §2，实现成本不可接受 |

选 sidecar。用 `stdio` / `localhost:1338` 与 Tauri 通信，plugin 负责起/停/健康检查/日志转发。这套模式 llama.cpp 自己就是这么跑的（llama.cpp 也是独立进程）。

### 4.2 Metal (MPS) 而不是 CUDA

KakeyaLattice v1.5 的 `V15KakeyaZamirE8GPU` 构造器签名:

```python
V15KakeyaZamirE8GPU(D=head_dim, q_range=Q, device="cuda")
```

Mac 上只要把 `device="cuda"` 换成 `device="mps"` 就能跑 — 因为所有算子都是
标准 PyTorch（`matmul`、`argmin`、`amax`、`clamp`、`round`），MPS backend 全部支持。
这是整个集成最幸运的点:**不需要写任何 Metal shader**。

> 验证方式：`integrations/atomic-chat/kakeya_sidecar/tests/test_mps_smoke.py`
> 会在 Apple Silicon 上跑 E8 roundtrip，对齐 CUDA 参考输出（相对误差 < 1e-5）。

性能预期（vs v1.5 on H200 @ 551µs/2048-vec slice）:
- M2 Pro / M3 Pro (16GB)：codec 开销约 4-8 ms/slice，decode step 整体 15-30ms
  的 20-30%。不是热点，但不可忽略。
- 优化方向（未来 PR）：Triton 不行，直接写 Metal Performance Shaders (MPS)
  的 fused E8 closest-point。当前 sidecar 预留接口 `codec.set_backend("metal")`。

### 4.3 per-model Q 档位（来自 v1.5 报告）

直接复用 `reports/v1_5_release/V15_FULL_4MODEL_REPORT.md` 的结论，每个模型
落成一张 "deployment profile" JSON，见 `kakeya_sidecar/model_registry.py`:

| Model (HF id) | head_dim | variant | 推荐 Q | CR | |Δppl| (v1.5 实测) | 备注 |
|:--|:-:|:-:|:-:|:-:|:-:|:--|
| Qwen/Qwen3-4B | 128 | e8 | 10 | 3.37× | 3.85% | 平衡点 |
| Qwen/Qwen3-4B | 128 | e8 | 38 | ~2.5× | <1% (paper) | 近无损 |
| meta-llama/Llama-3.2-3B-Instruct | 128 | e8 | 10 | 3.37× | 待测 (同类) | 报告未直测 |
| google/gemma-4-E4B | 256/512 | e8 | 10 | 3.47× | 1.56% | 异构 head_dim 已验证 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 128 | e8 | 10 | 3.37× | 需 boundary≥2 | 小模型结构敏感 |
| zai-org/GLM-4-9B-Chat | 128 | e8 | 10 | 3.37× | 6.96% | L=40，可用 |
| mistralai/Mistral-7B-Instruct-v0.3 | 128 | e8 | 10 | 3.37× | 待测 (同类) | head_dim 兼容 |

准则:
- **默认 Q=38**：近无损，Mac 首选（用户感知最稳）。
- **长上下文 / 内存吃紧时 Q=10**：3.37× 压缩，|Δppl|<7% 在主流 7B~9B。
- **Q=4 不作为默认**：报告里 GLM Q=4 |Δppl|=32%，只在 UI 高级选项里暴露。
- **DeepSeek-R1-Distill 家族强制 `boundary=2`**：避开报告里记录的
  no-boundary 灾难模式。

### 4.4 长上下文是真正的卖点

Mac 本地部署里"KV cache 撑爆内存"是高频失败。举例 Llama-3.2-3B 在 32k
ctx 的 KV：`2 × L=28 × head_dim=128 × num_kv_heads=8 × 32768 × 2(bf16) =
3.7 GB`。把它压 3.37× 后变 1.1 GB，Mac mini M2 16GB 就够跑。

这是"接 KakeyaLattice"带来的**用户可感知价值**，不是单纯的 PR 演示。

---

## 5. 多模型支持矩阵

所有通过 `head_dim % 8 == 0` 且 `head_dim` 是 2 的幂 的模型都即插即用。
`KakeyaLatticeCache` 的 `strict=True` 会在不兼容的模型（如 GPT-NeoX
head_dim=96）启动时报错并 fallback 到 `llama.cpp` 后端。

| Family | head_dim | E8 OK | 备注 |
|:--|:-:|:-:|:--|
| Llama-3.x (1B/3B/8B) | 128 | ✅ | Mac 主力 |
| Qwen2 / Qwen3 (1.5B/4B/7B) | 128 | ✅ | v1.5 报告首发 |
| Mistral / Mixtral | 128 | ✅ | |
| Gemma-3 / Gemma-4 | 256 (+512 MatFormer) | ✅ | 异构 head_dim 在 KakeyaLatticeCache 已覆盖 |
| DeepSeek-R1-Distill 系列 (1.5B/7B/14B) | 128 | ✅ | 强制 boundary=2 |
| DeepSeek-V2/V3 (MLA) | 128 + 64 rope | ⚠️ | 需要 MLA 专用路径，下一步 |
| GLM-4-9B-Chat | 128 | ✅ | |
| Phi-3 (4k/128k) | 96 | ✗ | 跳过 → llama.cpp |
| GPT-NeoX 旧模型 | 96 | ✗ | 跳过 → llama.cpp |

---

## 6. API 设计 — OpenAI 兼容

Sidecar 暴露最小 OpenAI 子集，Atomic-Chat 的 `localhost:1337` 前门只做 URL 改写。

### `GET /v1/models`
返回当前已加载 + 可加载的模型；每条多两个 KakeyaLattice 字段:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-4b@e8-q10",
      "object": "model",
      "owned_by": "kakeyalattice",
      "x_kakeya": {
        "variant": "e8",
        "q_range": 10,
        "boundary": 0,
        "est_compression": 3.37,
        "est_delta_ppl_pct": 3.85
      }
    }
  ]
}
```

### `POST /v1/chat/completions`
完全兼容 OpenAI 请求格式。sidecar 接到后:
1. 按 `model` 字段解析 `<hf-id>@<variant>-q<Q>`；
2. lazy-load 模型 + `KakeyaLatticeCache` (LRU，同时间只驻留 1~2 个)；
3. `model.generate(..., past_key_values=cache, streamer=...)` 跑出
   token 流；
4. 转成 OpenAI `chat.completion.chunk` SSE 流。

### `GET /v1/kakeya/stats`
扩展接口，给 UI 用，返回:
- 当前 cache 的 `codec_fired_per_layer` / `skip_fired_per_layer`
- 当前 KV HBM/URAM footprint 估算
- 上一次 decode step 的 codec 占用 %

---

## 7. 安装 & 运行（开发者视角）

本仓库提供的骨架可以**直接拉到 Atomic-Chat 仓库 `extensions/` 下使用**。

```bash
# 1. 装 Python sidecar
cd integrations/atomic-chat/kakeya_sidecar
pip install -e ".[mac]"
kakeya-sidecar --port 1338 --device mps --prewarm qwen3-4b@e8-q10

# 2. 拷扩展到 Atomic-Chat 仓库
cp -R integrations/atomic-chat/kakeyalattice-extension \
      ~/Atomic-Chat/extensions/
# 在 web-app 的 extension registry 里注册一行，重新 `make dev` 即可

# 3. 拷 Tauri plugin
cp -R integrations/atomic-chat/tauri-plugin-kakeyalattice \
      ~/Atomic-Chat/src-tauri/plugins/kakeyalattice
# 在 src-tauri/tauri.conf.json + Cargo.toml 加 plugin 注册
```

打包到 Atomic-Chat 安装包时，sidecar 需要用 `PyOxidizer` 或 `briefcase`
打成独立可执行，与 `llama.cpp` 二进制并列塞进 `.dmg`。本仓库不做
端到端打包（Atomic-Chat 仓库自己的 CI 做），只保证 sidecar 本身能跑。

---

## 8. 与项目原有 vLLM 路径的关系

本仓库 `vllm_backend/kakeya_v1_4_snapshot/` 是 **Linux + CUDA + vLLM** 的
snapshot-hook 插件路径，用于在 H200 上跑 `benchmarks/rigorous_eval.py`。

Atomic-Chat 集成 **不复用这条路径**，原因：
- vLLM 在 Mac 上没有 Metal 后端，跑不起来。
- vLLM 的 PagedAttention + KV cache 是 CUDA 内核，snapshot hook 依赖 CUDA tensor。

二者关系:
- `vllm_backend/` → 研究/benchmark 用，CI 数据来源，**保持不动**。
- `integrations/atomic-chat/kakeya_sidecar/` → 产品用，Mac 部署，新增路径。

两者共用的唯一组件是 `kakeyalattice` Python 包本身（纯 PyTorch，device 无关）。

---

## 9. 路线图

| 阶段 | 内容 | 依赖 |
|:-:|:--|:--|
| M1 | Sidecar MVP：OpenAI `/v1/chat/completions` 流式 + Qwen3/Llama3/Gemma/Mistral/DeepSeek/GLM 6 模型单测 | 本 PR 落地 |
| M2 | kakeyalattice-extension + Tauri plugin 接入 Atomic-Chat，UI 增加 "Backend: KakeyaLattice" 选项 | M1 |
| M3 | Metal fused E8 closest-point kernel，codec 延迟降到 bf16 decode 的 < 5% | Metal Shading Language 专家 |
| M4 | `KakeyaLatticeCache` 切换到真·存索引模式（非 roundtrip），HBM 节省从 "nominal" 转为 "实打实" | 自定义 attention kernel |
| M5 | 与 Atomic-Chat 的 MCP/agent 链路打通：long-context 检索可在 compressed KV 上直接跑 NIAH 风格取值 | M1 + M2 |

---

## 10. 风险 & 取舍

| 风险 | 缓解 |
|:--|:--|
| MPS 算子在 torch 2.x 有偶发 bug（`argmin` 在非连续 tensor 上） | sidecar 在 codec 前强制 `.contiguous()`，已经放进 smoke test |
| sidecar 进程崩掉导致整台 Atomic-Chat 推理断线 | Tauri plugin 负责 supervise + 自动重启（对齐 llama.cpp plugin 现有行为） |
| 用户误选 Q=4 在小模型上（如 DeepSeek-1.5B no-boundary 已知 5.4 万 % PPL） | UI 只暴露 `safe` Q 档；`aggressive` 档进高级菜单 + 警示 |
| HF 模型许可协议（Llama-3 门禁、GLM-4 的自有 license）| 沿用 Atomic-Chat 既有机制，sidecar 不做任何绕过 |
| `KakeyaLatticeCache` 的 `roundtrip` 落在 bf16 阵上，实际 HBM 节省是 "nominal"（见 `kakeyalattice/README.md` §What it is NOT） | M4 前，主卖点是**长上下文的 attention 质量下限**，不是 HBM 压缩率。UI 文案要诚实 |

---

## 11. 与 atomic.chat 宣传口径的对齐

atomic.chat 首页写的是 *"Google TurboQuant built-in"*。按 v1.5 报告:

- v1.5 E8 Q=4 (CR 4.57×) **vs** TQ b=3 (CR 4.92×)：v1.5 在 4 个模型上全面
  赢 3-6× 更低 |Δppl|。
- TQ b=2 (CR 7.11×) 在 4 个模型上结构性不可用（\|Δppl\| > 100% 到 14 万%）。

所以这次集成对 Atomic-Chat 产品而言是**把宣传里的 "TurboQuant built-in"
换成工程上更靠谱的 E8 nested-lattice**。两者可以共存（用 `backend` 参数
切），但在默认档位上 KakeyaLattice 明显更稳。

---

## 附：文件对应表

本 PR 新增的文件与它们在 Atomic-Chat 仓库里的最终归宿:

| 本仓库路径 | Atomic-Chat 仓库映射 |
|:--|:--|
| `integrations/atomic-chat/kakeya_sidecar/` | 独立安装，sidecar 包装后塞进 `.dmg` |
| `integrations/atomic-chat/kakeyalattice-extension/` | 放到 `extensions/` |
| `integrations/atomic-chat/tauri-plugin-kakeyalattice/` | 放到 `src-tauri/plugins/kakeyalattice/` |
| `docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md` | 存本仓库作设计依据 |

---

*作者：Cursor Cloud Agent · 分支 `AgentMemory/atomic-chat-kakeya-integration-04ae`.*

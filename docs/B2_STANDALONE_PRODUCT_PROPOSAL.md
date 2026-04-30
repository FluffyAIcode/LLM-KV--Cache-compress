# B2 Standalone Product Proposal — 从 Atomic-Chat backend 走向独立 Mac 本地推理产品

**Date**: 2026-04-30
**Scope**: 评估把 B2 (`MLX + DFlash + KakeyaLattice-MLX`) 从"Atomic-Chat
的第二个后端"升级为**独立 Mac 本地推理产品**的商业/工程可行性。
**Status**: 评估稿，作为 M6 原 "Atomic-Chat extension 追加 backend 选项"
的替代路线。

---

## 1. 为什么重估

原 M6 规划是把 B2 做成 Atomic-Chat 的第二个 backend：在 TS 扩展里加
`"variant": "mlx-dflash"`，Tauri plugin 同时托管 B1 + B2 两个 sidecar。
这条路径**工程上成立**，但在商业逻辑上有几个问题。

### 1.1 Atomic-Chat 的宣传与 B2 的产品承诺不对等

atomic.chat 首页的核心宣传：

- *"Google TurboQuant built-in — 8× Faster Inference"* — v1.5 报告里
  TurboQuant b=2 在 4 个模型上**结构性不可用**，b=3 被 E8 Q=4 全面压过
  3-6×。真实落地的是 llama.cpp 的 q4_0/q8_0 标量量化（2023 年的技术）。
- *"Compressed down to just 3 bits — with no retraining, no fine-tuning,
  and no trade-off in model performance"* — 同上，3 bit 级别对 <3B 模型
  在 llama.cpp 上真实可用性极差。

把 B2 挂进 Atomic-Chat 作为 backend 选项，等于帮它**兑现宣传承诺**。
但 Atomic-Chat 团队**不是我们**，合作节奏、PR review、version
compatibility 都有摩擦。我们交付 PR → 他们决定合不合 → 用户什么时候
能拿到——这条链路至少多一个季度。

### 1.2 B2 的价值点超过"chat UI 的后端"

B2 的三个能力：
- **MLX 原生推理** — Apple Silicon 第一等公民
- **DFlash 3-6× 无损加速** — 2026 年 speculative decoding SOTA
- **KakeyaLattice E8 KV 压缩** — 长上下文不 OOM

这三个能力的使用面 **不止 chat**。举例：
- IDE 集成（Cursor / VSCode 里本地跑 Qwen-Coder 30B）
- 命令行 agent / MCP-driven workflow（不需要 UI 的场景）
- 批量文本处理（合同 / 论文 / 代码库摘要，不交互）
- 其他 GUI 宿主（Raycast extension、Alfred workflow、Obsidian plugin）

把 B2 绑死到 Atomic-Chat 的 UI 里，等于把一个**通用推理加速层**锁进
一个特定 chat app。

### 1.3 atomic.chat 的长期演化风险

atomic.chat 是商业公司产品，UI / 扩展 API 路线图由他们决定。他们可能：
- 换推理引擎（目前完全绑 llama.cpp；将来改成 MLX 原生或自研）
- 改扩展 API（我们的 TS 扩展 + Rust plugin 需随之改）
- 决定不把 KakeyaLattice 放进默认发行（"Pro Mode for Mac" 选项可能永远
  藏在高级设置里）

这些风险不是零概率，且一旦发生我们的工程沉默成本很高。

## 2. 独立产品路线的三种形态

### Form A：CLI 工具 `kakeya-llm`

- `brew install kakeya-llm` / `pip install kakeya-llm`
- `kakeya-llm chat qwen3-8b --q 38 --dflash` 启交互
- `kakeya-llm serve --port 1339` 开 OpenAI 兼容 server
- `kakeya-llm bench gsm8k` 跑内置 M5 benchmark
- 目标用户：开发者 / ML 工程师 / 学生

**优势**：
- 分发简单，homebrew 生态现成
- 不依赖 UI 框架；Metal / CUDA / CPU 都能跑（基座是 mlx-lm + dflash）
- 完整控制 KakeyaLattice × DFlash 的每一个旋钮

**劣势**：
- 非技术用户门槛高
- 默认没有对话历史 / 多 assistant / MCP 集成，纯"跑模型"
- 需要自己维护包发布 + 版本升级

**适合场景**：作为底层 API 被其他 app 集成（IDE / 插件 / 工作流）。

### Form B：native macOS app `Kakeya Studio`

- DMG 包，不要求任何命令行能力
- 单窗口 chat UI + 模型管理 + 性能监控
- 内置 KakeyaLattice compression 的**可视化**（"你刚才这次对话省了
  X GB KV，速度比 baseline 快 Y×"）
- OpenAI 兼容 `:1339` 自带，让 Cursor / Raycast / Alfred 能连
- 目标用户：Mac 本地 AI 爱好者（和 atomic.chat 用户重叠但不完全相同）

**优势**：
- 完整控制 UI／叙事／宣传口径（"基于论文的 E8 格压缩"，可验证、学术
  可溯）
- 把 KakeyaLattice 的 "discrete Kakeya cover" 理论直接 surface 到
  UI（这是 atomic.chat 做不到也不会做的差异化）
- 省下 Atomic-Chat 团队合作的沟通成本

**劣势**：
- 从零搭 Tauri/Electron UI + signing + notarize + auto-update
- App Store 审核（如果走 MAS 发行）
- 和 atomic.chat 正面竞争 Mac 本地 AI app 市场 — 他们已经 500+ stars，
  我们要从 0 开始做 GTM

**适合场景**：想建立 KakeyaLattice 作为品牌的长期目标。

### Form C：SDK + 企业/开发者服务 `kakeyalattice-mlx` as a library

- 把 `kakeyalattice_mlx` + `kakeya_sidecar_mlx` + `cache_injection` 升级
  为严肃 Python 包，发到 PyPI
- 卖点不是 "app"，是 "the MLX-native long-context inference stack"
- 配套：文档站 + benchmark 报告 + 参考 Docker image
- 用户是开发团队（Cursor 这类 IDE 想集成本地推理、内部工具团队想给
  员工提供本地 AI）
- 目标客户：对"确定性 + 可审计 + 离线"有强诉求的企业（律所、医疗、
  投行、国防承包商）

**优势**：
- 最小 UI 投入，最大技术杠杆
- 和 Atomic-Chat（C 端 chat app）错位竞争，客户群完全不重叠
- Apache-2.0 吸引生态，有机会成为事实标准（类似 `transformers` 在
  HF 生态里的地位）

**劣势**：
- B2B 销售周期长
- 需要保持上游（MLX / DFlash / HF）兼容性的长期工程承诺
- 早期没有"用户界面"的东西，marketing 需要更专业渠道（Twitter/X、
  arXiv 引用、技术 blog）

**适合场景**：研究机构 / 行业 / 企业客户优先。

## 3. 推荐路径：A + C 串联，暂缓 B

基于以下几个现实判断：

1. **我们的工程资产天然贴合 A 和 C**：
   - `kakeya_sidecar_mlx` 已经是 OpenAI 兼容 server — **直接就是 A 的核心**。
   - `kakeyalattice_mlx` 已经是独立 pip 包，bit-identical parity 有测试
     保证 — **直接就是 C 的核心**。
   - 从"B1 + B2 集成到 Atomic-Chat"变成"把 B2 独立发布"，**工程改动不到
     10%**，主要是加发布 CI + 文档站 + 品牌 landing page。

2. **B 的 native app 路径成本最高且回报最晚**：
   - 成本：搭 UI / signing / notarize / auto-update / App Store / GTM /
     客服，全套做齐是半年级别工作。
   - 回报：必须打败 atomic.chat (500+ stars, 成熟 GTM) 才能起量。
   - 风险：一旦 atomic.chat 集成 KakeyaLattice（无论是通过我们的 PR
     还是自己实现），B 的差异化就瓦解。

3. **A + C 的组合可以"先卖给开发者 → 逐步渗透到 app"**：
   - Cursor / Raycast / Alfred / Obsidian 的插件开发者用我们 SDK 嵌
     本地推理 → 终端用户通过这些 app 间接用上 KakeyaLattice
   - **这条路径 atomic.chat 无法阻断**，因为他们的 UI 只是众多消费端之一

## 4. 具体执行建议

### Phase 1（M5 benchmark 之后立即做）— 公开发布 A + C 的底层包

1. **PyPI 发布**:
   - `kakeyalattice-mlx` v0.1.0
   - `kakeya-sidecar-mlx` v0.1.0
   - 两者都走 trusted publishing + GitHub release + 自动 wheel build
2. **文档站**（`kakeyalattice.dev` 或 `kakeya-mlx.dev` domain）:
   - Quick start（一条 `pip install` 命令）
   - API reference（auto-gen from docstrings）
   - 性能 benchmark 页面（绑到 `reports/b2_release/` 的 JSON）
   - 对比表格（atomic.chat / llama.cpp / ollama / MLC-LLM / 我们）
3. **CLI shim**: `kakeya-llm` 作为 `kakeya-sidecar-mlx` 的子命令包装，
   提供 `chat`/`serve`/`bench` 三个入口。
4. **GitHub 项目分拆**（可选）: `kakeyalattice` 主 repo 保留 codec +
   论文；`kakeya-mlx` 拆出独立 repo 做 sidecar + benchmark + 文档站。
   独立 issue tracker + release cadence。

### Phase 2（Phase 1 上线 4-8 周后）— 生态嵌入

1. **Cursor / Continue / Aider 集成** — 这些 IDE agent 工具都支持
   OpenAI 兼容的本地 server。我们的 sidecar 默认就能被它们连上，
   只需要文档 + 示例。
2. **Raycast extension**: Raycast 的 extension 生态对"本地 AI"有大需求；
   做一个 Raycast AI 替代品走 Kakeya sidecar。
3. **企业 POC**: 选 2-3 家有"隐私 + Mac 标配 + 长文档处理"需求的客户
  （律所、投行），做 3 个月 POC，产出案例。

### Phase 3（6 个月后再评估）— B 的 native app 要不要做

如果 Phase 2 证明了：
- 用户愿意把 KakeyaLattice 作为"本地 AI infra"集成进他们的工作流
- SDK/CLI 形态的 conversion 到 Apache-2.0 用户足够多（比如周下载过万）
- atomic.chat 没有把 KakeyaLattice 集成进去（无论出于什么原因）

**那时再评估要不要做 native app**。做的时候差异化应该非常明确（e.g.
"for researchers who want to see the E8 lattice compression happen in
real time"），而不是通用 chat UI 和 atomic.chat 正面硬刚。

### 不推荐的路径

- **不做原 M6**: 把 B2 做成 Atomic-Chat 的 backend 选项，然后等 PR
  合入。工程沉默成本过高，且回报受制于第三方。
- **不跳过 Phase 1 直接做 B**: "先做 UI，再补后端"是典型的精力错配；
  UI 是消费端表面，不是壁垒。KakeyaLattice 的壁垒在 `kakeyalattice_mlx`
  这一层，应该优先发布它。

## 5. 对 B1 PR #57 和 B2 PR #58/#59 的影响

这份评估不否定已有工作：

- **B1 (PR #57)** 依然有用 — HF + MPS 路径是跨平台的（Mac/Win/Linux），
  CLI `kakeya-llm` 会默认用它；Apple Silicon 用户自动切 B2。
- **B2 (PR #58 骨架 + PR #59 DFlash 集成)** 依然有用 — 就是 Phase 1
  要发布的 MLX 栈。
- **Atomic-Chat 集成** (原 M6) — **可以保留为可选**: 如果 atomic.chat
  团队愿意合我们的 PR，那是 bonus 分发渠道；但**不是独立产品的主路径**。

换言之，现有工程资产**全部都在独立产品路径上可直接复用**。本评估只是
**调整 GTM 策略和 branding focus**。

## 6. 下一步（如果采纳本评估）

1. **本 PR 合并后立即做**：
   - 给 `kakeyalattice_mlx` 和 `kakeya_sidecar_mlx` 加 PyPI 发布 CI
     (`.github/workflows/publish-*.yml`)
   - 注册 domain（`kakeyalattice.dev` 等）
   - 起草 landing page 内容（对比表 + quick start + FAQ）
2. **2 周内**：
   - 首个 PyPI release (v0.1.0)
   - 开 `kakeya-mlx` 独立 repo（可选，看我们 repo 策略偏好）
   - Twitter / arXiv / HN 发布公告
3. **1 个月内**：
   - Cursor/Continue/Aider 集成文档 + PR
   - 第一个企业 POC 接触

## 7. 结论（一句话）

> **把 B2 从"atomic.chat 的后端"升级为"Mac 本地推理的独立 SDK + CLI 发行"。
> 所有工程资产都能直接复用；差异化点（E8 格压缩 + DFlash 加速的
> MLX 原生组合）在独立发行下完全可控，且不受 Atomic-Chat 产品路线
> 影响。Native app 形态的决定推迟到 Phase 3，避免早期 UI 投入与
> atomic.chat 正面竞争。**

---

*作者：Cursor Cloud Agent · 分支
`AgentMemory/atomic-chat-b2-m5-acceptance-benchmark-04ae` · 作为 M6 评估
文档与 M5 benchmark 同 PR 交付。*

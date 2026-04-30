# @atomic-chat/kakeyalattice-extension

Atomic-Chat TypeScript 扩展，把 KakeyaLattice v1.5 sidecar 作为**第二个
一等推理后端** 注册进 Atomic-Chat 的 Extension System。

## 架构位置

```
 web-app (React)
    │
    ▼
 core SDK  ──┬── registerBackend(...)
             │
 extensions ─┤
             ├── llamacpp-extension           (既有，GGUF / llama.cpp)
             └── kakeyalattice-extension      ★ 本扩展
                       │
                       │ Tauri invoke
                       ▼
            plugins/kakeyalattice  (Rust, 下一级目录)
                       │
                       │ supervise + HTTP
                       ▼
            kakeya-sidecar (Python, localhost:1338)
```

## 关键文件

- `src/index.ts` — 扩展入口，向 Core SDK 注册一个新的 `Backend`。
- `src/backend.ts` — `Backend` 实现：`listModels()`、`chatCompletion()`、
  `healthCheck()`、`getStats()`，统一通过 Tauri invoke 过桥到 Rust 插件。
- `src/types.ts` — 与 Python sidecar `/v1/models` `x_kakeya` 字段一一对齐的
  类型定义。

## 真正落地到 Atomic-Chat 主仓库时要做什么

1. 把本目录整份 `rsync` 到 `extensions/kakeyalattice-extension/`。
2. 修 `@atomic-chat/core` 的 `peerDependency` 版本指向主仓库的实际版本。
3. 在 Atomic-Chat 的扩展注册入口（通常是 `core/src/extensions.ts` 或
   `web-app/src/App.tsx` 里 bootstrap 阶段的扩展列表）追加：
   ```ts
   import { register as registerKakeya } from "@atomic-chat/kakeyalattice-extension";
   registerKakeya();
   ```
4. Rust 插件同步复制到 `src-tauri/plugins/kakeyalattice/`，在
   `src-tauri/Cargo.toml` + `tauri.conf.json` 注册。

因为我们**不改 Atomic-Chat 主仓库本身**，主仓库的维护者审阅后决定
怎么合并；本扩展只保证"代码骨架能通过 tsc --noEmit"即可。

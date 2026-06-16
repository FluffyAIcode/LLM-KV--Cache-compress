# kakeya-sidecar-mlx  (B2)

B2 版 OpenAI 兼容 sidecar，与 B1 (`integrations/atomic-chat/kakeya_sidecar/`)
并列。二者 HTTP 接口完全一致，Atomic-Chat 的前端不区分。区别：

| | B1 `kakeya-sidecar` | B2 `kakeya-sidecar-mlx` |
|:-|:-|:-|
| 推理引擎 | HF transformers + torch MPS | MLX + (可选) DFlash drafter |
| KV 压缩 | `kakeyalattice.hf.KakeyaLatticeCache` | `kakeyalattice_mlx.KakeyaLatticeMLXCache` |
| 默认端口 | 1338 | **1339** |
| 平台 | Mac/Win/Linux | Mac (Apple Silicon) only |

## 当前 PR 范围

M1-M3 骨架 + 纯逻辑单测。真正的 MLX 模型加载、DFlash 接入、生成循环
在后续 PR（见 `../ROADMAP.md` M4-M6）里补。本目录的代码现在做到:

- CLI / pyproject / FastAPI 路由壳
- `model_registry_mlx.py` — 带 `dflash_draft_repo` 字段的 MLX 版模型档位
- 纯逻辑单元测试全绿（无 mlx 依赖也能跑）

## 启动（需 mlx）

```bash
pip install -e ".[mlx,dev]"
kakeya-sidecar-mlx --port 1339 --device mps --prewarm qwen3-4b@e8-q38
```

## 路由

与 B1 完全一致:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /v1/kakeya/stats`

`/v1/models` 返回的每条模型额外带 `x_kakeya.dflash_draft_repo` 字段
(B1 里保留 `null`，B2 填实际 z-lab/... id)。

## 与 B1 的共存

推荐的 Atomic-Chat 整合：Tauri plugin 同时启动两个 sidecar
(`:1338` B1 + `:1339` B2)，按用户在 UI 选的 backend 路由。Mac 用户默认
走 B2，Win/Linux 用户强制走 B1。

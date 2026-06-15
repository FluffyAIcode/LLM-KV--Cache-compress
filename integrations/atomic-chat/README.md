# Atomic-Chat × KakeyaLattice v1.5 — 本地 Mac 部署集成

把 [`kakeyalattice` v1.5 (E8 格 KV-cache codec)](../../kakeyalattice/) 作为
**第二个一等推理后端** 接入
[`AtomicBot-ai/Atomic-Chat`](https://github.com/AtomicBot-ai/Atomic-Chat)，
目标是 **Mac (Apple Silicon, Metal)** 的多模型离线部署。

> 完整设计依据见 [`docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md`](../../docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md)。
> 本目录只放"可直接拷进 Atomic-Chat 仓库"的工程骨架。

## 目录结构

```
integrations/atomic-chat/
├── kakeya_sidecar/              ★ Python 推理 sidecar
│   ├── pyproject.toml           —— 独立 pip 包 `kakeya-sidecar`
│   ├── kakeya_sidecar/
│   │   ├── __init__.py
│   │   ├── __main__.py          —— `python -m kakeya_sidecar ...`
│   │   ├── cli.py               —— argparse 入口 + uvicorn 启动
│   │   ├── server.py            —— FastAPI OpenAI 兼容接口
│   │   ├── engine.py            —— HF + KakeyaLatticeCache 推理核心
│   │   ├── model_registry.py    —— 多模型部署档位
│   │   └── schemas.py           —— OpenAI 请求/响应 dataclass
│   └── tests/
│       └── test_model_registry.py
├── kakeyalattice-extension/     ★ Atomic-Chat TypeScript 扩展
│   ├── package.json
│   ├── src/
│   │   ├── index.ts             —— 扩展入口（注册到 core SDK）
│   │   └── backend.ts           —— 走 Tauri plugin → sidecar
│   └── README.md
└── tauri-plugin-kakeyalattice/  ★ Rust Tauri 插件（桩）
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               —— `tauri::plugin::Builder` 注册
    │   └── commands.rs          —— sidecar 生命周期 + 代理调用
    └── README.md
```

## 快速验证 (Mac)

```bash
# 1. 安装 sidecar
cd integrations/atomic-chat/kakeya_sidecar
pip install -e ".[mac]"

# 2. 单元测试（不需要下载模型）
pytest tests/ -v

# 3. 跑起来
kakeya-sidecar --port 1338 --device mps
curl http://localhost:1338/v1/models | jq .

# 4. 发一次推理请求（会首次下载模型，小心硬盘）
curl http://localhost:1338/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3-4b@e8-q10",
  "messages": [{"role":"user","content":"Explain nested-lattice codes in one line."}],
  "stream": false, "max_tokens": 64
}'
```

## 集成到 Atomic-Chat 主仓库（步骤示意）

```bash
# 假设 Atomic-Chat 主仓库在 ~/code/Atomic-Chat
export ATOMIC=~/code/Atomic-Chat

# 1. 扩展
rsync -a kakeyalattice-extension/   $ATOMIC/extensions/kakeyalattice-extension/

# 2. Tauri plugin
rsync -a tauri-plugin-kakeyalattice/ $ATOMIC/src-tauri/plugins/kakeyalattice/

# 3. 向 Atomic-Chat 的 extension registry 加一行：
#    （详见 $ATOMIC/extensions/README / CONTRIBUTING.md）
#    { name: "kakeyalattice-extension", enabled: true }

# 4. Python sidecar 需要和 llama.cpp 一起打进安装包。
#    Atomic-Chat 在 `scripts/bundle-binaries.*` 有现成的二进制打包脚本，
#    对 sidecar 用 PyOxidizer / PyInstaller 产出单文件，挂上去即可。
```

> 本 PR 不改 Atomic-Chat 主仓库 — 只在本仓库提供可直接移植的骨架、
> 完整的设计文档、以及 sidecar 自己的单元测试。

## 与既有 vLLM 插件的关系

| 路径 | 场景 | 是否改动 |
|:--|:--|:--|
| `vllm_backend/kakeya_v1_4_snapshot/` | Linux / CUDA / H200 benchmark | 不动 |
| `integrations/atomic-chat/kakeya_sidecar/` | Mac / MPS / 产品端推理 | 本 PR 新增 |

两者共用的唯一组件是 Python 包 `kakeyalattice` 本身 — 标量算子都走
PyTorch，设备无关。

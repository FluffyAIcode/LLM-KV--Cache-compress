# kakeya-sidecar

OpenAI 兼容的本地推理 sidecar，给 [Atomic-Chat](https://github.com/AtomicBot-ai/Atomic-Chat)
用；推理走 HuggingFace `transformers` + `kakeyalattice.hf.KakeyaLatticeCache`
(E8 nested-lattice KV-cache 压缩)。

设计目标:

1. **零代码改动** — 对外是 `POST /v1/chat/completions`，Atomic-Chat 既有
   OpenAI 客户端直连即可。
2. **多模型本地部署** — Qwen3 / Llama-3.x / Gemma-4 / DeepSeek-R1-Distill /
   GLM-4-9B / Mistral，每个都有"出厂" Q 档配置。
3. **Mac 优先** — 默认 `--device mps`，Linux/CUDA 也支持。
4. **Tauri-friendly** — 纯 HTTP + JSON，Tauri plugin 负责起进程、转发调用。

详细设计：[`docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md`](../../../docs/ATOMIC_CHAT_KAKEYA_INTEGRATION.md)。

## Quick start

```bash
pip install -e ".[mac,dev]"
pytest tests/ -v                    # 不下载模型的纯逻辑单测
kakeya-sidecar --port 1338 --device mps
```

```bash
curl http://localhost:1338/v1/models
curl http://localhost:1338/v1/chat/completions -d '{...}' ...
```

## 支持的模型

参见 `kakeya_sidecar/model_registry.py`。通过 `GET /v1/models` 实时返回。

## License

Apache-2.0，与主仓库一致。

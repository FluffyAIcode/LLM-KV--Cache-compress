"""vLLM plugin entry point for the v1.4 KakeyaLattice snapshot harness.

Registered via `vllm.general_plugins` in `pyproject.toml`.  vLLM
invokes this callable once per process (both the parent
`vllm.entrypoints.llm.LLM(...)` process AND the spawned engine-core
subprocesses), so the attention-class monkey-patches take effect in
every worker.

Activation gate: set `KAKEYA_SNAPSHOT_QWEN3=1` before importing
vLLM.  Without the env var this plugin is a no-op (vLLM startup is
not perturbed for non-measurement runs).

Patches installed:
  * `Qwen3Attention`   (Qwen3 family, e.g. Qwen/Qwen3-4B)
  * `Qwen2Attention`   (DeepSeek-R1-Distill-Qwen-1.5B, base Qwen2)
  * `Gemma4Attention`  (google/gemma-4-E2B, E4B, 26B-A4B, 31B)
  * `GLMAttention`     (zai-org/GLM-4, THUDM/chatglm*)

Each patch is idempotent and fires only when the corresponding
model type is loaded, so installing all four is safe even if only
one is in use.
"""
from __future__ import annotations


def register_plugin() -> None:
    import logging
    import os

    logger = logging.getLogger("kakeya_v1_4_snapshot.plugin")

    if os.environ.get("KAKEYA_SNAPSHOT_QWEN3", "0") != "1":
        return

    try:
        from . import snapshot_hook
        snapshot_hook.install_all_snapshot_patches()
    except Exception as e:
        logger.exception("Failed to install snapshot patches: %s", e)
        raise

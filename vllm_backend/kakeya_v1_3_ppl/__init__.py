"""vLLM attention backend for the v1.3 PPL codec.

Usage (once the package is importable from vLLM's Python environment):

    import vllm_backend.kakeya_v1_3_ppl  # noqa: F401 — side-effect registers
    vllm serve Qwen/Qwen3-4B \\
        --kv-cache-dtype kakeya_v1_3_ppl \\
        --block-size 512 \\
        --attention-backend KAKEYA_V1_3_PPL

Public surface
--------------

  * `KakeyaV13PPLConfig`        — per-layer slot-size + bits/metric config
  * `KakeyaV13PPLAttentionSpec` — FullAttentionSpec subclass; declares
                                   the raw-byte cache shape
  * `KakeyaV13PPLAttentionBackend` — AttentionBackend subclass
  * `KakeyaV13PPLAttentionImpl`    — AttentionImpl subclass; owns
                                      store/decode dispatch
  * `register_kakeya_backend()`    — patches `CacheDType` literal +
                                      `AttentionBackendEnum.CUSTOM`.  Call
                                      once before `vllm serve` starts its
                                      engine (e.g. from a vLLM plugin).

All registration is opt-in (no monkey-patching on `import`) so that
running the tests on a CPU / non-vLLM environment doesn't accidentally
modify vllm's global state.
"""
from .config import KakeyaV13PPLConfig, KAKEYA_V1_3_PPL_NAME

# Lazy re-export: these modules import vllm internals and should only
# be touched when the user is actually running inside vLLM.  Import
# errors on CPU-only dev machines are acceptable — the M5 kernel tests
# don't need them.
__all__ = [
    "KakeyaV13PPLConfig",
    "KAKEYA_V1_3_PPL_NAME",
    "KakeyaV13PPLAttentionSpec",
    "KakeyaV13PPLAttentionBackend",
    "KakeyaV13PPLAttentionImpl",
    "register_kakeya_backend",
]


def __getattr__(name: str):
    if name == "KakeyaV13PPLAttentionSpec":
        from .spec import KakeyaV13PPLAttentionSpec
        return KakeyaV13PPLAttentionSpec
    if name == "KakeyaV13PPLAttentionBackend":
        from .backend import KakeyaV13PPLAttentionBackend
        return KakeyaV13PPLAttentionBackend
    if name == "KakeyaV13PPLAttentionImpl":
        from .impl import KakeyaV13PPLAttentionImpl
        return KakeyaV13PPLAttentionImpl
    if name == "register_kakeya_backend":
        from .registration import register_kakeya_backend
        return register_kakeya_backend
    raise AttributeError(name)

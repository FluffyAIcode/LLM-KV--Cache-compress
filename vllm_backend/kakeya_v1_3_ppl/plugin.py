"""vLLM plugin entry point.

Registered via `vllm.general_plugins` in `pyproject.toml`.  vLLM
invokes this callable once per process (both the parent
`vllm.entrypoints.llm.LLM(...)` process AND the spawned engine-core
subprocesses), so all backend-registration side effects become
effective in every worker.
"""
from __future__ import annotations


def register_plugin() -> None:
    # Import inside the function so that merely *discovering* the
    # entry point doesn't pull in our heavy deps (torch, triton,
    # pyo3 module) before vllm has initialised its device
    # environment.
    import os
    import logging
    logger = logging.getLogger("kakeya_v1_3_ppl.plugin")

    from .registration import register_kakeya_backend
    register_kakeya_backend()
    logger.info("kakeya_v1_3_ppl backend registered")

    # Auto-load calibration bundle if env vars are set.  This is how
    # the plugin gets calibration into every subprocess (engine-core,
    # workers) without the caller having to call set_global_calibration
    # explicitly.  Env vars are inherited across fork/spawn.
    sigma_q = os.environ.get("KAKEYA_SIGMA_Q_PATH")
    k_cent = os.environ.get("KAKEYA_K_CENTROIDS_PATH")
    v_cent = os.environ.get("KAKEYA_V_CENTROIDS_PATH")
    skip_str = os.environ.get("KAKEYA_SKIP_LAYERS", "")
    skip_layers = [int(s) for s in skip_str.split(",") if s.strip()] if skip_str else []

    if sigma_q and k_cent and v_cent:
        try:
            from .calibration import load_calibration_bundle
            from .impl import set_global_calibration
            bundle = load_calibration_bundle(
                sigma_q_safetensors=sigma_q,
                k_centroids_f32=k_cent,
                v_centroids_f32=v_cent,
                skip_layers=skip_layers,
            )
            set_global_calibration(bundle)
            logger.info(
                "kakeya calibration loaded: %d active layers",
                len(bundle.active_layers()),
            )
        except Exception as e:
            logger.exception("Failed to auto-load calibration: %s", e)
            raise

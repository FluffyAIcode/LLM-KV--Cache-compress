"""``kakeya-sidecar`` CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys

from .engine import EngineConfig
from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kakeya-sidecar",
        description="OpenAI-compatible local inference sidecar for Atomic-Chat "
        "(HuggingFace transformers + KakeyaLattice v1.5 E8 KV-cache).",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default 127.0.0.1 — localhost only).")
    p.add_argument("--port", type=int, default=1338,
                   help="Bind port (default 1338; Atomic-Chat's OpenAI front "
                        "door is 1337 so we sit one port over).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"],
                   help="Torch device. 'auto' picks mps on Mac / cuda on Linux.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"],
                   help="Model parameter dtype.")
    p.add_argument("--max-resident", type=int, default=1,
                   help="Max number of fully-loaded models kept in RAM/VRAM "
                        "at once (LRU).")
    p.add_argument("--hf-cache-dir", default=None,
                   help="Override HF_HOME / cache location.")
    p.add_argument("--prewarm", default=None,
                   help="Channel id to pre-load at startup, e.g. "
                        "'qwen3-4b@e8-q10'.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = EngineConfig(
        device=args.device,
        dtype=args.dtype,
        max_resident=args.max_resident,
        hf_cache_dir=args.hf_cache_dir,
    )

    engine_instance = None
    if args.prewarm:
        from .engine import KakeyaEngine

        log = logging.getLogger("kakeya_sidecar.cli")
        engine_instance = KakeyaEngine(cfg)
        log.info("prewarming %s on %s", args.prewarm, engine_instance._device)
        engine_instance.warmup(args.prewarm)

    app = create_app(
        cfg,
        lazy_engine=engine_instance is None,
        engine_instance=engine_instance,
    )

    try:
        import uvicorn  # type: ignore
    except ImportError:  # pragma: no cover
        print("uvicorn is required. `pip install uvicorn[standard]`.", file=sys.stderr)
        return 2

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0

"""``kakeya-sidecar-mlx`` CLI entry point.

Mirrors the B1 CLI API so Atomic-Chat's Tauri plugin can supervise
both sidecars with identical command-line patterns.
"""
from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kakeya-sidecar-mlx",
        description=(
            "B2 OpenAI-compatible local inference sidecar for Atomic-Chat "
            "(MLX + DFlash + KakeyaLattice v1.5 E8 KV-cache compression; "
            "Apple Silicon only)."
        ),
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=1339,
                   help="Bind port (default 1339; B1 sits at 1338, "
                        "Atomic-Chat front door at 1337).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cpu"],
                   help="MLX device. 'auto' picks Metal on Apple Silicon, "
                        "CPU otherwise.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--max-resident", type=int, default=1,
                   help="Max number of fully-loaded MLX models (LRU).")
    p.add_argument("--enable-dflash", action="store_true",
                   help="Enable DFlash speculative decoding when the "
                        "resolved channel has dflash_available=True.")
    p.add_argument("--prewarm", default=None,
                   help="Channel id to pre-load, e.g. "
                        "'qwen3-8b@e8-q38'.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("kakeya_sidecar_mlx.cli")

    # Import engine + server lazily so ``--help`` works without mlx.
    from .engine_mlx import MLXEngineConfig, MLXEngine
    from .server import create_app

    cfg = MLXEngineConfig(
        device=args.device,
        dtype=args.dtype,
        max_resident=args.max_resident,
        enable_dflash=args.enable_dflash,
    )

    engine_instance = None
    if args.prewarm:
        engine_instance = MLXEngine(cfg)
        log.info("prewarming %s", args.prewarm)
        engine_instance.warmup(args.prewarm)

    app = create_app(
        cfg,
        lazy_engine=engine_instance is None,
        engine_instance=engine_instance,
    )

    try:
        import uvicorn  # type: ignore
    except ImportError:  # pragma: no cover
        print("uvicorn is required. `pip install uvicorn[standard]`.",
              file=sys.stderr)
        return 2

    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level.lower())
    return 0

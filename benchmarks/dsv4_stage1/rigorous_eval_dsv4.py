"""Stage 1 rigorous-eval adapter for DeepSeek-V4 (Flash / Pro).

Status: **scaffold only** — see HARDWARE_REQUIREMENTS.md in this
directory for the hardware prerequisites (not satisfied by our
current dev H200 instance; V4-Flash requires 2+ H200 or 4×B200 with
≥ 500 GB disk).

This script does NOT re-implement ``benchmarks/rigorous_eval.py``;
it is a thin adapter that

1. installs the DSV4-aware snapshot patches (including the new
   ``DeepseekV4Attention`` hook)
2. invokes the standard rigorous eval harness with V4-appropriate
   defaults (head_dim=512, single-latent KV, YaRN RoPE)
3. adds one V4-specific CLI flag ``--kv-stream-filter`` to select
   which of the three KV-cache block types (SWA / c4a / c128a) to
   include in the final aggregate ΔMSE / Δppl rollup

When vLLM V4 support lands on vast.ai (or the user provisions
appropriate hardware), this script is the intended entry point.

Usage (example, NOT yet runnable on our dev H200)
-------------------------------------------------

    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export KAKEYA_SNAPSHOT_DSV4=1

    python benchmarks/dsv4_stage1/rigorous_eval_dsv4.py \\
        --model-path deepseek-ai/DeepSeek-V4-Flash \\
        --model-name dsv4_flash \\
        --mode inforward \\
        --no-boundary \\
        --q-values "" \\
        --v15-q-values 4,10,38 \\
        --tq-b-values "" \\
        --kv-modes KV \\
        --ctx-len 4096 \\
        --n-eval 64 \\
        --n-passages 32 \\
        --kv-stream-filter all \\
        --tensor-parallel-size 2 \\
        --trust-remote-code \\
        --out-dir reports/v1_5_release/dsv4_stage1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    # Ensure the V4 hook is installed before vLLM loads the model. We set
    # KAKEYA_SNAPSHOT_DSV4=1 so the plugin entry-point picks up the DSV4
    # patch variant (regular KAKEYA_SNAPSHOT_QWEN3=1 stays working for
    # the v1.4/v1.5 harness on non-V4 models).
    os.environ.setdefault("KAKEYA_SNAPSHOT_DSV4", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    parser = argparse.ArgumentParser(
        description=(
            "DSV4-aware wrapper around benchmarks/rigorous_eval.py. "
            "Adds the DSV4 snapshot hook (single-latent 512-dim KV) and "
            "--kv-stream-filter. Arguments not listed below are "
            "forwarded verbatim to the underlying harness."
        ),
    )
    parser.add_argument(
        "--kv-stream-filter",
        choices=["all", "swa", "c4a", "c128a"],
        default="all",
        help=(
            "Which DSV4 KV-cache block type to include in the final "
            "aggregate (filter is applied layer-wise using each layer's "
            "compress_ratio attribute: 1=SWA, 4=c4a, 128=c128a). "
            "'all' aggregates the three block types together."
        ),
    )
    parser.add_argument(
        "--dsv4-patch-only",
        action="store_true",
        help=(
            "Install only the DSV4 patch, skip the Qwen3/Qwen2/Gemma4/GLM "
            "patches. Useful when running exclusively against V4."
        ),
    )
    # Pre-parse just our flags so the rest can be forwarded to rigorous_eval.
    args, forwarded = parser.parse_known_args()

    # Install the DSV4 patch NOW (before rigorous_eval imports vLLM).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm_backend"))
    if args.dsv4_patch_only:
        from kakeya_v1_4_snapshot.dsv4_snapshot_hook import (
            install_dsv4_snapshot_patch,
        )
        install_dsv4_snapshot_patch()
    else:
        from kakeya_v1_4_snapshot.dsv4_snapshot_hook import (
            install_all_snapshot_patches_dsv4_aware,
        )
        install_all_snapshot_patches_dsv4_aware()

    # Stash the kv_stream_filter where the harness can see it. The
    # rigorous_eval harness reads HookState for per-layer routing; we
    # add a side-channel attribute for the filter flag so the final
    # rollup can subset the per-layer MSE dict.
    from kakeya_v1_4_snapshot.snapshot_hook import HookState
    HookState.dsv4_kv_stream_filter = args.kv_stream_filter  # type: ignore[attr-defined]

    # Forward to the real rigorous_eval main(). We re-exec sys.argv so
    # the harness's own argparse sees the forwarded flags only.
    sys.argv = [sys.argv[0]] + forwarded
    print(
        f"[dsv4-stage1] dsv4_kv_stream_filter={args.kv_stream_filter!r}"
        f" dsv4_patch_only={args.dsv4_patch_only}"
        f" forwarding to rigorous_eval with argv={forwarded}"
    )

    # Import + invoke the core harness
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rigorous_eval import main as rigorous_main  # type: ignore

    return rigorous_main()


if __name__ == "__main__":
    sys.exit(main())

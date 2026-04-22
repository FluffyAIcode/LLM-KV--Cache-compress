"""In-process kakeyaturbo reference codec.

This package is a pyo3 wrapper around the `kakeyaturbo` Rust crate.
It replaces the CLI-subprocess-over-disk-KKTV dance that the
`benchmarks/e2e_ppl_validation_vllm*.py` harnesses used to run (one
`kakeyaturbo-bench` fork per stream per layer per forward pass).

Semantic contract
-----------------

`roundtrip_layer(arr, **kwargs)` produces exactly the same decoded
tensor and exactly the same `mean_block_mse` that
`kakeyaturbo-bench --verify --dump-decoded` would produce when given
the same input KKTV file and the same CLI flags.  No randomness is
introduced by this wrapper — the only entropy source in the codec
is `rotation_seed`, which is exposed verbatim as a kwarg.

Typical use::

    from kakeyaturbo_py import roundtrip_layer
    decoded, report = roundtrip_layer(
        arr_f32_2d,
        metric="inner_product",
        block_size=512, bit_width=3,
        variance_ratio=0.95,
        pca_method="randomized",
        rsvd_target_rank=64, rsvd_oversample=8, rsvd_power_iters=2,
        rotation_seed=3405691582,
        centroids_file="reports/.../ds_K_b3_centroids.f32",
        outlier_threshold=2.0,
        share_basis=False,
    )
    # decoded has shape (M, D) with M = (N // block_size) * block_size
    # report is a dict: mean_block_mse, ratio_vs_bf16, ...

The Rust hot path releases the GIL, so multiple layers can be rolled
through in parallel from a thread pool if the caller wishes.  In the
in-forward vLLM harness the call is synchronous; the payoff there is
the elimination of `subprocess.Popen` + tmpfs I/O per layer per
forward pass.
"""
from ._core import (
    __version__,
    roundtrip_layer,
    # Primitive helpers for M4+ — byte-identical to the Rust reference
    # implementation, no re-derivation needed on the Python side.
    wht_sign_pattern,
    wht_rows,
    rotate_rows,
    inverse_rotate_rows,
    pack_bits,
    unpack_bits,
    centroids_gaussian,
    # Block-level structured encode / decode — exposes Rust's Skeleton +
    # Vec<Code> as numpy arrays.  The bridge that lets the PyTorch
    # reference and (eventually) Triton kernels consume Rust-fit
    # skeletons, so stages 2..=5 can be validated bit-exactly without
    # dragging skeleton-fit numerical noise into the diff.
    encode_block_codes,
    decode_block_from_parts,
)

__all__ = [
    "__version__",
    "roundtrip_layer",
    "wht_sign_pattern",
    "wht_rows",
    "rotate_rows",
    "inverse_rotate_rows",
    "pack_bits",
    "unpack_bits",
    "centroids_gaussian",
    "encode_block_codes",
    "decode_block_from_parts",
    # PyTorch reference encoder / decoder — lazy-imported to avoid
    # pulling torch into `import kakeyaturbo_py` in environments that
    # only need the primitives.
    "encode_block_torch_stage2",
    "decode_block_torch_from_parts",
    "Skeleton",
    "CodeBatch",
]


def __getattr__(name):
    if name in {
        "encode_block_torch_stage2",
        "decode_block_torch_from_parts",
        "Skeleton",
        "CodeBatch",
    }:
        from . import reference_torch  # noqa: F401
        return getattr(reference_torch, name)
    raise AttributeError(f"module 'kakeyaturbo_py' has no attribute {name!r}")

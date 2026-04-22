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
from ._core import roundtrip_layer, __version__

__all__ = ["roundtrip_layer", "__version__"]

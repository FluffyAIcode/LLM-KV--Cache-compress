"""Snapshot hooks for the v1.4 KakeyaLattice measurement harness.

This package provides the `snapshot_hook` module — three-phase
`forward` wrappers on Qwen3 / Qwen2 / Gemma4 / GLM Attention classes
that let the `benchmarks/multimodel_v14_*.py` harnesses capture
post-QK/V-norm, pre-RoPE K/V tensors, feed them through the v1.4
codec, and replace them inside a vLLM alt-forward pass to measure
|Δppl| on real FlashAttention output.

The hooks are installed via the `vllm.general_plugins` entry point
in `pyproject.toml`, gated by the env var `KAKEYA_SNAPSHOT_QWEN3=1`
(the env-var name is retained for historical compatibility with
existing harness scripts).

v1.4 does NOT require a custom vLLM AttentionBackend — it runs on
top of vanilla FlashAttention and records / replaces K/V inside the
unchanged forward path.  The previous v1.3 PPL backend code (which
did require a custom backend with streaming-RSVD stage-1 statistics)
has been removed since v1.4 makes it unnecessary.
"""

"""Unit tests for the DSV4 snapshot hook.

The hook monkey-patches the live vLLM ``DeepseekV4Attention`` class,
which is only available in the vLLM V4 wheel (PR #40760+). On dev
machines without that wheel we still want to validate:

1. The hook correctly invokes ``_snapshot_capture_replace`` under
   all three phases (capture / replace / inforward).
2. The ``_extract_layer_id_from_prefix`` helper correctly maps
   ``layers.<N>.self_attn``, ``mtp.<M>.attn``, and unrecognised
   prefixes to the right layer id.

For (1) we build a stand-in ``DeepseekV4Attention``-shaped Python
object with the minimum attributes the hook reads, monkey-patch its
forward using a slightly modified version of the real hook that
SKIPS the downstream custom-op + inverse-RoPE + FP8 einsum (those
only exist in the real wheel), and verify phase behaviour.

These tests run entirely on CPU and do NOT require CUDA, vLLM V4,
or the DeepSeek-V4 checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


# Make the vllm_backend package importable without a full vLLM install.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vllm_backend"))

from kakeya_v1_4_snapshot.dsv4_snapshot_hook import (  # noqa: E402
    _extract_layer_id_from_prefix,
)
from kakeya_v1_4_snapshot.snapshot_hook import HookState, _snapshot_capture_replace  # noqa: E402


# ---------------------------------------------------------------------------
# Layer-id extractor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix,expected", [
    ("model.layers.0.self_attn", 0),
    ("model.layers.7.self_attn", 7),
    ("model.layers.42.self_attn", 42),
    ("mtp.0.attn", 10_000),
    ("mtp.1.attn", 10_001),
    ("", 0),
    (None, 0),
    ("some.weird.prefix", 0),
])
def test_extract_layer_id(prefix, expected):
    assert _extract_layer_id_from_prefix(prefix) == expected


# ---------------------------------------------------------------------------
# Three-phase capture / replace / inforward against a stand-in
# DeepseekV4Attention object
# ---------------------------------------------------------------------------

class _FakeDSV4Attention:
    """Stand-in for vllm.model_executor.models.deepseek_v4.DeepseekV4Attention
    with just enough attributes for our hook's interception logic to
    run on CPU. We don't exercise the real downstream custom-op path;
    instead we call ``_snapshot_capture_replace`` directly via the
    simulation helper below.
    """

    def __init__(self, layer_id: int = 0, head_dim: int = 512,
                 q_lora_rank: int = 1024, n_heads: int = 64):
        self.layer_id = layer_id
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.n_heads = n_heads
        self.n_local_heads = n_heads
        self.prefix = f"model.layers.{layer_id}.self_attn"

        # Stand-in fused_wqa_wkv: output shape [N, q_lora_rank + head_dim]
        # produced by linear identity (so intercept sees a known value)
        self._fused_weight = torch.eye(4096, q_lora_rank + head_dim)

    def fused_wqa_wkv(self, hidden_states):
        # Returns (qr_kv, None) to match the real signature
        # [N, hidden] @ [hidden, qr+kv] → [N, qr+kv]
        qr_kv = hidden_states @ self._fused_weight
        return qr_kv, None


def _simulate_patched_forward(
    attn_obj: _FakeDSV4Attention, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Simulate the part of our patched forward() that intercepts
    kv between fused_wqa_wkv and the custom op, WITHOUT calling
    the real V4 custom op (which only exists in the V4 wheel).

    Returns the intercepted (possibly codec-mutated) kv for test
    inspection.
    """
    if HookState.phase == "off":
        return attn_obj.fused_wqa_wkv(hidden_states)[0]

    qr_kv, _ = attn_obj.fused_wqa_wkv(hidden_states)
    qr, kv = qr_kv.split(
        [attn_obj.q_lora_rank, attn_obj.head_dim], dim=-1,
    )

    HookState.head_size = attn_obj.head_dim
    HookState.num_kv_heads = 1
    HookState.num_heads = attn_obj.n_heads
    kv_new, _ = _snapshot_capture_replace(
        attn_obj.layer_id,
        kv,
        kv,
        nkv=1,
        hd=attn_obj.head_dim,
    )
    return kv_new


def _reset_hook_state():
    HookState.phase = "off"
    HookState.captured = {}
    HookState.replacements = {}
    HookState.replace_fired = {}
    HookState.replace_missing = {}
    HookState.replace_shape_mismatch = {}
    HookState.inforward_skip_layers = set()
    HookState.inforward_fired = {}
    HookState.codec_fn = None
    HookState.capture_gpu = False


def test_phase_off_is_transparent():
    _reset_hook_state()
    attn = _FakeDSV4Attention(layer_id=3)
    x = torch.randn(16, 4096)
    # HookState.phase is "off" → _simulate_patched_forward returns the
    # raw qr_kv output; kv is the last head_dim dims. We just check
    # nothing is captured / recorded.
    _simulate_patched_forward(attn, x)
    assert HookState.captured == {}
    assert HookState.replace_fired == {}
    assert HookState.inforward_fired == {}


def test_capture_records_kv_latent():
    _reset_hook_state()
    HookState.phase = "capture"
    HookState.capture_gpu = True
    attn = _FakeDSV4Attention(layer_id=5, head_dim=512, q_lora_rank=1024)
    x = torch.randn(16, 4096)
    kv_out = _simulate_patched_forward(attn, x)
    assert 5 in HookState.captured
    cap = HookState.captured[5]
    assert "K" in cap and "V" in cap
    # Shape check: the capture stores [N, nkv=1, hd=512]
    assert cap["K"].shape == (16, 1, 512)
    assert cap["K"].dtype == torch.float32
    # Shared-latent invariant: K tensor equals V tensor (we fed same
    # input to both in the hook).
    assert torch.equal(cap["K"], cap["V"])
    # kv_out should equal the input (no replacement)
    assert kv_out.shape == (16, 512)


def test_replace_splices_tensor_back():
    _reset_hook_state()
    HookState.phase = "replace"
    attn = _FakeDSV4Attention(layer_id=7, head_dim=512)
    # Pre-load a replacement: a known-good KV latent
    replacement = torch.full((16, 1, 512), fill_value=1.5)
    HookState.replacements[7] = {
        "K": replacement,
        "V": replacement,
    }
    x = torch.randn(16, 4096)
    kv_out = _simulate_patched_forward(attn, x)

    # kv_out should now be reshape(replacement) = [16, 512]
    assert kv_out.shape == (16, 512)
    assert torch.allclose(kv_out, torch.full((16, 512), 1.5))
    assert HookState.replace_fired.get(7) == 1


def test_replace_missing_layer_is_recorded():
    _reset_hook_state()
    HookState.phase = "replace"
    attn = _FakeDSV4Attention(layer_id=12, head_dim=512)
    # No replacement loaded for layer 12
    x = torch.randn(8, 4096)
    _simulate_patched_forward(attn, x)
    assert HookState.replace_missing.get(12) == 1
    assert 12 not in HookState.replace_fired


def test_inforward_roundtrip_through_codec():
    _reset_hook_state()
    HookState.phase = "inforward"

    # Codec: simple scale-by-0.5 roundtrip (visible in downstream)
    def codec_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 0.5

    HookState.codec_fn = codec_fn
    attn = _FakeDSV4Attention(layer_id=9, head_dim=512)
    x = torch.randn(8, 4096)

    # Recover the ORIGINAL kv by running with phase="off" first
    HookState.phase = "off"
    qr_kv_clean, _ = attn.fused_wqa_wkv(x)
    _, kv_clean = qr_kv_clean.split([attn.q_lora_rank, attn.head_dim], dim=-1)

    # Now run with phase="inforward"
    HookState.phase = "inforward"
    HookState.codec_fn = codec_fn
    kv_out = _simulate_patched_forward(attn, x)
    # Codec was applied: kv_out ≈ 0.5 * kv_clean
    # (The hook applies codec to both "K" and "V" which are aliases,
    #  then returns the "K" result.)
    assert torch.allclose(kv_out.flatten(0, -2), (kv_clean * 0.5).to(kv_out.dtype),
                          atol=1e-5)
    assert HookState.inforward_fired.get(9) == 1


def test_inforward_skip_layers_bypass_codec():
    _reset_hook_state()
    HookState.phase = "inforward"

    def codec_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 0.0  # Obviously destructive

    HookState.codec_fn = codec_fn
    HookState.inforward_skip_layers = {0, 1, 41, 42}  # boundary layers

    attn_boundary = _FakeDSV4Attention(layer_id=0, head_dim=512)
    x = torch.randn(8, 4096)
    # Original kv for reference (phase=off)
    HookState.phase = "off"
    qr_kv_clean, _ = attn_boundary.fused_wqa_wkv(x)
    _, kv_clean = qr_kv_clean.split(
        [attn_boundary.q_lora_rank, attn_boundary.head_dim], dim=-1)
    # Run boundary layer under inforward
    HookState.phase = "inforward"
    HookState.codec_fn = codec_fn
    HookState.inforward_skip_layers = {0, 1, 41, 42}
    kv_out = _simulate_patched_forward(attn_boundary, x)
    # Boundary layer 0 should pass through unchanged
    assert torch.allclose(kv_out, kv_clean)
    # inforward_fired should NOT be incremented for a boundary skip
    assert HookState.inforward_fired.get(0, 0) == 0


def test_patched_attribute_marker():
    """If the real vLLM V4 wheel is installed the hook should install
    idempotently. Skip gracefully when it isn't."""
    try:
        import vllm.model_executor.models.deepseek_v4 as mod  # noqa: F401
    except ImportError:
        pytest.skip("vLLM V4 wheel not installed; live-patch test skipped.")

    from kakeya_v1_4_snapshot.dsv4_snapshot_hook import install_dsv4_snapshot_patch
    install_dsv4_snapshot_patch()
    install_dsv4_snapshot_patch()  # idempotent
    import vllm.model_executor.models.deepseek_v4 as mod
    assert getattr(mod.DeepseekV4Attention, "_kk_snapshot_patched", False)


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))

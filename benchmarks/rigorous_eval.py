"""Rigorous in-forward eval harness: v1.4 vs TQ + ablations.

Upgraded protocol vs multimodel_v14_kv_128k_report:
  * n_passages >= 32 by default (vs the old 4 or 8)
  * reports mean ± 95% CI using Student's t
  * K-only, V-only, K+V modes selectable per channel
  * ablation codec variants supported (v14_full / no_unit_norm / no_hadamard /
    no_per_vec_qmax / per_block_qmax / scalar_quantise)
  * honest in-forward semantics: codec runs INSIDE each layer's forward
    (HookState.codec_fn / .phase=="inforward"), so cross-layer error
    accumulation IS included in Δppl.  A separate --mode snapshot flag
    runs the old snapshot semantics for apples-to-apples comparison.

Compliance: no mock / no simplification / no fallback / no overfit.

  * head_dim % 4 != 0 → raise (no silent fallback)
  * fires < expected → mark channel fatal (no silent passthrough)
  * no curve fitting in iso-PPL argmin-bits reporting
  * same boundary layer policy (first 2 + last 2 bf16) as prior harnesses,
    can be overridden with --no-boundary to include boundary layers in
    the codec sweep (for the boundary ablation measurement)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")


def raw_bits_per_token_per_head(head_dim: int) -> int:
    return 16 * head_dim


def _sylvester_hadamard_normalised(D: int, device) -> torch.Tensor:
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], dim=0)
    return H / math.sqrt(D)


def make_v14_codec_fn(D: int, q_range: int, device: str = "cuda"):
    """Canonical v1.4 KakeyaLattice codec_fn."""
    from kakeyaturbo_py import V14KakeyaZamirLatticeGPU
    if D % 4 != 0:
        raise ValueError(
            f"v1.4 requires head_dim % 4 == 0, got {D}. No fallback.",
        )
    cb = V14KakeyaZamirLatticeGPU(D=D, q_range=q_range, device=device)

    def codec_fn(X: torch.Tensor) -> torch.Tensor:
        assert X.is_cuda and X.dtype == torch.float32
        return cb.roundtrip(X)

    codec_fn.bits_per_token_per_head = cb.bits_per_token_per_head
    codec_fn.label = f"v1.4 Q={q_range}"
    codec_fn.channel_id = f"v14_Q{q_range}"
    return codec_fn


def make_tq_codec_fn(D: int, bits_per_coord: int, device: str = "cuda"):
    """TurboQuant K+V codec (Hadamard + fp16 qmax + uniform scalar quant).

    Bit-identical to the TQ channels in prior harnesses for apples-to-apples.
    """
    H = _sylvester_hadamard_normalised(D, device)
    qs = (1 << (bits_per_coord - 1)) - 1
    eps = torch.finfo(torch.float32).eps
    bits = D * bits_per_coord + 32

    def codec_fn(X: torch.Tensor) -> torch.Tensor:
        assert X.is_cuda and X.dtype == torch.float32
        N_tok, H_heads, _ = X.shape
        flat = X.reshape(-1, D)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        norms_f16 = norms.to(torch.float16).to(torch.float32)
        unit = flat / norms
        y = unit @ H
        qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
        qmax_f16 = qmax.to(torch.float16).to(torch.float32)
        scale = qmax_f16 / float(qs)
        q = torch.round(y / scale).clamp(-qs, qs) * scale
        unit_hat = q @ H
        return (unit_hat * norms_f16).reshape(N_tok, H_heads, D)

    codec_fn.bits_per_token_per_head = bits
    codec_fn.label = f"TQ b={bits_per_coord}"
    codec_fn.channel_id = f"tq_b{bits_per_coord}"
    return codec_fn


def make_ablation_codec_fn(variant: str, D: int, q_range: int,
                           device: str = "cuda"):
    """Ablation variant codec_fn (from kakeyaturbo_py.ablation_codecs)."""
    from kakeyaturbo_py.ablation_codecs import make_ablation_codec
    fn = make_ablation_codec(variant, D=D, q_range=q_range, device=device)
    return fn


def make_kv_masked_codec(
    base_fn: Callable[[torch.Tensor], torch.Tensor],
    mode: str,
):
    """Wrap a codec so it applies to K only, V only, or both.

    mode ∈ {"K", "V", "KV"}.  The returned fn dispatches on a tag argument
    passed by the harness at call time, indicating whether the current
    tensor is a K or V.  Pass-through for the other side.
    """
    if mode not in ("K", "V", "KV"):
        raise ValueError(f"mode must be K/V/KV, got {mode!r}")

    def fn(X: torch.Tensor, *, tag: str) -> torch.Tensor:
        if mode == "KV":
            return base_fn(X)
        if (mode == "K" and tag == "K") or (mode == "V" and tag == "V"):
            return base_fn(X)
        # Pass through: bf16 round-trip to reflect the storage precision
        # that a K-only/V-only scheme would actually use for the other side.
        return X.to(torch.bfloat16).to(torch.float32)

    fn.bits_per_token_per_head = base_fn.bits_per_token_per_head
    fn.label = f"{base_fn.label} [{mode}]"
    fn.channel_id = f"{base_fn.channel_id}_{mode}"
    fn.mode = mode
    return fn


# ------------------------------------------------------------------
# Statistical helpers
# ------------------------------------------------------------------
def mean_ci95(xs: list[float]) -> tuple[float, float, float]:
    """Return (mean, half_width_95, std).  Uses Student's t since n may be
    small.  For n==1 returns (x, 0, 0).
    """
    if len(xs) <= 1:
        return (xs[0] if xs else 0.0, 0.0, 0.0)
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(len(xs))
    # t critical value at 95% two-sided; tabulated for a few n.
    n = len(xs)
    # Approximate t_{0.975, n-1}: use scipy if present, else table.
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 32: 2.037,
        40: 2.021, 50: 2.009, 60: 2.000, 100: 1.984,
    }
    df = n - 1
    # Choose the largest df <= our df.
    keys_le = [k for k in t_table if k <= df]
    k = max(keys_le) if keys_le else 1
    t_crit = t_table[k]
    # If df is larger than table's max, use 1.96 (normal limit).
    if df > 100:
        t_crit = 1.96
    return (mu, t_crit * se, sd)


# ------------------------------------------------------------------
# WikiText + PPL utilities
# ------------------------------------------------------------------
def load_wikitext_passages(tok: Any, min_tokens: int, n_passages: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    passages, cur, approx = [], [], 0
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        cur.append(text)
        approx += int(len(text.split()) * 1.3)
        if approx >= min_tokens:
            passage = "".join(cur)
            if len(tok.encode(passage)) >= min_tokens:
                passages.append(passage)
                if len(passages) >= n_passages:
                    return passages
            cur, approx = [], 0
    return passages


def prompt_logprobs_for_ids(llm: Any, ids: list[int]) -> list[Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp, use_tqdm=False,
    )
    return out[0].prompt_logprobs


def ppl_and_top1(pls, ids, start, end) -> tuple[float, list[int]]:
    lps: list[float] = []
    top1_ids: list[int] = []
    for t in range(start, end):
        entry = pls[t]
        if entry is None:
            lps.append(0.0)
            top1_ids.append(-1)
            continue
        gold_id = ids[t]
        lps.append(entry[gold_id].logprob)
        best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
        top1_ids.append(best_id)
    mean_nll = -statistics.fmean(lps)
    return float(math.exp(mean_nll)), top1_ids


def default_boundary_for_model(num_layers: int) -> set[int]:
    return set(list(range(2)) + list(range(num_layers - 2, num_layers)))


def fmt_bytes(b: float) -> str:
    gi = b / 1024**3
    if gi >= 1.0:
        return f"{gi:.2f} GiB"
    return f"{b/1024**2:.0f} MiB"


# ------------------------------------------------------------------
# Channel wrappers: integrate codec_fn with the mode (K/V/KV).
# ------------------------------------------------------------------
def install_inforward_codec(
    hookstate_cls, codec_fn_kv_aware,
    inforward_skip_layers: set[int],
):
    """Wire HookState.codec_fn for a specific channel + mode.

    The snapshot hook module has a single `HookState.codec_fn` slot.  We
    inject a function that dispatches on the tag set by the model patch
    (currently the hook doesn't tag K vs V in the inforward branch, so we
    wrap by allocating each roundtrip's `tag` via a thread-local hint that
    the harness updates between the K encode and the V encode).

    CURRENT LIMITATION: the inforward path in snapshot_hook calls
    codec_fn(K) then codec_fn(V) back-to-back, so we can disambiguate by
    setting a module-level flag between the two.  That's fragile; the
    cleanest fix is to extend the hook interface to pass a tag.  We'll do
    the cleanest fix.
    """
    # The harness asks the hook to call codec_fn twice: once for K, once
    # for V.  Currently the hook doesn't distinguish.  We solve this by
    # adding the distinction in the hook itself (see K/V aware codec_fn).
    raise NotImplementedError(
        "Install wiring moved to harness main() — see install_kv_aware_codec",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--mode", choices=["inforward", "snapshot"], default="inforward")
    ap.add_argument("--q-values", type=str, default="10,38,152",
                    help="v1.4 Q_range sweep values (comma-sep)")
    ap.add_argument("--tq-b-values", type=str, default="4,6,8",
                    help="TQ bits_per_coord sweep (comma-sep)")
    ap.add_argument("--kv-modes", type=str, default="KV,K,V",
                    help="Comma-separated: K, V, KV — which tensors to compress")
    ap.add_argument("--ablation-variants", type=str, default="",
                    help="Comma-separated ablation variants to also run "
                         "(uses the first Q value)")
    ap.add_argument("--no-boundary", action="store_true",
                    help="Disable boundary-layer protection (include all layers "
                         "in the codec sweep — for boundary ablation)")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=32)
    ap.add_argument("--report-ctx-tokens", type=int, default=128 * 1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    Q_VALUES  = [int(x) for x in args.q_values.split(",")    if x.strip()]
    TQ_VALUES = [int(x) for x in args.tq_b_values.split(",") if x.strip()]
    KV_MODES  = [m.strip() for m in args.kv_modes.split(",") if m.strip()]
    ABL_VARIANTS = [v.strip() for v in args.ablation_variants.split(",")
                    if v.strip()]
    print(f"[config] mode: {args.mode}", flush=True)
    print(f"[config] v1.4 Q:        {Q_VALUES}", flush=True)
    print(f"[config] TQ   b:        {TQ_VALUES}", flush=True)
    print(f"[config] KV modes:      {KV_MODES}", flush=True)
    print(f"[config] ablations:     {ABL_VARIANTS or '(none)'}", flush=True)
    print(f"[config] no_boundary:   {args.no_boundary}", flush=True)
    print(f"[config] n_passages:    {args.n_passages}", flush=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    from kakeya_v1_4_snapshot.snapshot_hook import HookState

    HookState.capture_gpu = True

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=args.model_path,
        max_model_len=args.ctx_len + args.n_eval + 1,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
    )

    passages = load_wikitext_passages(
        tok, min_tokens=args.ctx_len + args.n_eval,
        n_passages=args.n_passages,
    )
    passages_ids = [tok.encode(p)[: args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    if not passages_ids:
        print("[ERROR] No passages long enough.", flush=True)
        return 1
    print(f"[data] loaded {len(passages_ids)} passages", flush=True)

    # ---- Discover model config via a single capture pass ----
    HookState.phase = "capture"
    HookState.captured = {}
    _ = prompt_logprobs_for_ids(llm, passages_ids[0][: args.ctx_len])
    HookState.phase = "off"
    cap_dry = dict(HookState.captured)
    num_layers = max(cap_dry.keys()) + 1 if cap_dry else 0
    if num_layers == 0:
        print(f"[ERROR] Snapshot hook didn't fire for {args.model_path}.",
              flush=True)
        return 1

    # Per-layer head_dim discovery.  Gemma-4 MatFormer alternates
    # sliding-attention (hd=256) and full-attention (hd=512) layers,
    # so we can't assume a single hd for the whole model.  Build an
    # integer→(hd, num_kv_heads) map from the capture pass.  Layers
    # that were NOT captured (e.g. kv_shared layers) are handled by
    # being skipped in the sweep (they pass through bf16 unchanged).
    per_layer_shape: dict[int, tuple[int, int]] = {}
    for lid, kv in cap_dry.items():
        hd = kv['K'].shape[-1]
        nkv = kv['K'].shape[1]
        per_layer_shape[lid] = (hd, nkv)

    distinct_hds = sorted({hd for hd, _ in per_layer_shape.values()})
    distinct_nkvs = sorted({nkv for _, nkv in per_layer_shape.values()})
    print(f"[model] {args.model_path}", flush=True)
    print(f"[model] L={num_layers}  captured={len(per_layer_shape)}  "
          f"distinct_hds={distinct_hds}  distinct_kvh={distinct_nkvs}",
          flush=True)
    if len(distinct_hds) > 1:
        print(f"[model] VARIABLE head_dim across layers — per-layer "
              f"codec dispatch active", flush=True)
        hd_counts = {}
        for hd, _ in per_layer_shape.values():
            hd_counts[hd] = hd_counts.get(hd, 0) + 1
        for hd, cnt in sorted(hd_counts.items()):
            print(f"[model]   hd={hd}: {cnt} layer(s)", flush=True)

    for hd in distinct_hds:
        if hd % 4 != 0:
            print(f"[FAIL] head_dim {hd} not divisible by 4.", flush=True)
            return 2

    # Legacy scalars for reporting: use the MAJORITY head_dim (largest
    # number of layers) as the "representative" for the aggregate
    # byte counts.  Per-channel bookkeeping uses per-layer hd below.
    majority_hd = max(distinct_hds, key=lambda h:
                      sum(1 for lhd, _ in per_layer_shape.values() if lhd == h))
    majority_nkv = max(distinct_nkvs, key=lambda n:
                       sum(1 for _, lnkv in per_layer_shape.values() if lnkv == n))
    head_dim = majority_hd
    num_kv_heads = majority_nkv

    # Save one reference K/V per distinct head_dim for pure-codec
    # rel-MSE (instead of a single layer-0 reference, which would
    # mean rel-MSE only covers one head_dim on a heterogeneous model).
    ref_K_by_hd: dict[int, torch.Tensor] = {}
    ref_V_by_hd: dict[int, torch.Tensor] = {}
    for lid, kv in cap_dry.items():
        hd = kv['K'].shape[-1]
        if hd not in ref_K_by_hd:
            ref_K_by_hd[hd] = kv['K'].clone()
            ref_V_by_hd[hd] = kv['V'].clone()
    del cap_dry
    HookState.captured = {}
    torch.cuda.empty_cache()

    if args.no_boundary:
        boundary = set()
    else:
        boundary = default_boundary_for_model(num_layers)
    # Layers that weren't captured (e.g. Gemma-4 kv_shared) are always
    # skipped by the hook naturally; we also add them to boundary for
    # bookkeeping so the "expected fires" count below is correct.
    uncaptured = set(range(num_layers)) - set(per_layer_shape.keys())
    if uncaptured:
        print(f"[model] uncaptured layers (kv_shared/other): "
              f"{sorted(uncaptured)}", flush=True)
    effective_boundary = boundary | uncaptured
    n_boundary_like = len(effective_boundary)
    n_compressed = num_layers - n_boundary_like

    raw_bits_per_h = raw_bits_per_token_per_head(head_dim)
    raw_bytes_per_h = raw_bits_per_h // 8

    # ---- Build channel list — PER-head_dim codec registry ----
    # For each (kind, Q/b) we build one codec_fn per distinct hd in the
    # model.  The per-layer dispatch wrapper (_make_per_layer_codec)
    # picks the right one at call time based on the input tensor shape.
    def _make_per_layer_codec(
        per_hd_fns: dict[int, Any], channel_id: str, label: str,
    ):
        """Wrap per-hd codec_fns into a single fn that dispatches on D.

        All wrapped fns must share the same per-token bit budget at a
        given (variant, Q/b).  Per-layer bit-budget variation is
        accounted for in the storage calculation (uses per_layer_shape).
        """
        # Representative bit budget for display (picked on majority hd).
        rep_fn = per_hd_fns[majority_hd]
        def fn(X: torch.Tensor) -> torch.Tensor:
            D = X.shape[-1]
            hd_fn = per_hd_fns.get(D)
            if hd_fn is None:
                raise RuntimeError(
                    f"per-layer codec dispatch: no codec for head_dim={D}; "
                    f"available: {sorted(per_hd_fns.keys())}. "
                    f"No fallback by design."
                )
            return hd_fn(X)
        # Use the majority hd's bit budget for the summary reporting.
        fn.bits_per_token_per_head = rep_fn.bits_per_token_per_head
        fn.channel_id = channel_id
        fn.label = label
        fn.per_hd_bits = {hd: f.bits_per_token_per_head
                          for hd, f in per_hd_fns.items()}
        return fn

    def make_base_codec_fns():
        out: list[tuple[str, Any]] = []
        for Q in Q_VALUES:
            per_hd = {hd: make_v14_codec_fn(hd, Q) for hd in distinct_hds}
            out.append(("v14",
                        _make_per_layer_codec(
                            per_hd, f"v14_Q{Q}", f"v1.4 Q={Q}",
                        )))
        for b in TQ_VALUES:
            per_hd = {hd: make_tq_codec_fn(hd, b) for hd in distinct_hds}
            out.append(("tq",
                        _make_per_layer_codec(
                            per_hd, f"tq_b{b}", f"TQ b={b}",
                        )))
        for variant in ABL_VARIANTS:
            Q = Q_VALUES[0] if Q_VALUES else 38
            per_hd = {hd: make_ablation_codec_fn(variant, hd, Q)
                      for hd in distinct_hds}
            out.append(("ablation",
                        _make_per_layer_codec(
                            per_hd, f"{variant}_Q{Q}",
                            f"{variant} Q={Q}",
                        )))
        return out

    base_codecs = make_base_codec_fns()

    # Channels: bf16_ref + base_codec × mode + (for snapshot mode, the
    # "bf16_pass" tight reference that round-trips bf16 through the codec
    # slot to measure FA bf16 noise floor).
    channels: list[tuple[str, str, Any, str | None]] = [
        ("bf16_ref", "bf16", None, None),
    ]
    for kind, codec_fn in base_codecs:
        for kv_mode in KV_MODES:
            ch_id = f"{codec_fn.channel_id}_{kv_mode}"
            masked = make_kv_masked_codec(codec_fn, kv_mode)
            channels.append((ch_id, kind, codec_fn, kv_mode))

    print(f"[channels] {len(channels)} total", flush=True)

    # Per-channel × per-passage storage.
    raw_records: list[dict] = []

    def run_one_channel(
        ids: list[int],
        pi: int,
        ch_id: str,
        kind: str,
        codec_fn: Any,
        kv_mode: str | None,
        ppl_ref: float,
        ref_top1: list[int],
    ):
        """Run a single (passage, channel) pair.  For bf16_ref the Δppl
        is 0 by construction.  For the other channels we run the
        appropriate pass and measure."""
        # bf16 reference channel: zero delta by construction.
        if ch_id == "bf16_ref":
            raw_records.append({
                "passage": pi, "channel": ch_id, "kind": kind,
                "kv_mode": None,
                "k_bits": raw_bits_per_h, "v_bits": raw_bits_per_h,
                "delta_ppl": 0.0, "top1_pair": 1.0,
                "k_mse_layer0": 0.0, "v_mse_layer0": 0.0,
                "ppl_ref": ppl_ref, "ppl_alt": ppl_ref,
                "fire_count": 0,
            })
            return

        if args.mode == "inforward":
            # Build a closure that applies codec_fn only to the selected
            # side (K or V or both) based on kv_mode.
            def tagged_codec(X: torch.Tensor) -> torch.Tensor:
                # HookState calls codec_fn(K) then codec_fn(V) back-to-back;
                # we distinguish by a tiny state machine flip.  This is
                # deterministic because K is ALWAYS called first in
                # _snapshot_capture_replace's inforward branch.
                nonlocal _call_counter
                is_k = (_call_counter % 2 == 0)
                _call_counter += 1
                if kv_mode == "KV":
                    return codec_fn(X)
                if kv_mode == "K" and is_k:
                    return codec_fn(X)
                if kv_mode == "V" and not is_k:
                    return codec_fn(X)
                # Pass-through: bf16 round-trip to match storage precision.
                return X.to(torch.bfloat16).to(torch.float32)

            _call_counter = 0

            HookState.phase = "inforward"
            HookState.codec_fn = tagged_codec
            # Skip both user-configured boundary layers AND layers that
            # weren't captured (e.g. Gemma-4 kv_shared layers that reuse
            # another layer's K/V cache; see gemma4.py is_kv_shared_layer).
            HookState.inforward_skip_layers = set(effective_boundary)
            HookState.inforward_fired = {}

            t0 = time.perf_counter()
            alt_pls = prompt_logprobs_for_ids(llm, ids)
            t_alt = time.perf_counter() - t0

            HookState.phase = "off"
            HookState.codec_fn = None

            n_fired = sum(HookState.inforward_fired.values())
            # One forward call per layer increments the counter once
            # (the hook calls codec_fn for K and V back-to-back within
            # a single layer.forward(); see snapshot_hook.py
            # _snapshot_capture_replace).
            expected = n_compressed
            if n_fired < expected:
                raw_records.append({
                    "passage": pi, "channel": ch_id, "kind": kind,
                    "kv_mode": kv_mode,
                    "fatal": f"fires {n_fired} < expected {expected}",
                    "fire_count": n_fired,
                })
                return

        elif args.mode == "snapshot":
            # Snapshot mode: we need captured K/V for this passage.  We
            # reuse the per-passage cache (see run_all_channels_for_passage).
            captured = _snapshot_cache["captured"]
            replacements: dict[int, dict[str, torch.Tensor]] = {}
            for lid, kv in captured.items():
                K_g, V_g = kv["K"], kv["V"]
                # Skip boundary layers (keep bf16) AND uncaptured-by-design
                # layers (Gemma-4 kv_shared re-use earlier layer's cache).
                if lid in effective_boundary:
                    replacements[lid] = {"K": K_g, "V": V_g}
                    continue
                if kv_mode in ("K", "KV"):
                    K_hat = codec_fn(K_g)
                else:
                    K_hat = K_g.to(torch.bfloat16).to(torch.float32)
                if kv_mode in ("V", "KV"):
                    V_hat = codec_fn(V_g)
                else:
                    V_hat = V_g.to(torch.bfloat16).to(torch.float32)
                replacements[lid] = {"K": K_hat, "V": V_hat}
            torch.cuda.synchronize()

            HookState.phase = "replace"
            HookState.replacements = replacements
            HookState.replace_fired = {}
            HookState.replace_shape_mismatch = {}
            HookState.replace_missing = {}

            t0 = time.perf_counter()
            alt_pls = prompt_logprobs_for_ids(llm, ids)
            t_alt = time.perf_counter() - t0
            HookState.phase = "off"
            HookState.replacements = {}

            n_fired = sum(HookState.replace_fired.values())
            expected = num_layers  # every layer's forward fires in replace
            if n_fired < expected - 4:  # boundary may have 0-fires, tolerate
                raw_records.append({
                    "passage": pi, "channel": ch_id, "kind": kind,
                    "kv_mode": kv_mode,
                    "fatal": f"fires {n_fired} < expected ≥{expected-4}",
                    "fire_count": n_fired,
                })
                return
        else:
            raise RuntimeError(f"unknown mode {args.mode!r}")

        ppl_alt, alt_top1 = ppl_and_top1(
            alt_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        top1_pair = float(
            sum(1 for a, r in zip(alt_top1, ref_top1)
                if a == r and a != -1) / max(len(alt_top1), 1)
        )
        delta_ppl = (ppl_alt - ppl_ref) / max(ppl_ref, 1e-9)

        # Pure-codec rel-MSE: average across distinct head_dims (not just
        # layer 0), so Gemma-4 sliding (hd=256) and full (hd=512) both
        # contribute. Weighted by the number of layers at each hd.
        def _compute_rel_mse(refs_by_hd: dict) -> float:
            if not refs_by_hd:
                return 0.0
            num, den = 0.0, 0.0
            for hd, X_ref in refs_by_hd.items():
                n_layers_at_hd = sum(1 for lhd, _ in per_layer_shape.values()
                                     if lhd == hd)
                X_hat = codec_fn(X_ref)
                diff_sq = ((X_hat - X_ref) ** 2).sum(-1).mean()
                norm_sq = (X_ref ** 2).sum(-1).mean().clamp(min=1e-12)
                num += float(diff_sq.item()) * n_layers_at_hd
                den += float(norm_sq.item()) * n_layers_at_hd
            return num / max(den, 1e-20)

        if kv_mode in ("K", "KV"):
            k_mse = _compute_rel_mse(ref_K_by_hd)
        else:
            k_mse = 0.0
        if kv_mode in ("V", "KV"):
            v_mse = _compute_rel_mse(ref_V_by_hd)
        else:
            v_mse = 0.0

        # Per-hd bit accounting: for each layer, its K (and V) bit budget
        # depends on that layer's head_dim.  We report the MAJORITY hd's
        # bit count as the scalar `k_bits` / `v_bits` for the summary
        # table, but the storage (kakeya_bytes_128k) below sums per-layer.
        k_bits = codec_fn.bits_per_token_per_head if kv_mode in ("K", "KV") else raw_bits_per_h
        v_bits = codec_fn.bits_per_token_per_head if kv_mode in ("V", "KV") else raw_bits_per_h

        raw_records.append({
            "passage": pi, "channel": ch_id, "kind": kind,
            "kv_mode": kv_mode,
            "k_bits": k_bits, "v_bits": v_bits,
            "delta_ppl": delta_ppl, "top1_pair": top1_pair,
            "k_mse_layer0": k_mse, "v_mse_layer0": v_mse,
            "ppl_ref": ppl_ref, "ppl_alt": ppl_alt,
            "t_alt": t_alt,
            "fire_count": n_fired,
        })

    # ---- Run passages ----
    _snapshot_cache: dict[str, Any] = {}
    for pi, ids in enumerate(passages_ids):
        print(f"\n=== passage {pi + 1}/{len(passages_ids)} ===", flush=True)

        # bf16 ref pass + optional capture.
        if args.mode == "snapshot":
            HookState.phase = "capture"
            HookState.captured = {}
            t0 = time.perf_counter()
            ref_pls = prompt_logprobs_for_ids(llm, ids)
            t_ref = time.perf_counter() - t0
            HookState.phase = "off"
            captured = dict(HookState.captured)
            _snapshot_cache["captured"] = captured
        else:
            HookState.phase = "off"
            t0 = time.perf_counter()
            ref_pls = prompt_logprobs_for_ids(llm, ids)
            t_ref = time.perf_counter() - t0

        ppl_ref, ref_top1 = ppl_and_top1(
            ref_pls, ids, args.ctx_len, args.ctx_len + args.n_eval,
        )
        print(f"  [ref] ppl={ppl_ref:.3f}  ({t_ref:.2f}s)", flush=True)

        # Each channel.
        for ch_id, kind, codec_fn, kv_mode in channels:
            try:
                run_one_channel(
                    ids, pi, ch_id, kind, codec_fn, kv_mode,
                    ppl_ref, ref_top1,
                )
            except Exception as e:
                print(f"  [{ch_id:<32}] ERROR: {type(e).__name__}: {e}",
                      flush=True)
                raw_records.append({
                    "passage": pi, "channel": ch_id, "kind": kind,
                    "kv_mode": kv_mode,
                    "fatal": f"{type(e).__name__}: {e}",
                })
                continue

            # Find the last record and print it.
            r = raw_records[-1]
            if "fatal" in r:
                print(f"  [{ch_id:<32}] FATAL: {r['fatal']}", flush=True)
                continue
            print(
                f"  [{ch_id:<32}] "
                f"Δppl={r['delta_ppl']*100:+7.3f}% "
                f"top1={r['top1_pair']*100:6.2f}% "
                f"K-MSE(l0)={r['k_mse_layer0']:.2e} "
                f"V-MSE(l0)={r['v_mse_layer0']:.2e} "
                f"fires={r.get('fire_count', '?')}",
                flush=True,
            )

        # Release snapshot cache for this passage.
        _snapshot_cache.clear()
        HookState.captured = {}
        torch.cuda.empty_cache()

    # ---- Aggregate with CIs ----
    by_ch: dict[str, list[dict]] = {}
    for r in raw_records:
        if "fatal" in r:
            continue
        by_ch.setdefault(r["channel"], []).append(r)

    agg_rows: list[dict] = []
    for ch, rs in by_ch.items():
        if not rs:
            continue
        abs_dppl = [abs(r["delta_ppl"]) for r in rs]
        dppl = [r["delta_ppl"] for r in rs]
        top1 = [r["top1_pair"] for r in rs]
        k_mse = [r["k_mse_layer0"] for r in rs]
        v_mse = [r["v_mse_layer0"] for r in rs]

        mu_abs, ci_abs, sd_abs = mean_ci95(abs_dppl)
        mu_d, ci_d, sd_d = mean_ci95(dppl)
        mu_t1, ci_t1, sd_t1 = mean_ci95(top1)
        mu_km, _, sd_km = mean_ci95(k_mse)
        mu_vm, _, sd_vm = mean_ci95(v_mse)

        sample = rs[0]
        k_bits = sample["k_bits"]
        v_bits = sample["v_bits"]

        # Per-layer storage accounting (handles heterogeneous head_dim
        # across layers for Gemma-4 MatFormer).  For each layer:
        #   baseline = num_kv_heads × (raw_K + raw_V)  with that layer's hd
        #   kakeya:
        #     if layer is boundary OR uncaptured → keep bf16
        #     else → use the codec's per-hd bit budget (K and/or V
        #            depending on kv_mode)
        kv_mode_for_ch = sample.get("kv_mode")
        k_bits_bf16 = raw_bits_per_h  # legacy — but we need per-layer
        # Find the channel's codec_fn to look up per_hd_bits.
        ch_codec_fn = None
        for _k, _cf in base_codecs:
            if _cf.channel_id == ch.replace(
                    f"_{kv_mode_for_ch}" if kv_mode_for_ch else "", ""):
                ch_codec_fn = _cf
                break
        per_hd_bits = getattr(ch_codec_fn, "per_hd_bits",
                              {hd: 16 * hd for hd in distinct_hds}) \
            if ch_codec_fn else {hd: 16 * hd for hd in distinct_hds}

        baseline_per_tok = 0
        kakeya_per_tok = 0
        for lid in range(num_layers):
            hd_lid, nkv_lid = per_layer_shape.get(lid, (majority_hd, majority_nkv))
            raw_bytes_this = 16 * hd_lid // 8  # bf16 per-head per-token
            baseline_per_tok += nkv_lid * (raw_bytes_this * 2)  # K + V
            if ch == "bf16_ref" or lid in effective_boundary or lid not in per_layer_shape:
                kakeya_per_tok += nkv_lid * (raw_bytes_this * 2)
            else:
                codec_bits = per_hd_bits.get(hd_lid, 16 * hd_lid)
                kbytes = codec_bits // 8 if kv_mode_for_ch in ("K", "KV") else raw_bytes_this
                vbytes = codec_bits // 8 if kv_mode_for_ch in ("V", "KV") else raw_bytes_this
                kakeya_per_tok += nkv_lid * (kbytes + vbytes)
        baseline_128k = baseline_per_tok * args.report_ctx_tokens
        kakeya_128k   = kakeya_per_tok   * args.report_ctx_tokens

        agg_rows.append({
            "channel": ch,
            "kind":    sample["kind"],
            "kv_mode": sample.get("kv_mode"),
            "n":       len(rs),
            "k_bits":  k_bits, "v_bits": v_bits,
            "baseline_bytes_128k": baseline_128k,
            "kakeya_bytes_128k":   kakeya_128k,
            "total_ratio_128k":    baseline_128k / kakeya_128k,
            "mean_abs_delta_ppl":  mu_abs, "ci95_abs_delta_ppl": ci_abs,
            "std_abs_delta_ppl":   sd_abs,
            "mean_delta_ppl":      mu_d, "ci95_delta_ppl": ci_d,
            "std_delta_ppl":       sd_d,
            "mean_top1_pair":      mu_t1, "ci95_top1_pair": ci_t1,
            "mean_k_mse_layer0":   mu_km, "std_k_mse_layer0": sd_km,
            "mean_v_mse_layer0":   mu_vm, "std_v_mse_layer0": sd_vm,
        })

    # ---- Pretty print ----
    print()
    print("=" * 128)
    print(f"v1.4 KakeyaLattice rigorous eval — model={args.model_path}  "
          f"mode={args.mode}  n={args.n_passages}")
    print(f"Boundary skip: {sorted(boundary) if boundary else '(none)'}  "
          f"(codec on {n_compressed}/{num_layers} layers; "
          f"uncaptured-by-arch: {sorted(uncaptured) if uncaptured else '(none)'})")
    print("=" * 128)
    print(f"{'Channel':<30} {'KV':>4} {'n':>3} "
          f"{'CR':>6} {'|Δppl|±95%CI':>18} {'Δppl±95%CI':>18} "
          f"{'top1%':>7} {'K-MSE(l0)':>12} {'V-MSE(l0)':>12}")
    print("-" * 128)
    for r in agg_rows:
        mode_tag = r["kv_mode"] or "-"
        ci_abs = r["ci95_abs_delta_ppl"] * 100
        mu_abs = r["mean_abs_delta_ppl"] * 100
        ci_d = r["ci95_delta_ppl"] * 100
        mu_d = r["mean_delta_ppl"] * 100
        print(f"{r['channel']:<30} {mode_tag:>4} {r['n']:>3} "
              f"{r['total_ratio_128k']:>5.2f}× "
              f"{mu_abs:6.3f}%±{ci_abs:5.3f}% "
              f"{mu_d:+6.3f}%±{ci_d:5.3f}% "
              f"{r['mean_top1_pair']*100:6.2f}% "
              f"{r['mean_k_mse_layer0']:12.3e} "
              f"{r['mean_v_mse_layer0']:12.3e}")

    out = {
        "mode": args.mode,
        "model": args.model_path,
        "model_name": args.model_name,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "distinct_head_dims": distinct_hds,
        "distinct_num_kv_heads": distinct_nkvs,
        "per_layer_shape": {str(lid): {"hd": hd, "kvh": nkvh}
                            for lid, (hd, nkvh) in per_layer_shape.items()},
        "uncaptured_layers": sorted(uncaptured),
        "effective_boundary_layers": sorted(effective_boundary),
        "boundary_skip_layers": sorted(boundary),
        "no_boundary": args.no_boundary,
        "raw_bits_per_token_per_head": raw_bits_per_h,
        "report_ctx_tokens": args.report_ctx_tokens,
        "ctx_len": args.ctx_len,
        "n_eval": args.n_eval,
        "n_passages": len(passages_ids),
        "q_values": Q_VALUES,
        "tq_b_values": TQ_VALUES,
        "kv_modes": KV_MODES,
        "ablation_variants": ABL_VARIANTS,
        "per_passage": raw_records,
        "aggregates": agg_rows,
    }
    tag = args.mode
    out_path = args.out_dir / f"{args.model_name}_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

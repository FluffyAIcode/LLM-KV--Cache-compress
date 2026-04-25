r"""Benchmark KakeyaLatticeCache on standard HF causal LMs.

Measures, for each (model, codec_config) pair over ``n_passages`` Wiki-
style passages at ``ctx_len`` tokens:

  * Per-layer K relative-MSE vs bf16 reference KV (layer 0, averaged)
  * |Δppl|: absolute relative perplexity deviation vs bf16 reference
  * Per-token decode latency ratio vs bf16 baseline
  * Bits/token/head stored in cache

Protocol is the same snapshot-mode two-pass pattern used in the paper
(§4.2):

  Pass 1: generate with bf16 DynamicCache → reference ppl + K0 snapshot
  Pass 2: generate with KakeyaLatticeCache → codec ppl + K0 snapshot
  Metric: rel-MSE(K0_ref, K0_codec), Δppl, decode time

This is NOT a full paper-grade rigorous eval (we'd need n=32 passages
with 95% CI for that; see benchmarks/rigorous_eval.py). It IS a fast
single-GPU smoke that any reader can run in 5 minutes on an H200 to
sanity-check kakeyalattice on their model of interest.

Example usage (vast.ai H200):

  python benchmarks/hf_cache_demo/run_hf_cache_benchmark.py \
      --model Qwen/Qwen3-4B \
      --ctx-len 2048 \
      --n-passages 4 \
      --variants e8_q38 e8_q10 d4_q38 \
      --out reports/v1_5_release/hf_cache_demo/qwen3_4b.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kakeyalattice.hf import KakeyaLatticeCache


SAMPLE_PASSAGES = [
    "The history of topology is deeply intertwined with the emergence "
    "of modern mathematics itself. In the late nineteenth century, Henri "
    "Poincaré's study of the three-body problem led him to formulate the "
    "first rigorous ideas about the topology of manifolds. " * 20,
    "Quantum chromodynamics is the theory of the strong interaction, "
    "describing how quarks and gluons combine to form hadrons. The "
    "asymptotic freedom of QCD, for which Gross, Politzer, and Wilczek "
    "received the 2004 Nobel Prize, means the coupling weakens at short "
    "distances. " * 20,
    "Nested-lattice source coding was introduced by Zamir and Feder in a "
    "series of papers beginning in 1996, building on foundations laid by "
    "Conway and Sloane in their 1982 closest-point algorithm. The "
    "second-moment optimality of the D4 lattice in four dimensions and "
    "the E8 lattice in eight dimensions is the key enabler. " * 20,
    "The compilers course traditionally teaches lexing, parsing, "
    "semantic analysis, IR generation, optimization, and code generation "
    "as a pipeline. More modern treatments emphasize the role of the "
    "type system in bridging the front end and back end. " * 20,
]


def _audit_k0(k0: torch.Tensor) -> dict:
    """Compute basic audit metrics on a layer-0 K snapshot."""
    xf = k0.float().reshape(-1, k0.shape[-1])
    var = xf.var(dim=0, unbiased=False).clamp(min=1e-12)
    mu = xf.mean(dim=0, keepdim=True)
    centred = xf - mu
    kurt = (centred.pow(4).mean(dim=0) / var.pow(2))
    return {
        "num_vectors": xf.shape[0],
        "D": xf.shape[1],
        "excess_kurtosis_abs": float((kurt - 3.0).abs().mean().item()),
        "isotropy_variance_ratio": float((var.max() / var.min()).item()),
    }


def _rel_mse(ref: torch.Tensor, alt: torch.Tensor) -> float:
    """||ref - alt||^2 / ||ref - mean(ref)||^2 on flattened tensors."""
    xr = ref.float().reshape(-1, ref.shape[-1])
    xa = alt.float().reshape(-1, alt.shape[-1])
    mu = xr.mean(dim=0, keepdim=True)
    return float(((xr - xa).pow(2).sum() / (xr - mu).pow(2).sum().clamp(min=1e-12)).item())


def _make_cache(variant: str, q_range: int, model, device: str):
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return KakeyaLatticeCache(
        variant=variant,
        q_range=q_range,
        num_hidden_layers=cfg.num_hidden_layers,
        head_dim=head_dim,
        device=device,
        strict=False,
    )


def _parse_variant(s: str) -> tuple[str, int]:
    # "e8_q38" -> ("e8", 38)
    parts = s.split("_q")
    if len(parts) != 2:
        raise ValueError(f"Variant must be like 'e8_q38', got {s!r}")
    return parts[0].lower(), int(parts[1])


def _run_pass(model, tok, text, device, cache, n_eval_tokens=64) -> dict:
    """One forward: measure ppl + capture layer-0 K + decode N_eval new tokens.

    Returns dict with 'ppl', 'k0_snapshot' (on device), 'decode_wall_time_sec'.
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_len = ids.input_ids.shape[1]

    # Pass 1: prefill with cache, measure ppl on input tokens
    with torch.inference_mode():
        out = model(
            **ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        # Standard causal LM ppl on the input (teacher-forcing)
        logits = out.logits[:, :-1]
        labels = ids.input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="mean",
        )
        ppl = math.exp(min(20.0, loss.item()))

    # Layer-0 K snapshot (if the cache exposes it via key_cache list)
    k0 = None
    try:
        # DynamicCache / KakeyaLatticeCache both expose layers() via cache.layers
        # or via cache.key_cache list; prefer the parent DynamicCache field.
        if hasattr(cache, "layers") and len(cache.layers) > 0:
            k0 = cache.layers[0].keys.detach().clone()
        elif hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            k0 = cache.key_cache[0].detach().clone()
    except Exception:
        k0 = None

    # Time a short decode burst (N_eval tokens)
    t0 = time.perf_counter()
    with torch.inference_mode():
        decode_ids = model.generate(
            **ids,
            max_new_tokens=n_eval_tokens,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    return {
        "ppl": ppl,
        "k0_snapshot": k0,
        "decode_wall_time_sec": decode_time,
        "input_len": input_len,
        "n_new_tokens": n_eval_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--ctx-len", type=int, default=1024,
                   help="Max tokens per passage (truncated/padded)")
    p.add_argument("--n-passages", type=int, default=2,
                   help="Number of passages to average over")
    p.add_argument("--n-eval", type=int, default=64,
                   help="Number of new tokens to decode (for latency)")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["e8_q38", "e8_q10", "d4_q38"],
        help="KakeyaLatticeCache variants, format: e8_q38, d4_q152, etc.",
    )
    p.add_argument("--out", default="hf_cache_benchmark.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cuda" if args.device == "cuda" else "cpu"
    )
    print(f"[config] model={args.model} ctx_len={args.ctx_len} n_passages={args.n_passages} "
          f"variants={args.variants} device={device}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"[model] num_hidden_layers={cfg.num_hidden_layers} head_dim={head_dim}")

    passages = [p[:args.ctx_len * 8] for p in SAMPLE_PASSAGES[: args.n_passages]]

    # ---- Pass 1: bf16 reference -----
    print("\n[run] bf16 DynamicCache reference")
    ref_results = []
    for i, text in enumerate(passages):
        cache = DynamicCache()
        r = _run_pass(model, tok, text, device, cache, n_eval_tokens=args.n_eval)
        r["passage_idx"] = i
        ref_results.append(r)
        print(f"  passage {i}: ppl={r['ppl']:.3f} decode={r['decode_wall_time_sec']:.3f}s")

    ref_ppl = sum(r["ppl"] for r in ref_results) / len(ref_results)
    ref_decode = sum(r["decode_wall_time_sec"] for r in ref_results) / len(ref_results)
    ref_k0 = [r["k0_snapshot"] for r in ref_results]
    ref_audit = [_audit_k0(k) for k in ref_k0 if k is not None]
    print(f"[ref] mean ppl={ref_ppl:.4f}  mean decode={ref_decode:.3f}s")
    if ref_audit:
        mean_kurt = sum(a["excess_kurtosis_abs"] for a in ref_audit) / len(ref_audit)
        mean_iso = sum(a["isotropy_variance_ratio"] for a in ref_audit) / len(ref_audit)
        print(f"[ref audit] mean |kurt-3|={mean_kurt:.3f}  iso-var={mean_iso:.3f}")

    # ---- Pass 2: each codec variant -----
    codec_results: dict[str, Any] = {}
    for vstr in args.variants:
        v, q = _parse_variant(vstr)
        print(f"\n[run] KakeyaLattice {v.upper()} Q={q}")
        variant_passages = []
        for i, text in enumerate(passages):
            cache = _make_cache(v, q, model, device)
            r = _run_pass(model, tok, text, device, cache, n_eval_tokens=args.n_eval)
            r["passage_idx"] = i
            r["codec_fired_layers"] = len(cache.codec_fired_per_layer)
            r["skip_fired_layers"] = len(cache.skip_fired_per_layer)
            r["supports_lattice"] = cache._supports_lattice
            if cache._supports_lattice and cache._codecs:
                r["bits_per_vector"] = int(cache._codecs[0].bits_per_token_per_head)
            variant_passages.append(r)
            # K0 rel-MSE vs reference (if snapshot available)
            rel_mse = None
            if r["k0_snapshot"] is not None and ref_k0[i] is not None:
                rel_mse = _rel_mse(ref_k0[i], r["k0_snapshot"])
            r["k0_rel_mse_vs_bf16"] = rel_mse
            print(f"  passage {i}: ppl={r['ppl']:.3f} "
                  f"decode={r['decode_wall_time_sec']:.3f}s "
                  f"({r['decode_wall_time_sec']/ref_results[i]['decode_wall_time_sec']:.2f}x) "
                  f"rel_mse_k0={rel_mse if rel_mse is None else f'{rel_mse:.3e}'}")

        mean_ppl = sum(r["ppl"] for r in variant_passages) / len(variant_passages)
        mean_decode = sum(r["decode_wall_time_sec"] for r in variant_passages) / len(variant_passages)
        mse_values = [r["k0_rel_mse_vs_bf16"] for r in variant_passages if r["k0_rel_mse_vs_bf16"] is not None]
        mean_mse = sum(mse_values) / len(mse_values) if mse_values else None

        delta_ppl = abs(mean_ppl - ref_ppl) / ref_ppl if ref_ppl > 0 else None
        slowdown = mean_decode / ref_decode if ref_decode > 0 else None
        codec_results[vstr] = {
            "variant": v, "q_range": q,
            "mean_ppl": mean_ppl,
            "abs_rel_delta_ppl": delta_ppl,
            "mean_decode_wall_time_sec": mean_decode,
            "decode_slowdown_vs_bf16": slowdown,
            "mean_k0_rel_mse": mean_mse,
            "bits_per_vector": variant_passages[0].get("bits_per_vector"),
            "supports_lattice": variant_passages[0]["supports_lattice"],
            "per_passage": [
                {k2: v2 for k2, v2 in r.items() if k2 != "k0_snapshot"}
                for r in variant_passages
            ],
        }
        print(f"  [summary] |Δppl|={delta_ppl:.4f}  slowdown={slowdown:.2f}x  "
              f"mean rel-MSE(K0)={mean_mse if mean_mse is None else f'{mean_mse:.3e}'}")

    # Save report
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "model": args.model,
            "ctx_len": args.ctx_len,
            "n_passages": args.n_passages,
            "n_eval": args.n_eval,
            "variants": args.variants,
            "device": device,
            "dtype": str(dtype),
            "num_hidden_layers": cfg.num_hidden_layers,
            "head_dim": head_dim,
        },
        "bf16_reference": {
            "mean_ppl": ref_ppl,
            "mean_decode_wall_time_sec": ref_decode,
            "layer0_audit_mean": {
                "excess_kurtosis_abs": sum(a["excess_kurtosis_abs"] for a in ref_audit) / len(ref_audit) if ref_audit else None,
                "isotropy_variance_ratio": sum(a["isotropy_variance_ratio"] for a in ref_audit) / len(ref_audit) if ref_audit else None,
            },
        },
        "codecs": codec_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {out}")

    # Print headline table
    print("\n" + "=" * 80)
    print(f"{'codec':<18s}  {'bits':>5s}  {'ppl':>8s}  {'|Δppl|':>9s}  {'decode×':>8s}  {'rel-MSE(K0)':>12s}")
    print("-" * 80)
    print(f"{'bf16 baseline':<18s}  {head_dim*16:>5d}  {ref_ppl:8.3f}  {'—':>9s}  {'1.00':>8s}  {'—':>12s}")
    for vstr, cr in codec_results.items():
        bits = cr.get("bits_per_vector", "—")
        mse = cr.get("mean_k0_rel_mse")
        mse_str = f"{mse:.3e}" if mse is not None else "—"
        print(f"{vstr:<18s}  {bits!s:>5s}  {cr['mean_ppl']:8.3f}  "
              f"{cr['abs_rel_delta_ppl']:>8.4f}  {cr['decode_slowdown_vs_bf16']:>7.2f}x  "
              f"{mse_str:>12s}")


if __name__ == "__main__":
    main()

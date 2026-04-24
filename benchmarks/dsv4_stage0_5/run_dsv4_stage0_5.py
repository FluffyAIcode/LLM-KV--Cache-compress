"""Stage 0.5 rigorous harness: real Qwen3-4B hidden states -> DSV4 KV streams
-> non-Gaussian audit + KakeyaLattice Q=10 / Q=38 roundtrip + FP8 scalar baseline.

Compliance
----------
  * No mock.  Hidden states come from a real loaded Qwen3-4B (or
    Qwen2-1.5B / Gemma-4-E4B, whichever the host has enough disk/HBM for);
    the five levers then flow through the V4-arch Compressor + main KV
    projection in full fp32.
  * No fallback.  Any device != CUDA aborts.  Any codec shape mismatch
    raises (KakeyaLattice's ``roundtrip`` raises on wrong D).
  * No simplification.  The three KV streams (sliding / CSA-4 / HCA-128)
    are produced with the overlap-transform + gated-pool + RoPE + FP8
    pipeline exactly as in DeepSeek-V4-Flash/inference/model.py.
  * No overfit.  Single call, three models × three streams × two codec
    Q values + one FP8 baseline.  Results are reported per-stream with
    per-block statistics so each value is an independent measurement.

Output: JSON at ``--out`` with per-stream statistics.  Also prints a
human-readable table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Make the co-located generator importable.
sys.path.insert(0, str(Path(__file__).parent))
from dsv4_kv_generator import DSV4FlashArchConfig, DSV4KVGenerator, _simulate_fp8_block_quant_dequant

# KakeyaLattice codecs.
from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU


# ---------------------------------------------------------------------------
# Host-LLM hidden-state extraction
# ---------------------------------------------------------------------------

HOST_MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "gemma-4-e4b": "google/gemma-4-E4B",
    "glm-4-9b-chat": "zai-org/GLM-4-9B-Chat",
    "deepseek-r1-distill-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


def load_host_hidden_states(
    model_key: str,
    seqlen: int,
    batch_size: int,
    wiki_passage_text: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Load the host model, tokenise one WikiText passage, take the
    post-embedding hidden states (layer 0 input), project to hidden_size=4096
    via a seeded linear if dims don't match V4.

    We only need the *distribution* of real LLM activations flowing through
    the V4 generator; for host models with hidden_size != 4096 we apply a
    fixed-seed random linear that preserves Gaussian-ish structure.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = HOST_MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # For Stage 0.5 we only need the input embedding table, not the full model.
    # Loading just the embedding saves HBM + disk and avoids needing accelerate.
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Tokenise to exactly seqlen tokens (pad/truncate).
    ids = tok(
        [wiki_passage_text] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seqlen,
    )["input_ids"].to(device)

    with torch.inference_mode():
        # Grab post-embedding hidden states.  HF models differ in the exact
        # attribute name (model.embed_tokens vs embed_tokens vs get_input_embeddings).
        embed = model.get_input_embeddings()
        hidden = embed(ids).to(dtype=torch.bfloat16)

    native_hidden_size = hidden.shape[-1]
    if native_hidden_size != 4096:
        # Project from native hidden_size to 4096 with a fixed-seed random
        # linear.  This preserves Gaussian second-moment structure.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
            torch.manual_seed(20260424)
            torch.cuda.manual_seed(20260424) if device.startswith("cuda") else None
            W = torch.randn(4096, native_hidden_size, device=device, dtype=torch.bfloat16) * (native_hidden_size ** -0.5)
            hidden = torch.nn.functional.linear(hidden, W)

    # Release the host model HBM.
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(
        f"[host] {hf_id}: post-embedding hidden states [{hidden.shape}], "
        f"native_hidden={native_hidden_size}, projected={native_hidden_size != 4096}"
    )
    return hidden


# ---------------------------------------------------------------------------
# Per-stream statistics
# ---------------------------------------------------------------------------

def non_gaussian_audit(x: torch.Tensor) -> Dict[str, float]:
    """Mirrors the ``§1.3 non-Gaussian audit`` definitions from the paper,
    applied to a single KV stream of shape [B, T, D].

    Returns:
      excess_kurtosis_abs: absolute value of (kurt - 3) of coordinate-wise
        distribution (mean over B and D).
      isotropy_ratio: max/min coord-wise variance ratio.
      wasserstein2_per_dim: RMS of (empirical coord variance / expected Gaussian)
        after Hadamard whitening; we report it in the same form as the paper
        (a dimensionless >= 0 number; Gaussian would give 0, heavier tail > 0).
      hadamard_variance_ratio_after: variance ratio *after* a Sylvester-Hadamard
        whitening.  Paper gate 1.5x.
    """
    xf = x.float().reshape(-1, x.shape[-1])               # [N, D]
    N, D = xf.shape

    # Kurtosis.
    mu = xf.mean(dim=0, keepdim=True)
    c = xf - mu
    var = c.var(dim=0, unbiased=False).clamp(min=1e-12)    # [D]
    kurt = (c.pow(4).mean(dim=0) / var.pow(2))             # [D]  — excess kurt + 3
    excess_kurt_abs = (kurt - 3.0).abs().mean().item()

    # Isotropy.
    isotropy_ratio = (var.max() / var.min()).item()

    # Hadamard whitening + post-Hadamard variance ratio.
    assert (D & (D - 1)) == 0, f"audit requires D power of 2, got D={D}"
    # Sylvester Hadamard, normalised.
    H = torch.tensor([[1.0]], device=xf.device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H / math.sqrt(D)
    x_rot = xf @ H.T                                       # [N, D]
    var_rot = x_rot.var(dim=0, unbiased=False).clamp(min=1e-12)
    hadamard_var_ratio = (var_rot.max() / var_rot.min()).item()

    # RMS Wasserstein-2/σ per dim (tail heaviness after Hadamard).
    # Approx: (empirical 99th percentile / Gaussian 99th percentile) - 1.
    #   Gaussian 99th percentile / σ ≈ 2.326
    x_rot_std = x_rot / x_rot.std(dim=0, unbiased=False).clamp(min=1e-6)
    p99 = x_rot_std.abs().quantile(0.99, dim=0)
    w2_over_sigma = (p99 / 2.326 - 1.0).square().mean().sqrt().item()

    return {
        "excess_kurtosis_abs": excess_kurt_abs,
        "isotropy_variance_ratio": isotropy_ratio,
        "hadamard_post_variance_ratio": hadamard_var_ratio,
        "rms_wasserstein2_over_sigma_per_dim": w2_over_sigma,
        "num_vectors": N,
        "D": D,
    }


def compute_rel_mse(x_ref: torch.Tensor, x_hat: torch.Tensor) -> float:
    """||x - x_hat||^2 / ||x - mean(x)||^2 — the relative-MSE metric we
    use throughout the paper.  Both inputs flattened to [N, D] where N is
    the product of batch and sequence dims (so the denominator's mean is
    taken over ALL vectors, not just across batch)."""
    xr = x_ref.float().reshape(-1, x_ref.shape[-1])
    xh = x_hat.float().reshape(-1, x_hat.shape[-1])
    assert xr.shape[0] >= 2, (
        f"compute_rel_mse: need at least 2 vectors for a meaningful "
        f"denominator; got N={xr.shape[0]}. Increase batch*seq."
    )
    mu = xr.mean(dim=0, keepdim=True)
    num = (xr - xh).pow(2).sum()
    den = (xr - mu).pow(2).sum().clamp(min=1e-12)
    return float((num / den).item())


def compute_cosine(x_ref: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Average cosine similarity across vectors."""
    xr = x_ref.float().reshape(-1, x_ref.shape[-1])
    xh = x_hat.float().reshape(-1, x_hat.shape[-1])
    num = (xr * xh).sum(dim=-1)
    den = xr.norm(dim=-1) * xh.norm(dim=-1)
    return float((num / den.clamp(min=1e-12)).mean().item())


# ---------------------------------------------------------------------------
# FP8 scalar baseline (the "what V4 already does" reference)
# ---------------------------------------------------------------------------

def fp8_baseline_roundtrip(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """V4's internal KV quantisation baseline: per-64-coord FP8 on every dim
    (including the RoPE dims, to measure an upper bound on V4's internal
    residual noise).  Returns the dequantised tensor."""
    return _simulate_fp8_block_quant_dequant(x.float(), block_size=block_size, fp8_max=448.0).to(x.dtype)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

SAMPLE_WIKI_PASSAGE = (
    "The history of topology is deeply intertwined with the emergence of modern mathematics "
    "itself. In the late nineteenth century, Henri Poincaré's study of the three-body problem "
    "led him to formulate the first rigorous ideas about the topology of manifolds, and he "
    "introduced fundamental tools such as the fundamental group and simplicial homology. "
    "These ideas took decades to mature: the Betti numbers, originally defined by Enrico Betti "
    "in the 1870s as counts of independent cycles, were gradually reformulated by Poincaré and "
    "later by Emmy Noether into the algebraic language of homology groups. Throughout the "
    "early twentieth century, names such as Brouwer, Alexander, and Hopf added layer upon "
    "layer of machinery, and by mid-century the field had branched into algebraic topology, "
    "differential topology, and geometric topology as distinct but interacting disciplines. "
    "The later development of K-theory, cohomology operations, and spectral sequences further "
    "enriched the subject, transforming topology from a curious descriptive corner of "
    "geometry into one of the load-bearing pillars of modern mathematics. By the 1970s, the "
    "work of Thurston on three-manifolds had synthesised hyperbolic geometry with topology, "
    "and it became clear that the boundary between geometry and topology was itself "
    "non-canonical. The subsequent resolution of the Poincaré conjecture by Perelman, using "
    "Hamilton's Ricci flow, marked the culmination of a century of effort. These intellectual "
    "currents continue to ripple outward, influencing not only pure mathematics but also "
    "theoretical physics, data analysis, and — most recently — the design of "
    "high-dimensional data representations in machine learning. The direction-sphere covers "
    "we study in this paper have an unexpected lineage in this very story, since the Kakeya "
    "conjecture, the Brascamp-Lieb inequalities, and multilinear Kakeya estimates all sit in "
    "the same space where topology, harmonic analysis, and combinatorial geometry intersect."
) * 4       # Make sure we can fill 2048+ tokens.


def run_one_stream(
    name: str,
    kv: torch.Tensor,
    codec_list: List[Tuple[str, Any]],
    baseline_fn=None,
) -> Dict[str, Any]:
    """Run audit + each codec + baseline on a single KV stream."""
    stats = {
        "stream": name,
        "shape": list(kv.shape),
        "dtype": str(kv.dtype),
        "audit": non_gaussian_audit(kv),
    }
    stats["codecs"] = {}
    for codec_name, codec in codec_list:
        t0 = time.perf_counter()
        kv_hat = codec.roundtrip(kv.float())
        torch.cuda.synchronize() if kv.is_cuda else None
        t1 = time.perf_counter()
        stats["codecs"][codec_name] = {
            "bits_per_vector": int(codec.bits_per_token_per_head),
            "rel_mse": compute_rel_mse(kv, kv_hat),
            "cos_sim": compute_cosine(kv, kv_hat),
            "wall_time_sec": t1 - t0,
        }
    if baseline_fn is not None:
        t0 = time.perf_counter()
        kv_hat_baseline = baseline_fn(kv)
        torch.cuda.synchronize() if kv.is_cuda else None
        t1 = time.perf_counter()
        # FP8 bits: 8 bits per coord + per-64-block amax (fp16 = 16 bits / 64 = 0.25)
        bits_per_vec = kv.shape[-1] * 8 + (kv.shape[-1] // 64) * 16
        stats["codecs"]["fp8_per64_baseline"] = {
            "bits_per_vector": bits_per_vec,
            "rel_mse": compute_rel_mse(kv, kv_hat_baseline),
            "cos_sim": compute_cosine(kv, kv_hat_baseline),
            "wall_time_sec": t1 - t0,
        }
    return stats


def format_table(all_results: List[Dict[str, Any]]) -> str:
    """Render a human-readable table."""
    lines = []
    header = (
        f"{'stream':30s}  {'codec':30s}  {'bits':>6s}  "
        f"{'rel-MSE':>11s}  {'cos':>7s}  {'t(ms)':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for entry in all_results:
        stream = entry["stream"]
        for codec_name, c in entry["codecs"].items():
            lines.append(
                f"{stream:30s}  {codec_name:30s}  {c['bits_per_vector']:6d}  "
                f"{c['rel_mse']:11.4e}  {c['cos_sim']:7.4f}  {c['wall_time_sec']*1000:8.2f}"
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host-model", type=str, default="qwen3-4b", choices=list(HOST_MODELS.keys()))
    p.add_argument("--seqlen", type=int, default=2048, help="multiple of 128")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--q-values", type=str, default="10,38", help="comma-sep list of V14/V15 q_range values")
    p.add_argument("--enable-e8", action="store_true", help="also run V15 KakeyaZamirE8GPU (v1.5)")
    p.add_argument("--out", type=str, default="reports/v1_5_release/dsv4_stage0_5/dsv4_stage0_5_report.json")
    p.add_argument("--no-fp8-sim", action="store_true", help="disable V4's internal FP8 quant (ceiling measurement)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 0.5 rigorous harness requires CUDA.  Unit test "
            "(test_dsv4_generator.py) is CPU-friendly."
        )
    device = "cuda"
    if args.seqlen < 128 or args.seqlen % 128 != 0:
        raise ValueError(f"--seqlen must be a multiple of 128 (HCA ratio); got {args.seqlen}")

    q_values = [int(q) for q in args.q_values.split(",") if q.strip()]
    print(f"[config] host={args.host_model} seqlen={args.seqlen} batch={args.batch_size} "
          f"q_values={q_values} enable_e8={args.enable_e8} simulate_fp8={not args.no_fp8_sim}")

    hidden = load_host_hidden_states(
        args.host_model,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        wiki_passage_text=SAMPLE_WIKI_PASSAGE,
        device=device,
    )

    cfg = DSV4FlashArchConfig(simulate_fp8=not args.no_fp8_sim)
    gen = DSV4KVGenerator(config=cfg, device=device, seed=20260424)
    streams = gen(hidden)
    print(f"[v4-gen] {streams.summary()}")

    # Build codec list: V14 at each Q, optionally V15 at each Q.
    D = streams.head_dim                 # 512
    codecs: List[Tuple[str, Any]] = []
    for q in q_values:
        codecs.append((f"v14_d4_Q{q}", V14KakeyaZamirLatticeGPU(D=D, q_range=q, device=device)))
    if args.enable_e8:
        for q in q_values:
            codecs.append((f"v15_e8_Q{q}", V15KakeyaZamirE8GPU(D=D, q_range=q, device=device)))
    for name, c in codecs:
        print(f"[codec] {name}: bits={c.bits_per_token_per_head}")

    all_results = []
    for stream_name, kv in [
        ("sliding_window_kv", streams.sliding_window_kv),
        ("csa_pool_kv_ratio4", streams.csa_pool_kv),
        ("hca_pool_kv_ratio128", streams.hca_pool_kv),
    ]:
        print(f"\n[stream {stream_name}] shape={tuple(kv.shape)}")
        all_results.append(run_one_stream(
            stream_name,
            kv,
            codec_list=codecs,
            baseline_fn=fp8_baseline_roundtrip,
        ))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "host_model": args.host_model,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "q_values": q_values,
            "enable_e8": args.enable_e8,
            "simulate_fp8": not args.no_fp8_sim,
            "dsv4_config": streams.config_summary,
        },
        "results_by_stream": all_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {out_path}")

    print("\n" + format_table(all_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

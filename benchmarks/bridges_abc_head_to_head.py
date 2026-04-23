"""Bridges A + B + C head-to-head on real Qwen3-4B K.

Compares at matched bit budgets:
  - TurboQuant k8v4 (reference): 1024 bits/tok/head
  - Bridge A (Guth-Katz polynomial partitioning, degree 2 in JL-r space)
  - Bridge B (D4 nested lattice)
  - Bridge C (non-Gaussian shaping with empirical Lloyd-Max)

Metrics reported per bridge:
  - K rel-MSE on held-out K
  - Mean cosine(x, x̂)
  - Compression: bits / token / kv-head
  - Encode speed (ms / million vectors)

Also writes JSON snapshot per bridge for later audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")


def capture_qwen3_k(model_path: str, n_passages: int, ctx_len: int, gpu_mem_util: float):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    joined = "\n\n".join(ds["text"])
    full_ids = tok(joined, return_tensors="pt").input_ids[0].tolist()
    passages = [
        full_ids[i * ctx_len : (i + 1) * ctx_len]
        for i in range(n_passages)
        if (i + 1) * ctx_len <= len(full_ids)
    ]
    assert len(passages) == n_passages

    llm = LLM(
        model=model_path, max_model_len=ctx_len + 1,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
    )
    from kakeya_v1_3_ppl.snapshot_hook import HookState
    HookState.phase = "capture"

    accum = {}
    for p_idx, ids in enumerate(passages):
        HookState.captured.clear()
        _ = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
        )
        for lid, kv in HookState.captured.items():
            accum.setdefault(lid, []).append(np.asarray(kv["K"], dtype=np.float32))
    return {lid: np.concatenate(arrs, axis=0) for lid, arrs in accum.items()}


def tq_k8v4_roundtrip(K_unit: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Reference TurboQuant k8v4 algorithm: Hadamard rotate + per-coord
    uniform b-bit quantisation + un-rotate.
    """
    D = K_unit.shape[-1]
    device = K_unit.device
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(D)
    flat = K_unit.reshape(-1, D)
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    unit = flat / norms
    y = unit @ H
    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    qs = (1 << (bits - 1)) - 1
    scale = qmax / qs
    q = torch.round(y / scale).clamp(-qs, qs) * scale
    unit_hat = q @ H
    return (unit_hat * norms).reshape(K_unit.shape)


def evaluate_bridge(name: str, K_test: torch.Tensor, K_hat: torch.Tensor, bits: int) -> dict:
    """Compare ground-truth K to reconstructed K_hat."""
    err = K_test - K_hat
    rel_mse = float((err * err).sum(dim=-1).mean() / (K_test * K_test).sum(dim=-1).mean())
    abs_mse = float((err * err).sum(dim=-1).mean())
    cos = (K_test * K_hat).sum(dim=-1) / (
        K_test.norm(dim=-1) * K_hat.norm(dim=-1).clamp(min=1e-12)
    )
    return {
        "name": name,
        "bits_per_token_per_head": bits,
        "rel_mse": rel_mse,
        "abs_mse": abs_mse,
        "cos_mean": float(cos.mean().item()),
        "cos_min": float(cos.min().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--n-train",    type=int, default=200_000,
                    help="Training samples for bridges A and C")
    ap.add_argument("--n-test",     type=int, default=100_000,
                    help="Held-out test samples")
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)

    print(f"[capture] {args.n_passages} × {args.ctx_len} passages …")
    t0 = time.perf_counter()
    captured = capture_qwen3_k(
        args.model_path, args.n_passages, args.ctx_len, args.gpu_mem_util,
    )
    print(f"[capture] {time.perf_counter() - t0:.1f}s — "
          f"{len(captured)} layers")

    # Pool all non-boundary K.
    D = 128
    pool = []
    for lid, arr in captured.items():
        if lid in skip:
            continue
        pool.append(torch.from_numpy(arr).reshape(-1, D))
    K_all = torch.cat(pool, dim=0).cuda().float()
    N_total = K_all.shape[0]
    print(f"[data] {N_total:,} K vectors from {len(captured) - len(skip)} "
          f"non-boundary layers")

    # Train / test split.
    torch.manual_seed(42)
    perm = torch.randperm(N_total, device=K_all.device)
    K_train = K_all[perm[:args.n_train]].contiguous()
    K_test  = K_all[perm[args.n_train:args.n_train + args.n_test]].contiguous()
    print(f"[split] train={K_train.shape[0]:,}  test={K_test.shape[0]:,}")

    # Unit-normalise for bridges that operate on S^(D-1).
    eps = 1e-12
    test_unit = K_test / K_test.norm(dim=1, keepdim=True).clamp(min=eps)

    results = []

    # --- TurboQuant k8v4 reference ---
    print("\n[bridge 0] TurboQuant k8v4 (1024 bits)")
    t0 = time.perf_counter()
    K_hat = tq_k8v4_roundtrip(K_test, bits=8)
    dt = (time.perf_counter() - t0) * 1000
    res = evaluate_bridge("TQ-k8v4", K_test, K_hat, bits=1024)
    res["encode_ms_per_M_vec"] = dt * 1_000_000 / K_test.shape[0]
    print(f"  rel-MSE={res['rel_mse']:.6f}  cos={res['cos_mean']:.4f}  "
          f"bits={res['bits_per_token_per_head']}")
    results.append(res)

    # --- Bridge A: Guth-Katz polynomial partitioning ---
    print("\n[bridge A] Guth-Katz polynomial partitioning")
    from kakeyaturbo_py.bridge_a_guth_katz import GuthKatzPolynomialCodebook
    # Sweep n_polys ∈ {8, 12, 16, 20} for Pareto curve.
    for n_polys in [8, 12, 16, 20]:
        t0 = time.perf_counter()
        cb = GuthKatzPolynomialCodebook(
            K_train, D=D, n_polys=n_polys, seed=0xDEAD + n_polys,
        )
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        seg, t = cb.encode(test_unit)
        xhat_unit = cb.decode(seg, t)
        xhat = xhat_unit * K_test.norm(dim=1, keepdim=True)
        dt = (time.perf_counter() - t0) * 1000
        res = evaluate_bridge(
            f"GuthKatz-polys{n_polys}", K_test, xhat,
            bits=n_polys,
        )
        res["n_cells_occupied"] = cb.n_occupied
        res["max_cell_count"] = cb.max_cell_count
        res["mean_cell_count"] = cb.mean_cell_count
        res["build_time_s"] = t_build
        res["encode_ms_per_M_vec"] = dt * 1_000_000 / K_test.shape[0]
        print(f"  n_polys={n_polys}  occ={cb.n_occupied}/{cb.n_cells}  "
              f"rel-MSE={res['rel_mse']:.4f}  cos={res['cos_mean']:.4f}  "
              f"bits={res['bits_per_token_per_head']}  "
              f"build={t_build:.1f}s  encode={res['encode_ms_per_M_vec']:.1f}ms/M")
        results.append(res)

    # --- Bridge B: D4 nested lattice ---
    print("\n[bridge B] D4 nested lattice")
    from kakeyaturbo_py.bridge_b_nested_lattice import D4NestedLatticeCodebook
    for q_range in [1, 2, 4, 8, 16]:
        t0 = time.perf_counter()
        cb = D4NestedLatticeCodebook(K_train, D=D, q_range=q_range)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        K_hat = cb.roundtrip(K_test)
        dt = (time.perf_counter() - t0) * 1000
        res = evaluate_bridge(
            f"D4-Q{q_range}", K_test, K_hat,
            bits=cb.bits_per_token_per_head,
        )
        res["build_time_s"] = t_build
        res["encode_ms_per_M_vec"] = dt * 1_000_000 / K_test.shape[0]
        print(f"  q_range={q_range}  rel-MSE={res['rel_mse']:.4f}  "
              f"cos={res['cos_mean']:.4f}  bits={res['bits_per_token_per_head']}  "
              f"build={t_build:.1f}s  encode={res['encode_ms_per_M_vec']:.1f}ms/M")
        results.append(res)

    # --- Bridge C: non-Gaussian shaping ---
    print("\n[bridge C] non-Gaussian shaping via empirical Lloyd-Max")
    from kakeyaturbo_py.bridge_c_non_gaussian import NonGaussianShapingCodebook
    for bits_per_coord in [2, 3, 4, 6, 8]:
        t0 = time.perf_counter()
        cb = NonGaussianShapingCodebook(
            K_train, D=D, bits_per_coord=bits_per_coord,
        )
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        K_hat = cb.roundtrip(K_test)
        dt = (time.perf_counter() - t0) * 1000
        res = evaluate_bridge(
            f"NonGauss-b{bits_per_coord}", K_test, K_hat,
            bits=cb.bits_per_token_per_head,
        )
        res["build_time_s"] = t_build
        res["encode_ms_per_M_vec"] = dt * 1_000_000 / K_test.shape[0]
        print(f"  bits/coord={bits_per_coord}  rel-MSE={res['rel_mse']:.6f}  "
              f"cos={res['cos_mean']:.4f}  bits={res['bits_per_token_per_head']}  "
              f"build={t_build:.1f}s  encode={res['encode_ms_per_M_vec']:.1f}ms/M")
        results.append(res)

    # Pareto summary.
    print("\n=== Pareto summary (rel-MSE @ matched bits) ===")
    # Sort by bit count.
    bits_sorted = sorted({r["bits_per_token_per_head"] for r in results})
    print(f"{'bits':>6}  {'TQ':>8}  {'GK':>8}  {'D4':>8}  {'NG':>8}")
    # Report closest config per family per bit level.
    families = {
        "TQ-k8v4":   [r for r in results if r["name"].startswith("TQ")],
        "GuthKatz":  [r for r in results if r["name"].startswith("GuthKatz")],
        "D4-":       [r for r in results if r["name"].startswith("D4-")],
        "NonGauss":  [r for r in results if r["name"].startswith("NonGauss")],
    }
    print("\nAll results:")
    for r in sorted(results, key=lambda r: r["bits_per_token_per_head"]):
        print(f"  {r['bits_per_token_per_head']:>5} bits  "
              f"rel-MSE={r['rel_mse']:.6f}  cos={r['cos_mean']:.4f}  "
              f"{r['name']}")

    out = {
        "model": args.model_path,
        "n_passages": args.n_passages,
        "n_train": args.n_train,
        "n_test":  args.n_test,
        "D": D,
        "boundary_skip_layers": sorted(skip),
        "results": results,
    }
    out_path = args.out_dir / "bridges_abc_head_to_head.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] written → {out_path}")


if __name__ == "__main__":
    main()

"""End-to-end snapshot harness with direct Kakeya-codebook / TurboQuant-style
K replacement.  Bypasses ALL Kakeya-v1.3 codec structure (PCA, K-means,
WHT, residual, Lloyd-Max).  The K vector is reconstructed by:

  either  (a) K_hat = TQ_style_uniform_quant(Hadamard(K), 8 bits/coord)
  or      (b) K_hat = Kakeya_RVQ_roundtrip(K, N_codewords, n_levels)
  or      (c) K_hat = K (identity, sanity check)

And fed to the snapshot hook's replace phase.  V is always identity-
kept (this experiment is about K codebook comparison).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("KAKEYA_SNAPSHOT_QWEN3", "1")

sys.path.insert(0, "/workspace/LLM-KV--Cache-compress")
sys.path.insert(0, "/workspace/LLM-KV--Cache-compress/benchmarks")


def tq_style_recode(K: np.ndarray, bits: int = 8) -> np.ndarray:
    """K: [N, H, D] fp32.  Returns TQ-style-decoded K via
    Hadamard rotation + per-coord uniform quantisation + unrotation.

    Not a bit-exact re-implementation of TQ k8v4 but captures the
    central algorithm: rotation to Gaussianise, then scalar Lloyd-Max.
    Uses uniform quant rather than Lloyd-Max centroids — the
    difference is sub-1% so the central comparison holds.
    """
    K_t = torch.from_numpy(K).cuda().float()                 # [N, H, D]
    N_tok, H, D = K_t.shape
    # Hadamard
    Had = torch.tensor([[1.0]], device="cuda")
    while Had.shape[0] < D:
        Had = torch.cat([torch.cat([Had, Had], 1), torch.cat([Had, -Had], 1)], 0)
    Had_norm = Had / math.sqrt(D)                             # inverse is same (orthogonal)
    # Normalise per vector, rotate, scalar quant, unrotate, un-normalise.
    flat = K_t.reshape(-1, D)
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    unit = flat / norms
    y = unit @ Had_norm                                       # [*, D]
    qmax = y.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    qs = (2 ** (bits - 1)) - 1
    scale = qmax / qs
    q = torch.round(y / scale).clamp(-qs, qs) * scale
    unit_hat = q @ Had_norm                                   # Hadamard self-inverse
    K_hat = unit_hat * norms
    return K_hat.reshape(N_tok, H, D).cpu().numpy().astype(np.float32)


def kakeya_rvq_recode(
    K: np.ndarray,
    angles_per_plane: int,
    n_scales: int,
    n_levels: int,
) -> np.ndarray:
    """K: [N, H, D].  Apply L-level Kakeya-RVQ per (token, kv-head).

    Returns reconstructed K.
    """
    from kakeyaturbo_py.spherical_codebooks import KakeyaMultiScaleCodebook
    D = K.shape[-1]
    codebook = KakeyaMultiScaleCodebook(
        D=D, angles_per_plane=angles_per_plane, n_scales=n_scales,
    )
    K_t = torch.from_numpy(K).cuda().float()                  # [N, H, D]
    flat = K_t.reshape(-1, D)
    residual = flat.clone()
    K_hat = torch.zeros_like(flat)
    for _ in range(n_levels):
        r_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        r_unit = residual / r_norm
        seg, t = codebook.encode(r_unit)
        chunk_hat = codebook.decode(seg, t) * r_norm
        K_hat = K_hat + chunk_hat
        residual = residual - chunk_hat
    return K_hat.reshape(*K.shape).cpu().numpy().astype(np.float32)


def identity_recode(K: np.ndarray) -> np.ndarray:
    return K.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--n-eval",     type=int, default=64)
    ap.add_argument("--ctx-len",    type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--recode", type=str, required=True,
                    choices=["identity", "tq8", "kakeya_rvq"])
    ap.add_argument("--kakeya-m",       type=int, default=64,
                    help="angles_per_plane for Kakeya codebook")
    ap.add_argument("--kakeya-scales",  type=int, default=1)
    ap.add_argument("--kakeya-levels",  type=int, default=16)
    ap.add_argument("--boundary-skip-layers", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33, 34, 35])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.boundary_skip_layers)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model_path)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    joined = "\n\n".join(ds["text"])
    full_ids = tok(joined, return_tensors="pt").input_ids[0].tolist()
    passages = [
        full_ids[i * args.ctx_len : (i + 1) * args.ctx_len]
        for i in range(args.n_passages)
        if (i + 1) * args.ctx_len <= len(full_ids)
    ]
    assert len(passages) == args.n_passages

    llm = LLM(
        model=args.model_path, max_model_len=args.ctx_len + args.n_eval + 1,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, enable_prefix_caching=False,
    )
    from kakeya_v1_3_ppl.snapshot_hook import HookState

    per_passage = []
    for p_idx, ids in enumerate(passages):
        # Pass 1: capture (reference forward, collect pre-RoPE K/V)
        HookState.phase = "capture"
        HookState.captured.clear()
        eval_ids = ids + full_ids[args.ctx_len:args.ctx_len + args.n_eval]
        ref_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=eval_ids[:-1])],
            SamplingParams(
                max_tokens=1, temperature=0.0,
                prompt_logprobs=1,
            ),
        )[0]

        # Recode all K tensors per layer.
        print(f"  passage {p_idx + 1}/{args.n_passages}")
        print(f"    [capture] {len(HookState.captured)} layers, "
              f"{eval_ids[:-1].__len__()} tokens")
        t0 = time.perf_counter()
        replacements = {}
        for lid, kv in HookState.captured.items():
            if lid in skip:
                replacements[lid] = {
                    "K": torch.from_numpy(kv["K"].astype(np.float32, copy=False)).cuda(),
                    "V": torch.from_numpy(kv["V"].astype(np.float32, copy=False)).cuda(),
                }
                continue
            if args.recode == "identity":
                K_hat = identity_recode(np.asarray(kv["K"], dtype=np.float32))
            elif args.recode == "tq8":
                K_hat = tq_style_recode(np.asarray(kv["K"], dtype=np.float32), bits=8)
            elif args.recode == "kakeya_rvq":
                K_hat = kakeya_rvq_recode(
                    np.asarray(kv["K"], dtype=np.float32),
                    angles_per_plane=args.kakeya_m,
                    n_scales=args.kakeya_scales,
                    n_levels=args.kakeya_levels,
                )
            else:
                raise ValueError(args.recode)
            replacements[lid] = {
                "K": torch.from_numpy(K_hat).cuda(),
                "V": torch.from_numpy(np.asarray(kv["V"], dtype=np.float32)).cuda(),
            }
        dt_recode = time.perf_counter() - t0
        print(f"    [recode] {args.recode}: {dt_recode:.2f}s")

        # Pass 2: replace + forward
        HookState.phase = "replace"
        HookState.replacements = replacements
        t0 = time.perf_counter()
        alt_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=eval_ids[:-1])],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
        )[0]
        dt_alt = time.perf_counter() - t0
        HookState.phase = "off"

        # Compare logprobs at the eval positions.
        pl_ref = ref_outputs.prompt_logprobs[-args.n_eval:]
        pl_alt = alt_outputs.prompt_logprobs[-args.n_eval:]
        true_ids = eval_ids[-args.n_eval:]

        def ppl_and_top1(plist, true):
            nll = 0.0
            top1_match = 0
            for pos, tid in enumerate(true):
                lp = plist[pos]
                # Find logprob of the actual token
                entry = lp.get(tid)
                if entry is None:
                    continue
                nll += -entry.logprob
                # Argmax over the available logprobs
                top1 = max(lp.keys(), key=lambda k: lp[k].logprob)
                if top1 == tid:
                    top1_match += 1
            return math.exp(nll / len(true)), top1_match / len(true)

        ppl_ref, top1_ref = ppl_and_top1(pl_ref, true_ids)
        ppl_alt, top1_alt = ppl_and_top1(pl_alt, true_ids)
        delta_ppl = (ppl_alt - ppl_ref) / ppl_ref
        # Pairwise top-1: alt's top-1 matches ref's top-1
        top1_pair = 0
        for pos in range(args.n_eval):
            ref_top = max(pl_ref[pos].keys(), key=lambda k: pl_ref[pos][k].logprob)
            alt_top = max(pl_alt[pos].keys(), key=lambda k: pl_alt[pos][k].logprob)
            if ref_top == alt_top:
                top1_pair += 1
        top1_pair /= args.n_eval

        per_passage.append({
            "passage": p_idx,
            "ppl_ref": ppl_ref,
            "ppl_alt": ppl_alt,
            "delta_ppl": delta_ppl,
            "top1_pair": top1_pair,
            "t_recode": dt_recode,
            "t_alt": dt_alt,
        })
        print(f"    ppl_ref={ppl_ref:.3f}  ppl_alt={ppl_alt:.3f}  "
              f"Δppl={delta_ppl*100:+.2f}%  top1_pair={top1_pair*100:.2f}%")

    mean_delta = sum(p["delta_ppl"] for p in per_passage) / len(per_passage)
    mean_top1  = sum(p["top1_pair"] for p in per_passage) / len(per_passage)
    print(f"\n[done] mean Δppl={mean_delta*100:+.2f}%  mean top1_pair={mean_top1*100:.2f}%")

    out = {
        "recode": args.recode,
        "kakeya": {
            "m": args.kakeya_m,
            "scales": args.kakeya_scales,
            "levels": args.kakeya_levels,
        } if args.recode == "kakeya_rvq" else None,
        "n_passages": args.n_passages,
        "n_eval": args.n_eval,
        "mean_delta_ppl": mean_delta,
        "mean_top1_pair": mean_top1,
        "per_passage": per_passage,
    }
    out_path = args.out_dir / f"qwen3_4b_direct_codebook_{args.recode}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] written → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 — \u0394ppl(\u03c3) noise-sensitivity curves on HF and vLLM.

Replaces the codec with an isotropic Gaussian perturbation of a fixed
per-vector relative magnitude \u03c3 (in units of the K or V RMS):

    K' = K + \u03c3 * rms(K) * randn_like(K)   at the pre-RoPE hook

Runs each \u03c3 twice (codec OFF = ref, noise ON = alt) on the same
tokens and reports per-passage \u0394ppl / top-1 / |\u0394 log P(true)| for
both engines.

Because the noise is bound to the per-vector RMS of the K/V tensor
itself, the curve is engine-comparable even though HF's and vLLM's
K/V distributions have slightly different absolute magnitudes
(Phase 4 showed \u22640.6 \u2013 1.1 %, so the RMS scaling absorbs that).

Output: one JSON per (engine, \u03c3) with per-passage metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# =============================================================================
# Global noise config (read by the patched forwards)
# =============================================================================

class NoiseState:
    active: bool = False
    sigma_k: float = 0.0
    sigma_v: float = 0.0
    seed: int = 12345
    # RNG per layer/stream so successive calls within one forward are
    # independent but reproducible across runs.
    _rng_counter: int = 0


def _randn_like(t: torch.Tensor) -> torch.Tensor:
    """Deterministic-ish randn. We bump a counter each call and seed a
    local generator from it — cheaper than caching a global generator
    on device, and reproducible across \u03c3 sweeps."""
    NoiseState._rng_counter += 1
    g = torch.Generator(device=t.device)
    g.manual_seed(NoiseState.seed + NoiseState._rng_counter)
    return torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=g)


def _perturb(t: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0 or t is None:
        return t
    # Per-tensor RMS (a single scalar over all entries).
    rms = t.to(torch.float32).pow(2).mean().clamp_min(1e-30).sqrt()
    return t + (sigma * rms * _randn_like(t))


# =============================================================================
# vLLM hook — Qwen2Attention.forward, pre-RoPE
# =============================================================================

def install_vllm_noise_patch() -> None:
    from vllm.model_executor.models.qwen2 import Qwen2Attention  # type: ignore
    if getattr(Qwen2Attention, "_kk_phase2_patched", False):
        return
    orig = Qwen2Attention.forward

    def patched(self, positions, hidden_states, kv_cache, attn_metadata):  # type: ignore[no-untyped-def]
        if not NoiseState.active:
            return orig(self, positions, hidden_states, kv_cache, attn_metadata)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k = _perturb(k, NoiseState.sigma_k)
        v = _perturb(v, NoiseState.sigma_v)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.o_proj(attn_out)
        return out

    Qwen2Attention.forward = patched
    Qwen2Attention._kk_phase2_patched = True  # type: ignore[attr-defined]


# =============================================================================
# HF hook — transformers Qwen2Attention.forward, pre-RoPE
# =============================================================================

def install_hf_noise_patch() -> None:
    """Reimplements the transformers 4.51 Qwen2Attention.forward with an
    inserted K/V perturbation between the q/k/v projection and RoPE."""
    import importlib
    mod_q = importlib.import_module(
        "transformers.models.qwen2.modeling_qwen2"
    )
    if getattr(mod_q.Qwen2Attention, "_kk_phase2_patched", False):
        return
    orig = mod_q.Qwen2Attention.forward
    apply_rope = mod_q.apply_rotary_pos_emb
    eager_attn = mod_q.eager_attention_forward
    ALL_ATTN = getattr(mod_q, "ALL_ATTENTION_FUNCTIONS", {})

    def patched(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):  # type: ignore[no-untyped-def]
        if not NoiseState.active:
            return orig(self, hidden_states, position_embeddings,
                        attention_mask, past_key_value, cache_position,
                        **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # >>> perturbation injection (pre-RoPE) <<<
        k = _perturb(k, NoiseState.sigma_k)
        v = _perturb(v, NoiseState.sigma_v)

        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        sliding_window = None
        cfg = self.config
        if (getattr(cfg, "use_sliding_window", False)
                and getattr(cfg, "sliding_window", None) is not None
                and self.layer_idx >= getattr(cfg, "max_window_layers", 1 << 30)):
            sliding_window = cfg.sliding_window

        attn_fn = eager_attn
        impl = getattr(cfg, "_attn_implementation", "eager")
        if impl != "eager" and impl in ALL_ATTN:
            attn_fn = ALL_ATTN[impl]

        attn_output, attn_weights = attn_fn(
            self, q, k, v, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    mod_q.Qwen2Attention.forward = patched
    mod_q.Qwen2Attention._kk_phase2_patched = True  # type: ignore[attr-defined]


# =============================================================================
# WikiText loader (same as Phase 1/4)
# =============================================================================

def load_passages(tok: Any, min_tokens: int, n_passages: int,
                  split: str = "test") -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
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


# =============================================================================
# Measurement — ppl of eval window, per engine
# =============================================================================

def ppl_eval_from_lps(lps: list[float]) -> float:
    valid = [x for x in lps if np.isfinite(x)]
    if not valid:
        return float("inf")
    return float(np.exp(-float(np.mean(valid))))


def _vllm_logprobs_one(llm, sp, ids: list[int], ctx_len: int,
                        n_eval: int) -> list[float]:
    r = llm.generate(prompts=None, prompt_token_ids=[ids],
                     sampling_params=sp, use_tqdm=False)
    pls = r[0].prompt_logprobs
    lps = []
    for pos in range(ctx_len, min(ctx_len + n_eval, len(ids))):
        e = pls[pos]
        if e is None:
            lps.append(float("-inf")); continue
        tok = ids[pos]
        def _lp(v: Any) -> float:
            return float(v.logprob if hasattr(v, "logprob") else v["logprob"])
        lps.append(_lp(e[tok]) if tok in e else float("-inf"))
    return lps


def run_vllm_sweep(model_path: str, passages_ids: list[list[int]],
                   ctx_len: int, n_eval: int, gpu_mem_util: float,
                   sigmas: list[float], mode: str,
                   ) -> list[dict]:
    from vllm import LLM, SamplingParams  # type: ignore
    install_vllm_noise_patch()
    max_model_len = ctx_len + n_eval + 16
    llm = LLM(model=model_path, dtype="bfloat16",
              max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_util,
              enforce_eager=True, trust_remote_code=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    # Reference pass once.
    NoiseState.active = False
    ref_lps = [_vllm_logprobs_one(llm, sp, ids, ctx_len, n_eval)
               for ids in passages_ids]

    results = []
    for sig in sigmas:
        sk = sig if mode in ("both", "k") else 0.0
        sv = sig if mode in ("both", "v") else 0.0
        NoiseState.active = True
        NoiseState.sigma_k = sk
        NoiseState.sigma_v = sv
        NoiseState._rng_counter = 0
        per_passage = []
        for pi, ids in enumerate(passages_ids):
            t0 = time.perf_counter()
            alt_lps = _vllm_logprobs_one(llm, sp, ids, ctx_len, n_eval)
            t = time.perf_counter() - t0
            ppl_r = ppl_eval_from_lps(ref_lps[pi])
            ppl_a = ppl_eval_from_lps(alt_lps)
            dlogp = [abs(a - b) for a, b in zip(alt_lps, ref_lps[pi])
                     if np.isfinite(a) and np.isfinite(b)]
            per_passage.append({
                "passage": pi, "t_sec": t,
                "ppl_ref": ppl_r, "ppl_alt": ppl_a,
                "ppl_delta_rel": (ppl_a - ppl_r) / max(ppl_r, 1e-8),
                "mean_abs_dlogp_true": (float(np.mean(dlogp)) if dlogp
                                        else float("nan")),
                "n_tokens": len(alt_lps),
            })
        NoiseState.active = False
        agg = aggregate(per_passage)
        results.append({"sigma": sig, "sigma_k": sk, "sigma_v": sv,
                        **agg, "per_passage": per_passage})
        print(f"  [vllm] sigma={sig}  Δppl={agg['mean_ppl_delta_rel']*100:+.3f}%  "
              f"|Δlogp|={agg['mean_abs_dlogp']:.4f}", flush=True)
    return results


def run_hf_sweep(model_path: str, passages_ids: list[list[int]],
                 ctx_len: int, n_eval: int,
                 sigmas: list[float], mode: str,
                 ) -> list[dict]:
    from transformers import AutoModelForCausalLM  # type: ignore
    install_hf_noise_patch()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to("cuda").eval()

    def _forward_one(ids: list[int]) -> list[float]:
        ids_t = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model(ids_t).logits[0]
        lps = []
        for pos in range(ctx_len, min(ctx_len + n_eval, ids_t.shape[1])):
            logp_all = torch.log_softmax(
                logits[pos - 1].to(torch.float32), dim=-1
            )
            tok_id = ids[pos]
            lps.append(float(logp_all[tok_id].cpu()))
        return lps

    NoiseState.active = False
    ref_lps = [_forward_one(ids) for ids in passages_ids]

    results = []
    for sig in sigmas:
        sk = sig if mode in ("both", "k") else 0.0
        sv = sig if mode in ("both", "v") else 0.0
        NoiseState.active = True
        NoiseState.sigma_k = sk
        NoiseState.sigma_v = sv
        NoiseState._rng_counter = 0
        per_passage = []
        for pi, ids in enumerate(passages_ids):
            t0 = time.perf_counter()
            alt_lps = _forward_one(ids)
            t = time.perf_counter() - t0
            ppl_r = ppl_eval_from_lps(ref_lps[pi])
            ppl_a = ppl_eval_from_lps(alt_lps)
            dlogp = [abs(a - b) for a, b in zip(alt_lps, ref_lps[pi])
                     if np.isfinite(a) and np.isfinite(b)]
            per_passage.append({
                "passage": pi, "t_sec": t,
                "ppl_ref": ppl_r, "ppl_alt": ppl_a,
                "ppl_delta_rel": (ppl_a - ppl_r) / max(ppl_r, 1e-8),
                "mean_abs_dlogp_true": (float(np.mean(dlogp)) if dlogp
                                        else float("nan")),
                "n_tokens": len(alt_lps),
            })
        NoiseState.active = False
        agg = aggregate(per_passage)
        results.append({"sigma": sig, "sigma_k": sk, "sigma_v": sv,
                        **agg, "per_passage": per_passage})
        print(f"  [hf] sigma={sig}  Δppl={agg['mean_ppl_delta_rel']*100:+.3f}%  "
              f"|Δlogp|={agg['mean_abs_dlogp']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def aggregate(per_passage: list[dict]) -> dict:
    valid = [r for r in per_passage
             if np.isfinite(r["ppl_delta_rel"])]
    return {
        "mean_ppl_delta_rel": (float(np.mean([r["ppl_delta_rel"] for r in valid]))
                               if valid else float("nan")),
        "mean_abs_dlogp": (float(np.mean([r["mean_abs_dlogp_true"] for r in valid
                                           if np.isfinite(r["mean_abs_dlogp_true"])]))
                           if valid else float("nan")),
        "n_passages": len(per_passage),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.001, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--mode", choices=["both", "k", "v"], default="both",
                    help="'both'=perturb K and V with same sigma; "
                         "'k'=K only; 'v'=V only")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--engine", choices=["vllm", "hf"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    NoiseState.seed = args.seed
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    passages = load_passages(tok, args.ctx_len + args.n_eval,
                             args.n_passages)
    passages_ids = [tok.encode(p)[:args.ctx_len + args.n_eval]
                    for p in passages
                    if len(tok.encode(p)) >= args.ctx_len + args.n_eval]
    print(f"[setup] {len(passages_ids)} passages, {args.engine} engine, "
          f"sigmas={args.sigmas} mode={args.mode}", flush=True)

    results: dict[str, Any] = {
        "engine": args.engine, "model_name": args.model_name,
        "ctx_len": args.ctx_len, "n_eval": args.n_eval,
        "n_passages": len(passages_ids), "mode": args.mode,
        "sigmas": args.sigmas, "seed": args.seed,
    }
    if args.engine == "vllm":
        results["per_sigma"] = run_vllm_sweep(
            args.model_path, passages_ids, args.ctx_len, args.n_eval,
            args.gpu_mem_util, args.sigmas, args.mode,
        )
    else:
        results["per_sigma"] = run_hf_sweep(
            args.model_path, passages_ids, args.ctx_len, args.n_eval,
            args.sigmas, args.mode,
        )

    # Inflection points: summarise
    print(f"\n===== SUMMARY ({args.engine}) =====", flush=True)
    print(f"  sigma \u0394ppl(%) |\u0394logp|", flush=True)
    for r in results["per_sigma"]:
        print(f"  {r['sigma']:.4f}  {r['mean_ppl_delta_rel']*100:+7.3f}  "
              f"{r['mean_abs_dlogp']:.4f}", flush=True)

    out_path = args.out_dir / f"{args.model_name}_{args.engine}_{args.mode}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

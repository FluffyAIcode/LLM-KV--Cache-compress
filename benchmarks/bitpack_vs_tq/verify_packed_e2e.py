"""End-to-end REAL bit-packed storage on a live model (per-operating-point CR).

Generates with the bit-packed caches on Qwen3-4B and reports the real packed KV
footprint vs bf16 DynamicCache, and verifies pack->unpack is lossless.

!!! NOT A FAIR HEAD-TO-HEAD !!!
The points below (D4/E8 @ Q=38, TurboQuant @ b=4) are at DIFFERENT bit budgets /
quality, so their raw CRs are NOT comparable: TurboQuant b=4 shows a higher CR
ONLY because it is a much more aggressive, much lower-quality point
(|Δppl| ~4.8% vs ~0.2% for KakeyaLattice Q=38). Comparing CR across unmatched
quality is meaningless.

>>> The canonical KakeyaLattice-vs-TurboQuant comparison is ISO-QUALITY (matched
    |Δppl|) and lives in `compare_real_cr.py`. At |Δppl| <= 2% on Qwen3-4B the
    real-byte winners are E8 +7.7% / D4 +5.0% over TurboQuant. <<<

This script is only a sanity check that each packed cache works end-to-end and
hits its expected real CR at its own operating point.
"""
from __future__ import annotations
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kakeyalattice.hf import KakeyaLatticePackedCache, TurboQuantPackedCache


def dyn_bytes(cache):
    total = 0
    for layer in cache.layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if t is not None:
                total += t.element_size() * t.numel()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default="/root/kakeyalattice-test/reports/v1_5_release/bitpack_vs_tq_2026-06-15/packed_e2e.json")
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    L = cfg.num_hidden_layers
    prompt = ("Summarise the core idea of nested-lattice quantisation for KV caches "
              "and why E8 improves on scalar quantisation.")
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    in_len = ids.shape[1]
    gen = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True)

    def run(make_cache, name):
        cache = make_cache()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(ids, past_key_values=cache, **gen)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        txt = tok.decode(out[0][in_len:], skip_special_tokens=True)
        info = {"name": name, "seq": int(out.shape[1]), "time_s": dt,
                "text": txt}
        if hasattr(cache, "kv_storage_bytes"):
            info["kv_bytes"] = int(cache.kv_storage_bytes())
        if hasattr(cache, "packed_pack_unpack_ok"):
            info["lossless"] = bool(cache.packed_pack_unpack_ok())
        del cache, out
        torch.cuda.empty_cache()
        return info

    # bf16 baseline first (for byte reference at same seq len)
    base = run(lambda: DynamicCache(), "bf16_DynamicCache")
    # recompute base bytes via a fresh forward-built DynamicCache of same length
    cacheA = DynamicCache()
    with torch.inference_mode():
        outA = model.generate(ids, past_key_values=cacheA, **gen)
    base_bytes = dyn_bytes(cacheA)
    seqA = int(outA.shape[1]); del cacheA, outA; torch.cuda.empty_cache()

    runs = [base]
    runs.append(run(lambda: KakeyaLatticePackedCache(variant="d4", q_range=38,
                num_hidden_layers=L, head_dim=hd, device=dev), "D4_Q38_packed"))
    runs.append(run(lambda: KakeyaLatticePackedCache(variant="e8", q_range=38,
                num_hidden_layers=L, head_dim=hd, device=dev), "E8_Q38_packed"))
    runs.append(run(lambda: TurboQuantPackedCache(bits_b=4,
                num_hidden_layers=L, head_dim=hd, device=dev), "TurboQuant_b4_packed"))

    print(f"model={args.model} layers={L} head_dim={hd} seq={seqA}")
    print(f"bf16 DynamicCache KV bytes = {base_bytes:,} ({base_bytes/2**20:.2f} MiB)")
    print("NOTE: per-operating-point raw CR — NOT quality-matched. Do not rank "
          "codecs by these numbers. Iso-quality comparison: compare_real_cr.py.")
    print(f"{'cache':<26} {'KV MiB':>9} {'real CR':>9} {'lossless':>9} {'time(s)':>8}")
    rows = []
    for r in runs:
        if "kv_bytes" not in r:
            continue
        cr = base_bytes / r["kv_bytes"]
        rows.append({**r, "real_cr": cr})
        print(f"{r['name']:<26} {r['kv_bytes']/2**20:>9.2f} {cr:>8.3f}x "
              f"{str(r.get('lossless','-')):>9} {r['time_s']:>8.2f}")
    print("\n[sample text — E8 Q=38]\n", next(r['text'][:300] for r in runs if r['name']=='E8_Q38_packed'))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "gpu": torch.cuda.get_device_name(0),
                   "head_dim": hd, "layers": L, "seq": seqA,
                   "bf16_kv_bytes": base_bytes, "runs": rows,
                   "note": ("per-operating-point raw CR, NOT quality-matched; "
                            "ranking codecs by these is meaningless. Iso-quality "
                            "(matched |Dppl|) comparison is in compare_real_cr.py."),
                   "iso_quality_comparison": "benchmarks/bitpack_vs_tq/compare_real_cr.py"},
                  f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()

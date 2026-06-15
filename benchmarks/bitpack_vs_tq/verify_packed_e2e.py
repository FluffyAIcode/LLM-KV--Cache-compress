"""End-to-end REAL bit-packed storage on a live model.

Generates with the bit-packed caches on Qwen3-4B and reports the real packed
KV footprint vs bf16 DynamicCache:
  * KakeyaLatticePackedCache (D4 Q=38)  -> ~2.46x
  * KakeyaLatticePackedCache (E8 Q=38)  -> ~2.42x
  * TurboQuantPackedCache    (b=4)      -> ~3.76x  (lower quality; see iso-ppl)
Also verifies the pack->unpack cycle is lossless (so quality == unpacked cache).
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
                   "bf16_kv_bytes": base_bytes, "runs": rows}, f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()

"""Gemma-4-26B heterogeneous-head_dim check / repro / fix verification.

Gemma-4 uses head_dim=256 (sliding_attention layers) and global_head_dim=512
(full_attention layers). This script:
  1. loads the model (text-only generate),
  2. inspects the per-layer K head_dim from a bf16 DynamicCache,
  3. tries KakeyaLatticePackedCache (E8 Q=38) and reports success/CR/coherence
     or the assertion (pre-fix repro).
"""
from __future__ import annotations
import argparse, json, os, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kakeyalattice.hf import KakeyaLatticePackedCache


def layer_kv_dims(cache):
    dims = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            dims.append(None if k is None else int(k.shape[-1]))
    return dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--out", default="/root/kakeyalattice-test/reports/v1_5_release/gemma4_hetero_headdim_2026-06-15/gemma4_check.json")
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev).eval()
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    L = tcfg.num_hidden_layers
    hd = getattr(tcfg, "head_dim", None)
    ghd = getattr(tcfg, "global_head_dim", None)
    print(f"[cfg] layers={L} head_dim={hd} global_head_dim={ghd}", flush=True)
    print(f"[cfg] layer_types={getattr(tcfg,'layer_types',None)}", flush=True)

    msgs = [{"role": "user", "content": "In one sentence, what is lattice quantization?"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    in_len = ids.shape[1]
    gen = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True)

    report = {"model": args.model, "layers": L, "head_dim": hd, "global_head_dim": ghd}

    # 1) bf16 baseline + per-layer K dims
    cacheA = DynamicCache()
    with torch.inference_mode():
        outA = model.generate(ids, past_key_values=cacheA, **gen)
    dimsA = layer_kv_dims(cacheA)
    base_bytes = sum(
        (layer.keys.element_size()*layer.keys.numel() + layer.values.element_size()*layer.values.numel())
        for layer in cacheA.layers if getattr(layer, "keys", None) is not None)
    textA = tok.decode(outA[0][in_len:], skip_special_tokens=True)
    print(f"[bf16] per-layer K head_dim = {dimsA}", flush=True)
    print(f"[bf16] distinct dims = {sorted(set(d for d in dimsA if d))}", flush=True)
    print(f"[bf16] text: {textA[:160]}", flush=True)
    report["per_layer_kv_dim"] = dimsA
    report["distinct_dims"] = sorted(set(d for d in dimsA if d))
    report["bf16_text"] = textA
    report["bf16_kv_bytes"] = base_bytes
    seqA = int(outA.shape[1]); del cacheA, outA; torch.cuda.empty_cache()

    # 2) packed cache (E8 Q=38)
    try:
        cacheB = KakeyaLatticePackedCache(variant="e8", q_range=38,
                                          num_hidden_layers=L, head_dim=hd or 256, device=dev)
        with torch.inference_mode():
            outB = model.generate(ids, past_key_values=cacheB, **gen)
        textB = tok.decode(outB[0][in_len:], skip_special_tokens=True)
        kb = cacheB.kv_storage_bytes()
        cr = base_bytes / kb if kb else None
        codec_dims = {li: (c.D_shape if c is not None else None)
                      for li, c in enumerate(cacheB._codecs)}
        print(f"[packed] OK seq={int(outB.shape[1])} kv={kb/2**20:.2f}MiB realCR={cr:.3f}x lossless={cacheB.packed_pack_unpack_ok()}", flush=True)
        print(f"[packed] per-layer codec D_shape = {codec_dims}", flush=True)
        print(f"[packed] text: {textB[:160]}", flush=True)
        report.update({"packed_ok": True, "packed_kv_bytes": kb, "packed_real_cr": cr,
                       "packed_text": textB, "codec_dims": codec_dims,
                       "lossless": cacheB.packed_pack_unpack_ok()})
    except Exception as e:
        print(f"[packed] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        report.update({"packed_ok": False, "error": f"{type(e).__name__}: {e}"})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()

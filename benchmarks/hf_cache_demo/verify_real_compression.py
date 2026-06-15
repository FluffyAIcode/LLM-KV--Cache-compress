"""End-to-end REAL-compression verification on a live model + GPU.

Question answered: does KakeyaLatticeQuantizedCache hold *fewer real bytes*
than transformers.DynamicCache during an actual generate(), while still
producing coherent text?

Protocol (identical prompt + greedy decode, same #new tokens for both):
  Run A: DynamicCache (bf16)                -> persistent KV bytes
  Run B: KakeyaLatticeQuantizedCache (int8) -> persistent KV bytes + text
Compares cache.kv_storage_bytes() (B) vs summed bf16 tensor bytes (A) at the
same final sequence length, plus torch.cuda.memory_allocated deltas, plus the
generated text from both for a coherence check.
"""
from __future__ import annotations
import argparse, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kakeyalattice.hf import KakeyaLatticeQuantizedCache


def dyn_bytes(cache):
    total = 0
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None:
                    total += t.element_size() * t.numel()
    else:
        for t in list(getattr(cache, "key_cache", [])) + list(getattr(cache, "value_cache", [])):
            if t is not None and hasattr(t, "element_size"):
                total += t.element_size() * t.numel()
    return total


def storage_devices(qc):
    devs = set()
    for entries in qc._k_quant_entries + qc._v_quant_entries:
        for (q, n, m) in entries:
            devs.add(str(q.device)); 
    return devs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--variant", default="e8")
    ap.add_argument("--q-range", type=int, default=38)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default="/root/kakeyalattice-test/reports/v1_5_release/gpu_compression_test_2026-06-15/qwen3_4b_real_compression.json")
    args = ap.parse_args()

    dev = "cuda"
    print(f"[load] {args.model} on {dev} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    print(f"[model] layers={cfg.num_hidden_layers} head_dim={head_dim} num_kv_heads={nkv}", flush=True)

    prompt = ("Explain, in detail, why nested-lattice quantization achieves a "
              "shaping gain over scalar quantization, and how the E8 lattice "
              "improves on D4 for compressing transformer KV caches.")
    msgs = [{"role": "user", "content": prompt}]
    input_ids = None
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") else enc).to(dev)
    except Exception as e:
        print("[warn] chat_template failed, plain encode:", repr(e), flush=True)
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    in_len = input_ids.shape[1]
    print(f"[input] {in_len} tokens", flush=True)

    gen_kw = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True)

    # ---- Run A: bf16 DynamicCache baseline ----
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    cacheA = DynamicCache()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outA = model.generate(input_ids, past_key_values=cacheA, **gen_kw)
    torch.cuda.synchronize(); tA = time.perf_counter() - t0
    bytesA = dyn_bytes(cacheA)
    textA = tok.decode(outA[0][in_len:], skip_special_tokens=True)
    seqA = outA.shape[1]
    print(f"[A bf16] seq={seqA} kv_bytes={bytesA:,} ({bytesA/2**20:.2f} MiB) time={tA:.2f}s", flush=True)

    del cacheA, outA
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- Run B: KakeyaLatticeQuantizedCache (int8 indices) ----
    cacheB = KakeyaLatticeQuantizedCache(
        variant=args.variant, q_range=args.q_range,
        num_hidden_layers=cfg.num_hidden_layers, head_dim=head_dim,
        device=dev, out_dtype=torch.bfloat16,
    )
    t0 = time.perf_counter()
    with torch.inference_mode():
        outB = model.generate(input_ids, past_key_values=cacheB, **gen_kw)
    torch.cuda.synchronize(); tB = time.perf_counter() - t0
    bytesB = cacheB.kv_storage_bytes()
    textB = tok.decode(outB[0][in_len:], skip_special_tokens=True)
    seqB = outB.shape[1]
    devs = storage_devices(cacheB)
    fired = sum(cacheB.codec_fired_per_layer.values())
    skipped = sum(cacheB.skip_fired_per_layer.values())
    print(f"[B int8] seq={seqB} kv_bytes={bytesB:,} ({bytesB/2**20:.2f} MiB) time={tB:.2f}s", flush=True)
    print(f"[B int8] storage_devices={devs} codec_fired={fired} skip_fired={skipped}", flush=True)

    ratio = bytesA / bytesB if bytesB else float("nan")
    per_vec_expected = (2*head_dim) / (head_dim + 4)
    print("="*70)
    print(f"REAL COMPRESSION RATIO (persistent KV bytes): {ratio:.3f}x")
    print(f"  expected per-vector ceiling at D={head_dim}: {per_vec_expected:.3f}x")
    print(f"  bytes saved: {(bytesA-bytesB):,} ({100*(1-bytesB/bytesA):.1f}%)")
    print("="*70)
    print("[A bf16 text]\n", textA[:400], flush=True)
    print("-"*70)
    print("[B int8 text]\n", textB[:400], flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rep = {
        "model": args.model, "variant": args.variant, "q_range": args.q_range,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_hidden_layers": cfg.num_hidden_layers, "head_dim": head_dim,
        "num_kv_heads": nkv, "input_tokens": in_len, "max_new_tokens": args.max_new,
        "seq_len_final": seqB,
        "bf16_kv_bytes": bytesA, "int8_kv_bytes": bytesB,
        "real_compression_ratio": ratio,
        "per_vector_expected_ratio": per_vec_expected,
        "bytes_saved_pct": 100*(1-bytesB/bytesA),
        "storage_devices": sorted(devs),
        "codec_fired": fired, "skip_fired": skipped,
        "decode_time_bf16_s": tA, "decode_time_int8_s": tB,
        "text_bf16": textA, "text_int8": textB,
    }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()

r"""Real-byte iso-perplexity comparison: KakeyaLattice (D4 & E8, bit-packed)
vs TurboQuant (bit-packed) on a live model.

Answers: with **real bit-packing** (not bit-rate accounting), does KakeyaLattice
still compress more than TurboQuant at a fixed |Δppl| quality target?

Method (apples-to-apples, identical model/text/layers):
  * Quality: run a single teacher-forcing forward over real Gutenberg prose with
    each packed cache as past_key_values; perplexity reflects the compression.
    |Δppl| = mean_p |ppl_codec - ppl_ref| / ppl_ref.
  * Real bytes: the same packed cache's kv_storage_bytes() after the forward
    (bit-packed lattice/scalar buffer + fp16 norm/qmax + exception side-channel).
    real CR = bf16 DynamicCache bytes / packed bytes (matched seq length).
The packed caches decode losslessly, so the measured ppl is exactly the codec's.

Sweeps (head_dim=128): D4 Q, E8 Q, TurboQuant b. Builds an iso-ppl Pareto and
reports, per quality threshold, each codec's best real-CR and the advantage.
"""
from __future__ import annotations
import argparse, json, math, os, re, time, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kakeyalattice.hf import KakeyaLatticePackedCache, TurboQuantPackedCache

GUTENBERG_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice


def load_passages(tok, ctx_len, n_passages, cache_path="/root/gutenberg_1342.txt"):
    if os.path.exists(cache_path):
        text = open(cache_path, encoding="utf-8", errors="ignore").read()
    else:
        text = urllib.request.urlopen(GUTENBERG_URL, timeout=30).read().decode("utf-8", "ignore")
        open(cache_path, "w", encoding="utf-8").write(text)
    # strip Gutenberg header/footer
    m = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* END OF", text)
    if m:
        text = text[:m.start()]
    text = re.sub(r"\s+", " ", text).strip()
    ids = tok(text, return_tensors="pt").input_ids[0]
    # take n non-overlapping chunks from the middle of the book
    start = 2000
    chunks = []
    for i in range(n_passages):
        s = start + i * ctx_len
        chunk = ids[s:s + ctx_len]
        if chunk.numel() < ctx_len:
            break
        chunks.append(chunk.unsqueeze(0))
    return chunks


def dyn_bytes(cache):
    total = 0
    for layer in cache.layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if t is not None:
                total += t.element_size() * t.numel()
    return total


@torch.inference_mode()
def ppl_and_bytes(model, input_ids, make_cache, device):
    cache = make_cache()
    out = model(input_ids.to(device), past_key_values=cache, use_cache=True)
    logits = out.logits[:, :-1].float()
    labels = input_ids[:, 1:].to(device)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    ppl = math.exp(min(20.0, loss.item()))
    kb = cache.kv_storage_bytes() if hasattr(cache, "kv_storage_bytes") else dyn_bytes(cache)
    del cache, out, logits
    torch.cuda.empty_cache()
    return ppl, int(kb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--n-passages", type=int, default=4)
    ap.add_argument("--d4-q", default="4,6,10,15,22,38,76,152")
    ap.add_argument("--e8-q", default="4,6,10,15,22,38,76,152")
    ap.add_argument("--tq-b", default="3,4,5,6,7,8")
    ap.add_argument("--out", default="/root/kakeyalattice-test/reports/v1_5_release/bitpack_vs_tq_2026-06-15/qwen3_4b_real_cr.json")
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    L = cfg.num_hidden_layers
    passages = load_passages(tok, args.ctx_len, args.n_passages)
    print(f"model={args.model} layers={L} head_dim={hd} passages={len(passages)} ctx={args.ctx_len}", flush=True)

    # bf16 reference
    ref_ppls, ref_bytes = [], []
    for p in passages:
        ppl, kb = ppl_and_bytes(model, p, lambda: DynamicCache(), dev)
        ref_ppls.append(ppl); ref_bytes.append(kb)
    ref_ppl = sum(ref_ppls) / len(ref_ppls)
    print(f"[bf16] mean ppl={ref_ppl:.3f}  KV={sum(ref_bytes)/len(ref_bytes)/2**20:.1f} MiB", flush=True)

    points = []
    for q in [int(x) for x in args.d4_q.split(",")]:
        points.append(("d4", q, lambda q=q: KakeyaLatticePackedCache(
            variant="d4", q_range=q, num_hidden_layers=L, head_dim=hd, device=dev)))
    for q in [int(x) for x in args.e8_q.split(",")]:
        points.append(("e8", q, lambda q=q: KakeyaLatticePackedCache(
            variant="e8", q_range=q, num_hidden_layers=L, head_dim=hd, device=dev)))
    for b in [int(x) for x in args.tq_b.split(",")]:
        points.append(("tq", b, lambda b=b: TurboQuantPackedCache(
            bits_b=b, num_hidden_layers=L, head_dim=hd, device=dev)))

    results = []
    for fam, param, mk in points:
        ppls, crs = [], []
        for i, p in enumerate(passages):
            ppl, kb = ppl_and_bytes(model, p, mk, dev)
            ppls.append(ppl); crs.append(ref_bytes[i] / kb)
        mean_ppl = sum(ppls) / len(ppls)
        dppl = sum(abs(pp - ref_ppls[i]) / ref_ppls[i] for i, pp in enumerate(ppls)) / len(ppls)
        real_cr = sum(crs) / len(crs)
        rec = {"family": fam, "param": param, "mean_ppl": mean_ppl,
               "abs_rel_delta_ppl": dppl, "real_cr": real_cr}
        results.append(rec)
        print(f"  {fam} {param:>4}: ppl={mean_ppl:7.3f} |Δppl|={dppl*100:6.3f}%  realCR={real_cr:.3f}x", flush=True)

    # iso-ppl Pareto (record the winning operating point per codec family)
    pareto = {}
    for T in [0.005, 0.01, 0.02, 0.05]:
        row = {}
        for fam in ("d4", "e8", "tq"):
            feas = [r for r in results if r["family"] == fam and r["abs_rel_delta_ppl"] <= T]
            best = max(feas, key=lambda r: r["real_cr"], default=None)
            row[fam] = {"real_cr": best["real_cr"], "param": best["param"]} if best else None
        pareto[f"{T*100:.1f}%"] = row

    def _cr(cell):
        return cell["real_cr"] if cell else None

    print("\n=== iso-ppl REAL-byte compression ratio (winning Q / b annotated) ===")
    print(f"{'|Δppl|≤':>8} {'D4':>14} {'E8':>14} {'TurboQuant':>14} {'D4 vs TQ':>10} {'E8 vs TQ':>10}")
    for T, row in pareto.items():
        d4, e8, tq = _cr(row["d4"]), _cr(row["e8"]), _cr(row["tq"])
        def adv(x, y):
            return f"{(x/y-1)*100:+.1f}%" if (x and y) else "—"
        def cell(c, lbl):
            return f"{c['real_cr']:.2f}x({lbl}={c['param']})" if c else "oor"
        print(f"{T:>8} {cell(row['d4'],'Q'):>14} {cell(row['e8'],'Q'):>14} "
              f"{cell(row['tq'],'b'):>14} {adv(d4,tq):>10} {adv(e8,tq):>10}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "gpu": torch.cuda.get_device_name(0),
                   "head_dim": hd, "layers": L, "ctx_len": args.ctx_len,
                   "n_passages": len(passages), "ref_ppl": ref_ppl,
                   "points": results, "iso_ppl_pareto_real_cr": pareto}, fh, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

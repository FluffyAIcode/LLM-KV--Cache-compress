"""Test how many layers we can compress before downstream quality dies.
If quality survives single-layer but fails all-layers, the problem is
error compounding through depth, not single-layer codec fidelity."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import sys
sys.path.insert(0, 'benchmarks')
from e2e_ppl_validation import rust_roundtrip, load_wikitext_passages

tok = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16, attn_implementation="eager")
model.eval()

passages = load_wikitext_passages(tok, min_tokens=1088, n_passages=1)
ids = tok(passages[0], return_tensors="pt")["input_ids"][:, :1024]
cont = tok(passages[0], return_tensors="pt")["input_ids"][:, 1024:1088]

cache_ref = DynamicCache(config=model.config)
with torch.no_grad():
    _ = model(input_ids=ids, past_key_values=cache_ref, use_cache=True)
ref_copy = copy.deepcopy(cache_ref)
with torch.no_grad():
    logits_ref = model(input_ids=cont, past_key_values=ref_copy, use_cache=True).logits

def run_with_compressed_layers(layer_set: set, bit_width=2, pca_method="randomized", vr=0.95):
    cache_alt = DynamicCache(config=model.config)
    for i, layer in enumerate(cache_ref.layers):
        if layer.keys is None or layer.keys.numel() == 0:
            continue
        if i not in layer_set:
            cache_alt.layers[i].update(layer.keys.clone(), layer.values.clone(), 0)
            continue
        k = layer.keys
        v = layer.values
        bsz, nkv, seq, hd = k.shape
        k_flat = k.to(torch.float32).cpu().numpy().reshape(-1, hd)
        v_flat = v.to(torch.float32).cpu().numpy().reshape(-1, hd)
        n_block = (k_flat.shape[0] // 512) * 512
        if n_block == 0:
            cache_alt.layers[i].update(k.clone(), v.clone(), 0)
            continue
        k_dec, _ = rust_roundtrip(k_flat[:n_block], block_size=512, bit_width=bit_width,
                                   rsvd_target_rank=hd//2, metric="inner_product",
                                   share_basis=False, pca_method=pca_method, variance_ratio=vr)
        v_dec, _ = rust_roundtrip(v_flat[:n_block], block_size=512, bit_width=bit_width,
                                   rsvd_target_rank=hd//2, metric="mse",
                                   share_basis=True, pca_method=pca_method, variance_ratio=vr)
        k_full = np.concatenate([k_dec, k_flat[n_block:]]) if n_block < k_flat.shape[0] else k_dec
        v_full = np.concatenate([v_dec, v_flat[n_block:]]) if n_block < v_flat.shape[0] else v_dec
        k_r = torch.from_numpy(k_full.copy()).reshape(bsz, nkv, seq, hd).to(k.dtype)
        v_r = torch.from_numpy(v_full.copy()).reshape(bsz, nkv, seq, hd).to(v.dtype)
        cache_alt.layers[i].update(k_r, v_r, 0)
    with torch.no_grad():
        logits_alt = model(input_ids=cont, past_key_values=cache_alt, use_cache=True).logits
    # metrics
    sr = logits_ref[..., :-1, :].float()
    sa = logits_alt[..., :-1, :].float()
    lab = cont[..., 1:]
    nll_r = F.cross_entropy(sr.reshape(-1, sr.size(-1)), lab.reshape(-1))
    nll_a = F.cross_entropy(sa.reshape(-1, sa.size(-1)), lab.reshape(-1))
    top1_r = sr.argmax(-1)
    top1_a = sa.argmax(-1)
    ppl_r = float(torch.exp(nll_r).item())
    ppl_a = float(torch.exp(nll_a).item())
    agree = float((top1_r == top1_a).float().mean().item())
    return ppl_r, ppl_a, agree

n_layers = len([l for l in cache_ref.layers if l.keys is not None])
print(f"total non-null layers: {n_layers}")

for label, bw, pca, vr in [
    ("paper default b=2 rsvd vr=0.95",  2, "randomized", 0.95),
    ("v1.2 b=3 exact vr=0.95",           3, "exact",      0.95),
    ("max fidelity b=4 exact vr=1.0",    4, "exact",      1.0),
]:
    print(f"\n=== {label} ===")
    for k in [0, 1, 2, 4, 8, 16, 24]:
        layer_set = set(range(k))
        ppl_r, ppl_a, agree = run_with_compressed_layers(layer_set, bit_width=bw, pca_method=pca, vr=vr)
        print(f"  first {k:2d} layers: ppl_ref={ppl_r:.2f}  ppl_alt={ppl_a:.2f}  Δ={((ppl_a/ppl_r)-1)*100:+.1f}%  top1={agree*100:.1f}%")

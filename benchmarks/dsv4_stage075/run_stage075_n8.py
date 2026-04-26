r"""Stage 0.75 — **n=8 passage** non-Gaussian audit of V4-Flash KV with
TRAINED weights.

Purpose
-------
Closes Caveat 1 of ``reports/v1_5_release/dsv4_stage075/FINDINGS.md``:

    "One passage, one layer of each type. V4-Flash has 21 c4a layers
    + 20 c128a layers + 3 SWA/MTP layers; we tested one of each.
    Per-layer statistics can vary across layers; for a paper-grade
    claim we'd need to audit all 43 layers (scaling this script is
    cheap on H200 once shards are pre-fetched)."

This harness keeps the same three representative V4 layers (0 = SWA,
2 = c4a, 3 = c128a) — per-layer expansion is a separate, larger PR —
but replaces the single passage with **n=8 semantically diverse
WikiText-style passages**.  For each passage we re-run the V4 forward,
recompute the non-Gaussian audit, roundtrip through the codec suite,
and aggregate the per-stream metrics with mean / std / 95% CI.

Output JSON shape
-----------------
    {
      "generated_at": ...,
      "config": { ... seed + n_passages + q_values + ... },
      "per_passage": [
        { "passage_id": 0, "results": <stage-0.75 per-passage block> },
        ...
      ],
      "aggregate_by_stream": {
        "<stream>": {
          "audit": { "<metric>": {"mean","std","ci95_hw","n"}, ... },
          "codecs": {
            "<codec_name>": {
              "rel_mse": {...},
              "cos_sim": {...},
              "bits_per_vector": int,
            }, ...
          }
        }, ...
      }
    }

Running
-------
``` bash
python3 benchmarks/dsv4_stage075/run_stage075_n8.py \
    --host-model Qwen/Qwen2-0.5B \
    --seqlen 2048 --batch-size 1 \
    --n-passages 8 \
    --q-values 10,38 \
    --hf-home /workspace/.hf_home \
    --out reports/v1_5_release/dsv4_stage075/stage075_n8.json
```
End-to-end wall time on 2x H200 with shards cached: ~2 minutes
(1 passage ≈ 15s; n=8 ≈ 120s incl. codec instantiation once).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage0_5"))
sys.path.insert(0, str(REPO / "benchmarks" / "dsv4_stage075"))

from dsv4_kv_generator import (  # type: ignore[import-not-found]
    DSV4Compressor,
    DSV4FlashArchConfig,
    DSV4MainKVProjection,
)
from dsv4_weight_loader import (  # type: ignore[import-not-found]
    inject_weights_into_compressor,
    inject_weights_into_main_kv,
    load_single_layer_weights,
    load_v4_shard_paths,
)
from run_dsv4_stage0_5 import (  # type: ignore[import-not-found]
    compute_cosine,
    compute_rel_mse,
    fp8_baseline_roundtrip,
    non_gaussian_audit,
)

from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU  # type: ignore


# ---------------------------------------------------------------------------
# 8 semantically diverse WikiText-style passages
#
# Chosen deliberately across disciplines to broaden the empirical
# support of the audit (math, history, biology, economics, physics,
# linguistics, music, engineering).  Each is ~1200 tokens of English
# prose after x8 replication in the host-hidden extractor.
# ---------------------------------------------------------------------------
PASSAGES: List[str] = [
    # 0. Topology / algebraic topology (original Stage 0.75 passage)
    "The history of topology is deeply intertwined with the emergence of "
    "modern mathematics itself. In the late nineteenth century, Henri "
    "Poincaré's study of the three-body problem led him to formulate the "
    "first rigorous ideas about the topology of manifolds. Betti numbers, "
    "originally defined by Enrico Betti in the 1870s as counts of "
    "independent cycles, were gradually reformulated by Poincaré and later "
    "by Emmy Noether into the algebraic language of homology groups. ",
    # 1. Renaissance history
    "The Italian Renaissance emerged from city-state prosperity in the "
    "fourteenth century, transforming European art, architecture, and "
    "scholarship. In Florence, patrons such as the Medici family funded "
    "workshops where Donatello, Brunelleschi, and Masaccio developed "
    "perspective, contrapposto, and chiaroscuro. Humanist scholars including "
    "Petrarch and Bruni revived classical Latin and Greek, while printers "
    "such as Aldus Manutius popularised portable editions of ancient texts. ",
    # 2. Molecular biology
    "The central dogma of molecular biology describes the unidirectional "
    "flow of sequence information from DNA to RNA to protein. Transcription "
    "begins when RNA polymerase binds a promoter upstream of a gene, unwinds "
    "the double helix, and synthesises a messenger RNA copy from the "
    "template strand. Messenger RNA is then translated at the ribosome, "
    "where transfer RNAs matched to codons deliver amino acids that are "
    "joined by peptide bonds to form the polypeptide chain. ",
    # 3. Macroeconomics
    "Modern macroeconomic theory distinguishes between short-run demand "
    "fluctuations and long-run supply-side growth. Keynesian models treat "
    "aggregate demand as the primary driver of output over business-cycle "
    "horizons, justifying counter-cyclical fiscal and monetary policy. In "
    "the long run, however, output is determined by capital accumulation, "
    "labour force growth, and total factor productivity; Solow's growth "
    "model formalises this with a Cobb-Douglas aggregate production function. ",
    # 4. Quantum mechanics
    "Quantum mechanics emerged in the early twentieth century to resolve "
    "phenomena that classical physics could not explain: blackbody radiation, "
    "the photoelectric effect, and the stability of atomic spectra. Planck's "
    "quantum hypothesis in 1900 introduced discrete energy packets; Einstein "
    "extended this to photons in 1905. Bohr's 1913 atomic model quantised "
    "angular momentum, and by 1925 Heisenberg and Schrödinger had formulated "
    "matrix mechanics and wave mechanics, later unified by Dirac and von Neumann. ",
    # 5. Linguistics / syntax
    "Generative grammar, pioneered by Noam Chomsky in the 1950s, treats the "
    "syntax of a natural language as a formal system generating the set of "
    "all grammatical sentences. Phrase-structure rules, later refined into "
    "X-bar theory and then the Minimalist Program, describe how hierarchical "
    "constituents combine through operations such as Merge and Move. "
    "Universal Grammar posits innate constraints shared across languages, "
    "explaining the rapid acquisition of complex grammar by children. ",
    # 6. Music theory
    "Western tonal harmony rests on the hierarchical organisation of "
    "consonance and dissonance within a key. The major-minor tonal system, "
    "codified by Rameau in the eighteenth century, treats the tonic triad "
    "as the point of resolution and the dominant-tonic cadence as the "
    "principal closure. Functional harmony classifies chords as tonic, "
    "predominant, or dominant according to their role in voice-leading "
    "toward the tonic, and modulations follow the circle of fifths. ",
    # 7. Structural engineering
    "Reinforced-concrete design combines the compressive strength of "
    "concrete with the tensile capacity of embedded steel reinforcement. "
    "Eurocode 2 and ACI 318 define partial safety factors, strain-limit "
    "design, and serviceability checks that govern the reinforcement layout "
    "of beams, slabs, and columns. For seismic loads, capacity design "
    "principles ensure plastic hinges form in ductile flexural members "
    "rather than brittle shear failures at connections. ",
]


def load_host_hidden_for_passage(
    model, tok, passage_text: str,
    seqlen: int, batch_size: int,
    target_hidden_size: int, device: str,
    projection_W: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B, seqlen, target_hidden_size] bf16 hiddens for a single passage.

    The projection matrix is passed in and shared across passages so the
    n=8 runs all see the same 2560→4096 (or 896→4096) linear map.
    """
    prompt = passage_text * 8
    ids = tok(
        [prompt] * batch_size,
        return_tensors="pt", padding="max_length",
        truncation=True, max_length=seqlen,
    )["input_ids"].to(device)

    with torch.inference_mode():
        hidden = model.get_input_embeddings()(ids).to(torch.bfloat16)
    native = hidden.shape[-1]
    if native != target_hidden_size:
        assert projection_W is not None, "projection_W required for native!=target"
        hidden = torch.nn.functional.linear(hidden, projection_W)
    return hidden


def build_projection_W(native: int, target: int, device: str) -> torch.Tensor:
    """Same fixed seed as Stage 0.75 single-passage run so n=8 is a
    superset of n=1 numerically."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
        torch.manual_seed(20260425)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(20260425)
        W = (torch.randn(target, native, device=device, dtype=torch.bfloat16)
             * native ** -0.5)
    return W


def build_and_load_dsv4_blocks(
    shard_paths: Dict[int, str], device: str, config: DSV4FlashArchConfig,
) -> Dict[str, object]:
    blocks: Dict[str, object] = {}
    # SWA layer 0
    params_layer0 = load_single_layer_weights(shard_paths[2], layer_id=0)
    swa_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 0})
    blocks["main_kv_swa"] = DSV4MainKVProjection(swa_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_swa"], params_layer0, layer_id=0, device=device)
    # c4a layer 2
    params_layer2 = load_single_layer_weights(shard_paths[4], layer_id=2)
    c4a_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 4})
    blocks["main_kv_c4a"] = DSV4MainKVProjection(c4a_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_c4a"], params_layer2, layer_id=2, device=device)
    blocks["compressor_c4a"] = DSV4Compressor(c4a_cfg, compress_ratio=4, rotate=False, device=device)
    inject_weights_into_compressor(blocks["compressor_c4a"], params_layer2, layer_id=2, device=device)
    # c128a layer 3
    params_layer3 = load_single_layer_weights(shard_paths[5], layer_id=3)
    c128a_cfg = DSV4FlashArchConfig(**{**config.__dict__, "compress_ratio": 128})
    blocks["main_kv_c128a"] = DSV4MainKVProjection(c128a_cfg, device=device)
    inject_weights_into_main_kv(blocks["main_kv_c128a"], params_layer3, layer_id=3, device=device)
    blocks["compressor_c128a"] = DSV4Compressor(c128a_cfg, compress_ratio=128, rotate=False, device=device)
    inject_weights_into_compressor(blocks["compressor_c128a"], params_layer3, layer_id=3, device=device)
    return blocks


def run_trio(blocks: Dict[str, object], hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.inference_mode():
        sliding_window_kv = blocks["main_kv_swa"](hidden)
        csa_pool_kv = blocks["compressor_c4a"](hidden)
        hca_pool_kv = blocks["compressor_c128a"](hidden)
    return {
        "sliding_window_kv": sliding_window_kv,
        "csa_pool_kv_ratio4": csa_pool_kv,
        "hca_pool_kv_ratio128": hca_pool_kv,
    }


def evaluate_stream(name: str, kv: torch.Tensor, codecs: List) -> Dict:
    result = {
        "stream": name,
        "shape": list(kv.shape),
        "dtype": str(kv.dtype),
        "audit": non_gaussian_audit(kv),
        "codecs": {},
    }
    for codec_name, c in codecs:
        kv_hat = c.roundtrip(kv.float())
        if kv.is_cuda:
            torch.cuda.synchronize()
        result["codecs"][codec_name] = {
            "bits_per_vector": int(c.bits_per_token_per_head),
            "rel_mse": compute_rel_mse(kv, kv_hat),
            "cos_sim": compute_cosine(kv, kv_hat),
        }
    fp8_hat = fp8_baseline_roundtrip(kv)
    bits_per_vec = kv.shape[-1] * 8 + (kv.shape[-1] // 64) * 16
    result["codecs"]["fp8_per64_baseline"] = {
        "bits_per_vector": bits_per_vec,
        "rel_mse": compute_rel_mse(kv, fp8_hat),
        "cos_sim": compute_cosine(kv, fp8_hat),
    }
    return result


# ---------------------------------------------------------------------------
# Aggregation helpers — mean / std / 95% CI half-width via Student t
# ---------------------------------------------------------------------------

# Student-t 95% critical values for small n (two-sided, α=0.05).
# Looked up once from a standard table — no scipy dependency needed.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}


def _t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    # Fall back to nearest larger tabulated df (conservative).
    for k in sorted(_T95.keys()):
        if k >= df:
            return _T95[k]
    return 1.960  # large-n normal approximation


def _agg(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci95_hw": float("nan"), "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95_hw": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    hw = _t95(n - 1) * se
    return {"mean": mean, "std": std, "ci95_hw": hw, "n": n}


def aggregate_per_passage(per_passage: List[Dict]) -> Dict[str, Dict]:
    """Given a list of per-passage reports (each `results_by_stream` list),
    produce mean/std/CI per stream per metric."""
    # Collect stream -> metric -> [values]
    stream_names = [r["stream"] for r in per_passage[0]["results"]]
    audit_keys = list(per_passage[0]["results"][0]["audit"].keys())
    codec_names = list(per_passage[0]["results"][0]["codecs"].keys())

    out: Dict[str, Dict] = {}
    for stream in stream_names:
        entry = {"audit": {}, "codecs": {}}
        # audit
        for k in audit_keys:
            vals = []
            for pp in per_passage:
                for r in pp["results"]:
                    if r["stream"] == stream:
                        v = r["audit"].get(k)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
            if vals:
                entry["audit"][k] = _agg(vals)
        # codecs
        for cn in codec_names:
            rel_mses: List[float] = []
            cos_sims: List[float] = []
            bits_pv = None
            for pp in per_passage:
                for r in pp["results"]:
                    if r["stream"] == stream:
                        c = r["codecs"].get(cn, {})
                        if "rel_mse" in c:
                            rel_mses.append(float(c["rel_mse"]))
                        if "cos_sim" in c:
                            cos_sims.append(float(c["cos_sim"]))
                        if "bits_per_vector" in c:
                            bits_pv = int(c["bits_per_vector"])
            entry["codecs"][cn] = {
                "bits_per_vector": bits_pv,
                "rel_mse": _agg(rel_mses),
                "cos_sim": _agg(cos_sims),
            }
        # E8/FP8 ratio per passage -> aggregate
        ratios_by_codec: Dict[str, List[float]] = {}
        fp8_per_pp: List[float] = []
        for pp in per_passage:
            for r in pp["results"]:
                if r["stream"] != stream:
                    continue
                fp8 = r["codecs"].get("fp8_per64_baseline", {}).get("rel_mse")
                if fp8 is None or fp8 == 0:
                    continue
                fp8_per_pp.append(float(fp8))
                for cn, c in r["codecs"].items():
                    if cn == "fp8_per64_baseline":
                        continue
                    rel = c.get("rel_mse")
                    if rel is None:
                        continue
                    ratios_by_codec.setdefault(cn, []).append(float(rel) / float(fp8))
        entry["ratios_vs_fp8"] = {cn: _agg(vals) for cn, vals in ratios_by_codec.items()}
        out[stream] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host-model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-passages", type=int, default=8)
    p.add_argument("--q-values", default="10,38")
    p.add_argument("--enable-e8", action="store_true", default=True)
    p.add_argument("--out", default="reports/v1_5_release/dsv4_stage075/stage075_n8.json")
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/.hf_home"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seqlen % 128 != 0:
        raise ValueError(f"seqlen must be multiple of 128 (HCA ratio); got {args.seqlen}")
    if args.n_passages > len(PASSAGES):
        raise ValueError(f"n_passages={args.n_passages} exceeds the {len(PASSAGES)} built-in passages")

    q_values = [int(q) for q in args.q_values.split(",") if q.strip()]
    print(f"[config] host={args.host_model} seqlen={args.seqlen} batch={args.batch_size} "
          f"n_passages={args.n_passages} q_values={q_values} device={device}", flush=True)

    # 1. V4-Flash shards
    shard_paths = load_v4_shard_paths(args.hf_home, "deepseek-ai/DeepSeek-V4-Flash")
    for needed in (2, 4, 5):
        if needed not in shard_paths:
            raise FileNotFoundError(
                f"Shard {needed} not found in HF cache at {args.hf_home}. "
                f"Re-run the download script before running Stage 0.75."
            )
    print(f"[shards] found {len(shard_paths)} V4 shards; needed: 2, 4, 5", flush=True)

    # 2. V4 blocks
    cfg = DSV4FlashArchConfig(simulate_fp8=True)
    t0 = time.perf_counter()
    blocks = build_and_load_dsv4_blocks(shard_paths, device=device, config=cfg)
    t1 = time.perf_counter()
    print(f"[load] V4 blocks loaded in {t1-t0:.2f}s", flush=True)

    # 3. Host model loaded once
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[host] loading {args.host_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.host_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.host_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    native_hidden = model.config.hidden_size
    W_proj = build_projection_W(native_hidden, cfg.hidden_size, device) \
        if native_hidden != cfg.hidden_size else None

    # 4. Codecs built ONCE (they're passage-independent)
    D = cfg.head_dim
    codecs = []
    for q in q_values:
        codecs.append((f"v14_d4_Q{q}", V14KakeyaZamirLatticeGPU(D=D, q_range=q, device=device)))
    if args.enable_e8:
        for q in q_values:
            codecs.append((f"v15_e8_Q{q}", V15KakeyaZamirE8GPU(D=D, q_range=q, device=device)))
    for name, c in codecs:
        print(f"[codec] {name}: bits={c.bits_per_token_per_head}", flush=True)

    # 5. Iterate passages
    per_passage: List[Dict] = []
    for i in range(args.n_passages):
        print(f"\n[passage {i}/{args.n_passages}] running…", flush=True)
        tpp0 = time.perf_counter()
        hidden = load_host_hidden_for_passage(
            model, tok, PASSAGES[i],
            args.seqlen, args.batch_size,
            target_hidden_size=cfg.hidden_size, device=device,
            projection_W=W_proj,
        )
        streams = run_trio(blocks, hidden)
        results = [evaluate_stream(n, kv, codecs) for n, kv in streams.items()]
        tpp1 = time.perf_counter()
        per_passage.append({
            "passage_id": i,
            "wall_time_sec": tpp1 - tpp0,
            "results": results,
        })
        # Print a one-line summary per passage
        for r in results:
            e8_q38 = r["codecs"].get("v15_e8_Q38", {}).get("rel_mse")
            fp8 = r["codecs"].get("fp8_per64_baseline", {}).get("rel_mse")
            ratio = (e8_q38 / fp8) if (e8_q38 and fp8) else float("nan")
            print(f"  [passage {i}] {r['stream']:<22s} E8Q38/FP8={ratio:.3f}  kurt={r['audit']['excess_kurtosis_abs']:.2f}",
                  flush=True)

    # 6. Aggregate
    aggregate = aggregate_per_passage(per_passage)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "host_model": args.host_model,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "n_passages": args.n_passages,
            "q_values": q_values,
            "enable_e8": args.enable_e8,
            "simulate_fp8": cfg.simulate_fp8,
            "device": device,
            "dsv4_config": {
                "hidden_size": cfg.hidden_size,
                "head_dim": cfg.head_dim,
                "qk_rope_head_dim": cfg.qk_rope_head_dim,
                "v4_layers_used": {0: "SWA", 2: "c4a", 3: "c128a"},
                "weight_source": "deepseek-ai/DeepSeek-V4-Flash safetensors shards 2/4/5",
                "trained_weights": True,
            },
            "passages_sha_first64": [
                p[:64].replace("\n", " ") for p in PASSAGES[: args.n_passages]
            ],
        },
        "per_passage": per_passage,
        "aggregate_by_stream": aggregate,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {out}", flush=True)

    # Human-readable summary
    print()
    print("=" * 96)
    print(f"AGGREGATE over n={args.n_passages} passages — mean ± 95% CI half-width")
    print("=" * 96)
    for stream, entry in aggregate.items():
        print(f"\n[{stream}]")
        # codec rel-MSE summary
        print(f"  {'codec':<22s}  {'bits':>5s}  {'rel-MSE':>22s}  {'ratio vs FP8':>20s}")
        for cn, c in entry["codecs"].items():
            rm = c["rel_mse"]
            bits = c["bits_per_vector"]
            bits_s = f"{bits:>5d}" if bits is not None else f"{'?':>5s}"
            ratio = entry["ratios_vs_fp8"].get(cn)
            if ratio is None or cn == "fp8_per64_baseline":
                ratio_s = f"{'—':>20s}"
            else:
                ratio_s = f"{ratio['mean']:.3f} ± {ratio['ci95_hw']:.3f}"
                ratio_s = f"{ratio_s:>20s}"
            print(f"  {cn:<22s}  {bits_s}  {rm['mean']:>9.3e} ± {rm['ci95_hw']:>9.3e}  {ratio_s}")
        # audit summary (three key gates)
        a = entry["audit"]
        for k in ("excess_kurtosis_abs", "isotropy_variance_ratio",
                  "hadamard_post_variance_ratio", "rms_wasserstein2_over_sigma_per_dim"):
            if k in a:
                v = a[k]
                print(f"    audit {k:<38s}  {v['mean']:>12.4g} ± {v['ci95_hw']:>9.4g}")


if __name__ == "__main__":
    main()
